"""Two-step cascade decay: pi0 -> gamma + A' -> chi + chibar.

    Step 1  Pi0ToDarkphoton:       pi0 -> gamma + A'       (pi0 decay vertex)
    Step 2  DarkphotonToDarkmatter: A' -> chi + chibar      (displaced vertex)

    Uses SIREN's Injector

    python plot_Pi0DarkphotonDarkmatter_distributions.py \
        --m_med 0.12 --alpha_D 0.5 --eps 1.8e-4 --n 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import siren

from ccm_siren_alps.channels_Pi0DarkphotonDarkmatter import (
    DarkphotonToDarkmatter,
    Pi0ToDarkphoton,
)

DEFAULT_PI0_CSV = Path(
    "/ceph/submit/data/user/y/yumeng1/workspaces/dataLinks/target_sim/ccm_pi0/pi0_list_CCM200.csv"
)
PI0_TOTAL_WIDTH_GEV = 7.8e-9

# CLI                                                                

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_PI0_CSV, help="Input pi0 CSV")
    parser.add_argument("--m_med", type=float, required=True, help="Dark photon mass [GeV]")
    parser.add_argument("--alpha_D", type=float, required=True, help="Dark fine-structure constant")
    parser.add_argument("--eps", type=float, required=True, help="Kinetic mixing epsilon")
    parser.add_argument("--n", type=int, default=1000, help="Number of events")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--detector", default="CCM")
    parser.add_argument("--detector-model", type=Path, default=None)
    return parser


def load_detector_model(args):
    if args.detector_model is not None:
        detector_model = siren.detector.DetectorModel()
        detector_model.LoadDetectorModel(str(args.detector_model))
        return detector_model
    return siren.utilities.load_detector(args.detector)


def default_outdir(m_med: float, alpha_D: float) -> Path:
    return Path(f"outputs/pi0_darkphoton_dm_med{m_med:g}")


def write_siren_csv_copy(input_csv: Path, outdir: Path) -> Path:
    output_csv = outdir / "pi0_list_siren_input.csv"
    with input_csv.open() as fin, output_csv.open("w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fout.write(line)
    return output_csv


def build_external_distribution(args):
    siren_csv = write_siren_csv_copy(args.csv, args.outdir)
    return siren.distributions.PrimaryExternalDistribution(str(siren_csv))

# Physics helpers                                                    

def p_abs(px: float, py: float, pz: float) -> float:
    return math.sqrt(px * px + py * py + pz * pz)


def cos_angle(p1: list[float], p2: list[float]) -> float:
    n1 = p_abs(*p1)
    n2 = p_abs(*p2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    dot = sum(p1[j] * p2[j] for j in range(3))
    return max(-1.0, min(1.0, dot / (n1 * n2)))

# Injector-based simulation                                          

def generate_rows(
    pi0_dist,
    detector_model,
    step1: Pi0ToDarkphoton,
    step2: DarkphotonToDarkmatter,
    n_events: int,
    seed: int,
) -> tuple[list[dict[str, float]], list[float]]:
    """Generate events using the SIREN Injector.

    The injector handles pi0 -> gamma + A' (step1) as the primary process,
    then automatically propagates A' to a physically-sampled displaced vertex
    via SecondaryPhysicalVertexDistribution before sampling A' -> chi + chibar
    (step2) as the secondary process.
    """
    pt = siren.dataclasses.Particle.ParticleType

    def stop_before_non_dark_photon(datum, i):
        secondary_type = datum.record.signature.secondary_types[i]
        return secondary_type != step1.dark_photon_type

    injector = siren.injection.Injector()
    injector.number_of_events = n_events
    injector.detector_model = detector_model
    injector.primary_type = pt.Pi0
    injector.primary_interactions = [step1]
    injector.primary_injection_distributions = [pi0_dist]
    injector.secondary_interactions = {step1.dark_photon_type: [step2]}
    injector.secondary_injection_distributions = {
        step1.dark_photon_type: [siren.distributions.SecondaryPhysicalVertexDistribution()]
    }
    injector.stopping_condition = stop_before_non_dark_photon
    injector.seed = seed
    injector._Injector__initialize_injector()

    # Secondary momentum index lookups (consistent with GetPossibleSignatures order)
    sig1 = step1.GetPossibleSignatures()[0]
    sec_types_1 = list(sig1.secondary_types)
    aprime_idx_1 = sec_types_1.index(step1.dark_photon_type)
    gamma_idx_1 = sec_types_1.index(pt.Gamma)

    sig2 = step2.GetPossibleSignatures()[0]
    sec_types_2 = list(sig2.secondary_types)
    chi_idx_2 = sec_types_2.index(step2.chi_type)
    chibar_idx_2 = sec_types_2.index(step2.chibar_type)

    rows: list[dict[str, float]] = []
    gen_times: list[float] = []
    missing_pi0 = 0
    missing_aprime = 0
    tree_shapes: dict[tuple[int, ...], int] = {}

    for i in range(n_events):
        t0 = time.time()
        event = injector.generate_event()
        gen_times.append(time.time() - t0)

        # Locate the pi0-decay and A'-decay nodes in the interaction tree
        pi0_datum = None
        aprime_datum = None
        for datum in event.tree:
            ptype = datum.record.signature.primary_type
            if ptype == pt.Pi0:
                pi0_datum = datum
            elif ptype == step1.dark_photon_type:
                aprime_datum = datum

        tree_shape = tuple(int(datum.record.signature.primary_type) for datum in event.tree)
        tree_shapes[tree_shape] = tree_shapes.get(tree_shape, 0) + 1

        if pi0_datum is None or aprime_datum is None:
            if pi0_datum is None:
                missing_pi0 += 1
            if aprime_datum is None:
                missing_aprime += 1
            continue

        # ---- Step 1 kinematics (pi0 -> gamma + A') ----
        pi0_p4 = list(pi0_datum.record.primary_momentum)
        pi0_vertex = list(pi0_datum.record.interaction_vertex)
        aprime_p4 = list(pi0_datum.record.secondary_momenta[aprime_idx_1])
        gamma_p4 = list(pi0_datum.record.secondary_momenta[gamma_idx_1])

        # ---- Step 2 kinematics (A' -> chi + chibar at displaced vertex) ----
        aprime_vertex = list(aprime_datum.record.interaction_vertex)
        chi_p4 = list(aprime_datum.record.secondary_momenta[chi_idx_2])
        chibar_p4 = list(aprime_datum.record.secondary_momenta[chibar_idx_2])

        # ---- Derived quantities ----
        pi0_p = p_abs(*pi0_p4[1:4])
        aprime_p3 = aprime_p4[1:4]
        aprime_p = p_abs(*aprime_p3)
        aprime_E = aprime_p4[0]
        aprime_mass = step2.m_dark_photon

        chi_p3 = chi_p4[1:4]
        chibar_p3 = chibar_p4[1:4]
        chi_p = p_abs(*chi_p3)
        chibar_p = p_abs(*chibar_p3)
        chi_E = chi_p4[0]
        chibar_E = chibar_p4[0]

        # Decay length: distance between pi0 vertex and A' decay vertex
        decay_length = p_abs(
            aprime_vertex[0] - pi0_vertex[0],
            aprime_vertex[1] - pi0_vertex[1],
            aprime_vertex[2] - pi0_vertex[2],
        )

        # Invariant mass of chi+chibar pair (should reconstruct m_A')
        pair_px = chi_p3[0] + chibar_p3[0]
        pair_py = chi_p3[1] + chibar_p3[1]
        pair_pz = chi_p3[2] + chibar_p3[2]
        inv_mass_sq = (chi_E + chibar_E) ** 2 - pair_px**2 - pair_py**2 - pair_pz**2
        inv_mass = math.sqrt(max(inv_mass_sq, 0.0))

        rows.append({
            "event_index": i,
            # pi0 primary
            "pi0_E": pi0_p4[0],
            "pi0_p": pi0_p,
            "pi0_vertex_x": pi0_vertex[0],
            "pi0_vertex_y": pi0_vertex[1],
            "pi0_vertex_z": pi0_vertex[2],
            # gamma from pi0 decay
            "gamma_E": gamma_p4[0],
            "gamma_px": gamma_p4[1],
            "gamma_py": gamma_p4[2],
            "gamma_pz": gamma_p4[3],
            # dark photon (A')
            "aprime_E": aprime_E,
            "aprime_px": aprime_p4[1],
            "aprime_py": aprime_p4[2],
            "aprime_pz": aprime_p4[3],
            "aprime_p": aprime_p,
            "aprime_kinetic_E": aprime_E - aprime_mass,
            "aprime_cos_beam": max(-1.0, min(1.0, aprime_p4[3] / max(aprime_p, 1e-30))),
            "aprime_cos_pi0": cos_angle(aprime_p3, pi0_p4[1:4]),
            "aprime_beta": aprime_p / max(aprime_E, 1e-30),
            "aprime_gamma_factor": aprime_E / max(aprime_mass, 1e-30),
            "aprime_decay_length": decay_length,
            # A' decay vertex (displaced, set by SecondaryPhysicalVertexDistribution)
            "aprime_vx": aprime_vertex[0],
            "aprime_vy": aprime_vertex[1],
            "aprime_vz": aprime_vertex[2],
            # chi
            "chi_E": chi_E,
            "chi_px": chi_p3[0],
            "chi_py": chi_p3[1],
            "chi_pz": chi_p3[2],
            "chi_p": chi_p,
            "chi_kinetic_E": chi_E - step2.m_dark_matter,
            "chi_cos_beam": max(-1.0, min(1.0, chi_p3[2] / max(chi_p, 1e-30))),
            "chi_cos_aprime": cos_angle(chi_p3, aprime_p3),
            # chibar
            "chibar_E": chibar_E,
            "chibar_px": chibar_p3[0],
            "chibar_py": chibar_p3[1],
            "chibar_pz": chibar_p3[2],
            "chibar_p": chibar_p,
            "chibar_kinetic_E": chibar_E - step2.m_dark_matter,
            "chibar_cos_beam": max(-1.0, min(1.0, chibar_p3[2] / max(chibar_p, 1e-30))),
            "chibar_cos_aprime": cos_angle(chibar_p3, aprime_p3),
            # chi+chibar pair
            "chi_chibar_inv_mass": inv_mass,
            "chi_chibar_cos_opening": cos_angle(chi_p3, chibar_p3),
        })

    if not rows:
        shape_summary = ", ".join(
            f"{shape}: {count}" for shape, count in sorted(tree_shapes.items(), key=lambda item: item[1], reverse=True)
        )
        raise RuntimeError(
            "No events generated with both pi0 and A' decay nodes. "
            f"missing_pi0={missing_pi0}, missing_aprime={missing_aprime}, "
            f"tree_shapes={shape_summary or 'none'}"
        )
    return rows, gen_times

# Output helpers                                                     

_CSV_KEYS = [
    "event_index",
    "pi0_E", "pi0_p", "pi0_vertex_x", "pi0_vertex_y", "pi0_vertex_z",
    "gamma_E", "gamma_px", "gamma_py", "gamma_pz",
    "aprime_E", "aprime_px", "aprime_py", "aprime_pz", "aprime_p",
    "aprime_kinetic_E", "aprime_cos_beam", "aprime_cos_pi0",
    "aprime_beta", "aprime_gamma_factor", "aprime_decay_length",
    "aprime_vx", "aprime_vy", "aprime_vz",
    "chi_E", "chi_px", "chi_py", "chi_pz", "chi_p",
    "chi_kinetic_E", "chi_cos_beam", "chi_cos_aprime",
    "chibar_E", "chibar_px", "chibar_py", "chibar_pz", "chibar_p",
    "chibar_kinetic_E", "chibar_cos_beam", "chibar_cos_aprime",
    "chi_chibar_inv_mass", "chi_chibar_cos_opening",
]


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.12g}" for k in _CSV_KEYS})


def column(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if math.isfinite(row[key])]


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def stats(values: list[float]) -> dict[str, float]:
    values = finite(values)
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def histogram_plot(
    values: list[float], path: Path, title: str, xlabel: str, bins: int = 80
) -> None:
    values = finite(values)
    if not values:
        raise RuntimeError(f"Cannot plot empty histogram: {title}")
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    ax.hist(values, bins=bins, histtype="stepfilled", color="tab:blue", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def two_panel_plot(
    vals1: list[float], vals2: list[float],
    path: Path, title: str,
    xlabel1: str, xlabel2: str,
    label1: str = "chi", label2: str = "chibar",
    bins: int = 80,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=150)
    for ax, vals, lbl, xlabel in (
        (ax1, vals1, label1, xlabel1),
        (ax2, vals2, label2, xlabel2),
    ):
        vals = finite(vals)
        ax.hist(vals, bins=bins, histtype="stepfilled", color="tab:orange", alpha=0.75)
        ax.set_title(f"{title} — {lbl}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# Main                                                                

def main() -> None:
    args = build_parser().parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args.m_med, args.alpha_D)
    args.outdir.mkdir(parents=True, exist_ok=True)

    detector_model = load_detector_model(args)

    step1 = Pi0ToDarkphoton(
        m_dark_photon=args.m_med,
        epsilon=args.eps,
        pi0_total_width=PI0_TOTAL_WIDTH_GEV,
    )
    step2 = DarkphotonToDarkmatter(
        m_dark_photon=args.m_med,
        alpha_D=args.alpha_D,
    )

    pi0_dist = build_external_distribution(args)
    rows, gen_times = generate_rows(
        pi0_dist, detector_model, step1, step2, args.n, args.seed
    )

    write_rows(args.outdir / "cascade_sample.csv", rows)

    # ---- Dark photon plots ----
    aprime_plots = {
        "aprime_energy.png": (
            "Dark photon lab energy", "E_{A'} [GeV]", column(rows, "aprime_E")
        ),
        "aprime_kinetic_energy.png": (
            "Dark photon lab kinetic energy", "K_{A'} [GeV]", column(rows, "aprime_kinetic_E")
        ),
        "aprime_momentum.png": (
            "Dark photon lab momentum", "|p_{A'}| [GeV]", column(rows, "aprime_p")
        ),
        "aprime_cos_beam.png": (
            "Dark photon cos(theta) w.r.t. beam", "cos(theta_z)", column(rows, "aprime_cos_beam")
        ),
        "aprime_cos_pi0.png": (
            "Dark photon cos(theta) w.r.t. pi0 (lab)", "cos(theta_{A', pi0})", column(rows, "aprime_cos_pi0")
        ),
        "aprime_beta.png": (
            "Dark photon beta", "beta_{A'}", column(rows, "aprime_beta")
        ),
        "aprime_gamma.png": (
            "Dark photon Lorentz gamma", "gamma_{A'}", column(rows, "aprime_gamma_factor")
        ),
        "aprime_decay_length.png": (
            "Dark photon lab decay length", "L_{A'} [m]", column(rows, "aprime_decay_length")
        ),
    }
    for filename, (title, xlabel, values) in aprime_plots.items():
        histogram_plot(values, args.outdir / filename, title, xlabel)

    # ---- A' decay vertex ----
    vertex_plots = {
        "aprime_vertex_x.png": ("A' decay vertex x", "x [m]", column(rows, "aprime_vx")),
        "aprime_vertex_y.png": ("A' decay vertex y", "y [m]", column(rows, "aprime_vy")),
        "aprime_vertex_z.png": ("A' decay vertex z", "z [m]", column(rows, "aprime_vz")),
    }
    for filename, (title, xlabel, values) in vertex_plots.items():
        histogram_plot(values, args.outdir / filename, title, xlabel)

    # ---- Dark matter (chi / chibar) ----
    two_panel_plot(
        column(rows, "chi_E"), column(rows, "chibar_E"),
        args.outdir / "dm_energy.png",
        "Dark matter lab energy", "E_chi [GeV]", "E_chibar [GeV]",
    )
    two_panel_plot(
        column(rows, "chi_p"), column(rows, "chibar_p"),
        args.outdir / "dm_momentum.png",
        "Dark matter lab momentum", "|p_chi| [GeV]", "|p_chibar| [GeV]",
    )
    two_panel_plot(
        column(rows, "chi_kinetic_E"), column(rows, "chibar_kinetic_E"),
        args.outdir / "dm_kinetic_energy.png",
        "Dark matter kinetic energy", "K_chi [GeV]", "K_chibar [GeV]",
    )
    two_panel_plot(
        column(rows, "chi_cos_beam"), column(rows, "chibar_cos_beam"),
        args.outdir / "dm_cos_beam.png",
        "Dark matter cos(theta) w.r.t. beam", "cos(theta_z) chi", "cos(theta_z) chibar",
    )
    two_panel_plot(
        column(rows, "chi_cos_aprime"), column(rows, "chibar_cos_aprime"),
        args.outdir / "dm_cos_aprime.png",
        "Dark matter cos(theta) w.r.t. A' (lab)", "cos(theta_{chi, A'})", "cos(theta_{chibar, A'})",
    )

    # ---- chi+chibar pair ----
    histogram_plot(
        column(rows, "chi_chibar_inv_mass"),
        args.outdir / "chi_chibar_inv_mass.png",
        f"Reconstructed chi+chibar invariant mass (expect m_A'={args.m_med:.4g} GeV)",
        "m_{chi chibar} [GeV]",
    )
    histogram_plot(
        column(rows, "chi_chibar_cos_opening"),
        args.outdir / "chi_chibar_cos_opening.png",
        "chi-chibar opening angle",
        "cos(theta_{chi, chibar})",
    )

    # ---- pi0 vertex ----
    histogram_plot(
        column(rows, "pi0_vertex_z"),
        args.outdir / "pi0_vertex_z.png",
        "pi0 decay vertex z",
        "z [m]",
    )

    # ---- Summary ----
    gamma_width_pi0 = step1._partial_width()
    gamma_width_aprime = step2._partial_width()
    hbar_c = 0.197326980e-15  # GeV*m
    ctau_aprime_m = hbar_c / gamma_width_aprime if gamma_width_aprime > 0.0 else float("inf")

    summary = {
        "input_csv": str(args.csv),
        "requested_events": args.n,
        "generated_rows": len(rows),
        "seed": args.seed,
        "detector": args.detector if args.detector_model is None else None,
        "m_dark_photon_GeV": args.m_med,
        "m_dark_matter_GeV": step2.m_dark_matter,
        "epsilon": args.eps,
        "alpha_D": args.alpha_D,
        # Step 1
        "pi0_partial_width_GeV": gamma_width_pi0,
        "pi0_to_aprime_branching_ratio": gamma_width_pi0 / PI0_TOTAL_WIDTH_GEV,
        # Step 2
        "aprime_partial_width_GeV": gamma_width_aprime,
        "aprime_ctau_m": ctau_aprime_m,
        "aprime_decay_length_m": stats(column(rows, "aprime_decay_length")),
        # Dark photon kinematics
        "aprime_energy_GeV": stats(column(rows, "aprime_E")),
        "aprime_cos_beam": stats(column(rows, "aprime_cos_beam")),
        "aprime_gamma_factor": stats(column(rows, "aprime_gamma_factor")),
        # Dark matter kinematics
        "chi_energy_GeV": stats(column(rows, "chi_E")),
        "chi_cos_beam": stats(column(rows, "chi_cos_beam")),
        "chi_chibar_inv_mass_GeV": stats(column(rows, "chi_chibar_inv_mass")),
        "mean_generation_time_s": statistics.fmean(gen_times),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Output: {args.outdir}")
    print(f"Events: {len(rows)} / {args.n}")
    print(f"BR(pi0->gamma A'):          {summary['pi0_to_aprime_branching_ratio']:.4e}")
    print(f"Gamma(A'->chi chibar) [GeV]: {gamma_width_aprime:.4e}")
    print(f"c*tau(A') [m]:               {ctau_aprime_m:.4e}")
    print(f"Mean A' decay length [m]:    {summary['aprime_decay_length_m']['mean']:.4e}")
    print(f"Mean chi E [GeV]:            {summary['chi_energy_GeV']['mean']:.4g}")


if __name__ == "__main__":
    main()
