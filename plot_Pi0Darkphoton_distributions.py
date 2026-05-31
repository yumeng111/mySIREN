"""pi0 -> gamma + dark photon with SIREN and plot daughter kinematics.

    PrimaryExternalDistribution(csv) -> Injector -> Pi0Darkphoton.SampleFinalState

    python tools/plot_Pi0Darkphoton_distributions.py --m_med 0.12 --eps 1.8e-4 --n 1000
    --m 3e-3 --eps 1.6e-5
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

from ccm_siren_alps.channels_Pi0Darkphoton import Pi0Darkphoton


DEFAULT_PI0_CSV = Path(
    "/ceph/submit/data/user/y/yumeng1/workspaces/dataLinks/target_sim/ccm_pi0/pi0_list_CCM200.csv"
)
PI0_TOTAL_WIDTH_GEV = 7.8e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_PI0_CSV, help="Input pi0 CSV")
    parser.add_argument("--m_med", type=float, required=True, help="Dark photon mass in GeV")
    parser.add_argument("--eps", type=float, required=True, help="Kinetic mixing epsilon")
    parser.add_argument("--n", type=int, default=1000, help="Number of SIREN events to generate")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--detector",
        default="CCM",
        help="Detector name for siren.utilities.load_detector when --detector-model is not set",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=None,
        help="Optional detector densities.dat path. If omitted, uses siren.utilities.load_detector.",
    )
    return parser


def load_detector_model(args):
    if args.detector_model is not None:
        detector_model = siren.detector.DetectorModel()
        detector_model.LoadDetectorModel(str(args.detector_model))
        return detector_model
    return siren.utilities.load_detector(args.detector)


def default_outdir(m_med: float) -> Path:
    return Path(f"outputs/pi0_darkphoton_siren_med{m_med:g}")


def build_pi0_decay(m_dark_photon: float, epsilon: float) -> Pi0Darkphoton:
    return Pi0Darkphoton(
        m_dark_photon=m_dark_photon,
        epsilon=epsilon,
        pi0_total_width=PI0_TOTAL_WIDTH_GEV,
    )


def write_siren_csv_copy(input_csv: Path, outdir: Path) -> Path:
    """
    PrimaryExternalDistribution expects the first line to be the column header.
    The CCM pi0 list starts with a comment line describing units, so pass SIREN
    a cleaned copy with comments and blank lines removed.
    """
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


def p_abs(px: float, py: float, pz: float) -> float:
    return math.sqrt(px * px + py * py + pz * pz)


def row_from_record(record, dark_photon_type) -> dict[str, float] | None:
    pt = siren.dataclasses.Particle.ParticleType
    if record.signature.primary_type != pt.Pi0:
        return None

    secondary_types = list(record.signature.secondary_types)
    if dark_photon_type not in secondary_types:
        return None

    idx = secondary_types.index(dark_photon_type)
    dark_p4 = list(record.secondary_momenta[idx])
    gamma_idx = secondary_types.index(pt.Gamma) if pt.Gamma in secondary_types else None
    gamma_p4 = list(record.secondary_momenta[gamma_idx]) if gamma_idx is not None else [math.nan] * 4

    pi0_p4 = list(record.primary_momentum)
    pi0_p = p_abs(pi0_p4[1], pi0_p4[2], pi0_p4[3])
    dark_p = p_abs(dark_p4[1], dark_p4[2], dark_p4[3])
    cos_parent = (
        (pi0_p4[1] * dark_p4[1] + pi0_p4[2] * dark_p4[2] + pi0_p4[3] * dark_p4[3])
        / max(pi0_p * dark_p, 1.0e-30)
    )
    cos_beam = max(min(dark_p4[3] / max(dark_p, 1.0e-30), 1.0), -1.0)
    vertex = list(record.interaction_vertex)
    dark_mass = list(record.secondary_masses)[idx]

    return {
        "pi0_E": pi0_p4[0],
        "pi0_px": pi0_p4[1],
        "pi0_py": pi0_p4[2],
        "pi0_pz": pi0_p4[3],
        "pi0_p": pi0_p,
        "vertex_x": vertex[0],
        "vertex_y": vertex[1],
        "vertex_z": vertex[2],
        "dark_type": int(dark_photon_type),
        "dark_m": dark_mass,
        "dark_E": dark_p4[0],
        "dark_px": dark_p4[1],
        "dark_py": dark_p4[2],
        "dark_pz": dark_p4[3],
        "dark_p": dark_p,
        "dark_kinetic_E": dark_p4[0] - dark_mass,
        "dark_cos_beam": cos_beam,
        "dark_cos_lab_parent": max(min(cos_parent, 1.0), -1.0),
        "dark_beta": dark_p / max(dark_p4[0], 1.0e-30),
        "dark_gamma": dark_p4[0] / max(dark_mass, 1.0e-30),
        "gamma_E": gamma_p4[0],
        "gamma_px": gamma_p4[1],
        "gamma_py": gamma_p4[2],
        "gamma_pz": gamma_p4[3],
    }
    return None


def generate_dark_photon_rows(
    pi0_dist,
    detector_model,
    pi0_decay: Pi0Darkphoton,
    n_events: int,
    seed: int,
) -> tuple[list[dict[str, float]], list[float]]:
    """Sample external pi0 records and decay them through SIREN record methods.

    The full SIREN Injector requires a separate VertexPositionDistribution, but
    PrimaryExternalDistribution is not one in the installed SIREN. So use the same SIREN distribution 
    and decay objects directly:
    sample a PrimaryDistributionRecord, finalize it to an InteractionRecord,
    call SampleFinalState, then finalize the daughter momenta.
    """
    pt = siren.dataclasses.Particle.ParticleType
    rand = siren.utilities.SIREN_random(seed)
    interactions = siren.interactions.InteractionCollection(pt.Pi0, [pi0_decay])
    signature = pi0_decay.GetPossibleSignatures()[0]
    rows: list[dict[str, float]] = []
    #how much wall-clock time the computer spent generating one event.
    gen_times: list[float] = []
    for i in range(n_events):
        t0 = time.time()
        primary_record = siren.dataclasses.PrimaryDistributionRecord(pt.Pi0)
        pi0_dist.Sample(rand, detector_model, interactions, primary_record)

        record = siren.dataclasses.InteractionRecord()
        primary_record.finalize(record)
        record.signature = signature

        decay_record = siren.dataclasses.CrossSectionDistributionRecord(record)
        pi0_decay.SampleFinalState(decay_record, rand)
        decay_record.finalize(record)

        gen_times.append(time.time() - t0)
        row = row_from_record(record, pi0_decay.dark_photon_type)
        if row is not None:
            row["event_index"] = i
            rows.append(row)
    if not rows:
        raise RuntimeError("No generated events contained a dark-photon secondary")
    return rows, gen_times


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    keys = [
        "event_index",
        "pi0_E",
        "pi0_px",
        "pi0_py",
        "pi0_pz",
        "pi0_p",
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "dark_type",
        "dark_m",
        "dark_E",
        "dark_px",
        "dark_py",
        "dark_pz",
        "dark_p",
        "dark_kinetic_E",
        "dark_cos_beam",
        "dark_cos_lab_parent",
        "dark_beta",
        "dark_gamma",
        "gamma_E",
        "gamma_px",
        "gamma_py",
        "gamma_pz",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.12g}" for key in keys})


def column(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if math.isfinite(row[key])]


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def histogram(values: list[float], bins: int) -> tuple[list[int], list[float]]:
    values = finite(values)
    if not values:
        raise RuntimeError("Cannot plot an empty histogram")
    v_min = min(values)
    v_max = max(values)
    if v_min == v_max:
        delta = abs(v_min) * 0.05 + 0.5
        v_min -= delta
        v_max += delta
    width = (v_max - v_min) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(int((value - v_min) / width), bins - 1)
        counts[index] += 1
    edges = [v_min + i * width for i in range(bins + 1)]
    return counts, edges


def histogram_plot(values: list[float], path: Path, title: str, xlabel: str, bins: int = 80) -> None:
    values = finite(values)
    if not values:
        raise RuntimeError(f"Cannot plot an empty histogram: {title}")

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    ax.hist(values, bins=bins, histtype="stepfilled", color="tab:blue", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stats(values: list[float]) -> dict[str, float]:
    values = finite(values)
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args.m_med)
    args.outdir.mkdir(parents=True, exist_ok=True)

    detector_model = load_detector_model(args)
    pi0_decay = build_pi0_decay(args.m_med, args.eps)
    pi0_dist = build_external_distribution(args)
    rows, gen_times = generate_dark_photon_rows(
        pi0_dist,
        detector_model,
        pi0_decay,
        args.n,
        args.seed,
    )

    write_rows(args.outdir / "dark_photon_siren_sample.csv", rows)

    plots = {
        "pi0_energy.png": ("SIREN input pi0 energy", "E_pi0 [GeV]", column(rows, "pi0_E")),
        "dark_photon_energy.png": ("SIREN dark photon lab energy", "E_dark photon [GeV]", column(rows, "dark_E")),
        "dark_photon_kinetic_energy.png": (
            "SIREN dark photon lab kinetic energy",
            "K_dark photon [GeV]",
            column(rows, "dark_kinetic_E"),
        ),
        "dark_photon_momentum.png": (
            "SIREN dark photon lab momentum",
            "|p_dark photon| [GeV]",
            column(rows, "dark_p"),
        ),
        "dark_photon_cos_beam.png": (
            "SIREN dark photon lab cos angle to +z",
            "cos(theta_z)",
            column(rows, "dark_cos_beam"),
        ),
        "dark_photon_cos_parent_lab.png": (
            "SIREN dark photon lab cos angle to pi0",
            "cos(theta_lab dark photon, pi0)",
            column(rows, "dark_cos_lab_parent"),
        ),
        "dark_photon_beta.png": ("SIREN dark photon beta", "beta", column(rows, "dark_beta")),
        "dark_photon_gamma.png": ("SIREN dark photon gamma", "gamma", column(rows, "dark_gamma")),
        "vertex_z.png": ("SIREN pi0 decay vertex z", "z [m]", column(rows, "vertex_z")),
    }
    for filename, (title, xlabel, values) in plots.items():
        histogram_plot(values, args.outdir / filename, title, xlabel)

    summary = {
        "input_csv": str(args.csv),
        "requested_events": args.n,
        "generated_dark_photon_rows": len(rows),
        "seed": args.seed,
        "detector": args.detector if args.detector_model is None else None,
        "detector_model": str(args.detector_model) if args.detector_model is not None else None,
        "m_dark_photon_GeV": args.m_med,
        "epsilon": args.eps,
        "dark_photon_particle_type": int(pi0_decay.dark_photon_type),
        "pi0_mass_GeV": pi0_decay.pi0_mass(),
        "pi0_to_dark_photon_branching_ratio": pi0_decay._branching_ratio(),
        "pi0_to_dark_photon_partial_width_GeV": pi0_decay._partial_width(),
        "mean_generation_time_s": statistics.fmean(gen_times),
        "dark_photon_energy_GeV": stats(column(rows, "dark_E")),
        "dark_photon_cos_beam": stats(column(rows, "dark_cos_beam")),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote SIREN daughter sample and plots to: {args.outdir}")
    print(f"Generated dark photon rows: {len(rows)} / {args.n}")
    print(f"BR(pi0 -> gamma dark photon): {summary['pi0_to_dark_photon_branching_ratio']:.6e}")
    print(f"Partial width [GeV]: {summary['pi0_to_dark_photon_partial_width_GeV']:.6e}")
    print(f"Mean dark photon E [GeV]: {summary['dark_photon_energy_GeV']['mean']:.6g}")


if __name__ == "__main__":
    main()
