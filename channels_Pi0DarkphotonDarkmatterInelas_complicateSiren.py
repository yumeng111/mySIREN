from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

import siren

try:
    from .channels_Pi0DarkphotonDarkmatter import Pi0ToDarkphoton, DarkphotonToDarkmatter
    from .constants import TWO_PI
    from .kinematics import LorentzVector, Vector3, lorentz_boost
except ImportError:
    from ccm_siren_alps.channels_Pi0DarkphotonDarkmatter import Pi0ToDarkphoton, DarkphotonToDarkmatter
    from ccm_siren_alps.constants import TWO_PI
    from ccm_siren_alps.kinematics import LorentzVector, Vector3, lorentz_boost


Ar40 = siren.dataclasses.Particle.ParticleType.Ar40Nucleus
Gamma = siren.dataclasses.Particle.ParticleType.Gamma
N4 = siren.dataclasses.Particle.ParticleType.N4
N4Bar = siren.dataclasses.Particle.ParticleType.N4Bar
MARLEY_MEV_TO_SIREN_GEV = 1.0e-3
# Ar40Nucleus is not in SIREN's GetParticleMass map; detector fills record.target_mass.
AR40_MASS_GEV = 37.2


@dataclass(frozen=True)
class ArgonTransition:
    gamma_energy: float
    branching_fraction: float
    final_level: int


@dataclass(frozen=True)
class ArgonLevel:
    index: int
    excitation_energy: float
    two_j: int
    parity: str
    transitions: tuple[ArgonTransition, ...]


@dataclass(frozen=True)
class GammaRecord:
    transition: ArgonTransition
    four_momentum: LorentzVector
    direction: np.ndarray
    position: np.ndarray
    time: float
    parent_rest_energy: float


@dataclass(frozen=True)
class CascadeResult:
    initial_level: ArgonLevel
    final_nucleus: LorentzVector
    gammas: tuple[GammaRecord, ...]


@dataclass(frozen=True)
class Pi0DarkPhotonDarkMatterInelasticChain:
    pi0_to_darkphoton: Pi0ToDarkphoton
    darkphoton_to_darkmatter: DarkphotonToDarkmatter
    darkmatter_argon_inelastic: "DarkMatterArgonInelastic"


def _uniform(random_like, low: float = 0.0, high: float = 1.0) -> float:
    if hasattr(random_like, "Uniform"):
        return float(random_like.Uniform(low, high))
    if hasattr(random_like, "uniform"):
        return float(random_like.uniform(low, high))
    return float(np.random.uniform(low, high))


def _safe_unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        if fallback is None:
            return np.asarray([0.0, 0.0, 1.0], dtype=float)
        return np.asarray(fallback, dtype=float)
    return np.asarray(vec / norm, dtype=float)


def _find_secondary_indices_by_type(record, particle_type) -> list[int]:
    return [
        idx for idx, sec_type in enumerate(getattr(record.signature, "secondary_types", []))
        if sec_type == particle_type
    ]


def _lv_array(lv: LorentzVector) -> np.ndarray:
    return np.asarray([lv.p0, lv.p1, lv.p2, lv.p3], dtype=float)


def _default_argon_level_file() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "marley" / "data" / "structure" / "Ar.dat"
        if candidate.exists():
            return candidate
    return here.parent.parent / "marley" / "data" / "structure" / "Ar.dat"


def load_argon_levels(argon_level_file: str | Path = "Ar.dat") -> dict[int, ArgonLevel]:
    """Read the Z=18, A=40 block from a MARLEY nuclear-structure file."""
    path = Path(argon_level_file).expanduser()
    if not path.exists() and str(argon_level_file) == "Ar.dat":
        path = _default_argon_level_file()
    if not path.exists():
        raise FileNotFoundError(f"Could not find argon level file: {argon_level_file}")

    tokens = path.read_text().split()
    cursor = 0
    while cursor < len(tokens):
        z = int(tokens[cursor])
        a = int(tokens[cursor + 1])
        number_of_levels = int(tokens[cursor + 2])
        cursor += 3
        levels: dict[int, ArgonLevel] = {}
        for index in range(number_of_levels):
            excitation = float(tokens[cursor]) * MARLEY_MEV_TO_SIREN_GEV
            two_j = int(tokens[cursor + 1])
            parity = tokens[cursor + 2]
            number_of_transitions = int(tokens[cursor + 3])
            cursor += 4
            transitions = []
            for _ in range(number_of_transitions):
                gamma_energy = float(tokens[cursor]) * MARLEY_MEV_TO_SIREN_GEV
                branching_fraction = float(tokens[cursor + 1])
                final_level = int(tokens[cursor + 2])
                cursor += 3
                transitions.append(ArgonTransition(gamma_energy, branching_fraction, final_level))
            levels[index] = ArgonLevel(index, excitation, two_j, parity, tuple(transitions))
        if z == 18 and a == 40:
            _validate_argon_levels(levels)
            return levels
    raise ValueError(f"No Z=18, A=40 block found in {path}")


def _validate_argon_levels(levels: dict[int, ArgonLevel]) -> None:
    if 0 not in levels:
        raise ValueError("Ar40 level table has no ground state")
    for level in levels.values():
        for transition in level.transitions:
            if transition.branching_fraction < 0.0:
                raise ValueError(f"Negative branching fraction at level {level.index}")
            if transition.final_level not in levels:
                raise ValueError(f"Transition from level {level.index} points to unknown level {transition.final_level}")
            if transition.final_level >= level.index:
                raise ValueError(f"Transition from level {level.index} does not point to a lower level")
        if level.index != 0 and level.transitions:
            total = sum(t.branching_fraction for t in level.transitions)
            if total <= 0.0:
                raise ValueError(f"Level {level.index} has no positive transition weight")


def sample_weighted_index(weights, random) -> int:
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0.0):
        raise ValueError("Cannot sample from negative weights")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot sample from empty or zero weights")
    return int(np.searchsorted(np.cumsum(weights), _uniform(random) * total, side="right"))


def cascade_multiplicity_options(levels: dict[int, ArgonLevel], level_index: int, memo=None) -> set[int]:
    if memo is None:
        memo = {}
    if level_index in memo:
        return memo[level_index]
    if level_index == 0:
        memo[level_index] = {0}
        return memo[level_index]
    options: set[int] = set()
    for transition in levels[level_index].transitions:
        if transition.branching_fraction > 0.0:
            for tail in cascade_multiplicity_options(levels, transition.final_level, memo):
                options.add(1 + tail)
    memo[level_index] = options
    return options


def cascade_multiplicity_probability(
    levels: dict[int, ArgonLevel],
    level_index: int,
    gamma_count: int,
    memo: dict[tuple[int, int], float] | None = None,
) -> float:
    """Probability that a cascade from level_index emits exactly gamma_count gammas.

    Branching fractions are treated as relative weights and normalized at each level. 
    Ground state contributes only to gamma_count == 0.
    """
    if memo is None:
        memo = {}
    key = (level_index, int(gamma_count))
    if key in memo:
        return memo[key]
    if gamma_count < 0:
        memo[key] = 0.0
        return 0.0
    if level_index == 0:
        result = 1.0 if gamma_count == 0 else 0.0
        memo[key] = result
        return result

    level = levels[level_index]
    positive = [t for t in level.transitions if t.branching_fraction > 0.0]
    if not positive:
        memo[key] = 0.0
        return 0.0
    br_sum = sum(t.branching_fraction for t in positive)
    total = 0.0
    for transition in positive:
        p_branch = transition.branching_fraction / br_sum
        total += p_branch * cascade_multiplicity_probability(
            levels, transition.final_level, gamma_count - 1, memo
        )
    memo[key] = float(total)
    return memo[key]


def sigma_placeholder(E_chi: float, level: ArgonLevel) -> float:
    """Placeholder level-resolved partial cross section in cm^2.

    current model: levels with excitation energy below the incoming DM total energy E_chi
    receive a weight proportional to excitation energy.
    """
    if level.index == 0:
        return 0.0
    energy = max(float(E_chi), 0.0)
    if energy <= level.excitation_energy:
        return 0.0
    return 1.0e-40 * max(level.excitation_energy, 1.0e-12)


def time_placeholder(level: ArgonLevel) -> float:
    """Placeholder proper lifetime in SIREN time units."""
    return 0.0


def kinematically_accessible_levels(
    E_chi: float,
    m_chi: float,
    target_mass: float,
    levels: dict[int, ArgonLevel],
) -> list[ArgonLevel]:
    s = m_chi * m_chi + target_mass * target_mass + 2.0 * target_mass * E_chi
    accessible = []
    for level in levels.values():
        if level.index == 0:
            continue
        excited_mass = target_mass + level.excitation_energy
        if s >= (m_chi + excited_mass) ** 2:
            accessible.append(level)
    return accessible


def levels_capable_of_gamma_count(levels: dict[int, ArgonLevel], gamma_count: int) -> list[ArgonLevel]:
    return [
        level for level in levels.values()
        if level.index != 0 and cascade_multiplicity_probability(levels, level.index, gamma_count) > 0.0
    ]


def partial_cross_section_weights(
    E_chi: float,
    m_chi: float,
    target_mass: float,
    levels: dict[int, ArgonLevel],
    gamma_count: int,
    sigma_model=None,
) -> list[tuple[ArgonLevel, float]]:
    """Exclusive fixed-N channel weights: w_i = sigma_i(Eχ) * P(Ngamma = N | i).

    These weights are used both for initial-level sampling
    and for TotalCrossSection. 
    Do not multiply P(N|i) again in FinalStateProbability.
    """
    sigma = sigma_placeholder if sigma_model is None else sigma_model
    multiplicity_memo: dict[tuple[int, int], float] = {}
    weighted = []
    for level in kinematically_accessible_levels(E_chi, m_chi, target_mass, levels):
        p_mult = cascade_multiplicity_probability(levels, level.index, gamma_count, multiplicity_memo)
        if p_mult <= 0.0:
            continue
        sigma_i = float(sigma(E_chi, level))
        if sigma_i < 0.0:
            raise ValueError(f"Negative partial cross section for level {level.index}")
        weight = sigma_i * p_mult
        if weight > 0.0:
            weighted.append((level, weight))
    return weighted


def sample_initial_excited_level(
    E_chi: float,
    m_chi: float,
    target_mass: float,
    levels: dict[int, ArgonLevel],
    gamma_count: int,
    random,
    sigma_model=None,
) -> ArgonLevel:
    weighted = partial_cross_section_weights(E_chi, m_chi, target_mass, levels, gamma_count, sigma_model)
    if not weighted:
        raise ValueError("No accessible Ar40 excited level has positive weight for this gamma multiplicity")
    idx = sample_weighted_index([weight for _, weight in weighted], random)
    return weighted[idx][0]


def sample_transition(
    levels: dict[int, ArgonLevel],
    level: ArgonLevel,
    random,
    remaining_gamma_count: int | None = None,
    multiplicity_memo: dict[tuple[int, int], float] | None = None,
) -> ArgonTransition:
    """Sample a de-excitation transition.

    Unconditional (remaining_gamma_count is None): weight proportional to BR.
    Fixed-N conditional: weight proportional to BR(a→b) * P(N_tail = remaining-1 | b).
    """
    transitions = []
    weights = []
    for transition in level.transitions:
        if transition.branching_fraction <= 0.0:
            continue
        if remaining_gamma_count is None:
            weight = transition.branching_fraction
        else:
            tail_count = remaining_gamma_count - 1
            tail_probability = cascade_multiplicity_probability(
                levels, transition.final_level, tail_count, multiplicity_memo
            )
            weight = transition.branching_fraction * tail_probability
        if weight <= 0.0:
            continue
        transitions.append(transition)
        weights.append(weight)
    if not transitions:
        raise ValueError(f"No valid transition from level {level.index}")
    return transitions[sample_weighted_index(weights, random)]


def sample_gamma_direction(random) -> np.ndarray:
    cos_theta = 1.0 - 2.0 * _uniform(random)
    sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    phi = TWO_PI * _uniform(random)
    return np.asarray([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=float)


def sample_isotropic_direction(random) -> np.ndarray:
    return sample_gamma_direction(random)


def two_body_deexcitation(
    parent_lab: LorentzVector,
    parent_excitation: float,
    daughter_excitation: float,
    random,
    tabulated_gamma_energy: float | None = None,
    energy_warning_tolerance: float = 1.0e-3,
) -> tuple[LorentzVector, LorentzVector, float]:
    """Sample Ar40* -> Ar40(*) + gamma using sequential two-body recoil.

    Gamma emission is currently isotropic in the parent-nucleus rest frame;
    nuclear alignment and gamma-gamma angular correlations are not included.
    """
    ground_mass = parent_lab.mass() - parent_excitation
    if ground_mass <= 0.0:
        raise ValueError("Invalid parent excitation: inferred ground-state mass is nonpositive")
    parent_mass = ground_mass + parent_excitation
    daughter_mass = ground_mass + daughter_excitation
    e_gamma = (parent_mass * parent_mass - daughter_mass * daughter_mass) / (2.0 * parent_mass)
    if e_gamma <= 0.0:
        raise ValueError("Nonpositive de-excitation gamma energy")
    if tabulated_gamma_energy is not None and abs(e_gamma - tabulated_gamma_energy) > energy_warning_tolerance:
        warnings.warn(
            f"Calculated Ar40 gamma energy {e_gamma:.6g} differs from table {tabulated_gamma_energy:.6g}",
            RuntimeWarning,
            stacklevel=2,
        )
    direction = sample_gamma_direction(random)
    gamma_rest = LorentzVector(e_gamma, *(e_gamma * direction).tolist())
    daughter_rest = LorentzVector(
        np.sqrt(daughter_mass * daughter_mass + e_gamma * e_gamma),
        *(-e_gamma * direction).tolist(),
    )
    boost_to_lab = -parent_lab.get_3velocity()
    return lorentz_boost(gamma_rest, boost_to_lab), lorentz_boost(daughter_rest, boost_to_lab), e_gamma


def generate_argon_cascade(
    levels: dict[int, ArgonLevel],
    initial_level: ArgonLevel,
    excited_nucleus_lab: LorentzVector,
    random,
    gamma_count: int,
    initial_position,
    initial_time: float,
    lifetime_model=None,
    max_attempts: int = 200,
) -> CascadeResult:
    lifetime = time_placeholder if lifetime_model is None else lifetime_model
    start_position = np.asarray(initial_position, dtype=float)
    multiplicity_memo: dict[tuple[int, int], float] = {}
    for _ in range(max_attempts):
        current_level = initial_level
        current_nucleus = excited_nucleus_lab
        current_position = start_position.copy()
        current_time = float(initial_time)
        gammas: list[GammaRecord] = []
        while current_level.index != 0:
            remaining = gamma_count - len(gammas)
            transition = sample_transition(
                levels, current_level, random, remaining, multiplicity_memo
            )
            tau = float(lifetime(current_level))
            if tau < 0.0:
                raise ValueError(f"Negative lifetime for level {current_level.index}")
            if tau > 0.0:
                proper_dt = -tau * np.log(max(_uniform(random), np.finfo(float).tiny))
                gamma_nucleus = current_nucleus.energy() / current_nucleus.mass()
                lab_dt = gamma_nucleus * proper_dt
                beta = current_nucleus.get_3velocity().vec
                current_position = current_position + beta * siren.utilities.Constants.c * lab_dt
                current_time += lab_dt
            daughter_level = levels[transition.final_level]
            gamma_lab, daughter_lab, rest_energy = two_body_deexcitation(
                current_nucleus,
                current_level.excitation_energy,
                daughter_level.excitation_energy,
                random,
                transition.gamma_energy,
            )
            p3 = np.asarray([gamma_lab.p1, gamma_lab.p2, gamma_lab.p3], dtype=float)
            gammas.append(GammaRecord(transition, gamma_lab, _safe_unit(p3), current_position.copy(), current_time, rest_energy))
            current_level = daughter_level
            current_nucleus = daughter_lab
            if len(gammas) > gamma_count:
                break
        if current_level.index == 0 and len(gammas) == gamma_count:
            return CascadeResult(initial_level, current_nucleus, tuple(gammas))
    raise RuntimeError(f"Could not sample an Ar40 cascade with {gamma_count} gammas after {max_attempts} attempts")


class DarkMatterArgonInelastic(siren.interactions.CrossSection):
    """chi + Ar40 -> chi + Ar40 ground state + configured de-excitation gammas.

    Gamma secondary times and interaction-parameter metadata are measured from the full event-chain start
    if upstream propagation has filled record.interaction_time accordingly. 
    This process treats that field as the DM-Ar40 scattering time seed.
    """

    def __init__(
        self,
        m_chi: float,
        gamma_count: int,
        sigma_model=None,
        argon_level_file: str | Path = "Ar.dat",
        lifetime_model=None,
        primary_types: list | None = None,
        max_cascade_attempts: int = 200,
    ):
        siren.interactions.CrossSection.__init__(self)
        self.m_chi = float(m_chi)
        self.gamma_count = int(gamma_count)
        if self.gamma_count < 0:
            raise ValueError("gamma_count must be nonnegative")
        self.sigma_model = sigma_model
        self.lifetime_model = lifetime_model
        self.argon_level_file = str(argon_level_file)
        self.levels = load_argon_levels(argon_level_file)
        self.primary_types = [N4, N4Bar] if primary_types is None else list(primary_types)
        self.max_cascade_attempts = int(max_cascade_attempts)
        self.last_cascade: CascadeResult | None = None

    def equal(self, other) -> bool:
        return (
            isinstance(other, DarkMatterArgonInelastic)
            and self.m_chi == other.m_chi
            and self.gamma_count == other.gamma_count
            and self.argon_level_file == other.argon_level_file
            and self.primary_types == other.primary_types
        )

    def _total_xs_for_record(self, record) -> float:
        signature = getattr(record, "signature", None)
        primary = getattr(signature, "primary_type", record)
        if primary not in self.primary_types:
            return 0.0
        target = getattr(signature, "target_type", Ar40)
        if target != Ar40:
            return 0.0
        if hasattr(record, "primary_momentum"):
            E_chi = float(record.primary_momentum[0])
            target_mass = float(record.target_mass)
        else:
            E_chi = float(getattr(record, "energy", 0.0))
            target_mass = AR40_MASS_GEV
        return sum(
            weight for _, weight in partial_cross_section_weights(
                E_chi, self.m_chi, target_mass, self.levels, self.gamma_count, self.sigma_model
            )
        )

    def _scatter_sim(self, record, random):
        initial_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_chi_in = LorentzVector(*initial_4vec.tolist())
        target_mass = float(record.target_mass)
        lv_target = LorentzVector(target_mass, 0.0, 0.0, 0.0)
        initial_level = sample_initial_excited_level(
            lv_chi_in.energy(),
            self.m_chi,
            target_mass,
            self.levels,
            self.gamma_count,
            random,
            self.sigma_model,
        )
        excited_mass = target_mass + initial_level.excitation_energy
        cm_p4 = lv_chi_in + lv_target
        s = cm_p4.mass2()
        if s < (self.m_chi + excited_mass) ** 2:
            return None
        p_out = np.sqrt(max(
            (s - (self.m_chi + excited_mass) ** 2) * (s - (self.m_chi - excited_mass) ** 2),
            0.0,
        )) / (2.0 * np.sqrt(s))
        e_chi_cm = np.sqrt(self.m_chi * self.m_chi + p_out * p_out)
        direction = sample_isotropic_direction(random)
        chi_cm = LorentzVector(e_chi_cm, *(p_out * direction).tolist())
        nucleus_cm = LorentzVector(np.sqrt(excited_mass * excited_mass + p_out * p_out), *(-p_out * direction).tolist())
        p_in = cm_p4.get_3momentum()
        boost_to_lab = Vector3(-p_in.v1 / cm_p4.energy(), -p_in.v2 / cm_p4.energy(), -p_in.v3 / cm_p4.energy())
        chi_lab = lorentz_boost(chi_cm, boost_to_lab)
        nucleus_lab = lorentz_boost(nucleus_cm, boost_to_lab)
        cascade = generate_argon_cascade(
            self.levels,
            initial_level,
            nucleus_lab,
            random,
            self.gamma_count,
            getattr(record, "interaction_vertex", [0.0, 0.0, 0.0]),
            getattr(record, "interaction_time", 0.0),
            self.lifetime_model,
            self.max_cascade_attempts,
        )
        self.last_cascade = cascade
        return chi_lab, cascade

    def TotalCrossSection(self, record) -> float:
        return self._total_xs_for_record(record)

    def DifferentialCrossSection(self, record) -> float:
        total = self.TotalCrossSection(record)
        return total / (4.0 * np.pi) if total > 0.0 else 0.0

    def InteractionThreshold(self, record) -> float:
        target_mass = float(getattr(record, "target_mass", AR40_MASS_GEV))
        capable = levels_capable_of_gamma_count(self.levels, self.gamma_count)
        if not capable:
            return np.inf
        min_excited_mass = target_mass + min(level.excitation_energy for level in capable)
        return ((self.m_chi + min_excited_mass) ** 2 - self.m_chi * self.m_chi - target_mass * target_mass) / (2.0 * target_mass)

    def SampleFinalState(self, record, random):
        sampled = self._scatter_sim(record, random)
        if sampled is None:
            return
        chi_lab, cascade = sampled
        secondary_particles = record.get_secondary_particle_records()
        chi_indices = [
            idx for idx, sec_type in enumerate(record.signature.secondary_types)
            if sec_type in self.primary_types
        ]
        nucleus_indices = _find_secondary_indices_by_type(record, Ar40)
        gamma_indices = _find_secondary_indices_by_type(record, Gamma)
        if len(chi_indices) != 1 or len(nucleus_indices) != 1 or len(gamma_indices) != self.gamma_count:
            return
        self._fill_secondary(secondary_particles[chi_indices[0]], chi_lab, self.m_chi, getattr(record, "interaction_time", 0.0))
        self._fill_secondary(secondary_particles[nucleus_indices[0]], cascade.final_nucleus, cascade.final_nucleus.mass(), cascade.gammas[-1].time if cascade.gammas else getattr(record, "interaction_time", 0.0))
        for idx, gamma in zip(gamma_indices, cascade.gammas):
            self._fill_secondary(secondary_particles[idx], gamma.four_momentum, 0.0, gamma.time)
        self._store_gamma_metadata(record, cascade)

    def _fill_secondary(self, particle, lv: LorentzVector, mass: float, time: float) -> None:
        p4 = _lv_array(lv)
        particle.four_momentum = p4.tolist()
        particle.mass = mass
        p3 = np.asarray([lv.p1, lv.p2, lv.p3], dtype=float)
        pnorm = float(np.linalg.norm(p3))
        if pnorm > 0.0:
            particle.direction = (p3 / pnorm).tolist()
        particle.time = float(time)

    def _store_gamma_metadata(self, record, cascade: CascadeResult) -> None:
        """Persist per-gamma positions/times through writable SIREN parameters.

        SecondaryParticleRecord.initial_position is readonly in current pybind
        bindings, so displaced gamma production positions are exported through
        interaction_parameters using stable keys. 
        With prompt lifetimes these positions are equal to record.interaction_vertex.
        """
        params = dict(getattr(record, "interaction_parameters", {}))
        params["argon_gamma_count"] = float(len(cascade.gammas))
        for idx, gamma in enumerate(cascade.gammas):
            prefix = f"argon_gamma_{idx}"
            params[f"{prefix}_energy"] = float(gamma.four_momentum.energy())
            params[f"{prefix}_px"] = float(gamma.four_momentum.p1)
            params[f"{prefix}_py"] = float(gamma.four_momentum.p2)
            params[f"{prefix}_pz"] = float(gamma.four_momentum.p3)
            params[f"{prefix}_dir_x"] = float(gamma.direction[0])
            params[f"{prefix}_dir_y"] = float(gamma.direction[1])
            params[f"{prefix}_dir_z"] = float(gamma.direction[2])
            params[f"{prefix}_x"] = float(gamma.position[0])
            params[f"{prefix}_y"] = float(gamma.position[1])
            params[f"{prefix}_z"] = float(gamma.position[2])
            params[f"{prefix}_time"] = float(gamma.time)
        record.interaction_parameters = params

    def GetPossibleTargets(self):
        return [Ar40]

    def GetPossibleTargetsFromPrimary(self, primary):
        if primary in self.primary_types:
            return [Ar40]
        return []

    def GetPossiblePrimaries(self):
        return list(self.primary_types)

    def GetPossibleSignatures(self):
        return [sig for primary in self.primary_types for sig in self.GetPossibleSignaturesFromParents(primary, Ar40)]

    def GetPossibleSignaturesFromParents(self, primary, target):
        if primary not in self.primary_types or target != Ar40:
            return []
        signature = siren.dataclasses.InteractionSignature()
        signature.primary_type = primary
        signature.target_type = Ar40
        signature.secondary_types = [primary, Ar40] + [Gamma] * self.gamma_count
        return [signature]

    def DensityVariables(self):
        return []

    def FinalStateProbability(self, record):
        # Angular / continuous density only. Exclusive multiplicity P(N|i) is
        # already included in TotalCrossSection via partial_cross_section_weights;
        # do not multiply it here again.
        total_xs = self.TotalCrossSection(record)
        differential_xs = self.DifferentialCrossSection(record)
        if differential_xs == 0.0 or total_xs == 0.0:
            return 0.0
        return differential_xs / total_xs


def build_pi0_darkphoton_dm_inelastic_chain(
    m_dark_photon: float,
    epsilon: float,
    alpha_D: float,
    gamma_count: int,
    *,
    m_chi: float | None = None,
    pi0_total_width: float = 7.8e-9,
    pi0_to_diphoton_br: float = 0.98823,
    sigma_model=None,
    argon_level_file: str | Path = "Ar.dat",
    lifetime_model=None,
    max_cascade_attempts: int = 200,
    dark_photon_type=None,
    chi_type=None,
    chibar_type=None,
) -> Pi0DarkPhotonDarkMatterInelasticChain:
    """Create the full pi0 -> A' -> chi/chibar -> Ar40 inelastic channel set.

    step3 is this script's fixed-gamma-multiplicity DM-Ar40 CrossSection.
    """
    step1 = Pi0ToDarkphoton(
        m_dark_photon=m_dark_photon,
        epsilon=epsilon,
        pi0_total_width=pi0_total_width,
        pi0_to_diphoton_br=pi0_to_diphoton_br,
        dark_photon_type=dark_photon_type,
    )
    step2 = DarkphotonToDarkmatter(
        m_dark_photon=m_dark_photon,
        alpha_D=alpha_D,
        dark_photon_type=step1.dark_photon_type,
        chi_type=chi_type,
        chibar_type=chibar_type,
    )
    step3 = DarkMatterArgonInelastic(
        m_chi=step2.m_dark_matter if m_chi is None else m_chi,
        gamma_count=gamma_count,
        sigma_model=sigma_model,
        argon_level_file=argon_level_file,
        lifetime_model=lifetime_model,
        primary_types=[step2.chi_type, step2.chibar_type],
        max_cascade_attempts=max_cascade_attempts,
    )
    return Pi0DarkPhotonDarkMatterInelasticChain(step1, step2, step3)


def build_pi0_darkphoton_dm_inelastic_injector(
    pi0_dist,
    detector_model,
    chain: Pi0DarkPhotonDarkMatterInelasticChain,
    n_events: int,
    seed: int,
    *,
    dark_photon_vertex_distribution=None,
    dark_matter_vertex_distribution=None,
):
    """Build a SIREN Injector for pi0 -> A' -> chi/chibar -> Ar40 + gammas.
    """
    pt = siren.dataclasses.Particle.ParticleType
    step1 = chain.pi0_to_darkphoton
    step2 = chain.darkphoton_to_darkmatter
    step3 = chain.darkmatter_argon_inelastic

    if dark_photon_vertex_distribution is None:
        dark_photon_vertex_distribution = siren.distributions.SecondaryPhysicalVertexDistribution()
    if dark_matter_vertex_distribution is None:
        chi_vertex_distribution = siren.distributions.SecondaryPhysicalVertexDistribution()
        chibar_vertex_distribution = siren.distributions.SecondaryPhysicalVertexDistribution()
    else:
        chi_vertex_distribution = dark_matter_vertex_distribution
        chibar_vertex_distribution = dark_matter_vertex_distribution

    dm_primary_types = set(step3.primary_types)

    def stopping_condition(datum, i):
        parent_type = datum.record.signature.primary_type
        secondary_type = datum.record.signature.secondary_types[i]
        if parent_type == pt.Pi0:
            return secondary_type != step1.dark_photon_type
        if parent_type == step1.dark_photon_type:
            return secondary_type not in dm_primary_types
        return True

    return siren.injection.Injector(
        number_of_events=n_events,
        detector_model=detector_model,
        seed=seed,
        primary_type=pt.Pi0,
        primary_interactions=[step1],
        primary_injection_distributions=[pi0_dist],
        secondary_interactions={
            step1.dark_photon_type: [step2],
            step2.chi_type: [step3],
            step2.chibar_type: [step3],
        },
        secondary_injection_distributions={
            step1.dark_photon_type: [dark_photon_vertex_distribution],
            step2.chi_type: [chi_vertex_distribution],
            step2.chibar_type: [chibar_vertex_distribution],
        },
        stopping_condition=stopping_condition,
    )
