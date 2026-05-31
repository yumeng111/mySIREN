# Not-used version; did not run for check
from __future__ import annotations

from collections import Counter

import numpy as np

import siren

from .constants import TWO_PI
from .kinematics import LorentzVector, lorentz_boost



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
    return vec / norm


def _find_secondary_index_by_type(record, particle_type) -> int | None:
    secondary_types = getattr(record.signature, "secondary_types", [])
    for idx, sec_type in enumerate(secondary_types):
        if sec_type == particle_type:
            return idx
    return None


def _two_body_decay_cm(m_parent: float, m1: float, m2: float, random) -> tuple[LorentzVector, LorentzVector]:
    """Sample isotropic two-body decay in the parent rest frame."""
    p_cm = np.sqrt(max(
        (m_parent**2 - (m1 + m2)**2) * (m_parent**2 - (m1 - m2)**2), 0.0
    )) / (2.0 * m_parent)
    e1 = np.sqrt(p_cm**2 + m1**2)
    e2 = np.sqrt(p_cm**2 + m2**2)

    cos_theta = 1.0 - 2.0 * _uniform(random)
    sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
    phi = TWO_PI * _uniform(random)

    p_vec = p_cm * np.asarray([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    return LorentzVector(e1, *p_vec.tolist()), LorentzVector(e2, *(-p_vec).tolist())


class Pi0DarkphotonDarkmatter(siren.interactions.DarkNewsDecay):
    """Cascade decay pi0 -> gamma + A'(on-shell) -> gamma+ chi + chibar.

    BR(A' -> chi chibar) = 1 (invisible decay only).

    Effective partial width:
        Gamma_eff = Gamma(pi0 -> gamma A')
                  = pi0_total_width * 2*eps^2 * (1 - m_A'^2/m_pi0^2)^3 * BR(pi0->gamma gamma)

    m_chi = m_dark_photon / 3.
    """

    def __init__(
        self,
        # unit: GeV (SIREN convention)
        m_dark_photon: float,
        epsilon: float,
        # PDG2025 values
        pi0_total_width: float = 7.8e-9,
        pi0_to_diphoton_br: float = 0.98823,
        dark_photon_type=None,
        chi_type=None,
        chibar_type=None,
    ):
        siren.interactions.DarkNewsDecay.__init__(self)
        self.m_dark_photon = float(m_dark_photon)
        self.m_dark_matter = float(m_dark_photon) / 3.0
        self.epsilon = float(epsilon)
        self.pi0_total_width = float(pi0_total_width)
        self.pi0_to_diphoton_br = float(pi0_to_diphoton_br)
        self.dark_photon_type = (
            siren.dataclasses.Particle.ParticleType.ZPrime
            if dark_photon_type is None
            else dark_photon_type
        )
        self.chi_type = (
            siren.dataclasses.Particle.ParticleType.N4
            if chi_type is None
            else chi_type
        )
        self.chibar_type = (
            siren.dataclasses.Particle.ParticleType.N4Bar
            if chibar_type is None
            else chibar_type
        )

    def equal(self, other) -> bool:
        return (
            isinstance(other, Pi0DarkphotonDarkmatter)
            and self.m_dark_photon == other.m_dark_photon
            and self.epsilon == other.epsilon
            and self.pi0_total_width == other.pi0_total_width
            and self.pi0_to_diphoton_br == other.pi0_to_diphoton_br
            and self.dark_photon_type == other.dark_photon_type
            and self.chi_type == other.chi_type
            and self.chibar_type == other.chibar_type
        )

    def pi0_mass(self) -> float:
        return siren.dataclasses.GetParticleMass(siren.dataclasses.Particle.ParticleType.Pi0)

    # Width calculations                                                   #

    def _pi0_partial_width(self) -> float:
        """Gamma(pi0 -> gamma A') in GeV."""
        mpi0 = self.pi0_mass()
        mA = self.m_dark_photon
        if mA <= 0.0 or mA >= mpi0:
            return 0.0
        phase_space = (1.0 - (mA / mpi0) ** 2) ** 3
        br_pi0_to_aprime = 2.0 * self.epsilon**2 * phase_space * self.pi0_to_diphoton_br
        return self.pi0_total_width * br_pi0_to_aprime

    def _partial_width(self) -> float:
        """Effective Gamma(pi0 -> gamma chi chibar) via on-shell A', BR(A'->chi chibar)=1."""
        return self._pi0_partial_width()

    # Kinematics                                                           

    def _scatter_sim(self, record, random):
        """Cascade two-body decay: pi0->gamma+A' then A'->chi+chibar."""
        parent_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_parent = LorentzVector(*parent_4vec.tolist())
        mpi0 = lv_parent.mass()
        mA = self.m_dark_photon
        mchi = self.m_dark_matter

        if mpi0 <= 0.0 or mA >= mpi0 or mA <= 2.0 * mchi:
            return None

        # --- Step 1: pi0 -> gamma + A' in pi0 rest frame ---
        gamma_pi0rf, aprime_pi0rf = _two_body_decay_cm(mpi0, 0.0, mA, random)

        # Boost gamma and A' to lab frame
        v_pi0 = -lv_parent.get_3velocity()
        gamma_lab = lorentz_boost(gamma_pi0rf, v_pi0)
        aprime_lab = lorentz_boost(aprime_pi0rf, v_pi0)

        # --- Step 2: A' -> chi + chibar in A' rest frame ---
        chi_aprf, chibar_aprf = _two_body_decay_cm(mA, mchi, mchi, random)

        # Boost chi and chibar from A' rest frame to lab frame
        v_aprime = -aprime_lab.get_3velocity()
        chi_lab = lorentz_boost(chi_aprf, v_aprime)
        chibar_lab = lorentz_boost(chibar_aprf, v_aprime)

        return gamma_lab, chi_lab, chibar_lab

    # SIREN interface                                                    

    def TotalDecayWidth(self, record):
        pi0 = siren.dataclasses.Particle.ParticleType.Pi0
        if record == pi0:
            return self._partial_width()
        if hasattr(record, "signature") and record.signature.primary_type == pi0:
            return self._partial_width()
        return 0.0

    def TotalDecayWidthForFinalState(self, record):
        pi0 = siren.dataclasses.Particle.ParticleType.Pi0
        if record.signature.primary_type != pi0:
            return 0.0
        secondaries = list(record.signature.secondary_types)
        expected = [siren.dataclasses.Particle.ParticleType.Gamma, self.chi_type, self.chibar_type]
        if Counter(secondaries) != Counter(expected):
            return 0.0
        return self._partial_width()

    def DifferentialDecayWidth(self, record):
        # Phase space for two independent isotropic two-body decays:
        # d^2Gamma / (dcos(theta1) dcos(theta2)) = Gamma / 4
        return 0.25 * self.TotalDecayWidthForFinalState(record)

    def SampleFinalState(self, record, random):
        sampled = self._scatter_sim(record, random)
        if sampled is None:
            return
        gamma_lab, chi_lab, chibar_lab = sampled

        secondary_particles = record.get_secondary_particle_records()
        if len(secondary_particles) < 3:
            return

        gamma_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.Gamma)
        chi_idx = _find_secondary_index_by_type(record, self.chi_type)
        chibar_idx = _find_secondary_index_by_type(record, self.chibar_type)
        if None in (gamma_idx, chi_idx, chibar_idx):
            return

        for idx, lv, mass in (
            (gamma_idx, gamma_lab, 0.0),
            (chi_idx, chi_lab, self.m_dark_matter),
            (chibar_idx, chibar_lab, self.m_dark_matter),
        ):
            secondary_particles[idx].mass = mass
            secondary_particles[idx].energy = lv.energy()
            p3 = np.asarray([lv.p1, lv.p2, lv.p3], dtype=float)
            pnorm = float(np.linalg.norm(p3))
            if pnorm <= 0.0:
                continue
            secondary_particles[idx].direction = (p3 / pnorm).tolist()

    def GetPossibleSignatures(self):
        signature = siren.dataclasses.InteractionSignature()
        signature.primary_type = siren.dataclasses.Particle.ParticleType.Pi0
        signature.secondary_types = [
            siren.dataclasses.Particle.ParticleType.Gamma,
            self.chi_type,
            self.chibar_type,
        ]
        return [signature]

    def GetPossibleSignaturesFromParent(self, primary):
        if primary == siren.dataclasses.Particle.ParticleType.Pi0:
            return self.GetPossibleSignatures()
        return []

    def GetPossibleSignaturesFromParents(self, primary):
        return self.GetPossibleSignaturesFromParent(primary)

    def DensityVariables(self):
        return []

    def FinalStateProbability(self, record):
        total = self.TotalDecayWidthForFinalState(record)
        if total == 0.0:
            return 0.0
        return self.DifferentialDecayWidth(record) / total
