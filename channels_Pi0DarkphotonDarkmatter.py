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

# Step 1: pi0 -> gamma + A'                                              

class Pi0ToDarkphoton(siren.interactions.DarkNewsDecay):
    """First step of cascade: pi0 -> gamma + A' (on-shell dark photon).

    Partial width:
        Gamma(pi0 -> gamma A') = pi0_total_width * 2*eps^2 * (1 - m_A'^2/m_pi0^2)^3 * BR(pi0->gamma gamma)

    The on-shell A' is propagated by SIREN to a displaced vertex where
    DarkphotonToDarkmatter handles A' -> chi + chibar.
    """

    def __init__(
        self,
        m_dark_photon: float,           # GeV(SIREN convention)
        epsilon: float,
        pi0_total_width: float = 7.8e-9,  # GeV, PDG2025
        pi0_to_diphoton_br: float = 0.98823,
        dark_photon_type=None,
    ):
        siren.interactions.DarkNewsDecay.__init__(self)
        self.m_dark_photon = float(m_dark_photon)
        self.epsilon = float(epsilon)
        self.pi0_total_width = float(pi0_total_width)
        self.pi0_to_diphoton_br = float(pi0_to_diphoton_br)
        self.dark_photon_type = (
            siren.dataclasses.Particle.ParticleType.ZPrime
            if dark_photon_type is None
            else dark_photon_type
        )

    def equal(self, other) -> bool:
        return (
            isinstance(other, Pi0ToDarkphoton)
            and self.m_dark_photon == other.m_dark_photon
            and self.epsilon == other.epsilon
            and self.pi0_total_width == other.pi0_total_width
            and self.pi0_to_diphoton_br == other.pi0_to_diphoton_br
            and self.dark_photon_type == other.dark_photon_type
        )

    def pi0_mass(self) -> float:
        return siren.dataclasses.GetParticleMass(siren.dataclasses.Particle.ParticleType.Pi0)

    def _partial_width(self) -> float:
        """Gamma(pi0 -> gamma A') in GeV."""
        mpi0 = self.pi0_mass()
        mA = self.m_dark_photon
        if mA <= 0.0 or mA >= mpi0:
            return 0.0
        phase_space = (1.0 - (mA / mpi0) ** 2) ** 3
        br_pi0_to_aprime = 2.0 * self.epsilon**2 * phase_space * self.pi0_to_diphoton_br
        return self.pi0_total_width * br_pi0_to_aprime

    def _scatter_sim(self, record, random):
        parent_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_parent = LorentzVector(*parent_4vec.tolist())
        mpi0 = lv_parent.mass()
        mA = self.m_dark_photon

        if mpi0 <= 0.0 or mA >= mpi0:
            return None

        gamma_pi0rf, aprime_pi0rf = _two_body_decay_cm(mpi0, 0.0, mA, random)

        v_pi0 = -lv_parent.get_3velocity()
        gamma_lab = lorentz_boost(gamma_pi0rf, v_pi0)
        aprime_lab = lorentz_boost(aprime_pi0rf, v_pi0)
        return gamma_lab, aprime_lab

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
        expected = [siren.dataclasses.Particle.ParticleType.Gamma, self.dark_photon_type]
        if Counter(secondaries) != Counter(expected):
            return 0.0
        return self._partial_width()

    def DifferentialDecayWidth(self, record):
        # Isotropic two-body decay: dGamma / dcos(theta) = Gamma / 2
        return 0.5 * self.TotalDecayWidthForFinalState(record)

    def SampleFinalState(self, record, random):
        sampled = self._scatter_sim(record, random)
        if sampled is None:
            return
        gamma_lab, aprime_lab = sampled

        secondary_particles = record.get_secondary_particle_records()
        if len(secondary_particles) < 2:
            return

        gamma_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.Gamma)
        aprime_idx = _find_secondary_index_by_type(record, self.dark_photon_type)
        if gamma_idx is None or aprime_idx is None:
            return

        for idx, lv, mass in (
            (gamma_idx, gamma_lab, 0.0),
            (aprime_idx, aprime_lab, self.m_dark_photon),
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
            self.dark_photon_type,
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

# Step 2: A' -> chi + chibar                                              

class DarkphotonToDarkmatter(siren.interactions.DarkNewsDecay):
    """Second step of cascade: A' -> chi + chibar (invisible dark photon decay).

    Partial width for a vector boson decaying to a Dirac fermion pair:
        Gamma(A' -> chi chibar) = (alpha_D / 3) * m_A' * beta * (1 + 2*m_chi^2 / m_A'^2)
    where beta = sqrt(1 - 4*m_chi^2 / m_A'^2) and alpha_D = g_D^2 / (4*pi).

    m_chi = m_dark_photon / 3.
    """

    def __init__(
        self,
        m_dark_photon: float,           # GeV
        alpha_D: float,                 # dark fine-structure constant
        dark_photon_type=None,
        chi_type=None,
        chibar_type=None,
    ):
        siren.interactions.DarkNewsDecay.__init__(self)
        self.m_dark_photon = float(m_dark_photon)
        self.m_dark_matter = float(m_dark_photon) / 3.0
        self.alpha_D = float(alpha_D)
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
            isinstance(other, DarkphotonToDarkmatter)
            and self.m_dark_photon == other.m_dark_photon
            and self.alpha_D == other.alpha_D
            and self.dark_photon_type == other.dark_photon_type
            and self.chi_type == other.chi_type
            and self.chibar_type == other.chibar_type
        )

    def _partial_width(self) -> float:
        """Gamma(A' -> chi chibar) in GeV."""
        mA = self.m_dark_photon
        mchi = self.m_dark_matter
        if mA <= 0.0 or mchi <= 0.0 or mA <= 2.0 * mchi:
            return 0.0
        x = (mchi / mA) ** 2
        beta = np.sqrt(max(1.0 - 4.0 * x, 0.0))
        return (self.alpha_D / 3.0) * mA * beta * (1.0 + 2.0 * x)

    def _scatter_sim(self, record, random):
        parent_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_parent = LorentzVector(*parent_4vec.tolist())
        mA = lv_parent.mass()
        mchi = self.m_dark_matter

        if mA <= 0.0 or mA <= 2.0 * mchi:
            return None

        chi_aprf, chibar_aprf = _two_body_decay_cm(mA, mchi, mchi, random)

        v_aprime = -lv_parent.get_3velocity()
        chi_lab = lorentz_boost(chi_aprf, v_aprime)
        chibar_lab = lorentz_boost(chibar_aprf, v_aprime)
        return chi_lab, chibar_lab

    def TotalDecayWidth(self, record):
        aprime = self.dark_photon_type
        if record == aprime:
            return self._partial_width()
        if hasattr(record, "signature") and record.signature.primary_type == aprime:
            return self._partial_width()
        return 0.0

    def TotalDecayWidthForFinalState(self, record):
        if record.signature.primary_type != self.dark_photon_type:
            return 0.0
        secondaries = list(record.signature.secondary_types)
        expected = [self.chi_type, self.chibar_type]
        if Counter(secondaries) != Counter(expected):
            return 0.0
        return self._partial_width()

    def DifferentialDecayWidth(self, record):
        # Isotropic two-body decay: dGamma / dcos(theta) = Gamma / 2
        return 0.5 * self.TotalDecayWidthForFinalState(record)

    def SampleFinalState(self, record, random):
        sampled = self._scatter_sim(record, random)
        if sampled is None:
            return
        chi_lab, chibar_lab = sampled

        secondary_particles = record.get_secondary_particle_records()
        if len(secondary_particles) < 2:
            return

        chi_idx = _find_secondary_index_by_type(record, self.chi_type)
        chibar_idx = _find_secondary_index_by_type(record, self.chibar_type)
        if chi_idx is None or chibar_idx is None:
            return

        for idx, lv, mass in (
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
        signature.primary_type = self.dark_photon_type
        signature.secondary_types = [self.chi_type, self.chibar_type]
        return [signature]

    def GetPossibleSignaturesFromParent(self, primary):
        if primary == self.dark_photon_type:
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