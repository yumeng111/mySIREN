# Not-used version; did not run for check
from __future__ import annotations

import functools
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
    return np.asarray(vec / norm, dtype=float)


def _find_secondary_index_by_type(record, particle_type) -> int | None:
    secondary_types = getattr(record.signature, "secondary_types", [])
    for idx, sec_type in enumerate(secondary_types):
        if sec_type == particle_type:
            return idx
    return None


class DarkphotonDarkmatter(siren.interactions.DarkNewsDecay):
    """Two-body dark photon decay A' -> chi + chibar.

    Partial width for a vector boson decaying to a Dirac fermion pair:
        Gamma(A' -> chi chibar) = (alpha_D / 3) * m_A' * beta * (1 + 2*m_chi^2 / m_A'^2)
    where beta = sqrt(1 - 4*m_chi^2 / m_A'^2) and alpha_D = g_D^2 / (4*pi).
    """

    def __init__(
        self,
        # unit: GeV (SIREN convention); m_dark_matter = m_dark_photon / 3
        m_dark_photon: float,
        # dark sector fine-structure constant alpha_D = g_D^2 / (4 pi)
        alpha_D: float,
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
        # chi and chibar default to N4 / N4Bar (lightest dark fermions in SIREN)
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
            isinstance(other, DarkphotonDarkmatter)
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
        """Sample two-body kinematics in the A' rest frame and boost to lab."""
        parent_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_parent = LorentzVector(*parent_4vec.tolist())
        mA = lv_parent.mass()
        mchi = self.m_dark_matter

        if mA <= 0.0 or mA <= 2.0 * mchi:
            return None

        p_cm = np.sqrt(max(mA * mA / 4.0 - mchi * mchi, 0.0))
        e_chi_cm = np.sqrt(p_cm * p_cm + mchi * mchi)

        # Isotropic angular distribution in rest frame
        phi = TWO_PI * _uniform(random)
        cos_theta = 1.0 - 2.0 * _uniform(random)
        sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))

        # Build orthonormal frame aligned with A' momentum direction
        k_hat = _safe_unit(lv_parent.get_3momentum().vec)
        x_hat = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=float), k_hat)
        if np.linalg.norm(x_hat) < 1.0e-8:
            x_hat = np.cross(np.asarray([0.0, 1.0, 0.0], dtype=float), k_hat)
        x_hat = _safe_unit(x_hat)
        y_hat = _safe_unit(np.cross(k_hat, x_hat))

        p_chi_cm_vec = p_cm * (
            sin_theta * np.cos(phi) * x_hat
            + sin_theta * np.sin(phi) * y_hat
            + cos_theta * k_hat
        )

        chi_cm = LorentzVector(e_chi_cm, *p_chi_cm_vec.tolist())
        chibar_cm = LorentzVector(e_chi_cm, *(-p_chi_cm_vec).tolist())

        v_in = -lv_parent.get_3velocity()
        return lorentz_boost(chi_cm, v_in), lorentz_boost(chibar_cm, v_in)

    # SIREN interface                                                     

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
