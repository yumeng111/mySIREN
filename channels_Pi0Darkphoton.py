from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import siren

from .constants import ALPLIB_TO_SIREN, HBARC_MEV_CM, SIREN_TO_ALPLIB, TWO_PI
from .kinematics import LorentzVector, Vector3, lorentz_boost
from .matrix_elements import M2InversePrimakoff, M2Primakoff
from .spline_tables import find_xs_table_file, load_diff_xs_evaluator, load_total_xs_evaluator

from collections import Counter


@functools.cache
def extract_z(pdg_code: int) -> int:
    """Extract nuclear Z from PDG nucleus code."""
    pdg_code = int(pdg_code)
    _excitation = pdg_code % 10
    pdg_code //= 10
    nucleon_count = pdg_code % 1000
    pdg_code //= 1000
    proton_count = pdg_code % 1000
    return proton_count


class per_instance_cache:
    """Like functools.cache, but scoped per instance."""

    def __init__(self, func):
        self.func = func
        self._slot = f"__cached_{func.__name__}"
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cached = obj.__dict__.get(self._slot)
        if cached is None:
            cached = functools.cache(self.func.__get__(obj, objtype))
            obj.__dict__[self._slot] = cached
        return cached


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

class Pi0Darkphoton(siren.interactions.DarkNewsDecay):
    """Two-body pi0 -> gamma + A' decay channel.

    The partial width follows
    BR(pi0 -> gamma A') = 2 eps^2 (1 - m_A'^2 / m_pi0^2)^3 BR(pi0 -> gamma gamma).
    """

    def __init__(
        self,
        #Q. unit, m_dark_photon=3 m_dm
        m_dark_photon: float,
        epsilon: float,
        #Q. unit in SIREN GeV. PDG2025
        pi0_total_width: float = 7.8e-9,
        pi0_to_diphoton_br: float = 0.98823,
        dark_photon_type=None,
    ):
        # Q1: DarkNewsDecay is a Decay subclass with trampoline/init support
        #siren.interactions.Decay.__init__(self)
        siren.interactions.DarkNewsDecay.__init__(self)
        #Q. unit? *SIREN_TO_ALPLIB
        self.m_dark_photon = m_dark_photon
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
            isinstance(other, Pi0Darkphoton)
            and self.m_dark_photon == other.m_dark_photon
            and self.epsilon == other.epsilon
            and self.pi0_total_width == other.pi0_total_width
            and self.pi0_to_diphoton_br == other.pi0_to_diphoton_br
            and self.dark_photon_type == other.dark_photon_type
        )

    def pi0_mass(self) -> float:
        #Q: *unit? *SIREN_TO_ALPLIB
        return siren.dataclasses.GetParticleMass(siren.dataclasses.Particle.ParticleType.Pi0)

    def _branching_ratio(self) -> float:
        mpi0 = self.pi0_mass()
        if self.m_dark_photon <= 0.0 or self.m_dark_photon >= mpi0:
            return 0.0
        phase_space = 1.0 - (self.m_dark_photon / mpi0) ** 2
        return 2.0 * self.epsilon * self.epsilon * phase_space**3 * self.pi0_to_diphoton_br

    def _partial_width(self) -> float:
        return self.pi0_total_width * self._branching_ratio()

    def scatter_sim(self, record, random):
        #Q: *unit? *SIREN_TO_ALPLIB
        parent_4vec = np.asarray(record.primary_momentum, dtype=float)
        lv_parent = LorentzVector(*parent_4vec.tolist())
        mp = lv_parent.mass()
        m_gamma = 0.0
        m_dark_photon = self.m_dark_photon
        if mp <= 0.0 or m_dark_photon >= mp:
            return None

        p_cm = np.sqrt(max((mp * mp - (m_dark_photon - m_gamma) ** 2) * (mp * mp - (m_dark_photon + m_gamma) ** 2), 0.0)) / (2.0 * mp)
        e_gamma_cm = p_cm
        e_dark_photon_cm = np.sqrt(p_cm * p_cm + m_dark_photon * m_dark_photon)

        phi_rnd = TWO_PI * _uniform(random)
        theta_rnd = float(np.arccos(1.0 - 2.0 * _uniform(random)))
        v_in = -lv_parent.get_3velocity()

        #pi0 coordinate, don't have to do this if isotropic in rest frame
        k_hat, x_hat, y_hat = _safe_unit(lv_parent.get_3momentum().vec), None, None
        x_hat = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=float), k_hat)
        if np.linalg.norm(x_hat) < 1.0e-8:
            x_hat = np.cross(np.asarray([0.0, 1.0, 0.0], dtype=float), k_hat)
        x_hat = _safe_unit(x_hat)
        y_hat = _safe_unit(np.cross(k_hat, x_hat))

        p_gamma_cm_vec = p_cm * (
            np.sin(theta_rnd) * np.cos(phi_rnd) * x_hat
            + np.sin(theta_rnd) * np.sin(phi_rnd) * y_hat
            + np.cos(theta_rnd) * k_hat
        )
        p_dark_photon_cm_vec = -p_gamma_cm_vec

        gamma_cm = LorentzVector(e_gamma_cm, *p_gamma_cm_vec.tolist())
        dark_photon_cm = LorentzVector(e_dark_photon_cm, *p_dark_photon_cm_vec.tolist())
        return lorentz_boost(gamma_cm, v_in), lorentz_boost(dark_photon_cm, v_in)

    def TotalDecayWidth(self, record):
        pi0 = siren.dataclasses.Particle.ParticleType.Pi0
        #Q. unit? *ALPLIB_TO_SIREN
        if record == pi0:
            return self._partial_width()
        #Q. unit? *ALPLIB_TO_SIREN
        if hasattr(record, "signature") and record.signature.primary_type == pi0:
            return self._partial_width()
        return 0.0

    def TotalDecayWidthForFinalState(self, record):
        pi0 = siren.dataclasses.Particle.ParticleType.Pi0
        gamma = siren.dataclasses.Particle.ParticleType.Gamma
        if record.signature.primary_type != pi0:
            return 0.0
        secondaries = list(record.signature.secondary_types)
        expected = [gamma, self.dark_photon_type]
        if Counter(secondaries) != Counter(expected):
            return 0.0
        #Q. unit? *ALPLIB_TO_SIREN
        return self._partial_width()

    def DifferentialDecayWidth(self, record):
        # Isotropic two-body decay: dGamma / dcos(theta) = Gamma / 2.
        return 0.5 * self.TotalDecayWidthForFinalState(record)

    def SampleFinalState(self, record, random):
        sampled = self.scatter_sim(record, random)
        if sampled is None:
            return
        gamma_lab, dark_photon_lab = sampled
        secondary_particles = record.get_secondary_particle_records()
        if len(secondary_particles) < 2:
            return

        gamma_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.Gamma)
        dark_photon_idx = _find_secondary_index_by_type(record, self.dark_photon_type)
        if gamma_idx is None or dark_photon_idx is None:
            return

        for idx, lv, mass in (
            (gamma_idx, gamma_lab, 0.0),
            (dark_photon_idx, dark_photon_lab, self.m_dark_photon),
        ):
            #Q. unit? *ALPLIB_TO_SIREN
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
