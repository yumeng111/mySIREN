"""ALP interaction channels used by the CCM simulation workflow."""

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


class _PrimakoffBase(siren.interactions.CrossSection):
    target_mats: dict[int, str]

    def __init__(
        self,
        *,
        ma: float,
        g: float,
        process_name: str,
        target_nuclei: list | None,
        default_target_nuclei: list,
        target_mats: dict[int, str],
        spline_dir: str | Path | None = None,
        allow_analytic_fallback: bool = True,
    ):
        siren.interactions.CrossSection.__init__(self)
        self.ma = ma * SIREN_TO_ALPLIB
        self.g = g * ALPLIB_TO_SIREN
        self.process_name = process_name
        self.target_nuclei = default_target_nuclei if target_nuclei is None else target_nuclei
        self.target_mats = target_mats
        self.allow_analytic_fallback = allow_analytic_fallback
        self.spline_dir = self._resolve_spline_dir(spline_dir)
        self.totxs_evaluators: dict[int, callable] = {}
        self.diffxs_evaluators: dict[int, callable] = {}
        self._load_tables()

    def _resolve_spline_dir(self, spline_dir: str | Path | None) -> Path | None:
        if spline_dir is not None:
            path = Path(spline_dir).expanduser()
            return path if path.exists() else None
        env_dir = os.environ.get("CCM_SIREN_ALPS_SPLINE_DIR")
        if env_dir:
            path = Path(env_dir).expanduser()
            if path.exists():
                return path
        return None

    def _load_tables(self) -> None:
        if self.spline_dir is None:
            return
        for z, material in self.target_mats.items():
            total_file = find_xs_table_file(
                spline_dir=self.spline_dir,
                process_name=self.process_name,
                material_tag=material,
                kind="tot",
                mass_mev=self.ma,
            )
            diff_file = find_xs_table_file(
                spline_dir=self.spline_dir,
                process_name=self.process_name,
                material_tag=material,
                kind="diff",
                mass_mev=self.ma,
            )
            if total_file is not None:
                self.totxs_evaluators[z] = load_total_xs_evaluator(total_file)
            if diff_file is not None:
                self.diffxs_evaluators[z] = load_diff_xs_evaluator(diff_file)

    def p1_cm(self, m1: float, m2: float, s: float) -> float:
        return float(np.sqrt((np.power(s - m1 * m1 - m2 * m2, 2) - np.power(2 * m1 * m2, 2)) / (4 * s)))

    def p3_cm(self, m3: float, m4: float, s: float) -> float:
        return float(np.sqrt((np.power(s - m3 * m3 - m4 * m4, 2) - np.power(2 * m3 * m4, 2)) / (4 * s)))

    def _table_total_xs(self, target_z: int, energy_mev: float) -> float | None:
        evaluator = self.totxs_evaluators.get(target_z)
        if evaluator is None:
            return None
        energy_mev = max(energy_mev, 1.0e-12)
        return float(evaluator(np.log10(energy_mev)))

    def _table_diff_xs(self, target_z: int, energy_mev: float, theta: float | np.ndarray) -> np.ndarray | None:
        evaluator = self.diffxs_evaluators.get(target_z)
        if evaluator is None:
            return None
        theta_arr = np.asarray(theta, dtype=float)
        theta_arr = np.clip(theta_arr, 1.0e-12, np.pi - 1.0e-12)
        loge = np.full_like(theta_arr, np.log10(max(energy_mev, 1.0e-12)))
        values = evaluator(loge, np.log10(theta_arr))
        return np.asarray(values, dtype=float)

    def _convert_mev2_to_cm2(self, xs_mev2: float) -> float:
        xs_cm2 = float(xs_mev2) * (HBARC_MEV_CM * HBARC_MEV_CM)
        if not np.isfinite(xs_cm2) or xs_cm2 < 0.0:
            return 0.0
        return xs_cm2

    def _build_basis_from_direction(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_hat = _safe_unit(direction, fallback=np.asarray([0.0, 0.0, 1.0], dtype=float))
        x_hat = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=float), k_hat)
        if np.linalg.norm(x_hat) < 1.0e-8:
            x_hat = np.cross(np.asarray([0.0, 1.0, 0.0], dtype=float), k_hat)
        x_hat = _safe_unit(x_hat)
        y_hat = _safe_unit(np.cross(k_hat, x_hat))
        return k_hat, x_hat, y_hat

    def equal(self, other) -> bool:
        return False


class AxionLikeParticlePrimakoffPhysicalWeighting(_PrimakoffBase):
    def __init__(
        self,
        ma: float = 1.0e-2,
        g: float = 1.0e-5,
        target_nuclei: list | None = None,
        spline_dir: str | Path | None = None,
        allow_analytic_fallback: bool = True,
    ):
        target_mats = {
            74: "W",
            8: "O",
            11: "Na",
            14: "Si",
            20: "Ca",
            13: "Al",
            26: "Fe",
            6: "C",
            25: "Mn",
            82: "Pb",
            29: "Cu",
            4: "Be",
        }
        default_targets = [
            siren.dataclasses.Particle.ParticleType.W183Nucleus,
            siren.dataclasses.Particle.ParticleType.O16Nucleus,
            siren.dataclasses.Particle.ParticleType.Na23Nucleus,
            siren.dataclasses.Particle.ParticleType.Si28Nucleus,
            siren.dataclasses.Particle.ParticleType.Ca40Nucleus,
            siren.dataclasses.Particle.ParticleType.Al27Nucleus,
            siren.dataclasses.Particle.ParticleType.Fe56Nucleus,
            siren.dataclasses.Particle.ParticleType.C12Nucleus,
            siren.dataclasses.Particle.ParticleType.Mn55Nucleus,
            siren.dataclasses.Particle.ParticleType.Pb208Nucleus,
            siren.dataclasses.Particle.ParticleType.Cu63Nucleus,
            siren.dataclasses.Particle.ParticleType.Be9Nucleus,
        ]
        _PrimakoffBase.__init__(
            self,
            ma=ma,
            g=g,
            process_name="primakoff",
            target_nuclei=target_nuclei,
            default_target_nuclei=default_targets,
            target_mats=target_mats,
            spline_dir=spline_dir,
            allow_analytic_fallback=allow_analytic_fallback,
        )

    def _analytic_diff_xs_mev2(self, record, theta: float | np.ndarray) -> np.ndarray:
        target_z = extract_z(int(record.signature.target_type))
        target_mass = float(record.target_mass) * SIREN_TO_ALPLIB
        initial_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_p1 = LorentzVector(*initial_4vec.tolist())
        lv_p2 = LorentzVector(target_mass, 0.0, 0.0, 0.0)
        cm_p4 = lv_p1 + lv_p2
        s = cm_p4.mass2()

        mtrx2 = M2Primakoff(self.ma, target_mass, target_z)
        m1, m2, m3, m4 = mtrx2.m1, mtrx2.m2, mtrx2.m3, mtrx2.m4
        p1_cm = self.p1_cm(m1, m2, s)
        p3_cm = self.p3_cm(m3, m4, s)
        e1_cm = np.sqrt(p1_cm * p1_cm + m1 * m1)
        e3_cm = np.sqrt(p3_cm * p3_cm + m3 * m3)

        theta_arr = np.asarray(theta, dtype=float)
        t = m1 * m1 + m3 * m3 + 2.0 * (p1_cm * p3_cm * np.cos(theta_arr) - e1_cm * e3_cm)
        denom = 16.0 * np.pi * (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
        if denom == 0.0:
            return np.zeros_like(theta_arr)
        dsigma_dt = np.asarray([mtrx2(s, float(tt), self.g) for tt in np.ravel(t)], dtype=float).reshape(theta_arr.shape)
        dsigma_dt /= denom
        dt_dtheta = 2.0 * p1_cm * p3_cm * np.sin(theta_arr)
        return np.maximum(dsigma_dt * dt_dtheta, 0.0)

    def _total_xs_mev2(self, record) -> float:
        target_z = extract_z(int(record.signature.target_type))
        photon_energy_mev = float(record.primary_momentum[0]) * SIREN_TO_ALPLIB
        table_value = self._table_total_xs(target_z, photon_energy_mev)
        if table_value is not None:
            return max(0.0, float(table_value * (self.g * self.g)))
        if not self.allow_analytic_fallback:
            return 0.0
        integrand = lambda th: float(self._analytic_diff_xs_mev2(record, th))
        return max(0.0, float(quad(integrand, 0.0, np.pi, limit=300)[0]))

    def _diff_xs_mev2(self, record, theta: float | np.ndarray) -> np.ndarray:
        target_z = extract_z(int(record.signature.target_type))
        photon_energy_mev = float(record.primary_momentum[0]) * SIREN_TO_ALPLIB
        table_value = self._table_diff_xs(target_z, photon_energy_mev, theta)
        if table_value is not None:
            return np.maximum(0.0, table_value * (self.g * self.g))
        if not self.allow_analytic_fallback:
            theta_arr = np.asarray(theta, dtype=float)
            return np.zeros_like(theta_arr)
        return self._analytic_diff_xs_mev2(record, theta)

    def scatter_sim(self, record, random):
        target_z = extract_z(int(record.signature.target_type))
        target_mass = float(record.target_mass) * SIREN_TO_ALPLIB

        mtrx2 = M2Primakoff(self.ma, target_mass, target_z)
        m1, m2, m3, m4 = mtrx2.m1, mtrx2.m2, mtrx2.m3, mtrx2.m4
        initial_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_p1 = LorentzVector(*initial_4vec.tolist())
        lv_p2 = LorentzVector(target_mass, 0.0, 0.0, 0.0)

        cm_p4 = lv_p1 + lv_p2
        e_in = cm_p4.energy()
        p_in = cm_p4.get_3momentum()
        v_in = Vector3(p_in.v1 / e_in, p_in.v2 / e_in, p_in.v3 / e_in)
        s = cm_p4.mass2()
        if s < (m3 + m4) ** 2:
            return None

        p1_cm = self.p1_cm(m1, m2, s)
        p3_cm = self.p3_cm(m3, m4, s)
        e3_cm = np.sqrt(p3_cm * p3_cm + m3 * m3)

        eps = 1.0e-12
        cos_theta_vals = np.linspace(-1.0 + eps, 1.0 - eps, 500)
        theta_vals = np.arccos(cos_theta_vals)
        pdf_vals = self._diff_xs_mev2(record, theta_vals) * np.sin(theta_vals)
        pdf_vals = np.maximum(pdf_vals, 0.0)
        pdf_sum = float(np.sum(pdf_vals))
        if not np.isfinite(pdf_sum) or pdf_sum <= 0.0:
            return None
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        theta_rnd = float(np.interp(_uniform(random), cdf_vals, theta_vals))
        phi_rnd = TWO_PI * _uniform(random)

        k_hat, x_hat, y_hat = self._build_basis_from_direction(lv_p1.get_3momentum().vec)
        p3_cm_vec = p3_cm * (
            np.sin(theta_rnd) * np.cos(phi_rnd) * x_hat
            + np.sin(theta_rnd) * np.sin(phi_rnd) * y_hat
            + np.cos(theta_rnd) * k_hat
        )
        p3_cm_4vector = LorentzVector(e3_cm, *p3_cm_vec.tolist())
        p3_lab_4vector = lorentz_boost(p3_cm_4vector, -v_in)
        p4_lab_4vector = lv_p1 + lv_p2 - p3_lab_4vector
        return p3_lab_4vector, p4_lab_4vector

    def TotalCrossSection(self, record) -> float:
        return self._convert_mev2_to_cm2(self._total_xs_mev2(record))

    def DifferentialCrossSection(self, record) -> float:
        p1_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        p1_3vec = p1_4vec[1:]

        alp_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.ALP)
        if alp_idx is None:
            return 0.0
        p3_4vec = np.asarray(record.secondary_momenta[alp_idx], dtype=float) * SIREN_TO_ALPLIB
        p3_3vec = p3_4vec[1:]
        denom = float(np.linalg.norm(p1_3vec) * np.linalg.norm(p3_3vec))
        if denom <= 0.0:
            return 0.0
        costheta_lab = np.clip(float(np.dot(p1_3vec, p3_3vec) / denom), -1.0, 1.0)
        theta_lab = float(np.arccos(costheta_lab))

        diff_mev2 = float(self._diff_xs_mev2(record, theta_lab))
        diff_cm2 = self._convert_mev2_to_cm2(diff_mev2)
        return diff_cm2 / TWO_PI

    def InteractionThreshold(self, record) -> float:
        return 0.0

    def SampleFinalState(self, record, random):
        sampled = self.scatter_sim(record, random)
        if sampled is None:
            return
        alp_lab_4vector, nucleus_lab_4vector = sampled
        secondary_particles = record.get_secondary_particle_records()

        alp_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.ALP)
        if alp_idx is None:
            return
        secondary_particles[alp_idx].mass = self.ma * ALPLIB_TO_SIREN
        secondary_particles[alp_idx].energy = alp_lab_4vector.energy() * ALPLIB_TO_SIREN
        momentum_3vec = np.asarray([alp_lab_4vector.p1, alp_lab_4vector.p2, alp_lab_4vector.p3], dtype=float) * ALPLIB_TO_SIREN
        momentum_norm = float(np.linalg.norm(momentum_3vec))
        if momentum_norm <= 0.0:
            return
        secondary_particles[alp_idx].direction = (momentum_3vec / momentum_norm).tolist()

        nuclear_idx = 1 - alp_idx
        secondary_particles[nuclear_idx].mass = nucleus_lab_4vector.mass() * ALPLIB_TO_SIREN
        secondary_particles[nuclear_idx].energy = nucleus_lab_4vector.energy() * ALPLIB_TO_SIREN
        nucleus_p = np.asarray([nucleus_lab_4vector.p1, nucleus_lab_4vector.p2, nucleus_lab_4vector.p3], dtype=float) * ALPLIB_TO_SIREN
        nucleus_p_norm = float(np.linalg.norm(nucleus_p))
        if nucleus_p_norm <= 0.0:
            return
        secondary_particles[nuclear_idx].direction = (nucleus_p / nucleus_p_norm).tolist()

    def GetPossibleTargets(self):
        return list(self.target_nuclei)

    def GetPossibleTargetsFromPrimary(self, primary):
        if primary == siren.dataclasses.Particle.ParticleType.Gamma:
            return list(self.target_nuclei)
        return []

    def GetPossiblePrimaries(self):
        return [siren.dataclasses.Particle.ParticleType.Gamma]

    @per_instance_cache
    def GetPossibleSignatures(self):
        signatures = []
        for target in self.target_nuclei:
            signature = siren.dataclasses.InteractionSignature()
            signature.primary_type = siren.dataclasses.Particle.ParticleType.Gamma
            signature.target_type = target
            signature.secondary_types = [siren.dataclasses.Particle.ParticleType.ALP, target]
            signatures.append(signature)
        return signatures

    @per_instance_cache
    def GetPossibleSignaturesFromParents(self, primary, target):
        if primary == siren.dataclasses.Particle.ParticleType.Gamma and target in self.target_nuclei:
            signature = siren.dataclasses.InteractionSignature()
            signature.primary_type = siren.dataclasses.Particle.ParticleType.Gamma
            signature.target_type = target
            signature.secondary_types = [siren.dataclasses.Particle.ParticleType.ALP, target]
            return [signature]
        return []

    def DensityVariables(self):
        return []

    def FinalStateProbability(self, record):
        total_xs = self.TotalCrossSection(record)
        differential_xs = self.DifferentialCrossSection(record)
        if differential_xs == 0.0 or total_xs == 0.0:
            return 0.0
        return differential_xs / total_xs


class AxionLikeParticlePrimakoffBiasedInjection(AxionLikeParticlePrimakoffPhysicalWeighting):
    def __init__(
        self,
        ma: float = 1.0e-2,
        g: float = 1.0e-5,
        target_nuclei: list | None = None,
        spline_dir: str | Path | None = None,
        allow_analytic_fallback: bool = True,
    ):
        AxionLikeParticlePrimakoffPhysicalWeighting.__init__(
            self,
            ma=ma,
            g=g,
            target_nuclei=target_nuclei,
            spline_dir=spline_dir,
            allow_analytic_fallback=allow_analytic_fallback,
        )
        self.unrotated_detector_axis = np.asarray([0.99966189, 0.0, -0.02600208], dtype=float)
        self.detector_rotation = (TWO_PI / 24.0) * (3.5 - 1.0)
        self.rotated_detector_axis = np.asarray(
            [
                self.unrotated_detector_axis[0] * np.cos(self.detector_rotation)
                - self.unrotated_detector_axis[1] * np.sin(self.detector_rotation),
                self.unrotated_detector_axis[0] * np.sin(self.detector_rotation)
                + self.unrotated_detector_axis[1] * np.cos(self.detector_rotation),
                self.unrotated_detector_axis[2],
            ],
            dtype=float,
        )
        self.rotated_detector_axis = _safe_unit(self.rotated_detector_axis)
        self.max_cone_half_angle = np.arctan(np.sqrt(1.0**2 + 0.6**2) / 23.0) * 1.5

        tmp = np.asarray([0.0, 0.0, 1.0], dtype=float)
        self.det_x_hat = _safe_unit(np.cross(tmp, self.rotated_detector_axis))
        self.det_y_hat = _safe_unit(np.cross(self.rotated_detector_axis, self.det_x_hat))

    def scatter_sim(self, record, random):
        target_z = extract_z(int(record.signature.target_type))
        target_mass = float(record.target_mass) * SIREN_TO_ALPLIB

        mtrx2 = M2Primakoff(self.ma, target_mass, target_z)
        m1, m2, m3, m4 = mtrx2.m1, mtrx2.m2, mtrx2.m3, mtrx2.m4
        initial_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_p1 = LorentzVector(*initial_4vec.tolist())
        lv_p2 = LorentzVector(target_mass, 0.0, 0.0, 0.0)

        cm_p4 = lv_p1 + lv_p2
        e_in = cm_p4.energy()
        p_in = cm_p4.get_3momentum()
        v_in = Vector3(p_in.v1 / e_in, p_in.v2 / e_in, p_in.v3 / e_in)
        s = cm_p4.mass2()
        if s < (m3 + m4) ** 2:
            return None

        p3_cm = self.p3_cm(m3, m4, s)
        e3_cm = np.sqrt(p3_cm * p3_cm + m3 * m3)

        phi_rnd = TWO_PI * _uniform(random)
        cos_theta = 1.0 - _uniform(random) * (1.0 - np.cos(self.max_cone_half_angle))
        theta_rnd = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        p3_cm_vec = p3_cm * (
            np.sin(theta_rnd) * np.cos(phi_rnd) * self.det_x_hat
            + np.sin(theta_rnd) * np.sin(phi_rnd) * self.det_y_hat
            + np.cos(theta_rnd) * self.rotated_detector_axis
        )
        p3_cm_4vector = LorentzVector(e3_cm, *p3_cm_vec.tolist())
        p3_lab_4vector = lorentz_boost(p3_cm_4vector, -v_in)
        p4_lab_4vector = lv_p1 + lv_p2 - p3_lab_4vector
        return p3_lab_4vector, p4_lab_4vector

    def DifferentialCrossSection(self, record) -> float:
        sampling_cone_pdf = 1.0 / (TWO_PI * (1.0 - np.cos(self.max_cone_half_angle)))
        gamma_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        gamma_3vec = gamma_4vec[1:]
        gamma_direction = _safe_unit(gamma_3vec, fallback=self.rotated_detector_axis)
        costheta = float(np.clip(np.dot(gamma_direction, self.rotated_detector_axis), -1.0, 1.0))
        cone_to_physical_jacobian = float(np.sqrt(max(0.0, 1.0 - costheta * costheta)))
        total_xs = self.TotalCrossSection(record)
        return total_xs * sampling_cone_pdf * cone_to_physical_jacobian


class AxionLikeParticleInversePrimakoff(_PrimakoffBase):
    def __init__(
        self,
        ma: float = 1.0e-2,
        g: float = 1.0e-5,
        target_nuclei: list | None = None,
        spline_dir: str | Path | None = None,
        allow_analytic_fallback: bool = True,
    ):
        target_mats = {18: "Ar"}
        default_targets = [siren.dataclasses.Particle.ParticleType.Ar40Nucleus]
        siren.interactions.CrossSection.__init__(self)
        super().__init__(
            ma=ma,
            g=g,
            process_name="invprimakoff",
            target_nuclei=target_nuclei,
            default_target_nuclei=default_targets,
            target_mats=target_mats,
            spline_dir=spline_dir,
            allow_analytic_fallback=allow_analytic_fallback,
        )

    def _analytic_diff_xs_mev2(self, record, theta: float | np.ndarray) -> np.ndarray:
        target_z = extract_z(int(record.signature.target_type))
        target_mass = float(record.target_mass) * SIREN_TO_ALPLIB
        initial_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_p1 = LorentzVector(*initial_4vec.tolist())
        lv_p2 = LorentzVector(target_mass, 0.0, 0.0, 0.0)
        cm_p4 = lv_p1 + lv_p2
        s = cm_p4.mass2()

        mtrx2 = M2InversePrimakoff(self.ma, target_mass, target_z)
        m1, m2, m3, m4 = mtrx2.m1, mtrx2.m2, mtrx2.m3, mtrx2.m4
        p1_cm = self.p1_cm(m1, m2, s)
        p3_cm = self.p3_cm(m3, m4, s)
        e1_cm = np.sqrt(p1_cm * p1_cm + m1 * m1)
        e3_cm = np.sqrt(p3_cm * p3_cm + m3 * m3)

        theta_arr = np.asarray(theta, dtype=float)
        t = m1 * m1 + m3 * m3 + 2.0 * (p1_cm * p3_cm * np.cos(theta_arr) - e1_cm * e3_cm)
        denom = 16.0 * np.pi * (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
        if denom == 0.0:
            return np.zeros_like(theta_arr)
        dsigma_dt = np.asarray([mtrx2(s, float(tt), self.g) for tt in np.ravel(t)], dtype=float).reshape(theta_arr.shape)
        dsigma_dt /= denom
        dt_dtheta = 2.0 * p1_cm * p3_cm * np.sin(theta_arr)
        return np.maximum(dsigma_dt * dt_dtheta, 0.0)

    def _total_xs_mev2(self, record) -> float:
        target_z = extract_z(int(record.signature.target_type))
        alp_energy_mev = float(record.primary_momentum[0]) * SIREN_TO_ALPLIB
        table_value = self._table_total_xs(target_z, alp_energy_mev)
        if table_value is not None:
            return max(0.0, float(table_value * (self.g * self.g)))
        if not self.allow_analytic_fallback:
            return 0.0
        integrand = lambda th: float(self._analytic_diff_xs_mev2(record, th))
        return max(0.0, float(quad(integrand, 0.0, np.pi, limit=300)[0]))

    def _diff_xs_mev2(self, record, theta: float | np.ndarray) -> np.ndarray:
        target_z = extract_z(int(record.signature.target_type))
        alp_energy_mev = float(record.primary_momentum[0]) * SIREN_TO_ALPLIB
        table_value = self._table_diff_xs(target_z, alp_energy_mev, theta)
        if table_value is not None:
            return np.maximum(0.0, table_value * (self.g * self.g))
        if not self.allow_analytic_fallback:
            theta_arr = np.asarray(theta, dtype=float)
            return np.zeros_like(theta_arr)
        return self._analytic_diff_xs_mev2(record, theta)

    def scatter_sim(self, record, random):
        target_z = extract_z(int(record.signature.target_type))
        target_mass = float(record.target_mass) * SIREN_TO_ALPLIB

        mtrx2 = M2InversePrimakoff(self.ma, target_mass, target_z)
        m1, m2, m3, m4 = mtrx2.m1, mtrx2.m2, mtrx2.m3, mtrx2.m4
        initial_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_p1 = LorentzVector(*initial_4vec.tolist())
        lv_p2 = LorentzVector(target_mass, 0.0, 0.0, 0.0)

        cm_p4 = lv_p1 + lv_p2
        e_in = cm_p4.energy()
        p_in = cm_p4.get_3momentum()
        v_in = Vector3(p_in.v1 / e_in, p_in.v2 / e_in, p_in.v3 / e_in)
        s = cm_p4.mass2()
        if s < (m3 + m4) ** 2:
            return None

        p3_cm = self.p3_cm(m3, m4, s)
        e3_cm = np.sqrt(p3_cm * p3_cm + m3 * m3)

        eps = 1.0e-12
        cos_theta_vals = np.linspace(-1.0 + eps, 1.0 - eps, 500)
        theta_vals = np.arccos(cos_theta_vals)
        pdf_vals = self._diff_xs_mev2(record, theta_vals) * np.sin(theta_vals)
        pdf_vals = np.maximum(pdf_vals, 0.0)
        pdf_sum = float(np.sum(pdf_vals))
        if not np.isfinite(pdf_sum) or pdf_sum <= 0.0:
            return None
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        theta_rnd = float(np.interp(_uniform(random), cdf_vals, theta_vals))
        phi_rnd = TWO_PI * _uniform(random)

        k_hat, x_hat, y_hat = self._build_basis_from_direction(lv_p1.get_3momentum().vec)
        p3_cm_vec = p3_cm * (
            np.sin(theta_rnd) * np.cos(phi_rnd) * x_hat
            + np.sin(theta_rnd) * np.sin(phi_rnd) * y_hat
            + np.cos(theta_rnd) * k_hat
        )
        p3_cm_4vector = LorentzVector(e3_cm, *p3_cm_vec.tolist())
        p3_lab_4vector = lorentz_boost(p3_cm_4vector, -v_in)
        p4_lab_4vector = lv_p1 + lv_p2 - p3_lab_4vector
        return p3_lab_4vector, p4_lab_4vector

    def TotalCrossSection(self, record) -> float:
        return self._convert_mev2_to_cm2(self._total_xs_mev2(record))

    def DifferentialCrossSection(self, record) -> float:
        p1_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        p1_3vec = p1_4vec[1:]
        gamma_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.Gamma)
        if gamma_idx is None:
            return 0.0
        p3_4vec = np.asarray(record.secondary_momenta[gamma_idx], dtype=float) * SIREN_TO_ALPLIB
        p3_3vec = p3_4vec[1:]
        denom = float(np.linalg.norm(p1_3vec) * np.linalg.norm(p3_3vec))
        if denom <= 0.0:
            return 0.0
        costheta_lab = np.clip(float(np.dot(p1_3vec, p3_3vec) / denom), -1.0, 1.0)
        theta_lab = float(np.arccos(costheta_lab))
        return self._convert_mev2_to_cm2(float(self._diff_xs_mev2(record, theta_lab)))

    def InteractionThreshold(self, record) -> float:
        return 0.0

    def SampleFinalState(self, record, random):
        sampled = self.scatter_sim(record, random)
        if sampled is None:
            return
        gamma_lab_4vector, nucleus_lab_4vector = sampled
        secondary_particles = record.get_secondary_particle_records()

        gamma_idx = _find_secondary_index_by_type(record, siren.dataclasses.Particle.ParticleType.Gamma)
        if gamma_idx is None:
            return
        secondary_particles[gamma_idx].mass = 0.0
        secondary_particles[gamma_idx].energy = gamma_lab_4vector.energy() * ALPLIB_TO_SIREN
        gamma_p = np.asarray([gamma_lab_4vector.p1, gamma_lab_4vector.p2, gamma_lab_4vector.p3], dtype=float) * ALPLIB_TO_SIREN
        gamma_p_norm = float(np.linalg.norm(gamma_p))
        if gamma_p_norm <= 0.0:
            return
        secondary_particles[gamma_idx].direction = (gamma_p / gamma_p_norm).tolist()

        nuclear_idx = 1 - gamma_idx
        secondary_particles[nuclear_idx].mass = nucleus_lab_4vector.mass() * ALPLIB_TO_SIREN
        secondary_particles[nuclear_idx].energy = nucleus_lab_4vector.energy() * ALPLIB_TO_SIREN
        nucleus_p = np.asarray([nucleus_lab_4vector.p1, nucleus_lab_4vector.p2, nucleus_lab_4vector.p3], dtype=float) * ALPLIB_TO_SIREN
        nucleus_p_norm = float(np.linalg.norm(nucleus_p))
        if nucleus_p_norm <= 0.0:
            return
        secondary_particles[nuclear_idx].direction = (nucleus_p / nucleus_p_norm).tolist()

    def GetPossibleTargets(self):
        return list(self.target_nuclei)

    def GetPossibleTargetsFromPrimary(self, primary):
        if primary == siren.dataclasses.Particle.ParticleType.ALP:
            return list(self.target_nuclei)
        return []

    def GetPossiblePrimaries(self):
        return [siren.dataclasses.Particle.ParticleType.ALP]

    @per_instance_cache
    def GetPossibleSignatures(self):
        signatures = []
        for target in self.target_nuclei:
            signature = siren.dataclasses.InteractionSignature()
            signature.primary_type = siren.dataclasses.Particle.ParticleType.ALP
            signature.target_type = target
            signature.secondary_types = [siren.dataclasses.Particle.ParticleType.Gamma, target]
            signatures.append(signature)
        return signatures

    @per_instance_cache
    def GetPossibleSignaturesFromParents(self, primary, target):
        if primary == siren.dataclasses.Particle.ParticleType.ALP and target in self.target_nuclei:
            signature = siren.dataclasses.InteractionSignature()
            signature.primary_type = siren.dataclasses.Particle.ParticleType.ALP
            signature.target_type = target
            signature.secondary_types = [siren.dataclasses.Particle.ParticleType.Gamma, target]
            return [signature]
        return []

    def DensityVariables(self):
        return []

    def FinalStateProbability(self, record):
        total_xs = self.TotalCrossSection(record)
        differential_xs = self.DifferentialCrossSection(record)
        if differential_xs == 0.0 or total_xs == 0.0:
            return 0.0
        return differential_xs / total_xs


class AxionLikeParticleDiphoton(siren.interactions.Decay):
    def __init__(self, ma: float = 1.0e-3, g: float = 1.0e-5):
        siren.interactions.Decay.__init__(self)
        self.ma = ma * SIREN_TO_ALPLIB
        self.g = g * ALPLIB_TO_SIREN

    def equal(self, other) -> bool:
        return False

    def scatter_sim(self, record, random):
        initial_axion_4vec = np.asarray(record.primary_momentum, dtype=float) * SIREN_TO_ALPLIB
        lv_parent = LorentzVector(*initial_axion_4vec.tolist())
        mp = lv_parent.mass()
        m1 = 0.0
        m2 = 0.0
        if mp <= 0.0:
            return None
        p_cm = np.sqrt(max((mp * mp - (m2 - m1) ** 2) * (mp * mp - (m2 + m1) ** 2), 0.0)) / (2.0 * mp)
        e1_cm = np.sqrt(p_cm * p_cm + m1 * m1)
        e2_cm = np.sqrt(p_cm * p_cm + m2 * m2)

        phi_rnd = TWO_PI * _uniform(random)
        theta_rnd = float(np.arccos(1.0 - 2.0 * _uniform(random)))
        v_in = -lv_parent.get_3velocity()

        k_hat, x_hat, y_hat = _safe_unit(lv_parent.get_3momentum().vec), None, None
        x_hat = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=float), k_hat)
        if np.linalg.norm(x_hat) < 1.0e-8:
            x_hat = np.cross(np.asarray([0.0, 1.0, 0.0], dtype=float), k_hat)
        x_hat = _safe_unit(x_hat)
        y_hat = _safe_unit(np.cross(k_hat, x_hat))

        p1_cm_vec = p_cm * (
            np.sin(theta_rnd) * np.cos(phi_rnd) * x_hat
            + np.sin(theta_rnd) * np.sin(phi_rnd) * y_hat
            + np.cos(theta_rnd) * k_hat
        )
        p2_cm_vec = -p1_cm_vec

        p1_cm_4vector = LorentzVector(e1_cm, *p1_cm_vec.tolist())
        p2_cm_4vector = LorentzVector(e2_cm, *p2_cm_vec.tolist())
        p1_lab = lorentz_boost(p1_cm_4vector, v_in)
        p2_lab = lorentz_boost(p2_cm_4vector, v_in)
        return p1_lab, p2_lab

    def TotalDecayWidth(self, record):
        total_decay_width = self.g * self.g * self.ma**3 / (64.0 * np.pi)
        return total_decay_width * ALPLIB_TO_SIREN

    def DifferentialDecayWidth(self, record):
        return 1.0

    def SampleFinalState(self, record, random):
        sampled = self.scatter_sim(record, random)
        if sampled is None:
            return
        gamma1_lab, gamma2_lab = sampled
        secondary_particles = record.get_secondary_particle_records()
        if len(secondary_particles) < 2:
            return
        gamma_vectors = [gamma1_lab, gamma2_lab]
        for idx, gv in enumerate(gamma_vectors):
            secondary_particles[idx].mass = 0.0
            secondary_particles[idx].energy = gv.energy() * ALPLIB_TO_SIREN
            p3 = np.asarray([gv.p1, gv.p2, gv.p3], dtype=float) * ALPLIB_TO_SIREN
            pnorm = float(np.linalg.norm(p3))
            if pnorm <= 0.0:
                continue
            secondary_particles[idx].direction = (p3 / pnorm).tolist()

    def GetPossibleSignatures(self):
        signature = siren.dataclasses.InteractionSignature()
        signature.primary_type = siren.dataclasses.Particle.ParticleType.ALP
        signature.secondary_types = [
            siren.dataclasses.Particle.ParticleType.Gamma,
            siren.dataclasses.Particle.ParticleType.Gamma,
        ]
        return [signature]

    def GetPossibleSignaturesFromParent(self, primary):
        if primary == siren.dataclasses.Particle.ParticleType.ALP:
            return self.GetPossibleSignatures()
        return []

    def DensityVariables(self):
        return []

    def FinalStateProbability(self, record):
        return 1.0
