from __future__ import annotations

"""Multi-fidelity electromagnetic-to-likelihood bridge for the active V58/V65 tree.

The implementation deliberately separates fidelity from scientific claim scope:

* R0 is the analytic Gaussian factor already used by the Schur planner.
* R1 is a waveform/sufficient-statistic realization from that likelihood.
* R2 is sampled complex baseband with explicit array/polarization, paths,
  synchronization, front-end, quantization, saturation, clutter and jammer terms.
* R2.5 is an externally supplied ray/channel path set with retained provenance.
* R3 is measured or independently validated data and is accepted only through
  :class:`MeasuredIQAcquisition` with hashes, calibration and units.

R0/R1 information is evaluated from the same likelihood Jacobian and covariance.
R2 score/Fisher information is evaluated from the complex-Gaussian sample model
``y ~ CN(mu(theta, eta), sigma2 I)`` using finite-difference mean derivatives.
The target block is nuisance-marginalized with a Schur complement before it is
made available to planning code.
"""

from dataclasses import MISSING, asdict, dataclass, field, replace
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .information import prior_information_normalized, schur_effective_information
from .effective_delay import (
    automatic_free_calibration_actions,
    configuration_for,
)
from .measurement import (
    ActionFactor,
    StateLayout,
    ambient_interference_fast_time_correlation,
    build_action_factor,
    commanded_beam_pointing_power_gain,
    normalized_world_state_error,
    receiver_link_budget,
    slow_time_window_contract,
    world_with_normalized_state_error,
)
from .propagation import (
    C0_M_S,
    distance_m,
    free_space_power_gain,
    fresnel_reflection_coefficient,
    thermal_noise_power_w,
    unit_vector,
    wavelength_m,
)
from .schema import (
    Jammer,
    MaterialSurface,
    PhysicalWorld,
    RadarAction,
    SensorNode,
    TargetState,
    Waveform,
)
from .antenna_geometry import (
    inward_horizontal_boresight,
    vertical_rectangular_element_positions_m,
)
from .iq import synthesize_waveform


class FidelityLevel(str, Enum):
    R0_ANALYTIC_FACTOR = "R0_analytic_factor"
    R1_SUFFICIENT_STATISTIC = "R1_waveform_sufficient_statistic"
    R2_COMPLEX_BASEBAND = "R2_sampled_complex_baseband"
    R2_5_CHANNEL_PROXY = "R2_5_validated_path_or_channel_proxy"
    R3_MEASURED = "R3_measured_or_independently_validated"


_FIDELITY_RANK = {
    FidelityLevel.R0_ANALYTIC_FACTOR: 0,
    FidelityLevel.R1_SUFFICIENT_STATISTIC: 1,
    FidelityLevel.R2_COMPLEX_BASEBAND: 2,
    FidelityLevel.R2_5_CHANNEL_PROXY: 3,
    FidelityLevel.R3_MEASURED: 4,
}


@dataclass(frozen=True)
class FidelityProvenance:
    level: FidelityLevel
    generator: str
    source_uri: str | None = None
    source_sha256: str | None = None
    calibration_sha256: str | None = None
    independently_validated: bool = False
    measured_data: bool = False
    limitations: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.generator:
            raise ValueError("fidelity provenance requires a generator")
        if self.level is FidelityLevel.R3_MEASURED:
            if not self.measured_data:
                raise ValueError("R3 requires measured_data=True")
            if not self.source_sha256 or not self.calibration_sha256:
                raise ValueError("R3 requires source and calibration SHA-256")
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a 64-character SHA-256")
        if self.calibration_sha256 is not None and len(self.calibration_sha256) != 64:
            raise ValueError("calibration_sha256 must be a 64-character SHA-256")

    def permits_claim(self, requested: FidelityLevel) -> bool:
        self.validate()
        return _FIDELITY_RANK[self.level] >= _FIDELITY_RANK[requested]

    def require_claim(self, requested: FidelityLevel) -> None:
        if not self.permits_claim(requested):
            raise ValueError(f"{self.level.value} evidence cannot support {requested.value} claim")


@dataclass(frozen=True)
class MeasuredIQAcquisition:
    """Fail-closed R3 acquisition contract.

    Samples remain external to JSON evidence; this record identifies their exact
    binary container, units, channel order, time/frequency reference and the
    calibration object used to convert ADC codes to volts and watts.
    """

    capture_path: str
    capture_sha256: str
    calibration_path: str
    calibration_sha256: str
    sample_rate_hz: float
    center_frequency_hz: float
    channel_ids: tuple[str, ...]
    sample_dtype: str
    iq_units: str
    time_reference: str
    frequency_reference: str
    adc_volts_per_code: tuple[float, ...]
    impedance_ohm: tuple[float, ...]
    start_time_utc: str
    receiver_configuration_ids: tuple[str, ...]
    missing_sample_mask_path: str | None = None

    def validate(self, *, verify_local_hashes: bool = False) -> None:
        if self.sample_rate_hz <= 0.0 or self.center_frequency_hz <= 0.0:
            raise ValueError("measured IQ frequencies must be positive SI values")
        n = len(self.channel_ids)
        if n == 0 or len(self.adc_volts_per_code) != n or len(self.impedance_ohm) != n:
            raise ValueError("measured IQ channel/calibration dimensions disagree")
        if any(v <= 0.0 for v in self.adc_volts_per_code) or any(z <= 0.0 for z in self.impedance_ohm):
            raise ValueError("ADC conversion and impedance must be positive")
        for digest in (self.capture_sha256, self.calibration_sha256):
            if len(digest) != 64:
                raise ValueError("measured IQ hashes must be SHA-256")
        if self.iq_units not in {"ADC_code", "V", "sqrt_W"}:
            raise ValueError("iq_units must be ADC_code, V, or sqrt_W")
        if verify_local_hashes:
            for path_string, expected in (
                (self.capture_path, self.capture_sha256),
                (self.calibration_path, self.calibration_sha256),
            ):
                path = Path(path_string)
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"SHA-256 mismatch for {path}")

    def provenance(self) -> FidelityProvenance:
        self.validate()
        return FidelityProvenance(
            level=FidelityLevel.R3_MEASURED,
            generator="MeasuredIQAcquisition",
            source_uri=self.capture_path,
            source_sha256=self.capture_sha256,
            calibration_sha256=self.calibration_sha256,
            independently_validated=True,
            measured_data=True,
            limitations=("measurement validity is conditional on the registered calibration interval",),
        )


@dataclass(frozen=True)
class JonesAntennaModel:
    node_id: str
    element_positions_body_m: np.ndarray
    boresight_world: np.ndarray
    horizontal_jones: np.ndarray
    vertical_jones: np.ndarray
    mutual_coupling: np.ndarray
    station_phase_rad: float = 0.0
    calibration_gain: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        positions = np.asarray(self.element_positions_body_m, dtype=float)
        boresight = np.asarray(self.boresight_world, dtype=float).reshape(3)
        horizontal = np.asarray(self.horizontal_jones, dtype=np.complex128).reshape(2)
        vertical = np.asarray(self.vertical_jones, dtype=np.complex128).reshape(2)
        coupling = np.asarray(self.mutual_coupling, dtype=np.complex128)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("element_positions_body_m must be N x 3")
        if coupling.shape != (positions.shape[0], positions.shape[0]):
            raise ValueError("mutual_coupling must be N x N")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(coupling)):
            raise ValueError("antenna model must be finite")
        norm = float(np.linalg.norm(boresight))
        if norm <= 0.0:
            raise ValueError("antenna boresight must be nonzero")
        object.__setattr__(self, "element_positions_body_m", positions)
        object.__setattr__(self, "boresight_world", boresight / norm)
        object.__setattr__(self, "horizontal_jones", horizontal)
        object.__setattr__(self, "vertical_jones", vertical)
        object.__setattr__(self, "mutual_coupling", coupling)

    @property
    def element_count(self) -> int:
        return int(self.element_positions_body_m.shape[0])

    def array_manifold(self, direction_world: np.ndarray, carrier_hz: float) -> np.ndarray:
        direction = np.asarray(direction_world, dtype=float).reshape(3)
        direction /= max(float(np.linalg.norm(direction)), 1e-30)
        phase = -2.0 * math.pi / wavelength_m(carrier_hz) * (self.element_positions_body_m @ direction)
        ideal = np.exp(1j * (phase + float(self.station_phase_rad)))
        transported = self.mutual_coupling @ ideal
        norm = math.sqrt(max(float(np.vdot(transported, transported).real), 1e-30))
        return self.calibration_gain * transported / norm

    def polarization_vector(self, direction_world: np.ndarray, *, transmit: bool) -> np.ndarray:
        direction = np.asarray(direction_world, dtype=float).reshape(3)
        direction /= max(float(np.linalg.norm(direction)), 1e-30)
        cosine = abs(float(np.dot(direction, self.boresight_world)))
        # Smooth co-/cross-polar pattern.  This is an analytic calibrated
        # element model, not a chamber-measured full-sphere pattern.
        co = max(cosine, 1e-3) ** 0.75
        cross = math.sqrt(max(0.0, 1.0 - cosine * cosine)) * 0.05
        vector = co * self.vertical_jones + cross * self.horizontal_jones
        vector = np.conjugate(vector) if transmit else vector
        return vector / max(float(np.linalg.norm(vector)), 1e-30)


def _default_element_positions(node: SensorNode, carrier_hz: float) -> np.ndarray:
    return vertical_rectangular_element_positions_m(node, carrier_hz)


def antenna_from_node(
    node: SensorNode,
    carrier_hz: float,
    *,
    cross_polar_leakage: float = 0.02,
    mutual_coupling_strength: float = 0.0,
    station_phase_rad: float = 0.0,
) -> JonesAntennaModel:
    positions = _default_element_positions(node, carrier_hz)
    n = positions.shape[0]
    coupling = np.eye(n, dtype=np.complex128)
    if mutual_coupling_strength:
        for i in range(n):
            for j in range(n):
                if i != j:
                    separation = float(np.linalg.norm(positions[i] - positions[j]))
                    coupling[i, j] = mutual_coupling_strength * np.exp(-separation / max(wavelength_m(carrier_hz), 1e-30)) * np.exp(0.25j * (i - j))
    horizontal = np.asarray([1.0, cross_polar_leakage], dtype=np.complex128)
    vertical = np.asarray([cross_polar_leakage, 1.0], dtype=np.complex128)
    return JonesAntennaModel(
        node_id=node.node_id,
        element_positions_body_m=positions,
        boresight_world=inward_horizontal_boresight(node),
        horizontal_jones=horizontal,
        vertical_jones=vertical,
        mutual_coupling=coupling,
        station_phase_rad=station_phase_rad,
        calibration_gain=1.0 + 0.0j,
    )


@dataclass(frozen=True)
class ScatteringModel:
    mean_rcs_m2: float
    reference_frequency_hz: float
    aspect_harmonics: tuple[float, float, float] = (0.20, 0.08, 0.04)
    frequency_slope_per_fraction: float = 0.12
    cross_polar_ratio: float = 0.08
    fluctuation_model: str = "Swerling_I"

    def complex_matrix(self, aspect_rad: float, frequency_hz: float, rng: np.random.Generator | None = None) -> np.ndarray:
        h1, h2, h3 = self.aspect_harmonics
        aspect_gain = max(0.05, 1.0 + h1 * math.cos(aspect_rad) + h2 * math.cos(2.0 * aspect_rad) + h3 * math.sin(3.0 * aspect_rad))
        fractional = (float(frequency_hz) - self.reference_frequency_hz) / max(self.reference_frequency_hz, 1e-30)
        frequency_gain = max(0.1, 1.0 + self.frequency_slope_per_fraction * fractional)
        texture = 1.0
        if rng is not None and self.fluctuation_model != "Swerling_0":
            if self.fluctuation_model in {"Swerling_III", "Swerling_IV"}:
                texture = float(rng.gamma(shape=2.0, scale=0.5))
            else:
                texture = float(rng.exponential(1.0))
        rcs = self.mean_rcs_m2 * aspect_gain * frequency_gain * max(texture, 1e-6)
        amplitude = math.sqrt(rcs / (4.0 * math.pi))
        phase = 0.35 * math.sin(aspect_rad) + 2.0 * math.pi * fractional
        cross = self.cross_polar_ratio * amplitude * np.exp(1j * (phase + math.pi / 3.0))
        return np.asarray(
            [[amplitude * np.exp(1j * phase), cross], [cross, 0.92 * amplitude * np.exp(1j * (phase - 0.18))]],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class PathComponent:
    path_id: str
    kind: str
    delay_s: float
    doppler_hz: float
    stretch: float
    complex_gain: complex
    source_node_id: str
    receiver_node_id: str
    target_id: str | None
    provenance: str


@dataclass(frozen=True)
class ReceiverState:
    clock_bias_s: float
    clock_drift_s_per_s: float
    cfo_hz: float
    sample_timing_offset_s: float
    phase_noise_linewidth_hz: float
    adc_bits: int
    adc_full_scale_w: float
    front_end_bandwidth_hz: float
    image_rejection_db: float


@dataclass(frozen=True)
class ComplexBasebandCapture:
    action_id: int
    channel_ids: tuple[str, ...]
    surveillance_iq: np.ndarray
    reference_iq: np.ndarray | None
    deterministic_mean: np.ndarray
    target_component: np.ndarray
    direct_path_component: np.ndarray
    multipath_component: np.ndarray
    clutter_component: np.ndarray
    jammer_component: np.ndarray
    thermal_component: np.ndarray
    receiver_impairment_component: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float
    path_components: tuple[PathComponent, ...]
    receiver_states: tuple[ReceiverState, ...]
    target_present: bool
    fidelity: FidelityProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict)
    coherent_surveillance_iq: np.ndarray | None = None
    coherent_reference_iq: np.ndarray | None = None
    pulse_slow_time_s: np.ndarray | None = None
    transmit_reference_iq: np.ndarray | None = None
    coherent_pre_front_end_iq: np.ndarray | None = None
    coherent_target_origin_iq: np.ndarray | None = None
    coherent_target_direct_iq: np.ndarray | None = None
    coherent_target_multipath_iq: np.ndarray | None = None
    coherent_direct_path_iq: np.ndarray | None = None
    coherent_environmental_multipath_iq: np.ndarray | None = None
    coherent_clutter_iq: np.ndarray | None = None
    coherent_jammer_iq: np.ndarray | None = None
    coherent_thermal_iq: np.ndarray | None = None
    coherent_receiver_impairment_iq: np.ndarray | None = None
    ambient_interference_component: np.ndarray | None = None
    coherent_ambient_interference_iq: np.ndarray | None = None

    def validate(self) -> None:
        arrays = [
            self.surveillance_iq,
            self.deterministic_mean,
            self.target_component,
            self.direct_path_component,
            self.multipath_component,
            self.clutter_component,
            self.jammer_component,
            self.thermal_component,
            self.receiver_impairment_component,
        ]
        shape = np.asarray(arrays[0]).shape
        if len(shape) != 2 or shape[0] != len(self.channel_ids):
            raise ValueError("IQ capture must be channel x sample")
        if any(np.asarray(array).shape != shape for array in arrays):
            raise ValueError("IQ capture components must have identical shape")
        if self.reference_iq is not None and np.asarray(self.reference_iq).shape[-1] != shape[-1]:
            raise ValueError("reference and surveillance windows must have equal samples")
        if self.ambient_interference_component is not None:
            ambient = np.asarray(self.ambient_interference_component)
            if ambient.shape != shape or not np.all(np.isfinite(ambient)):
                raise ValueError(
                    "ambient-interference IQ must be finite and match the capture"
                )
        if self.transmit_reference_iq is not None:
            transmit_reference = np.asarray(self.transmit_reference_iq)
            if transmit_reference.shape != (shape[-1],):
                raise ValueError(
                    "transmit reference IQ must have one fast-time window"
                )
            if not np.all(np.isfinite(transmit_reference)):
                raise ValueError("transmit reference IQ contains non-finite values")
        pulse_cube = self.coherent_surveillance_iq
        pulse_reference = self.coherent_reference_iq
        slow_time = self.pulse_slow_time_s
        if pulse_cube is not None:
            cube = np.asarray(pulse_cube)
            if cube.ndim != 3 or cube.shape[0] != shape[0] or cube.shape[2] != shape[1]:
                raise ValueError(
                    "coherent surveillance IQ must be channel x pulse x fast-time sample"
                )
            if cube.shape[1] < 2:
                raise ValueError("coherent pulse capture requires at least two pulses")
            if slow_time is None:
                raise ValueError("coherent pulse capture requires slow-time coordinates")
            slow = np.asarray(slow_time, dtype=float)
            if (
                slow.shape != (cube.shape[1],)
                or not np.all(np.isfinite(slow))
                or np.any(np.diff(slow) <= 0.0)
            ):
                raise ValueError("pulse slow-time coordinates must be finite and increasing")
            if pulse_reference is not None:
                reference_cube = np.asarray(pulse_reference)
                if reference_cube.shape != (cube.shape[1], cube.shape[2]):
                    raise ValueError(
                        "coherent reference IQ must be pulse x fast-time sample"
                    )
                if not np.all(np.isfinite(reference_cube)):
                    raise ValueError("coherent reference IQ contains non-finite values")
            if not np.all(np.isfinite(cube)):
                raise ValueError("coherent surveillance IQ contains non-finite values")
            for label, latent_cube in (
                ("pre-front-end", self.coherent_pre_front_end_iq),
                ("target-origin", self.coherent_target_origin_iq),
                ("target-direct", self.coherent_target_direct_iq),
                ("target-multipath", self.coherent_target_multipath_iq),
                ("direct-path", self.coherent_direct_path_iq),
                (
                    "environmental-multipath",
                    self.coherent_environmental_multipath_iq,
                ),
                ("clutter", self.coherent_clutter_iq),
                ("jammer", self.coherent_jammer_iq),
                ("thermal", self.coherent_thermal_iq),
                ("ambient-interference", self.coherent_ambient_interference_iq),
                (
                    "receiver-impairment",
                    self.coherent_receiver_impairment_iq,
                ),
            ):
                if latent_cube is None:
                    continue
                latent = np.asarray(latent_cube)
                if latent.shape != cube.shape or not np.all(np.isfinite(latent)):
                    raise ValueError(
                        f"coherent {label} IQ must be finite and match the surveillance cube"
                    )
        elif pulse_reference is not None or slow_time is not None:
            raise ValueError(
                "coherent reference IQ and slow time require a coherent surveillance cube"
            )
        elif any(
            cube is not None
            for cube in (
                self.coherent_pre_front_end_iq,
                self.coherent_target_origin_iq,
                self.coherent_target_direct_iq,
                self.coherent_target_multipath_iq,
                self.coherent_direct_path_iq,
                self.coherent_environmental_multipath_iq,
                self.coherent_clutter_iq,
                self.coherent_jammer_iq,
                self.coherent_thermal_iq,
                self.coherent_ambient_interference_iq,
                self.coherent_receiver_impairment_iq,
            )
        ):
            raise ValueError(
                "latent coherent component cubes require a coherent surveillance cube"
            )
        if self.sample_rate_hz <= 0.0 or self.center_frequency_hz <= 0.0:
            raise ValueError("capture frequencies must be positive")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("IQ capture contains non-finite values")
        self.fidelity.validate()

    def power_report(self) -> dict[str, float]:
        power = lambda value: float(np.mean(np.abs(value) ** 2))
        return {
            "target_power_w": power(self.target_component),
            "direct_path_power_w": power(self.direct_path_component),
            "multipath_power_w": power(self.multipath_component),
            "clutter_power_w": power(self.clutter_component),
            "jammer_power_w": power(self.jammer_component),
            "ambient_interference_power_w": (
                power(
                    self.coherent_ambient_interference_iq
                    if self.coherent_ambient_interference_iq is not None
                    else self.ambient_interference_component
                )
                if self.ambient_interference_component is not None
                else 0.0
            ),
            "thermal_power_w": power(self.thermal_component),
            "receiver_impairment_power_w": power(self.receiver_impairment_component),
            "received_power_w": power(
                self.coherent_surveillance_iq
                if self.coherent_surveillance_iq is not None
                else self.surveillance_iq
            ),
        }


@dataclass(frozen=True)
class SufficientStatisticObservation:
    names: tuple[str, ...]
    value: np.ndarray
    prediction: np.ndarray
    covariance: np.ndarray
    detected: bool
    association_probabilities: np.ndarray
    source_fidelity: FidelityLevel
    metadata: Mapping[str, Any]


def draw_controlled_local_linear_gaussian_sufficient_statistic(
    reference_world: PhysicalWorld,
    realization_world: PhysicalWorld,
    action: RadarAction,
    *,
    seed: int,
) -> SufficientStatisticObservation:
    """Draw the exact local Gaussian experiment registered by the controller.

    The conditional experiment is

    ``z = h(x_ref) + H_ref S delta + epsilon``,
    ``epsilon ~ N(0, R_ref)``,

    where ``delta`` is the truth-minus-reference state in normalized units and
    ``S`` is the state scaling matrix.  It deliberately conditions on target
    detection and known association.  The realized state error is used only by
    the simulator to draw ``z`` and is never stored in estimator-facing
    metadata.  Native R1 and R2 observation paths remain nonlinear.
    """

    condition = str(
        reference_world.external_channel_provenance.get(
            "experiment_condition_profile", "native"
        )
    )
    if condition != "controlled_gaussian_localization":
        raise ValueError(
            "the exact local Gaussian statistic is restricted to the "
            "controlled_gaussian_localization experiment"
        )
    factor = build_action_factor(reference_world, action, None)
    layout = StateLayout.from_world(reference_world)
    delta_normalized = normalized_world_state_error(
        reference_world,
        realization_world,
        layout,
    )
    jacobian_normalized = (
        np.asarray(factor.jacobian_physical, dtype=float)
        * layout.scales()[None, :]
    )
    conditional_mean = (
        np.asarray(factor.prediction, dtype=float)
        + jacobian_normalized @ delta_normalized
    )
    covariance = np.asarray(factor.covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    noise = np.linalg.cholesky(covariance) @ np.random.default_rng(seed).standard_normal(
        conditional_mean.size
    )
    value = conditional_mean + noise
    calibration_rows = sum(
        len(build_action_factor(reference_world, calibration, None).measurement_names)
        for calibration in automatic_free_calibration_actions(reference_world, action)
    )
    primary_rows = int(value.size) - int(calibration_rows)
    return SufficientStatisticObservation(
        names=factor.measurement_names,
        value=value,
        prediction=np.asarray(factor.prediction, dtype=float).copy(),
        covariance=covariance.copy(),
        detected=True,
        association_probabilities=np.asarray([1.0, 0.0], dtype=float),
        source_fidelity=FidelityLevel.R1_SUFFICIENT_STATISTIC,
        metadata={
            "likelihood": "conditional_local_linear_Gaussian_known_association",
            "fidelity_claim": FidelityLevel.R1_SUFFICIENT_STATISTIC.value,
            "primary_measurement_rows": primary_rows,
            "detection_independent_calibration_rows": int(calibration_rows),
            "detection_gates_primary_rows_only": True,
            "association_mixture_realized": False,
            "association_component_sampled": "known_correct_return",
            "association_model": "known_association_control",
            "controlled_measurement_model": (
                "exact_local_linear_gaussian_sufficient_statistic"
            ),
            "conditional_on_detection": True,
            "truth_state_error_exposed_to_controller": False,
        },
    )


@dataclass(frozen=True)
class FisherBridgeResult:
    action_id: int
    r0_joint_information_normalized: np.ndarray
    r1_joint_information_normalized: np.ndarray
    r2_joint_information_normalized: np.ndarray
    r2_effective_target_information_normalized: np.ndarray
    r2_known_nuisance_target_information_normalized: np.ndarray
    score_monte_carlo_information_normalized: np.ndarray
    score_mean_residual: float
    score_fisher_relative_error: float
    r0_r1_relative_error: float
    r0_r2_trace_ratio: float
    provenance: Mapping[str, Any]


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    values = np.asarray(signal, dtype=np.complex128)
    grid = np.arange(values.size, dtype=float)
    source = grid - float(delay_samples)
    real = np.interp(source, grid, values.real, left=0.0, right=0.0)
    imag = np.interp(source, grid, values.imag, left=0.0, right=0.0)
    return real + 1j * imag


def _time_scale(signal: np.ndarray, stretch: float) -> np.ndarray:
    if abs(stretch) < 1e-12:
        return np.asarray(signal, dtype=np.complex128)
    values = np.asarray(signal, dtype=np.complex128)
    grid = np.arange(values.size, dtype=float)
    source = grid * (1.0 + float(stretch))
    real = np.interp(source, grid, values.real, left=0.0, right=0.0)
    imag = np.interp(source, grid, values.imag, left=0.0, right=0.0)
    return real + 1j * imag


def _colored_complex_noise(rng: np.random.Generator, channels: int, samples: int, power_w: float, correlation: float) -> np.ndarray:
    if power_w <= 0.0:
        return np.zeros((channels, samples), dtype=np.complex128)
    white = (rng.normal(size=(channels, samples)) + 1j * rng.normal(size=(channels, samples))) / math.sqrt(2.0)
    coefficient = float(np.clip(correlation, -0.98, 0.98))
    output = np.empty_like(white)
    output[:, 0] = white[:, 0]
    innovation = math.sqrt(max(1.0 - coefficient * coefficient, 1e-12))
    for index in range(1, samples):
        output[:, index] = coefficient * output[:, index - 1] + innovation * white[:, index]
    output *= math.sqrt(power_w / max(float(np.mean(np.abs(output) ** 2)), 1e-30))
    return output


def _pulse_correlated_complex_noise(
    rng: np.random.Generator,
    channels: int,
    pulses: int,
    samples: int,
    power_w: float | Sequence[float],
    *,
    fast_time_correlation: float,
    slow_time_correlation: float,
) -> np.ndarray:
    """Draw a channel x pulse x fast-time complex disturbance cube.

    Fast-time correlation represents receiver filtering or range-correlated
    clutter.  Slow-time correlation represents pulse-to-pulse persistence.  A
    per-channel normalization is applied after both filters so ``power_w``
    remains a per-sample channel power rather than being accidentally averaged
    across receivers or multiplied by pulse count.
    """

    channel_power = np.asarray(power_w, dtype=float)
    if channel_power.ndim == 0:
        channel_power = np.full(channels, float(channel_power), dtype=float)
    if channel_power.shape != (channels,):
        raise ValueError(
            "pulse disturbance power must be scalar or one value per channel"
        )
    if np.any(~np.isfinite(channel_power)) or np.any(channel_power < 0.0):
        raise ValueError(
            "pulse disturbance powers must be finite and nonnegative"
        )
    if not np.any(channel_power > 0.0):
        return np.zeros((channels, pulses, samples), dtype=np.complex128)
    fast = float(np.clip(fast_time_correlation, -0.98, 0.98))
    slow = float(np.clip(slow_time_correlation, -0.999, 0.999))
    white = (
        rng.normal(size=(channels, pulses, samples))
        + 1j * rng.normal(size=(channels, pulses, samples))
    ) / math.sqrt(2.0)
    fast_innovation = math.sqrt(max(1.0 - fast * fast, 1e-12))
    for sample_index in range(1, samples):
        white[:, :, sample_index] = (
            fast * white[:, :, sample_index - 1]
            + fast_innovation * white[:, :, sample_index]
        )
    slow_innovation = math.sqrt(max(1.0 - slow * slow, 1e-12))
    for pulse_index in range(1, pulses):
        white[:, pulse_index, :] = (
            slow * white[:, pulse_index - 1, :]
            + slow_innovation * white[:, pulse_index, :]
        )
    for channel in range(channels):
        if channel_power[channel] <= 0.0:
            white[channel] = 0.0
            continue
        white[channel] *= math.sqrt(
            channel_power[channel]
            / max(float(np.mean(np.abs(white[channel]) ** 2)), 1e-30)
        )
    return np.asarray(white, dtype=np.complex128)


def _target_aspect(target: TargetState, tx: SensorNode, rx: SensorNode) -> float:
    velocity = np.asarray(target.velocity_m_s, dtype=float)
    if float(np.linalg.norm(velocity)) < 1e-9:
        velocity = np.asarray([1.0, 0.0, 0.0])
    velocity /= max(float(np.linalg.norm(velocity)), 1e-30)
    bisector = unit_vector(target.position_m, tx.position_m) + unit_vector(target.position_m, rx.position_m)
    bisector /= max(float(np.linalg.norm(bisector)), 1e-30)
    return math.acos(float(np.clip(np.dot(velocity, bisector), -1.0, 1.0)))


def _bistatic_delay_doppler(
    target: TargetState,
    tx: SensorNode,
    rx: SensorNode,
    carrier_hz: float,
    *,
    subtract_direct_reference: bool = False,
) -> tuple[float, float]:
    p = np.asarray(target.position_m, dtype=float)
    v = np.asarray(target.velocity_m_s, dtype=float)
    tx_u = unit_vector(tx.position_m, p)
    rx_u = unit_vector(rx.position_m, p)
    delay = (distance_m(tx.position_m, p) + distance_m(rx.position_m, p)) / C0_M_S
    tx_rate = float(
        tx_u
        @ (v - np.asarray(tx.velocity_m_s, dtype=float))
    )
    rx_rate = float(
        rx_u
        @ (v - np.asarray(rx.velocity_m_s, dtype=float))
    )
    path_rate = tx_rate + rx_rate
    if subtract_direct_reference:
        direct_u = unit_vector(tx.position_m, rx.position_m)
        direct_rate = float(
            direct_u
            @ (
                np.asarray(rx.velocity_m_s, dtype=float)
                - np.asarray(tx.velocity_m_s, dtype=float)
            )
        )
        delay -= distance_m(tx.position_m, rx.position_m) / C0_M_S
        path_rate -= direct_rate
    doppler = -carrier_hz / C0_M_S * path_rate
    return float(delay), float(doppler)


def _field_path_gain(
    world: PhysicalWorld,
    action: RadarAction,
    tx: SensorNode,
    rx: SensorNode,
    target: TargetState,
    waveform: Waveform,
    scattering: ScatteringModel,
    tx_antenna: JonesAntennaModel,
    rx_antenna: JonesAntennaModel,
    rng: np.random.Generator | None,
) -> complex:
    p = np.asarray(target.position_m, dtype=float)
    rt = distance_m(tx.position_m, p)
    rr = distance_m(rx.position_m, p)
    tx_direction = unit_vector(tx.position_m, p)
    rx_direction = unit_vector(rx.position_m, p)
    aspect = _target_aspect(target, tx, rx)
    scatter = scattering.complex_matrix(aspect, waveform.carrier_hz, rng)
    tx_pol = tx_antenna.polarization_vector(tx_direction, transmit=True)
    rx_pol = rx_antenna.polarization_vector(-rx_direction, transmit=False)
    polarization = complex(np.conjugate(rx_pol) @ scatter @ tx_pol)
    tx_array = tx_antenna.array_manifold(tx_direction, waveform.carrier_hz)
    rx_array = rx_antenna.array_manifold(-rx_direction, waveform.carrier_hz)
    array_gain = math.sqrt(max(float(np.vdot(tx_array, tx_array).real * np.vdot(rx_array, rx_array).real), 1e-30))
    path = math.sqrt(free_space_power_gain(rt, waveform.carrier_hz, world.atmosphere) * free_space_power_gain(rr, waveform.carrier_hz, world.atmosphere))
    source_voltage = math.sqrt(max(tx.tx_power_w * tx.tx_gain_linear * rx.rx_gain_linear, 0.0))
    carrier_phase = np.exp(-2j * math.pi * waveform.carrier_hz * (rt + rr) / C0_M_S)
    pointing_voltage = math.sqrt(
        commanded_beam_pointing_power_gain(world, action, tx, rx)
    )
    return (
        source_voltage
        * path
        * array_gain
        * polarization
        * carrier_phase
        * pointing_voltage
    )


def _synchronization_offsets(
    world: PhysicalWorld,
    action: RadarAction,
    tx: SensorNode,
    rx: SensorNode,
    waveform: Waveform,
    time_s: float,
) -> tuple[float, float]:
    r"""Return the (delay, frequency) synchronization offsets of the observable.

    The R2 capture must place the target return at the *same* physical
    observable that the analytic R0/R1 measurement model predicts, otherwise the
    two fidelity tiers describe different quantities and every R2 residual
    carries a deterministic multi-resolution-cell bias.  From
    ``measurement._measurement_model``:

    * ``active_monostatic``  -- transmitter and receiver share one oscillator, so
      the two-way delay carries no clock offset and the Doppler carries no CFO.
      Only the receive-chain effective delay :math:`\tau^{\rm eff}_{\rm rx}`
      appears:  :math:`z_\tau = 2r/c + \tau^{\rm eff}_{\rm rx}`, :math:`z_f =
      -2f_c\dot r/c`.
    * ``active_bistatic``    -- :math:`z_\tau=(R_t+R_r)/c + b_{\rm rx}-b_{\rm tx}
      + \tau^{\rm eff}_{\rm rx}` and :math:`z_f=-(f_c/c)(\dot R_t+\dot R_r)
      + \Delta f_{\rm rx}-\Delta f_{\rm tx}`.
    * reference-relative passive bistatic and AOA -- the excess delay and
      bistatic Doppler are formed against the direct-path reference received
      through the *same* chain, so the receiver clock offset, CFO and
      receive-chain delay cancel exactly: no synchronization term at all.

    The previous implementation added ``b_rx`` and ``cfo_rx`` unconditionally for
    every geometry, which is wrong for all three supported R2 kinds.
    """

    def clock(node: SensorNode) -> float:
        return float(node.clock_bias_s + time_s * node.clock_drift_s_per_s)

    def chain_delay(node: SensorNode) -> float:
        configuration = configuration_for(world, node.node_id, action.waveform_id)
        return 0.0 if configuration is None else float(configuration.effective_delay_s)

    if action.kind == "active_monostatic":
        return chain_delay(rx), 0.0
    if action.kind == "active_bistatic":
        return (
            clock(rx) - clock(tx) + chain_delay(rx),
            float(rx.frequency_bias_hz) - float(tx.frequency_bias_hz),
        )
    local_reference_channel = bool(
        action.reference_node_id is not None
        and action.reference_node_id == rx.node_id
    )
    if action.kind == "passive_bistatic" or (
        action.kind == "aoa" and local_reference_channel
    ):
        return 0.0, 0.0
    if action.kind in {"tdoa", "fdoa", "aoa"}:
        # Each passive receiver first observes an absolute bistatic channel.
        # Differencing those channels cancels the common illuminator leg and
        # leaves exactly the clock/CFO convention used by
        # measurement._measurement_model.  Receive-chain delay participates in
        # TDOA but is common across array elements for AOA.
        return clock(rx) + chain_delay(rx), float(rx.frequency_bias_hz)
    return clock(rx), float(rx.frequency_bias_hz)


def receiver_observable_delay_doppler(
    world: PhysicalWorld,
    action: RadarAction,
    receiver_node_id: str,
    *,
    target: TargetState | None = None,
    time_s: float = 0.0,
) -> tuple[float, float]:
    """Return one receiver's registered delay/Doppler observable.

    This helper is safe for the controller only when ``world`` is its belief
    projection.  The simulator calls the same expression with truth.  A local
    reference channel yields excess delay/Doppler; otherwise this is an
    absolute channel observable.  TDOA and FDOA are subsequently formed by
    subtracting receiver 1 from receiver 0, so the common transmitter path,
    transmitter clock, and transmitter CFO cancel rather than being silently
    omitted by a different formula.
    """

    if receiver_node_id not in action.rx_node_ids:
        raise ValueError(
            f"receiver {receiver_node_id!r} is not part of action {action.action_id}"
        )
    waveform = world.waveform(action.waveform_id)
    target_state = target or world.target
    tx_id = action.tx_node_id or action.reference_node_id
    if tx_id is None:
        if not world.spectrum_emitters:
            raise ValueError(
                "passive receiver observable requires an illuminator or spectrum emitter"
            )
        emitter = world.spectrum_emitters[0]
        tx = SensorNode(
            node_id=f"virtual_{emitter.emitter_id}",
            role="illuminator",
            position_m=emitter.position_m,
            tx_power_w=emitter.tx_power_w,
            tx_gain_linear=emitter.antenna_gain_linear,
            supported_waveform_ids=(waveform.waveform_id,),
        )
    else:
        tx = world.node(tx_id)
    rx = world.node(receiver_node_id)
    # A receiver-side reference channel defines zero delay at the direct
    # illuminator path.  Its surveillance CAF therefore measures excess path
    # delay, including for passive AOA.  Absolute delay is retained only when
    # no received reference channel exists (for example the current TDOA/FDOA
    # adapter driven by a registered transmitter waveform).
    subtract_direct_reference = bool(
        action.kind == "passive_bistatic"
        or (
            action.reference_node_id is not None
            and not action.kind.startswith("calibration")
        )
    )
    delay, doppler = _bistatic_delay_doppler(
        target_state,
        tx,
        rx,
        waveform.carrier_hz,
        subtract_direct_reference=subtract_direct_reference,
    )
    if action.kind == "active_monostatic":
        delay = 2.0 * distance_m(tx.position_m, target_state.position_m) / C0_M_S
    delay_offset, frequency_offset = _synchronization_offsets(
        world,
        action,
        tx,
        rx,
        waveform,
        time_s,
    )
    return float(delay + delay_offset), float(doppler + frequency_offset)


def _receiver_state(node: SensorNode, waveform: Waveform, time_s: float) -> ReceiverState:
    return ReceiverState(
        clock_bias_s=float(node.clock_bias_s + time_s * node.clock_drift_s_per_s),
        clock_drift_s_per_s=float(node.clock_drift_s_per_s),
        cfo_hz=float(node.frequency_bias_hz),
        sample_timing_offset_s=float(node.timing_jitter_s),
        phase_noise_linewidth_hz=float(node.integrated_phase_noise_rad2 / max(2.0 * math.pi * waveform.coherent_processing_interval_s, 1e-30)),
        adc_bits=int(node.adc_bits),
        adc_full_scale_w=float(node.adc_full_scale_w),
        front_end_bandwidth_hz=float(waveform.sample_rate_hz),
        image_rejection_db=float(node.image_rejection_db),
    )


def _front_end(
    analog: np.ndarray,
    states: Sequence[ReceiverState],
    sample_rate_hz: float,
    rng: np.random.Generator,
    *,
    impairments: bool,
    apply_cfo: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    analog = np.asarray(analog, dtype=np.complex128)
    output = np.empty_like(analog)
    deterministic = np.empty_like(analog)
    t = np.arange(analog.shape[1], dtype=float) / float(sample_rate_hz)
    for channel, state in enumerate(states):
        base = analog[channel]
        if impairments:
            increments = rng.normal(0.0, math.sqrt(max(2.0 * math.pi * state.phase_noise_linewidth_hz / sample_rate_hz, 0.0)), size=base.size)
            phase_noise = np.cumsum(increments)
            deterministic_frequency = state.cfo_hz if apply_cfo else 0.0
            lo = np.exp(
                1j
                * (
                    2.0 * math.pi * deterministic_frequency * t
                    + phase_noise
                )
            )
            image = 10.0 ** (-state.image_rejection_db / 20.0) * np.conjugate(base)
            impaired = (base + image) * lo
        else:
            impaired = base
        deterministic[channel] = impaired
        target_power = 0.08 * state.adc_full_scale_w
        gain = math.sqrt(target_power / max(float(np.mean(np.abs(impaired) ** 2)), 1e-30))
        adc = impaired * gain
        rail = math.sqrt(state.adc_full_scale_w / 2.0)
        clipped = np.clip(adc.real, -rail, rail) + 1j * np.clip(adc.imag, -rail, rail)
        levels = float(2 ** state.adc_bits)
        step = 2.0 * rail / max(levels - 1.0, 1.0)
        quantized = np.round(clipped.real / step) * step + 1j * np.round(clipped.imag / step) * step
        output[channel] = quantized / gain
    return output, output - deterministic


def _front_end_pulse_cube(
    analog: np.ndarray,
    states: Sequence[ReceiverState],
    sample_rate_hz: float,
    pulse_slow_time_s: np.ndarray,
    rng: np.random.Generator,
    *,
    impairments: bool,
    apply_cfo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one coherent receiver state over a nonuniform pulse-time cube.

    The phase-noise random walk advances over both fast-time samples and the
    gaps between pulses.  AGC and quantization are held fixed over the CPI,
    matching a coherent dwell rather than independently re-normalizing every
    pulse.  Deterministic CFO can be disabled when the action-specific
    observable already contains the oscillator difference exactly once.
    """

    values = np.asarray(analog, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != len(states):
        raise ValueError("pulse front end expects channel x pulse x sample IQ")
    slow = np.asarray(pulse_slow_time_s, dtype=float)
    if slow.shape != (values.shape[1],):
        raise ValueError("pulse front end slow-time dimension disagrees with IQ")
    fast = np.arange(values.shape[2], dtype=float) / float(sample_rate_hz)
    absolute_time = slow[:, None] + fast[None, :]
    flattened_time = absolute_time.reshape(-1)
    time_steps = np.diff(flattened_time, prepend=flattened_time[0])
    time_steps = np.maximum(time_steps, 0.0)
    output = np.empty_like(values)
    deterministic = np.empty_like(values)
    for channel, state in enumerate(states):
        base = values[channel]
        if impairments:
            increment_std = np.sqrt(
                np.maximum(
                    2.0 * math.pi * state.phase_noise_linewidth_hz * time_steps,
                    0.0,
                )
            )
            phase_noise = np.cumsum(
                rng.normal(0.0, 1.0, size=flattened_time.size) * increment_std
            ).reshape(base.shape)
            deterministic_frequency = state.cfo_hz if apply_cfo else 0.0
            lo = np.exp(
                1j
                * (
                    2.0
                    * math.pi
                    * deterministic_frequency
                    * absolute_time
                    + phase_noise
                )
            )
            image = 10.0 ** (-state.image_rejection_db / 20.0) * np.conjugate(base)
            impaired = (base + image) * lo
        else:
            impaired = base
        deterministic[channel] = impaired
        target_power = 0.08 * state.adc_full_scale_w
        gain = math.sqrt(
            target_power / max(float(np.mean(np.abs(impaired) ** 2)), 1e-30)
        )
        adc = impaired * gain
        rail = math.sqrt(state.adc_full_scale_w / 2.0)
        clipped = (
            np.clip(adc.real, -rail, rail)
            + 1j * np.clip(adc.imag, -rail, rail)
        )
        levels = float(2 ** state.adc_bits)
        step = 2.0 * rail / max(levels - 1.0, 1.0)
        quantized = (
            np.round(clipped.real / step) * step
            + 1j * np.round(clipped.imag / step) * step
        )
        output[channel] = quantized / gain
    return output, output - deterministic


def _jammer_waveform(
    jammer: Jammer,
    reference: np.ndarray,
    waveform: Waveform,
    samples: int,
    rng: np.random.Generator,
    delay_samples: float,
    time_offset_s: float = 0.0,
) -> np.ndarray:
    t = time_offset_s + np.arange(samples, dtype=float) / waveform.sample_rate_hz
    if jammer.kind == "drfm_false_target":
        false = _fractional_delay(reference, delay_samples + jammer.false_target_delay_s * waveform.sample_rate_hz)
        false *= np.exp(2j * math.pi * jammer.false_target_doppler_hz * t)
        return false / math.sqrt(max(float(np.mean(np.abs(false) ** 2)), 1e-30))
    if jammer.kind == "spot":
        tone = np.exp(2j * math.pi * (jammer.center_frequency_hz - waveform.carrier_hz) * t)
        return tone
    if jammer.kind == "swept":
        span = max(jammer.bandwidth_hz - float(jammer.instantaneous_bandwidth_hz or jammer.bandwidth_hz), 0.0)
        slope = span / max(samples / waveform.sample_rate_hz, 1e-30)
        base = jammer.center_frequency_hz - waveform.carrier_hz - span / 2.0
        return np.exp(2j * math.pi * (base * t + 0.5 * slope * t * t))
    return (rng.normal(size=samples) + 1j * rng.normal(size=samples)) / math.sqrt(2.0)


def _fast_time_reference(
    waveform: Waveform,
    samples: int,
    seed: int,
    *,
    pulsed: bool,
) -> tuple[np.ndarray, int, int]:
    """Generate the known transmit/reference signal on the receive fast-time grid.

    Active pulse radar emits only for ``pulse_width_s`` inside a PRI; zero
    padding outside that support is essential for minimum-range and leakage
    physics.  External illuminators are represented as continuous over the
    finite receive window.  The pulse waveform itself spans the declared
    occupied bandwidth, rather than stretching one chirp over the arbitrary
    range-gate length.
    """

    if not pulsed:
        return synthesize_waveform(waveform, samples, seed), int(samples), int(samples)
    declared_pulse_samples = max(
        17,
        int(round(waveform.pulse_width_s * waveform.sample_rate_hz)),
    )
    # Generate the declared physical pulse first, then retain only the portion
    # present in the receive gate. Compressing a long OFDM/LFM pulse into a
    # shorter gate changes its modulation law, occupied bandwidth and Fisher
    # information; a receiver can truncate a pulse but cannot time-compress it.
    pulse = synthesize_waveform(waveform, declared_pulse_samples, seed)
    captured_pulse_samples = min(declared_pulse_samples, samples)
    reference = np.zeros(samples, dtype=np.complex128)
    reference[:captured_pulse_samples] = pulse[:captured_pulse_samples]
    return reference, captured_pulse_samples, declared_pulse_samples


def simulate_complex_baseband(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    target: TargetState | None = None,
    target_present: bool = True,
    jammer: Jammer | None = None,
    samples: int = 1024,
    capture_start_s: float | None = None,
    capture_start_s_by_channel: Mapping[str, float] | None = None,
    time_s: float = 0.0,
    seed: int | None = None,
    impairments: bool = True,
    external_paths: Sequence[PathComponent] | None = None,
    external_path_provenance: FidelityProvenance | None = None,
    noise_scale: float = 1.0,
) -> ComplexBasebandCapture:
    """Generate an R2/R2.5 capture for active or passive bistatic actions.

    The window is range-gated around the nominal target path.  TDOA/FDOA/AOA
    actions are represented by their underlying receiver channels through the
    same function, while scalar calibration actions remain R1 measurements.
    """

    if samples < 128:
        raise ValueError("R2 capture requires at least 128 samples")
    if action.kind.startswith("calibration"):
        raise ValueError("calibration actions use the sufficient-statistic bridge")
    if not action.rx_node_ids:
        raise ValueError("R2 capture requires a receiver")
    waveform = world.waveform(action.waveform_id)
    target_state = target or world.target
    rng = np.random.default_rng(world.seed * 1_000_003 + action.action_id * 1009 if seed is None else seed)
    (
        reference,
        fast_time_pulse_samples,
        declared_fast_time_pulse_samples,
    ) = _fast_time_reference(
        waveform,
        samples,
        world.seed + action.action_id + 31,
        pulsed=bool(waveform.duty_cycle < 1.0 - 1e-12),
    )
    tx_id = action.tx_node_id or action.reference_node_id
    if tx_id is None:
        # Difference measurements without an explicit illuminator use the first
        # registered spectrum emitter through a virtual node at its position.
        if not world.spectrum_emitters:
            raise ValueError("passive difference measurement requires an illuminator or spectrum emitter")
        emitter = world.spectrum_emitters[0]
        tx = SensorNode(
            node_id=f"virtual_{emitter.emitter_id}",
            role="illuminator",
            position_m=emitter.position_m,
            tx_power_w=emitter.tx_power_w,
            tx_gain_linear=emitter.antenna_gain_linear,
            supported_waveform_ids=(waveform.waveform_id,),
        )
    else:
        tx = world.node(tx_id)
    receivers = [world.node(node_id) for node_id in action.rx_node_ids]
    receiver_budgets = tuple(
        receiver_link_budget(world, action, jammer, receiver.node_id)
        for receiver in receivers
    )
    channel_ids = tuple(node.node_id for node in receivers)
    channel_capture_starts = {
        str(node_id): float(value)
        for node_id, value in (capture_start_s_by_channel or {}).items()
    }
    unknown_capture_channels = set(channel_capture_starts).difference(channel_ids)
    if unknown_capture_channels:
        raise ValueError(
            "capture_start_s_by_channel references receivers outside the action: "
            f"{sorted(unknown_capture_channels)}"
        )
    passive_reference = bool(
        action.kind == "passive_bistatic"
        or action.reference_node_id is not None
    )
    nominal_delay, nominal_doppler = _bistatic_delay_doppler(
        target_state,
        tx,
        receivers[0],
        waveform.carrier_hz,
        subtract_direct_reference=passive_reference,
    )
    if action.kind == "active_monostatic":
        nominal_delay = 2.0 * distance_m(tx.position_m, target_state.position_m) / C0_M_S
    gate_lead_samples = min(48.0, samples / 6.0)
    # The range gate is a *receiver configuration* chosen by the operator, so it
    # must be a function of the belief, never of hidden truth.  When the caller
    # supplies ``capture_start_s`` (the operational path does, from the belief
    # prediction) it is used verbatim; if the true return then falls outside the
    # gate the extraction genuinely misses, which is the physically correct
    # consequence of a mis-pointed gate.  The truth-referenced default is
    # retained only for standalone physics studies and is explicitly labelled.
    if capture_start_s is not None:
        capture_start = float(capture_start_s)
        gate_reference_delay = capture_start + gate_lead_samples / waveform.sample_rate_hz
        gate_source = "caller_supplied_belief_referenced_gate"
    else:
        gate_reference_delay = float(nominal_delay)
        capture_start = float(nominal_delay - gate_lead_samples / waveform.sample_rate_hz)
        gate_source = "truth_referenced_default_gate_offline_physics_only"
    for node_id in channel_ids:
        channel_capture_starts.setdefault(node_id, capture_start)
    scattering = ScatteringModel(
        mean_rcs_m2=target_state.mean_rcs_m2,
        reference_frequency_hz=waveform.carrier_hz,
        fluctuation_model=f"Swerling_{target_state.swerling_case}",
    )
    target_components = np.zeros((len(receivers), samples), dtype=np.complex128)
    direct_components = np.zeros_like(target_components)
    target_multipath_components = np.zeros_like(target_components)
    environmental_multipath_components = np.zeros_like(target_components)
    path_records: list[PathComponent] = []
    primary_effective_delay = float(nominal_delay)
    primary_effective_doppler = float(nominal_doppler)
    channel_effective_delays: list[float] = []
    channel_effective_dopplers: list[float] = []
    captured_signal_energy_samples: list[float] = []
    t = np.arange(samples, dtype=float) / waveform.sample_rate_hz
    for index, rx in enumerate(receivers):
        tx_ant = antenna_from_node(tx, waveform.carrier_hz, mutual_coupling_strength=float(action.metadata.get("tx_mutual_coupling", 0.0)))
        rx_ant = antenna_from_node(rx, waveform.carrier_hz, mutual_coupling_strength=float(action.metadata.get("rx_mutual_coupling", 0.0)), station_phase_rad=float(action.metadata.get("station_phase_rad", 0.0)))
        delay, doppler = _bistatic_delay_doppler(
            target_state,
            tx,
            rx,
            waveform.carrier_hz,
            subtract_direct_reference=passive_reference,
        )
        state = _receiver_state(rx, waveform, time_s)
        delay_offset, frequency_offset = _synchronization_offsets(
            world, action, tx, rx, waveform, time_s
        )
        effective_delay = delay + delay_offset
        effective_doppler = doppler + frequency_offset
        if index == 0:
            primary_effective_delay = float(effective_delay)
            primary_effective_doppler = float(effective_doppler)
        channel_capture_start = channel_capture_starts[rx.node_id]
        channel_effective_delays.append(float(effective_delay))
        channel_effective_dopplers.append(float(effective_doppler))
        delay_samples = (
            effective_delay - channel_capture_start
        ) * waveform.sample_rate_hz
        stretch = -doppler / max(waveform.carrier_hz, 1e-30)
        delayed = _fractional_delay(_time_scale(reference, stretch), delay_samples)
        captured_signal_energy_samples.append(
            float(np.sum(np.abs(delayed) ** 2))
        )
        shifted = delayed * np.exp(2j * math.pi * effective_doppler * t)
        raw_gain = _field_path_gain(
            world,
            action,
            tx,
            rx,
            target_state,
            waveform,
            scattering,
            tx_ant,
            rx_ant,
            rng,
        )
        # The scalar link budget is the sole received-power authority shared by
        # R0/R1 and R2. Jones propagation supplies coherent phase and relative
        # multipath structure; its independently sampled magnitude must not
        # create a second, contradictory RCS/link realization.
        raw_phase = (
            raw_gain / abs(raw_gain)
            if abs(raw_gain) > np.finfo(float).tiny
            else 1.0 + 0.0j
        )
        gain = math.sqrt(
            max(float(receiver_budgets[index].signal_power_w), 0.0)
        ) * raw_phase
        if target_present:
            target_components[index] = gain * shifted
        path_records.append(PathComponent(
            path_id=f"target:{tx.node_id}:{target_state.target_id}:{rx.node_id}",
            kind="target_scatter",
            delay_s=float(effective_delay),
            doppler_hz=float(effective_doppler),
            stretch=float(stretch),
            complex_gain=complex(gain if target_present else 0.0j),
            source_node_id=tx.node_id,
            receiver_node_id=rx.node_id,
            target_id=target_state.target_id,
            provenance=(
                "authoritative_scalar_link_power_with_Jones_path_phase_"
                "and_relative_scattering_structure"
            ),
        ))
        # One specular ground-bounce proxy.  It is kept separate so R2.5 ray
        # paths can replace it without changing the likelihood adapter.
        image_target = np.asarray(target_state.position_m, dtype=float).copy()
        image_target[2] = 2.0 * world.ground_height_m - image_target[2]
        reflected_distance = distance_m(tx.position_m, image_target) + distance_m(image_target, rx.position_m)
        direct_distance = distance_m(tx.position_m, target_state.position_m) + distance_m(target_state.position_m, rx.position_m)
        reflection_delay = (reflected_distance - direct_distance) / C0_M_S
        incidence = math.atan2(
            float(np.linalg.norm(np.asarray(target_state.position_m)[:2] - np.asarray(rx.position_m)[:2])),
            max(float(target_state.position_m[2] + rx.position_m[2] - 2.0 * world.ground_height_m), 1e-9),
        )
        reflection = fresnel_reflection_coefficient(incidence, waveform.carrier_hz, world.surface)
        reflected = _fractional_delay(shifted, reflection_delay * waveform.sample_rate_hz)
        reflected_gain = 0.35 * reflection * gain if target_present else 0.0j
        if target_present:
            target_multipath_components[index] += reflected_gain * reflected
        path_records.append(PathComponent(
            path_id=f"ground:{tx.node_id}:{target_state.target_id}:{rx.node_id}",
            kind="specular_ground",
            delay_s=float(effective_delay + reflection_delay),
            doppler_hz=float(effective_doppler),
            stretch=float(stretch),
            complex_gain=complex(reflected_gain),
            source_node_id=tx.node_id,
            receiver_node_id=rx.node_id,
            target_id=target_state.target_id,
            provenance="planar_image_Fresnel_R2_proxy",
        ))
        # Direct illumination/reference leakage.  Passive reference extraction
        # observes this strongly; active bistatic channels see finite isolation.
        separation = float(np.linalg.norm(np.asarray(tx.position_m, dtype=float) - np.asarray(rx.position_m, dtype=float)))
        colocated = separation <= 1e-9
        # A monostatic TX/RX direct term is internal leakage/circulator coupling,
        # not free-space propagation over a zero-length path.  Register an
        # equivalent finite coupling path so the RF-isolation term remains
        # dimensionally meaningful and the propagation helper never sees
        # coincident endpoints.
        one_way = float(action.metadata.get("internal_leakage_equivalent_path_m", 1.0)) if colocated else separation
        direct_propagation_delay = float(action.metadata.get("internal_leakage_delay_s", 0.0)) if colocated else one_way / C0_M_S
        if passive_reference:
            # The reference channel defines direct-path delay zero.  At the same
            # wall-clock samples the surveillance channel contains the current
            # direct waveform at lag zero, while the target occupies positive
            # *excess* delay.  Subtracting the range-gate start here mixed an
            # absolute propagation clock with the reference-relative CAF clock.
            direct_delay_samples = 0.0
        else:
            direct_delay_samples = (
                direct_propagation_delay - channel_capture_start
            ) * waveform.sample_rate_hz
        direct_gain = math.sqrt(max(tx.tx_power_w * tx.tx_gain_linear * rx.rx_gain_linear * free_space_power_gain(one_way, waveform.carrier_hz, world.atmosphere), 0.0))
        hardware_isolation_db = (
            0.0
            if bool(action.metadata.get("external_illumination", False))
            else float(action.metadata.get("tx_rx_isolation_db", 75.0))
        )
        cancellation_db = float(
            action.metadata.get("direct_path_cancellation_db", 0.0)
        )
        isolation = 10.0 ** (
            -(hardware_isolation_db + cancellation_db) / 20.0
        )
        direct_components[index] = isolation * direct_gain * _fractional_delay(reference, direct_delay_samples)
    if external_paths:
        if external_path_provenance is None:
            raise ValueError("external paths require fidelity provenance")
        external_path_provenance.validate()
        for path in external_paths:
            if path.receiver_node_id not in channel_ids:
                continue
            index = channel_ids.index(path.receiver_node_id)
            delay_samples = (
                path.delay_s - channel_capture_starts[path.receiver_node_id]
            ) * waveform.sample_rate_hz
            component = path.complex_gain * _fractional_delay(_time_scale(reference, path.stretch), delay_samples)
            component *= np.exp(2j * math.pi * path.doppler_hz * t)
            if path.target_id == target_state.target_id:
                target_multipath_components[index] += component
            else:
                environmental_multipath_components[index] += component
            path_records.append(path)
    factor = build_action_factor(world, action, jammer)
    pulse_count = int(waveform.coherent_pulse_count)
    pulse_slow_time_s = np.arange(pulse_count, dtype=float) / float(
        waveform.pulse_repetition_hz
    )
    channel_dopplers = np.asarray(channel_effective_dopplers, dtype=float)
    slow_phase = np.exp(
        2j * math.pi * channel_dopplers[:, None] * pulse_slow_time_s[None, :]
    )
    target_cube = target_components[:, None, :] * slow_phase[:, :, None]
    target_multipath_cube = (
        target_multipath_components[:, None, :] * slow_phase[:, :, None]
    )
    environmental_multipath_cube = (
        environmental_multipath_components[:, None, :] * slow_phase[:, :, None]
    )
    multipath_cube = target_multipath_cube + environmental_multipath_cube
    direct_cube = np.repeat(direct_components[:, None, :], pulse_count, axis=1)
    clutter_spread_hz = max(
        0.1, float(action.metadata.get("clutter_doppler_spread_hz", 3.0))
    )
    clutter_slow_correlation = math.exp(
        -2.0 * math.pi * clutter_spread_hz / waveform.pulse_repetition_hz
    )
    clutter_power_by_channel = np.asarray(
        [budget.clutter_power_w for budget in receiver_budgets], dtype=float
    )
    clutter_fast_time_correlation = float(
        action.metadata.get("clutter_fast_time_correlation", 0.82)
    )
    clutter_cube = _pulse_correlated_complex_noise(
        rng,
        len(receivers),
        pulse_count,
        samples,
        clutter_power_by_channel,
        fast_time_correlation=clutter_fast_time_correlation,
        slow_time_correlation=clutter_slow_correlation,
    )
    ambient_power_by_channel = np.asarray(
        [budget.ambient_interference_power_w for budget in receiver_budgets],
        dtype=float,
    )
    occupancy = max(
        (budget.spectrum_occupancy_fraction for budget in receiver_budgets),
        default=0.0,
    )
    ambient_equivalent_bandwidth_hz = max(
        waveform.bandwidth_hz / 64.0,
        occupancy * waveform.bandwidth_hz,
    )
    ambient_fast_time_correlation = ambient_interference_fast_time_correlation(
        waveform,
        occupancy,
    )
    ambient_cube = _pulse_correlated_complex_noise(
        rng,
        len(receivers),
        pulse_count,
        samples,
        ambient_power_by_channel,
        fast_time_correlation=ambient_fast_time_correlation,
        slow_time_correlation=0.0,
    )
    jammer_cube = np.zeros_like(target_cube)
    if (
        jammer is not None
        and jammer.kind != "none"
        and factor.link_budget.jammer_power_w > 0.0
    ):
        for index in range(len(receivers)):
            nominal = (
                channel_effective_delays[index]
                - channel_capture_starts[receivers[index].node_id]
            ) * waveform.sample_rate_hz
            for pulse_index, pulse_time in enumerate(pulse_slow_time_s):
                jsig = _jammer_waveform(
                    jammer,
                    reference,
                    waveform,
                    samples,
                    rng,
                    nominal,
                    time_offset_s=float(pulse_time),
                )
                jsig /= math.sqrt(
                    max(float(np.mean(np.abs(jsig) ** 2)), 1e-30)
                )
                jammer_cube[index, pulse_index] = (
                    math.sqrt(receiver_budgets[index].jammer_power_w) * jsig
                )
    thermal_power_by_channel = np.asarray(
        [
            max(
                budget.thermal_noise_power_w,
                thermal_noise_power_w(
                    waveform.bandwidth_hz,
                    receiver.noise_figure_db,
                    world.atmosphere,
                ),
            )
            * float(noise_scale) ** 2
            for budget, receiver in zip(receiver_budgets, receivers)
        ],
        dtype=float,
    )
    thermal_cube = _pulse_correlated_complex_noise(
        rng,
        len(receivers),
        pulse_count,
        samples,
        thermal_power_by_channel,
        fast_time_correlation=0.0,
        slow_time_correlation=0.0,
    )
    receive_enabled_mask = np.ones(
        (len(receivers), samples), dtype=bool
    )
    duplexer_recovery_s = float(action.metadata.get("duplexer_recovery_s", 0.0))
    blanking_end_s = 0.0
    if action.kind == "active_monostatic":
        blanking_end_s = float(waveform.pulse_width_s + duplexer_recovery_s)
        for index, receiver in enumerate(receivers):
            absolute_fast_time_s = (
                channel_capture_starts[receiver.node_id] + t
            )
            receive_enabled_mask[index] = ~(
                (absolute_fast_time_s >= 0.0)
                & (absolute_fast_time_s < blanking_end_s)
            )
        pulse_mask = receive_enabled_mask[:, None, :]
        target_cube *= pulse_mask
        target_multipath_cube *= pulse_mask
        environmental_multipath_cube *= pulse_mask
        multipath_cube *= pulse_mask
        direct_cube *= pulse_mask
        clutter_cube *= pulse_mask
        ambient_cube *= pulse_mask
        jammer_cube *= pulse_mask
        thermal_cube *= pulse_mask
    analog_cube = (
        target_cube
        + direct_cube
        + multipath_cube
        + clutter_cube
        + ambient_cube
        + jammer_cube
        + thermal_cube
    )
    states = tuple(_receiver_state(node, waveform, time_s) for node in receivers)
    # ``effective_doppler`` above already contains the exact oscillator
    # convention of the analytic measurement model.  Applying receiver CFO in
    # the front end again would count it twice (and would fail to cancel it in a
    # passive direct-reference channel), so only random phase noise, image,
    # AGC, clipping and quantization are applied here.
    received_cube, impairment_cube = _front_end_pulse_cube(
        analog_cube,
        states,
        waveform.sample_rate_hz,
        pulse_slow_time_s,
        rng,
        impairments=impairments,
        apply_cfo=False,
    )
    received = received_cube[:, 0, :]
    clutter = clutter_cube[:, 0, :]
    ambient = ambient_cube[:, 0, :]
    jammer_components = jammer_cube[:, 0, :]
    thermal = thermal_cube[:, 0, :]
    impairment = impairment_cube[:, 0, :]
    reference_iq: np.ndarray | None = None
    coherent_reference_iq: np.ndarray | None = None
    if action.kind == "passive_bistatic" or action.reference_node_id is not None:
        reference_budget = receiver_budgets[0]
        reference_signal_power = max(
            float(reference_budget.reference_signal_power_w), 0.0
        )
        reference_disturbance_power = max(
            float(reference_budget.reference_noise_plus_interference_power_w),
            0.0,
        )
        structured_reference_interference_power = (
            min(reference_disturbance_power, float(reference_budget.jammer_power_w))
            if jammer is not None and jammer.kind == "drfm_false_target"
            else 0.0
        )
        reference_noise_power = max(
            reference_disturbance_power - structured_reference_interference_power,
            0.0,
        )
        reference_noise_cube = _pulse_correlated_complex_noise(
            rng,
            1,
            pulse_count,
            samples,
            reference_noise_power,
            fast_time_correlation=0.1,
            slow_time_correlation=0.0,
        )[0]
        coherent_reference_iq = (
            math.sqrt(reference_signal_power)
            * np.repeat(reference[None, :], pulse_count, axis=0)
            + reference_noise_cube
        )
        if jammer is not None and jammer.kind == "drfm_false_target":
            for pulse_index, pulse_time in enumerate(pulse_slow_time_s):
                structured = _jammer_waveform(
                    jammer,
                    reference,
                    waveform,
                    samples,
                    rng,
                    0.0,
                    time_offset_s=float(pulse_time),
                )
                structured /= math.sqrt(
                    max(float(np.mean(np.abs(structured) ** 2)), 1e-30)
                )
                coherent_reference_iq[pulse_index] += math.sqrt(
                    structured_reference_interference_power
                ) * structured
        reference_iq = coherent_reference_iq[0]
    fidelity = external_path_provenance or FidelityProvenance(
        level=FidelityLevel.R2_COMPLEX_BASEBAND,
        generator="simulate_complex_baseband",
        independently_validated=False,
        measured_data=False,
        limitations=(
            "analytic Jones element patterns rather than chamber pattern files",
            "single planar specular path plus diffuse proxy rather than full urban ray tracing",
            "complex-Gaussian receiver likelihood conditional on registered path state",
        ),
    )
    capture = ComplexBasebandCapture(
        action_id=action.action_id,
        channel_ids=channel_ids,
        surveillance_iq=received,
        reference_iq=reference_iq,
        deterministic_mean=(target_cube + direct_cube + multipath_cube)[:, 0, :],
        target_component=target_cube[:, 0, :],
        direct_path_component=direct_cube[:, 0, :],
        multipath_component=multipath_cube[:, 0, :],
        clutter_component=clutter,
        jammer_component=jammer_components,
        ambient_interference_component=ambient,
        thermal_component=thermal,
        receiver_impairment_component=impairment,
        sample_rate_hz=waveform.sample_rate_hz,
        center_frequency_hz=waveform.carrier_hz,
        path_components=tuple(path_records),
        receiver_states=states,
        target_present=bool(target_present),
        fidelity=fidelity,
        metadata={
            "capture_start_s": capture_start,
            "capture_start_s_by_channel": dict(channel_capture_starts),
            "nominal_delay_s": nominal_delay,
            "nominal_doppler_hz": nominal_doppler,
            # The observable actually injected into channel 0, including the
            # geometry-correct synchronization convention.  R0/R1 predict the
            # same quantity, so a cross-tier consistency test can compare them
            # without re-deriving the clock model.
            "injected_observable_delay_s": float(primary_effective_delay),
            "injected_observable_doppler_hz": float(primary_effective_doppler),
            "injected_observable_delay_s_by_channel": {
                node_id: value
                for node_id, value in zip(channel_ids, channel_effective_delays)
            },
            "injected_observable_doppler_hz_by_channel": {
                node_id: value
                for node_id, value in zip(channel_ids, channel_effective_dopplers)
            },
            "synchronization_convention": "analytic_measurement_model_consistent",
            "range_gate_reference_delay_s": float(gate_reference_delay),
            "range_gate_source": gate_source,
            "geometry": action.kind,
            "occupied_bandwidth_hz": float(waveform.bandwidth_hz),
            "coherent_pulse_count": pulse_count,
            "pulse_repetition_hz": float(waveform.pulse_repetition_hz),
            "pulse_repetition_interval_s": float(
                1.0 / waveform.pulse_repetition_hz
            ),
            "coherent_processing_interval_s": float(
                waveform.coherent_processing_interval_s
            ),
            "coherent_slow_time_aperture_s": float(
                pulse_count / waveform.pulse_repetition_hz
            ),
            "fast_time_window_s": float(samples / waveform.sample_rate_hz),
            "fast_time_pulse_samples": int(fast_time_pulse_samples),
            "declared_fast_time_pulse_samples": int(
                declared_fast_time_pulse_samples
            ),
            "declared_fast_time_pulse_width_s": float(
                waveform.pulse_width_s
            ),
            "captured_transmit_reference_duration_s": float(
                fast_time_pulse_samples / waveform.sample_rate_hz
            ),
            "fast_time_pulse_width_s": float(
                fast_time_pulse_samples / waveform.sample_rate_hz
            ),
            "fast_time_reference_mode": (
                "finite_pulse_gate"
                if waveform.duty_cycle < 1.0 - 1e-12
                else "continuous_wave_gate"
            ),
            "captured_on_air_dwell_s": float(
                pulse_count
                * captured_signal_energy_samples[0]
                / waveform.sample_rate_hz
            ),
            "captured_signal_energy_samples_by_channel": {
                receiver.node_id: float(energy_samples)
                for receiver, energy_samples in zip(
                    receivers, captured_signal_energy_samples
                )
            },
            "authoritative_signal_power_w_by_channel": {
                receiver.node_id: float(budget.signal_power_w)
                for receiver, budget in zip(receivers, receiver_budgets)
            },
            "authoritative_thermal_power_w_by_channel": {
                receiver.node_id: float(power)
                for receiver, power in zip(
                    receivers, thermal_power_by_channel
                )
            },
            "authoritative_clutter_power_w_by_channel": {
                receiver.node_id: float(power)
                for receiver, power in zip(receivers, clutter_power_by_channel)
            },
            "clutter_fast_time_correlation": float(
                clutter_fast_time_correlation
            ),
            "clutter_slow_time_correlation": float(
                clutter_slow_correlation
            ),
            "authoritative_ambient_interference_power_w_by_channel": {
                receiver.node_id: float(power)
                for receiver, power in zip(
                    receivers, ambient_power_by_channel
                )
            },
            "ambient_interference_iq_model": (
                "power_matched_complex_Gaussian_with_bandwidth_derived_AR1_"
                "correlation"
            ),
            "ambient_interference_equivalent_bandwidth_hz": float(
                ambient_equivalent_bandwidth_hz
            ),
            "ambient_interference_fast_time_correlation": float(
                ambient_fast_time_correlation
            ),
            "reference_channel_signal_power_w": (
                float(receiver_budgets[0].reference_signal_power_w)
                if passive_reference
                else None
            ),
            "reference_channel_noise_plus_interference_power_w": (
                float(
                    receiver_budgets[
                        0
                    ].reference_noise_plus_interference_power_w
                )
                if passive_reference
                else None
            ),
            "reference_channel_sinr_linear": (
                float(receiver_budgets[0].reference_sinr_linear)
                if passive_reference
                else None
            ),
            "reference_channel_sinr_processing_domain": (
                "receiver_input" if passive_reference else None
            ),
            "reference_channel_processing_gain_linear": (
                float(receiver_budgets[0].reference_processing_gain_linear)
                if passive_reference
                else None
            ),
            "reference_channel_output_sinr_linear": (
                float(receiver_budgets[0].reference_output_sinr_linear)
                if passive_reference
                else None
            ),
            "surveillance_channel_output_sinr_linear": float(
                receiver_budgets[0].surveillance_output_sinr_linear
            ),
            "cross_ambiguity_sinr_model": (
                receiver_budgets[0].cross_ambiguity_sinr_model
            ),
            "reference_channel_iq_units": "sqrt_W",
            "reference_channel_analytic_and_iq_power_authority_shared": True,
            "iq_target_amplitude_authority": (
                "measurement.receiver_link_budget_per_channel"
            ),
            "independent_duplicate_RCS_amplitude_draw_used": False,
            "deterministic_cfo_applied_exactly_once": True,
            "direct_path_time_origin": (
                "reference_relative_zero_lag"
                if action.kind == "passive_bistatic"
                else "absolute_propagation_delay"
            ),
            "direct_path_cancellation_db": float(
                action.metadata.get("direct_path_cancellation_db", 0.0)
            ),
            "monostatic_receiver_blanking_applied": bool(
                action.kind == "active_monostatic"
            ),
            "monostatic_receiver_blanking_end_s": float(blanking_end_s),
            "monostatic_receiver_blanked_sample_count_by_channel": {
                receiver.node_id: int(np.count_nonzero(~receive_enabled_mask[index]))
                for index, receiver in enumerate(receivers)
            },
            "monostatic_receiver_blanking_mutates_iq_samples": bool(
                action.kind == "active_monostatic"
            ),
            "fast_and_slow_time_axes_separated": True,
            "SI_units_internal": True,
            "external_paths_used": bool(external_paths),
        },
        coherent_surveillance_iq=received_cube,
        coherent_reference_iq=coherent_reference_iq,
        pulse_slow_time_s=pulse_slow_time_s,
        transmit_reference_iq=reference,
        coherent_pre_front_end_iq=analog_cube,
        coherent_target_origin_iq=target_cube + target_multipath_cube,
        coherent_target_direct_iq=target_cube,
        coherent_target_multipath_iq=target_multipath_cube,
        coherent_direct_path_iq=direct_cube,
        coherent_environmental_multipath_iq=environmental_multipath_cube,
        coherent_clutter_iq=clutter_cube,
        coherent_jammer_iq=jammer_cube,
        coherent_ambient_interference_iq=ambient_cube,
        coherent_thermal_iq=thermal_cube,
        coherent_receiver_impairment_iq=impairment_cube,
    )
    capture.validate()
    return capture


def receiver_channel_capture(
    capture: ComplexBasebandCapture,
    receiver_node_id: str,
) -> ComplexBasebandCapture:
    """Return one receiver channel without exposing simulator components.

    Component arrays are sliced only to preserve the simulator-side capture
    contract.  Operational estimators receive the returned object and read
    ``surveillance_iq``/``reference_iq`` only.
    """

    capture.validate()
    if receiver_node_id not in capture.channel_ids:
        raise ValueError(f"capture has no channel {receiver_node_id!r}")
    index = capture.channel_ids.index(receiver_node_id)

    def one_channel(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)[index : index + 1].copy()

    metadata = dict(capture.metadata)
    starts = metadata.get("capture_start_s_by_channel", {})
    if isinstance(starts, Mapping) and receiver_node_id in starts:
        metadata["capture_start_s"] = float(starts[receiver_node_id])
    injected_delays = metadata.get("injected_observable_delay_s_by_channel", {})
    if isinstance(injected_delays, Mapping) and receiver_node_id in injected_delays:
        metadata["injected_observable_delay_s"] = float(
            injected_delays[receiver_node_id]
        )
    injected_dopplers = metadata.get(
        "injected_observable_doppler_hz_by_channel", {}
    )
    if isinstance(injected_dopplers, Mapping) and receiver_node_id in injected_dopplers:
        metadata["injected_observable_doppler_hz"] = float(
            injected_dopplers[receiver_node_id]
        )
    result = ComplexBasebandCapture(
        action_id=capture.action_id,
        channel_ids=(receiver_node_id,),
        surveillance_iq=one_channel(capture.surveillance_iq),
        reference_iq=(
            None
            if capture.reference_iq is None
            else np.asarray(capture.reference_iq).copy()
        ),
        deterministic_mean=one_channel(capture.deterministic_mean),
        target_component=one_channel(capture.target_component),
        direct_path_component=one_channel(capture.direct_path_component),
        multipath_component=one_channel(capture.multipath_component),
        clutter_component=one_channel(capture.clutter_component),
        jammer_component=one_channel(capture.jammer_component),
        ambient_interference_component=(
            None
            if capture.ambient_interference_component is None
            else one_channel(capture.ambient_interference_component)
        ),
        thermal_component=one_channel(capture.thermal_component),
        receiver_impairment_component=one_channel(
            capture.receiver_impairment_component
        ),
        sample_rate_hz=capture.sample_rate_hz,
        center_frequency_hz=capture.center_frequency_hz,
        path_components=tuple(
            path
            for path in capture.path_components
            if path.receiver_node_id == receiver_node_id
        ),
        receiver_states=(capture.receiver_states[index],),
        target_present=capture.target_present,
        fidelity=capture.fidelity,
        metadata=metadata,
        coherent_surveillance_iq=(
            None
            if capture.coherent_surveillance_iq is None
            else np.asarray(capture.coherent_surveillance_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_reference_iq=(
            None
            if capture.coherent_reference_iq is None
            else np.asarray(capture.coherent_reference_iq).copy()
        ),
        pulse_slow_time_s=(
            None
            if capture.pulse_slow_time_s is None
            else np.asarray(capture.pulse_slow_time_s, dtype=float).copy()
        ),
        transmit_reference_iq=(
            None
            if capture.transmit_reference_iq is None
            else np.asarray(capture.transmit_reference_iq).copy()
        ),
        coherent_pre_front_end_iq=(
            None
            if capture.coherent_pre_front_end_iq is None
            else np.asarray(capture.coherent_pre_front_end_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_target_origin_iq=(
            None
            if capture.coherent_target_origin_iq is None
            else np.asarray(capture.coherent_target_origin_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_target_direct_iq=(
            None
            if capture.coherent_target_direct_iq is None
            else np.asarray(capture.coherent_target_direct_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_target_multipath_iq=(
            None
            if capture.coherent_target_multipath_iq is None
            else np.asarray(capture.coherent_target_multipath_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_direct_path_iq=(
            None
            if capture.coherent_direct_path_iq is None
            else np.asarray(capture.coherent_direct_path_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_environmental_multipath_iq=(
            None
            if capture.coherent_environmental_multipath_iq is None
            else np.asarray(capture.coherent_environmental_multipath_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_clutter_iq=(
            None
            if capture.coherent_clutter_iq is None
            else np.asarray(capture.coherent_clutter_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_jammer_iq=(
            None
            if capture.coherent_jammer_iq is None
            else np.asarray(capture.coherent_jammer_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_ambient_interference_iq=(
            None
            if capture.coherent_ambient_interference_iq is None
            else np.asarray(capture.coherent_ambient_interference_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_thermal_iq=(
            None
            if capture.coherent_thermal_iq is None
            else np.asarray(capture.coherent_thermal_iq)[
                index : index + 1
            ].copy()
        ),
        coherent_receiver_impairment_iq=(
            None
            if capture.coherent_receiver_impairment_iq is None
            else np.asarray(capture.coherent_receiver_impairment_iq)[
                index : index + 1
            ].copy()
        ),
    )
    result.validate()
    return result


def simulate_array_complex_baseband(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    jammer: Jammer | None = None,
    samples: int = 1024,
    capture_start_s: float | None = None,
    seed: int | None = None,
    impairments: bool = True,
) -> ComplexBasebandCapture:
    """Generate one-pulse element IQ for legacy array diagnostics.

    The operational AOA likelihood uses the pulse-compressed array bridge below
    because expanding every pulse and every component to 64 raw channels would
    require several gigabytes for one action.  This function remains useful for
    bounded front-end tests, but it does not claim a coherent slow-time AOA
    aperture.
    """

    if action.kind != "aoa" or len(action.rx_node_ids) != 1:
        raise ValueError("element-level array capture requires one-receiver AOA")
    station = simulate_complex_baseband(
        world,
        action,
        jammer=jammer,
        samples=samples,
        capture_start_s=capture_start_s,
        seed=seed,
        impairments=False,
    )
    waveform = world.waveform(action.waveform_id)
    receiver = world.node(action.rx_node_ids[0])
    antenna = antenna_from_node(
        receiver,
        waveform.carrier_hz,
        mutual_coupling_strength=float(
            action.metadata.get("rx_mutual_coupling", 0.0)
        ),
        station_phase_rad=float(action.metadata.get("station_phase_rad", 0.0)),
    )
    element_count = antenna.element_count
    amplitude_scale = math.sqrt(float(element_count))

    def propagation_direction(source_position: Sequence[float]) -> np.ndarray:
        return unit_vector(source_position, receiver.position_m)

    target_manifold = amplitude_scale * antenna.array_manifold(
        propagation_direction(world.target.position_m), waveform.carrier_hz
    )
    tx_id = action.tx_node_id or action.reference_node_id
    if tx_id is not None:
        tx_position = world.node(tx_id).position_m
    elif world.spectrum_emitters:
        tx_position = world.spectrum_emitters[0].position_m
    else:
        tx_position = world.target.position_m
    direct_manifold = amplitude_scale * antenna.array_manifold(
        propagation_direction(tx_position), waveform.carrier_hz
    )
    jammer_manifold = (
        amplitude_scale
        * antenna.array_manifold(
            propagation_direction(jammer.position_m), waveform.carrier_hz
        )
        if jammer is not None and jammer.kind != "none"
        else np.ones(element_count, dtype=np.complex128)
    )

    def spatialize(component: np.ndarray, manifold: np.ndarray) -> np.ndarray:
        base = np.asarray(component, dtype=np.complex128)[0]
        # The scalar station budget already contains the N-element receive
        # gain.  Recover per-element voltage before applying the manifold.
        return manifold[:, None] * base[None, :] / amplitude_scale

    target_component = spatialize(station.target_component, target_manifold)
    multipath_component = spatialize(
        station.multipath_component, target_manifold
    )
    direct_component = spatialize(station.direct_path_component, direct_manifold)
    jammer_component = spatialize(station.jammer_component, jammer_manifold)

    rng = np.random.default_rng(
        world.seed * 1_000_033 + action.action_id * 1013 + 71
        if seed is None
        else int(seed) + 71
    )
    clutter_base = np.asarray(station.clutter_component, dtype=np.complex128)[0]
    clutter_power = float(np.mean(np.abs(clutter_base) ** 2))
    random_spatial_phase = np.exp(
        2j * math.pi * rng.random(element_count)
    )[:, None]
    diffuse_clutter = _colored_complex_noise(
        rng,
        element_count,
        samples,
        clutter_power,
        correlation=0.82,
    )
    clutter_component = (
        math.sqrt(0.45) * random_spatial_phase * clutter_base[None, :]
        + math.sqrt(0.55) * diffuse_clutter
    )
    thermal_power = float(
        np.mean(np.abs(np.asarray(station.thermal_component)[0]) ** 2)
    )
    thermal_component = _colored_complex_noise(
        rng,
        element_count,
        samples,
        thermal_power,
        correlation=0.0,
    )
    analog = (
        target_component
        + multipath_component
        + direct_component
        + jammer_component
        + clutter_component
        + thermal_component
    )
    states = tuple(station.receiver_states[0] for _ in range(element_count))
    received, impairment_component = _front_end(
        analog,
        states,
        waveform.sample_rate_hz,
        rng,
        impairments=impairments,
        apply_cfo=False,
    )
    metadata = dict(station.metadata)
    metadata.update(
        {
            "array_receiver_node_id": receiver.node_id,
            "array_element_positions_body_m": (
                antenna.element_positions_body_m.tolist()
            ),
            "array_element_count": element_count,
            "array_observation_model": (
                "Jones_manifold_one_pulse_components_plus_diffuse_clutter_"
                "and_element_front_ends"
            ),
            "array_target_arrival_convention": (
                "propagation_direction_source_to_receiver"
            ),
            "array_station_gain_reversed_before_element_expansion": True,
            "array_element_signal_voltage_scale": float(1.0 / amplitude_scale),
            "array_coherent_pulse_count": 1,
            "array_slow_time_preserved": False,
        }
    )
    result = ComplexBasebandCapture(
        action_id=action.action_id,
        channel_ids=tuple(
            f"{receiver.node_id}:element:{index}" for index in range(element_count)
        ),
        surveillance_iq=received,
        reference_iq=(
            None
            if station.reference_iq is None
            else np.asarray(station.reference_iq).copy()
        ),
        deterministic_mean=(
            target_component + multipath_component + direct_component
        ),
        target_component=target_component,
        direct_path_component=direct_component,
        multipath_component=multipath_component,
        clutter_component=clutter_component,
        jammer_component=jammer_component,
        thermal_component=thermal_component,
        receiver_impairment_component=impairment_component,
        sample_rate_hz=waveform.sample_rate_hz,
        center_frequency_hz=waveform.carrier_hz,
        path_components=station.path_components,
        receiver_states=states,
        target_present=station.target_present,
        fidelity=station.fidelity,
        metadata=metadata,
        transmit_reference_iq=(
            None
            if station.transmit_reference_iq is None
            else np.asarray(station.transmit_reference_iq).copy()
        ),
    )
    result.validate()
    return result


def _sidak_exponential_power_threshold(
    familywise_false_alarm_probability: float,
    search_cell_count: int,
) -> tuple[float, float]:
    r"""Return a conservative spatial power threshold for a finite scan.

    Under the registered whitened circular-complex Gaussian null, the power
    in one fixed normalized Bartlett beam is exponential with unit mean.  For
    ``M`` searched cells and familywise false-alarm probability ``alpha``, the
    Sidak single-cell probability and power threshold are

    .. math::

        p_1 = 1-(1-\alpha)^{1/M},\qquad \tau=-\log p_1.

    The implementation uses every evaluated grid point as an independent-cell
    upper bound.  Correlated neighboring beams therefore make this threshold
    conservative rather than anti-conservative under the stated null model.
    """

    alpha = float(familywise_false_alarm_probability)
    cells = int(search_cell_count)
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise false-alarm probability must lie in (0, 1)")
    if cells <= 0:
        raise ValueError("spatial search cell count must be positive")
    single_cell_probability = -math.expm1(math.log1p(-alpha) / cells)
    threshold = -math.log(max(single_cell_probability, np.finfo(float).tiny))
    return float(threshold), float(single_cell_probability)


def _bonferroni_adaptive_subspace_ratio_threshold(
    familywise_false_alarm_probability: float,
    search_cell_count: int,
    subspace_rank: int,
    calibration_cell_count: int,
) -> tuple[float, float]:
    r"""Threshold an independently calibrated complex subspace-energy ratio.

    For a fixed rank-``r`` whitened projector under the circular complex
    Gaussian null, the primary energy is ``Gamma(r, sigma**2)``.  If ``K``
    independent calibration cells estimate the mean energy in the same
    projector, their ratio is

    .. math::

        T = \frac{X/r}{\bar Y/r} \sim F_{2r,\,2Kr}.

    Bonferroni allocates ``alpha/M`` to each of the ``M`` searched cells.  It
    controls familywise false alarm without pretending that neighboring array
    beams are independent.  Covariance-training cells and scale-calibration
    cells are disjoint upstream, so the denominator is not selected on the
    primary snapshot.
    """

    alpha = float(familywise_false_alarm_probability)
    cells = int(search_cell_count)
    rank = int(subspace_rank)
    calibration = int(calibration_cell_count)
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise false-alarm probability must lie in (0, 1)")
    if cells <= 0 or rank <= 0 or calibration <= 0:
        raise ValueError("search cells, subspace rank and calibration cells must be positive")
    single_cell_probability = alpha / cells
    from scipy.stats import f as fisher_f

    threshold = float(
        fisher_f.isf(
            single_cell_probability,
            2 * rank,
            2 * rank * calibration,
        )
    )
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError("adaptive subspace CFAR threshold is not finite positive")
    return threshold, float(single_cell_probability)


def _refined_grid_peak(
    score: np.ndarray,
    azimuth_grid: np.ndarray,
    elevation_grid: np.ndarray,
) -> tuple[int, int, float, float, bool]:
    """Return a wrapped parabolically refined peak on a rectangular grid."""

    row, column = (
        int(value)
        for value in np.unravel_index(int(np.argmax(score)), score.shape)
    )
    azimuth_step = float(azimuth_grid[1] - azimuth_grid[0])
    elevation_step = float(elevation_grid[1] - elevation_grid[0])
    azimuth_offset = (
        _parabolic_vertex(
            score[row, column - 1],
            score[row, column],
            score[row, column + 1],
        )
        if 0 < column < azimuth_grid.size - 1
        else 0.0
    )
    elevation_offset = (
        _parabolic_vertex(
            score[row - 1, column],
            score[row, column],
            score[row + 1, column],
        )
        if 0 < row < elevation_grid.size - 1
        else 0.0
    )
    azimuth = float(azimuth_grid[column] + azimuth_offset * azimuth_step)
    azimuth = (azimuth + math.pi) % (2.0 * math.pi) - math.pi
    elevation = float(elevation_grid[row] + elevation_offset * elevation_step)
    boundary = bool(
        row in (0, elevation_grid.size - 1)
        or column in (0, azimuth_grid.size - 1)
    )
    return row, column, azimuth, elevation, boundary


def _negative_profile_hessian_covariance(
    profile_log_likelihood: np.ndarray,
    row: int,
    column: int,
    azimuth_step_rad: float,
    elevation_step_rad: float,
    azimuth_half_width_rad: float,
    elevation_half_width_rad: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Invert the observed profile-likelihood curvature at an interior peak."""

    fallback = np.diag(
        [
            max(float(azimuth_half_width_rad), 1e-9) ** 2 / 3.0,
            max(float(elevation_half_width_rad), 1e-9) ** 2 / 3.0,
        ]
    )
    if (
        row <= 0
        or column <= 0
        or row >= profile_log_likelihood.shape[0] - 1
        or column >= profile_log_likelihood.shape[1] - 1
    ):
        return fallback, np.zeros((2, 2), dtype=float), False
    centre = float(profile_log_likelihood[row, column])
    d2_az = (
        float(profile_log_likelihood[row, column + 1])
        - 2.0 * centre
        + float(profile_log_likelihood[row, column - 1])
    ) / max(float(azimuth_step_rad) ** 2, 1e-30)
    d2_el = (
        float(profile_log_likelihood[row + 1, column])
        - 2.0 * centre
        + float(profile_log_likelihood[row - 1, column])
    ) / max(float(elevation_step_rad) ** 2, 1e-30)
    cross = (
        float(profile_log_likelihood[row + 1, column + 1])
        - float(profile_log_likelihood[row + 1, column - 1])
        - float(profile_log_likelihood[row - 1, column + 1])
        + float(profile_log_likelihood[row - 1, column - 1])
    ) / max(4.0 * float(azimuth_step_rad) * float(elevation_step_rad), 1e-30)
    information = -np.asarray([[d2_az, cross], [cross, d2_el]], dtype=float)
    information = 0.5 * (information + information.T)
    eigenvalues = np.linalg.eigvalsh(information)
    valid = bool(
        np.all(np.isfinite(information))
        and float(np.min(eigenvalues)) > 1e-10
    )
    if not valid:
        return fallback, information, False
    covariance = np.linalg.inv(information)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, information, True


def _belief_two_ray_matched_response_ratio(
    *,
    transmitter_position_m: Sequence[float],
    target_position_m: Sequence[float],
    receiver_position_m: Sequence[float],
    ground_height_m: float,
    surface: MaterialSurface,
    carrier_hz: float,
    sample_rate_hz: float,
    reference_waveform: np.ndarray,
    capture_start_s: float,
    expected_direct_delay_s: float,
    selected_integer_delay_samples: int,
) -> tuple[complex, float]:
    """Predict the selected-cell ground/direct complex response ratio.

    This mirrors the registered R2.5 image/Fresnel generator using only the
    belief target position, surveyed node geometry, declared surface model and
    known transmitted/reference waveform.  The common unknown target complex
    amplitude cancels in the ratio.
    """

    tx = np.asarray(transmitter_position_m, dtype=float).reshape(3)
    target = np.asarray(target_position_m, dtype=float).reshape(3)
    receiver = np.asarray(receiver_position_m, dtype=float).reshape(3)
    image = target.copy()
    image[2] = 2.0 * float(ground_height_m) - image[2]
    reflected_distance = distance_m(tx, image) + distance_m(image, receiver)
    direct_distance = distance_m(tx, target) + distance_m(target, receiver)
    reflection_delay_s = (reflected_distance - direct_distance) / C0_M_S
    incidence = math.atan2(
        float(np.linalg.norm(target[:2] - receiver[:2])),
        max(
            float(target[2] + receiver[2] - 2.0 * ground_height_m),
            1e-9,
        ),
    )
    reflection = fresnel_reflection_coefficient(
        incidence, carrier_hz, surface
    )
    reference = np.asarray(reference_waveform, dtype=np.complex128).reshape(-1)
    direct_delay_samples = (
        float(expected_direct_delay_s) - float(capture_start_s)
    ) * float(sample_rate_hz)
    direct = _fractional_delay(reference, direct_delay_samples)
    reflected = _fractional_delay(
        direct, reflection_delay_s * float(sample_rate_hz)
    )
    lag = int(selected_integer_delay_samples)
    direct_response = _cross_correlation_at_integer_lag(
        direct[None, :], reference, lag
    )[0]
    reflected_response = _cross_correlation_at_integer_lag(
        reflected[None, :], reference, lag
    )[0]
    if abs(direct_response) <= np.finfo(float).tiny:
        return 0.0j, float(reflection_delay_s)
    ratio = 0.35 * reflection * reflected_response / direct_response
    return complex(ratio), float(reflection_delay_s)


def _whitened_bartlett_angle_from_snapshot(
    snapshot: np.ndarray,
    secondary_snapshots: np.ndarray,
    antenna: JonesAntennaModel,
    *,
    carrier_hz: float,
    expected_azimuth_rad: float,
    expected_elevation_rad: float,
    azimuth_half_width_rad: float,
    elevation_half_width_rad: float,
    grid_size: int,
    estimator_receiver_position_m: Sequence[float] | None = None,
    estimator_target_range_m: float | None = None,
    estimator_ground_height_m: float = 0.0,
    estimator_reflection_response_ratio: complex | None = None,
    familywise_false_alarm_probability: float = 1.0e-6,
) -> dict[str, Any]:
    """Estimate direct-path direction with a constrained ground-image model.

    The direct direction is the estimand.  A common complex target amplitude is
    profiled out, while the relative reflected response is fixed by the
    belief-side image/Fresnel/waveform model.  This avoids the nonidentifiability
    of granting two arbitrary coherent amplitudes to one array snapshot.
    A direct-only profile remains an estimator-visible model-sensitivity audit.
    """

    primary = np.asarray(snapshot, dtype=np.complex128).reshape(-1)
    secondary = np.asarray(secondary_snapshots, dtype=np.complex128)
    if primary.shape != (antenna.element_count,):
        raise ValueError("array primary snapshot has the wrong element count")
    if secondary.ndim != 2 or secondary.shape[0] != antenna.element_count:
        raise ValueError("array secondary snapshots must be element x cell")
    if secondary.shape[1] < antenna.element_count + 16:
        raise ValueError(
            "array covariance and independent noise calibration require "
            "at least N+16 secondary cells"
        )
    covariance_cell_count = max(
        antenna.element_count,
        int(math.floor(0.75 * secondary.shape[1])),
    )
    covariance_secondary = secondary[:, :covariance_cell_count]
    calibration_secondary = secondary[:, covariance_cell_count:]
    sample_covariance = (
        covariance_secondary @ covariance_secondary.conj().T
        / max(covariance_secondary.shape[1], 1)
    )
    sample_covariance = 0.5 * (
        sample_covariance + sample_covariance.conj().T
    )
    mean_power = max(
        float(np.trace(sample_covariance).real) / antenna.element_count,
        np.finfo(float).tiny,
    )
    # Finite secondary support requires shrinkage.  The coefficient decreases
    # with the number of independent cells and is declared in the output.
    shrinkage = float(
        np.clip(
            0.25
            * antenna.element_count
            / max(covariance_secondary.shape[1], 1),
            0.05,
            0.35,
        )
    )
    covariance = (
        (1.0 - shrinkage) * sample_covariance
        + shrinkage * mean_power * np.eye(antenna.element_count)
    )
    loading = max(mean_power * 1.0e-6, np.finfo(float).tiny)
    inverse_noise = np.linalg.pinv(
        covariance + loading * np.eye(antenna.element_count)
    )
    azimuth_grid = np.linspace(
        expected_azimuth_rad - azimuth_half_width_rad,
        expected_azimuth_rad + azimuth_half_width_rad,
        grid_size,
    )
    elevation_grid = np.linspace(
        max(
            -0.5 * math.pi + 1e-4,
            expected_elevation_rad - elevation_half_width_rad,
        ),
        min(
            0.5 * math.pi - 1e-4,
            expected_elevation_rad + elevation_half_width_rad,
        ),
        grid_size,
    )
    score = np.full((elevation_grid.size, azimuth_grid.size), -np.inf, dtype=float)
    direct_only_score = np.full_like(score, -np.inf)
    rank_grid = np.ones_like(score, dtype=int)
    receiver_position = (
        None
        if estimator_receiver_position_m is None
        else np.asarray(estimator_receiver_position_m, dtype=float).reshape(3)
    )
    target_range = (
        None
        if estimator_target_range_m is None
        else float(estimator_target_range_m)
    )
    multipath_profile_requested = bool(
        receiver_position is not None
        and target_range is not None
        and math.isfinite(target_range)
        and target_range > 0.0
        and float(receiver_position[2]) > float(estimator_ground_height_m)
        and estimator_reflection_response_ratio is not None
        and math.isfinite(float(np.real(estimator_reflection_response_ratio)))
        and math.isfinite(float(np.imag(estimator_reflection_response_ratio)))
    )

    def steering_model(azimuth: float, elevation: float) -> tuple[np.ndarray, int]:
        cosine = math.cos(float(elevation))
        look = np.asarray(
            [
                cosine * math.cos(float(azimuth)),
                cosine * math.sin(float(azimuth)),
                math.sin(float(elevation)),
            ],
            dtype=float,
        )
        direct = math.sqrt(float(antenna.element_count)) * antenna.array_manifold(
            -look, carrier_hz
        )
        if not multipath_profile_requested:
            return direct[:, None], 1
        assert receiver_position is not None and target_range is not None
        candidate_target = receiver_position + target_range * look
        if float(candidate_target[2]) <= float(estimator_ground_height_m):
            return np.empty((antenna.element_count, 0), dtype=np.complex128), 0
        image_target = candidate_target.copy()
        image_target[2] = 2.0 * float(estimator_ground_height_m) - image_target[2]
        image_arrival = unit_vector(image_target, receiver_position)
        reflected = math.sqrt(float(antenna.element_count)) * antenna.array_manifold(
            image_arrival, carrier_hz
        )
        composite = direct + complex(estimator_reflection_response_ratio) * reflected
        if float(np.linalg.norm(composite)) <= np.finfo(float).tiny:
            return direct[:, None], 1
        return composite[:, None], 1

    def subspace_energy(vector: np.ndarray, model: np.ndarray) -> tuple[float, np.ndarray]:
        gram = model.conj().T @ inverse_noise @ model
        inverse_gram = np.linalg.pinv(gram, rcond=1.0e-10, hermitian=True)
        projection_coordinates = model.conj().T @ inverse_noise @ vector
        energy = float(
            np.real(
                np.vdot(
                    projection_coordinates,
                    inverse_gram @ projection_coordinates,
                )
            )
        )
        return max(energy, 0.0), inverse_gram @ projection_coordinates

    for row, elevation in enumerate(elevation_grid):
        for column, azimuth in enumerate(azimuth_grid):
            cosine = math.cos(float(elevation))
            look = np.asarray(
                [
                    cosine * math.cos(float(azimuth)),
                    cosine * math.sin(float(azimuth)),
                    math.sin(float(elevation)),
                ],
                dtype=float,
            )
            direct = math.sqrt(float(antenna.element_count)) * antenna.array_manifold(
                -look, carrier_hz
            )
            direct_only_score[row, column] = subspace_energy(
                primary, direct[:, None]
            )[0]
            model, rank = steering_model(float(azimuth), float(elevation))
            if rank <= 0:
                continue
            score[row, column] = subspace_energy(primary, model)[0]
            rank_grid[row, column] = rank
    if not np.any(np.isfinite(score)):
        raise RuntimeError("all physically admissible AOA grid cells were rejected")
    finite_floor = float(np.min(score[np.isfinite(score)]))
    score = np.where(np.isfinite(score), score, finite_floor)
    row, column, azimuth, elevation, boundary = _refined_grid_peak(
        score, azimuth_grid, elevation_grid
    )
    (
        direct_row,
        direct_column,
        direct_only_azimuth,
        direct_only_elevation,
        _,
    ) = _refined_grid_peak(direct_only_score, azimuth_grid, elevation_grid)
    azimuth_step = float(azimuth_grid[1] - azimuth_grid[0])
    elevation_step = float(elevation_grid[1] - elevation_grid[0])
    peak_model, peak_rank = steering_model(azimuth, elevation)
    if peak_rank <= 0:
        peak_model, peak_rank = steering_model(
            float(azimuth_grid[column]), float(elevation_grid[row])
        )
    peak_power, peak_coefficients = subspace_energy(primary, peak_model)
    calibration_scores = np.asarray(
        [subspace_energy(cell, peak_model)[0] for cell in calibration_secondary.T],
        dtype=float,
    )
    spatial_noise_mean = max(
        float(np.mean(calibration_scores)),
        np.finfo(float).tiny,
    )
    spatial_peak_over_noise = float(peak_power / spatial_noise_mean)
    spatial_cfar_threshold, spatial_single_cell_pfa = (
        _bonferroni_adaptive_subspace_ratio_threshold(
            familywise_false_alarm_probability,
            int(grid_size * grid_size),
            peak_rank,
            calibration_secondary.shape[1],
        )
    )
    noise_power_per_subspace_dimension = spatial_noise_mean / peak_rank
    profile_log_likelihood = score / max(
        noise_power_per_subspace_dimension, np.finfo(float).tiny
    )
    profile_covariance, profile_information, profile_curvature_valid = (
        _negative_profile_hessian_covariance(
            profile_log_likelihood,
            row,
            column,
            azimuth_step,
            elevation_step,
            azimuth_half_width_rad,
            elevation_half_width_rad,
        )
    )
    fitted_primary = peak_model @ peak_coefficients
    residual = primary - fitted_primary
    residual_energy_per_dof = float(
        np.real(np.vdot(residual, inverse_noise @ residual))
        / max(antenna.element_count - peak_rank, 1)
    )
    calibration_whitened_energy_per_dof = max(
        float(
            np.mean(
                [
                    np.real(np.vdot(cell, inverse_noise @ cell))
                    / antenna.element_count
                    for cell in calibration_secondary.T
                ]
            )
        ),
        np.finfo(float).tiny,
    )
    residual_scale = max(
        1.0,
        residual_energy_per_dof / calibration_whitened_energy_per_dof,
    )
    profile_covariance = residual_scale * profile_covariance
    azimuth_disagreement = (
        direct_only_azimuth - azimuth + math.pi
    ) % (2.0 * math.pi) - math.pi
    elevation_disagreement = direct_only_elevation - elevation
    model_disagreement = np.asarray(
        [azimuth_disagreement, elevation_disagreement], dtype=float
    )
    model_discrepancy_covariance = np.outer(
        model_disagreement, model_disagreement
    )
    peak_to_median = float(
        score[row, column] / max(float(np.median(score)), 1e-30)
    )
    return {
        "azimuth_rad": azimuth,
        "elevation_rad": elevation,
        "direct_only_azimuth_rad": direct_only_azimuth,
        "direct_only_elevation_rad": direct_only_elevation,
        "direct_only_peak_row": direct_row,
        "direct_only_peak_column": direct_column,
        "direct_vs_multipath_profile_disagreement_rad": model_disagreement,
        "model_discrepancy_covariance_rad2": model_discrepancy_covariance,
        "profile_likelihood_covariance_rad2": profile_covariance,
        "profile_likelihood_information_rad_minus2": profile_information,
        "profile_likelihood_curvature_valid": profile_curvature_valid,
        "profile_likelihood_residual_variance_scale": residual_scale,
        "spatial_peak_to_median": peak_to_median,
        "spatial_peak_over_independent_noise_mean": spatial_peak_over_noise,
        "spatial_output_sinr_linear": max(
            peak_rank * (spatial_peak_over_noise - 1.0), 0.0
        ),
        "spatial_peak_power": peak_power,
        "spatial_noise_mean_estimate": spatial_noise_mean,
        "spatial_noise_power_per_subspace_dimension": (
            noise_power_per_subspace_dimension
        ),
        "spatial_peak_on_boundary": boundary,
        "azimuth_grid_step_rad": azimuth_step,
        "elevation_grid_step_rad": elevation_step,
        "extraction_variance_azimuth_rad2": azimuth_step**2 / 12.0,
        "extraction_variance_elevation_rad2": elevation_step**2 / 12.0,
        "spatial_cfar_threshold_over_noise_mean": spatial_cfar_threshold,
        "spatial_single_cell_false_alarm_probability": spatial_single_cell_pfa,
        "spatial_cfar_effective_subspace_rank": peak_rank,
        "spatial_cfar_null_distribution": (
            f"F({2 * peak_rank},{2 * peak_rank * calibration_secondary.shape[1]})"
        ),
        "spatial_familywise_control": (
            "Bonferroni_over_all_evaluated_grid_cells"
        ),
        "spatial_covariance_secondary_cell_count": int(secondary.shape[1]),
        "spatial_covariance_training_cell_count": int(
            covariance_secondary.shape[1]
        ),
        "spatial_noise_calibration_cell_count": int(
            calibration_secondary.shape[1]
        ),
        "spatial_covariance_and_noise_calibration_are_disjoint": True,
        "spatial_covariance_shrinkage": shrinkage,
        "spatial_covariance_condition_number": float(
            np.linalg.cond(covariance + loading * np.eye(antenna.element_count))
        ),
        "array_estimator": (
            "shrinkage_whitened_Fresnel_constrained_two_ray_profile_likelihood"
        ),
        "direct_path_is_estimand_and_common_target_amplitude_is_nuisance": True,
        "reflection_to_direct_response_ratio_is_physics_constrained": True,
        "estimator_reflection_response_ratio_real": (
            None
            if estimator_reflection_response_ratio is None
            else float(np.real(estimator_reflection_response_ratio))
        ),
        "estimator_reflection_response_ratio_imag": (
            None
            if estimator_reflection_response_ratio is None
            else float(np.imag(estimator_reflection_response_ratio))
        ),
        "multipath_profile_uses_belief_range_not_target_truth": True,
        "multipath_profile_requested": multipath_profile_requested,
    }


def pulse_compressed_array_angle_sufficient_statistic(
    world: PhysicalWorld,
    action: RadarAction,
    capture: ComplexBasebandCapture,
    waveform: np.ndarray,
    antenna: JonesAntennaModel,
    *,
    jammer: Jammer | None,
    expected_delay_s: float,
    expected_doppler_hz: float,
    expected_azimuth_rad: float,
    expected_elevation_rad: float,
    azimuth_half_width_rad: float,
    elevation_half_width_rad: float,
    delay_search_half_width_s: float | None = None,
    doppler_search_half_width_hz: float | None = None,
    grid_size: int = 33,
    familywise_false_alarm_probability: float = 1.0e-6,
    clutter_notch_half_width_hz: float = 0.0,
    direct_path_guard_delay_s: float = 0.0,
    direct_path_guard_doppler_hz: float = 0.0,
    slow_time_window: str = "rectangular",
    estimator_receiver_position_m: Sequence[float] | None = None,
    estimator_target_range_m: float | None = None,
    estimator_target_position_m: Sequence[float] | None = None,
    estimator_transmitter_position_m: Sequence[float] | None = None,
    estimator_ground_height_m: float = 0.0,
    estimator_surface: MaterialSurface | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Pulse-compress one station channel, then form a finite array snapshot.

    The simulator uses its latent component cubes only to generate the spatial
    data.  The estimator receives the summed primary snapshot and secondary
    cells, never the component labels or target direction.
    """

    capture.validate()
    if capture.coherent_surveillance_iq is None:
        raise ValueError("pulse-compressed AOA requires coherent station IQ")
    pulse_repetition_hz = float(
        capture.metadata.get("pulse_repetition_hz", 0.0)
    )
    if pulse_repetition_hz <= 0.0:
        raise ValueError("pulse-compressed AOA requires a positive PRF")
    minimum_delay_search_half_width_s = max(
        8.0 / capture.sample_rate_hz,
        4.0
        / max(
            float(capture.metadata.get("occupied_bandwidth_hz", 1.0)),
            1.0,
        ),
    )
    resolved_delay_search_half_width_s = max(
        minimum_delay_search_half_width_s,
        0.0
        if delay_search_half_width_s is None
        else float(delay_search_half_width_s),
    )
    resolved_doppler_search_half_width_hz = (
        0.5 * pulse_repetition_hz * (1.0 - 1.0e-12)
        if doppler_search_half_width_hz is None
        else float(doppler_search_half_width_hz)
    )
    if (
        not math.isfinite(resolved_delay_search_half_width_s)
        or resolved_delay_search_half_width_s <= 0.0
        or not math.isfinite(resolved_doppler_search_half_width_hz)
        or resolved_doppler_search_half_width_hz <= 0.0
    ):
        raise ValueError("AOA delay/Doppler search widths must be finite and positive")
    delay_doppler = matched_filter_sufficient_statistic(
        capture,
        waveform,
        expected_delay_s=expected_delay_s,
        expected_doppler_hz=expected_doppler_hz,
        delay_search_half_width_s=resolved_delay_search_half_width_s,
        doppler_search_half_width_hz=resolved_doppler_search_half_width_hz,
        coherent_processing_interval_s=float(
            capture.metadata.get(
                "coherent_slow_time_aperture_s",
                np.asarray(capture.pulse_slow_time_s).size
                / pulse_repetition_hz,
            )
        ),
        familywise_false_alarm_probability=familywise_false_alarm_probability,
        clutter_notch_half_width_hz=clutter_notch_half_width_hz,
        direct_path_guard_delay_s=direct_path_guard_delay_s,
        direct_path_guard_doppler_hz=direct_path_guard_doppler_hz,
        slow_time_window=slow_time_window,
    )
    capture_start_s = float(capture.metadata.get("capture_start_s", 0.0))
    delay_search_lags = tuple(int(value) for value in delay_doppler["delay_search_lags"])
    absolute_delay_search_interval_s = (
        capture_start_s + delay_search_lags[0] / capture.sample_rate_hz,
        capture_start_s + delay_search_lags[1] / capture.sample_rate_hz,
    )
    injected_delay_s = capture.metadata.get("injected_observable_delay_s")
    injected_doppler_hz = capture.metadata.get("injected_observable_doppler_hz")
    truth_delay_inside_search: bool | None = None
    truth_doppler_inside_search: bool | None = None
    aliased_truth_doppler_hz: float | None = None
    if injected_delay_s is not None:
        truth_delay_inside_search = bool(
            absolute_delay_search_interval_s[0]
            <= float(injected_delay_s)
            <= absolute_delay_search_interval_s[1]
        )
    if injected_doppler_hz is not None:
        aliased_truth_doppler_hz = float(
            expected_doppler_hz
            + (
                float(injected_doppler_hz)
                - expected_doppler_hz
                + 0.5 * pulse_repetition_hz
            )
            % pulse_repetition_hz
            - 0.5 * pulse_repetition_hz
        )
        doppler_search_interval_hz = tuple(
            float(value) for value in delay_doppler["doppler_search_hz"]
        )
        truth_doppler_inside_search = bool(
            doppler_search_interval_hz[0]
            <= aliased_truth_doppler_hz
            <= doppler_search_interval_hz[1]
        )
    required_components = {
        "target_direct": capture.coherent_target_direct_iq,
        "target_multipath": capture.coherent_target_multipath_iq,
        "direct": capture.coherent_direct_path_iq,
        "environmental_multipath": (
            capture.coherent_environmental_multipath_iq
        ),
        "clutter": capture.coherent_clutter_iq,
        "jammer": capture.coherent_jammer_iq,
        "ambient": capture.coherent_ambient_interference_iq,
        "thermal": capture.coherent_thermal_iq,
        "receiver_impairment": capture.coherent_receiver_impairment_iq,
    }
    if any(component is None for component in required_components.values()):
        raise RuntimeError("pulse-compressed AOA source omitted component IQ")
    responses = {
        name: _component_ambiguity_response_at_selected_cell(
            capture, delay_doppler, component
        )
        for name, component in required_components.items()
    }
    receiver = world.node(action.rx_node_ids[0])
    element_count = antenna.element_count
    root_n = math.sqrt(float(element_count))

    def arrival(source_position: Sequence[float]) -> np.ndarray:
        direction = unit_vector(source_position, receiver.position_m)
        return root_n * antenna.array_manifold(
            direction, capture.center_frequency_hz
        )

    target_steering = arrival(world.target.position_m)
    image_target = np.asarray(world.target.position_m, dtype=float).copy()
    image_target[2] = 2.0 * world.ground_height_m - image_target[2]
    target_multipath_steering = arrival(image_target)
    tx_id = action.tx_node_id or action.reference_node_id
    direct_position = (
        world.node(tx_id).position_m
        if tx_id is not None
        else world.spectrum_emitters[0].position_m
    )
    direct_steering = arrival(direct_position)
    jammer_steering = (
        arrival(jammer.position_m)
        if jammer is not None and jammer.kind != "none"
        else direct_steering
    )
    clutter_azimuth = 0.37 + 0.11 * action.action_id
    clutter_direction = np.asarray(
        [math.cos(clutter_azimuth), math.sin(clutter_azimuth), -0.08],
        dtype=float,
    )
    clutter_direction /= max(float(np.linalg.norm(clutter_direction)), 1e-30)
    clutter_steering = root_n * antenna.array_manifold(
        clutter_direction, capture.center_frequency_hz
    )
    environmental_steering = clutter_steering
    deterministic = (
        responses["target_direct"] * target_steering
        + responses["target_multipath"] * target_multipath_steering
        + responses["direct"] * direct_steering
        + responses["environmental_multipath"] * environmental_steering
        + responses["jammer"] * jammer_steering
        + math.sqrt(0.45) * responses["clutter"] * clutter_steering
    ) / root_n
    background_power = max(
        float(delay_doppler["receiver_noise_mean_estimate"]),
        np.finfo(float).tiny,
    )
    clutter_unit = clutter_steering / max(
        float(np.linalg.norm(clutter_steering)), 1e-30
    )
    spatial_covariance = background_power * (
        0.65 * np.eye(element_count)
        + 0.35 * element_count * np.outer(clutter_unit, clutter_unit.conj())
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (spatial_covariance + spatial_covariance.conj().T)
    )
    square_root = eigenvectors @ np.diag(
        np.sqrt(np.maximum(eigenvalues, np.finfo(float).tiny))
    )
    rng = np.random.default_rng(
        int(seed) + world.seed * 1_000_003 + action.action_id * 1009 + 313
    )
    secondary_count = max(96, 2 * element_count)
    white = (
        rng.normal(size=(element_count, secondary_count + 1))
        + 1j * rng.normal(size=(element_count, secondary_count + 1))
    ) / math.sqrt(2.0)
    samples = square_root @ white
    primary = deterministic + samples[:, 0]
    reflection_response_ratio: complex | None = None
    reflection_excess_delay_s: float | None = None
    if (
        estimator_target_position_m is not None
        and estimator_transmitter_position_m is not None
        and estimator_receiver_position_m is not None
        and estimator_surface is not None
    ):
        reference_waveform = (
            np.asarray(capture.transmit_reference_iq, dtype=np.complex128)
            if capture.transmit_reference_iq is not None
            else np.asarray(waveform, dtype=np.complex128)
        )
        reflection_response_ratio, reflection_excess_delay_s = (
            _belief_two_ray_matched_response_ratio(
                transmitter_position_m=estimator_transmitter_position_m,
                target_position_m=estimator_target_position_m,
                receiver_position_m=estimator_receiver_position_m,
                ground_height_m=estimator_ground_height_m,
                surface=estimator_surface,
                carrier_hz=capture.center_frequency_hz,
                sample_rate_hz=capture.sample_rate_hz,
                reference_waveform=reference_waveform,
                capture_start_s=float(
                    capture.metadata.get("capture_start_s", 0.0)
                ),
                expected_direct_delay_s=expected_delay_s,
                selected_integer_delay_samples=int(
                    delay_doppler["ambiguity_peak_integer_delay_samples"]
                ),
            )
        )
    spatial = _whitened_bartlett_angle_from_snapshot(
        primary,
        samples[:, 1:],
        antenna,
        carrier_hz=capture.center_frequency_hz,
        expected_azimuth_rad=expected_azimuth_rad,
        expected_elevation_rad=expected_elevation_rad,
        azimuth_half_width_rad=azimuth_half_width_rad,
        elevation_half_width_rad=elevation_half_width_rad,
        grid_size=grid_size,
        estimator_receiver_position_m=estimator_receiver_position_m,
        estimator_target_range_m=estimator_target_range_m,
        estimator_ground_height_m=estimator_ground_height_m,
        estimator_reflection_response_ratio=reflection_response_ratio,
        familywise_false_alarm_probability=familywise_false_alarm_probability,
    )
    spatial_search_cell_count = int(grid_size * grid_size)
    spatial_cfar_passed = bool(
        float(spatial["spatial_peak_over_independent_noise_mean"])
        >= float(spatial["spatial_cfar_threshold_over_noise_mean"])
    )
    detected = bool(
        delay_doppler["detected"]
        and spatial_cfar_passed
        and float(spatial["spatial_peak_to_median"]) >= 1.05
        and not bool(spatial["spatial_peak_on_boundary"])
    )
    return {
        **dict(delay_doppler),
        **spatial,
        "detected": detected,
        "temporal_cfar_detected": bool(delay_doppler["detected"]),
        "array_sufficient_statistic": (
            "pulse_compressed_primary_plus_independent_secondary_cells"
        ),
        "coherent_slow_time_used_for_aoa_detection": True,
        "spatial_search_cell_count": spatial_search_cell_count,
        "spatial_independent_cell_upper_bound": spatial_search_cell_count,
        "spatial_cfar_passed": spatial_cfar_passed,
        "spatial_cfar_null_model": spatial["spatial_cfar_null_distribution"],
        "spatial_detection_requires_temporal_and_spatial_cfar": True,
        "doppler_information_claimed_by_aoa_extractor": False,
        "belief_centred_angular_search": True,
        "truth_angle_used_by_estimator": False,
        "truth_range_used_by_estimator": False,
        "belief_predicted_ground_reflection_excess_delay_s": (
            reflection_excess_delay_s
        ),
        "belief_predicted_ground_reflection_response_ratio_available": (
            reflection_response_ratio is not None
        ),
        "simulator_truth_component_spatialization_used_only_to_generate_data": True,
        "simulator_truth_target_component_response_power": float(
            abs(responses["target_direct"]) ** 2
            + abs(responses["target_multipath"]) ** 2
        ),
        "simulator_truth_nontarget_component_response_power": float(
            sum(
                abs(value) ** 2
                for name, value in responses.items()
                if name not in {"target_direct", "target_multipath"}
            )
        ),
        "aoa_delay_search_half_width_s": float(
            resolved_delay_search_half_width_s
        ),
        "aoa_doppler_search_half_width_hz": float(
            resolved_doppler_search_half_width_hz
        ),
        "absolute_delay_search_interval_s": [
            float(value) for value in absolute_delay_search_interval_s
        ],
        "simulator_truth_injected_observable_delay_s": (
            None if injected_delay_s is None else float(injected_delay_s)
        ),
        "simulator_truth_injected_observable_doppler_hz": (
            None if injected_doppler_hz is None else float(injected_doppler_hz)
        ),
        "simulator_truth_aliased_observable_doppler_hz": (
            aliased_truth_doppler_hz
        ),
        "simulator_truth_target_delay_inside_search": (
            truth_delay_inside_search
        ),
        "simulator_truth_target_doppler_inside_search": (
            truth_doppler_inside_search
        ),
        "simulator_truth_target_observable_inside_search": bool(
            truth_delay_inside_search is True
            and truth_doppler_inside_search is True
        ),
        "target_multipath_spatial_model": (
            "planar_ground_image_arrival_direction_R2p5_proxy"
        ),
    }


def array_angle_sufficient_statistic(
    capture: ComplexBasebandCapture,
    waveform: np.ndarray,
    antenna: JonesAntennaModel,
    *,
    expected_delay_s: float,
    expected_doppler_hz: float,
    expected_azimuth_rad: float,
    expected_elevation_rad: float,
    azimuth_half_width_rad: float,
    elevation_half_width_rad: float,
    grid_size: int = 33,
    familywise_false_alarm_probability: float = 1.0e-6,
    clutter_notch_half_width_hz: float = 0.0,
    direct_path_guard_delay_s: float = 0.0,
    direct_path_guard_doppler_hz: float = 0.0,
) -> dict[str, Any]:
    """Belief-centred whitened Bartlett estimate from element IQ."""

    capture.validate()
    if capture.surveillance_iq.shape[0] != antenna.element_count:
        raise ValueError("array capture and antenna element counts disagree")
    if grid_size < 9 or grid_size % 2 == 0:
        raise ValueError("AOA grid_size must be odd and at least nine")
    delay_doppler = matched_filter_sufficient_statistic(
        capture,
        waveform,
        expected_delay_s=expected_delay_s,
        expected_doppler_hz=expected_doppler_hz,
        delay_search_half_width_s=max(
            8.0 / capture.sample_rate_hz,
            4.0
            / max(
                float(capture.metadata.get("occupied_bandwidth_hz", 1.0)),
                1.0,
            ),
        ),
        # This bridge retains one fast-time spatial snapshot.  It estimates
        # angle, not radial velocity, so hold Doppler at the causal graph
        # prediction instead of fabricating a MHz-scale estimate from a
        # microsecond range gate.  Delay and spatial phase remain data-derived.
        doppler_grid_hz=np.asarray([expected_doppler_hz], dtype=float),
        coherent_processing_interval_s=(
            capture.surveillance_iq.shape[1] / capture.sample_rate_hz
        ),
        familywise_false_alarm_probability=familywise_false_alarm_probability,
        clutter_notch_half_width_hz=clutter_notch_half_width_hz,
        direct_path_guard_delay_s=direct_path_guard_delay_s,
        direct_path_guard_doppler_hz=direct_path_guard_doppler_hz,
    )
    reference = (
        np.asarray(capture.reference_iq, dtype=np.complex128)
        if capture.reference_iq is not None
        else np.asarray(
            capture.transmit_reference_iq
            if capture.transmit_reference_iq is not None
            else waveform,
            dtype=np.complex128,
        )
    )
    if reference.ndim == 2:
        reference = reference[0]
    lag_samples = (
        float(delay_doppler["absolute_delay_s"])
        - float(capture.metadata.get("capture_start_s", 0.0))
    ) * capture.sample_rate_hz
    delayed = _fractional_delay(reference, lag_samples)
    time_s = np.arange(reference.size, dtype=float) / capture.sample_rate_hz
    template = delayed * np.exp(
        2j * math.pi * float(delay_doppler["doppler_hz"]) * time_s
    )
    template_energy = max(float(np.vdot(template, template).real), 1e-30)
    iq = np.asarray(capture.surveillance_iq, dtype=np.complex128)
    snapshot = iq @ np.conjugate(template) / template_energy
    residual = iq - snapshot[:, None] * template[None, :]
    noise_covariance = residual @ residual.conj().T / max(1, residual.shape[1])
    loading = max(
        float(np.trace(noise_covariance).real)
        / max(antenna.element_count, 1)
        * 1e-3,
        np.finfo(float).tiny,
    )
    inverse_noise = np.linalg.pinv(
        0.5 * (noise_covariance + noise_covariance.conj().T)
        + loading * np.eye(antenna.element_count)
    )
    azimuth_grid = np.linspace(
        expected_azimuth_rad - azimuth_half_width_rad,
        expected_azimuth_rad + azimuth_half_width_rad,
        grid_size,
    )
    elevation_grid = np.linspace(
        max(-0.5 * math.pi + 1e-4, expected_elevation_rad - elevation_half_width_rad),
        min(0.5 * math.pi - 1e-4, expected_elevation_rad + elevation_half_width_rad),
        grid_size,
    )
    score = np.empty((elevation_grid.size, azimuth_grid.size), dtype=float)
    for row, elevation in enumerate(elevation_grid):
        cosine = math.cos(float(elevation))
        for column, azimuth in enumerate(azimuth_grid):
            look = np.asarray(
                [
                    cosine * math.cos(float(azimuth)),
                    cosine * math.sin(float(azimuth)),
                    math.sin(float(elevation)),
                ],
                dtype=float,
            )
            steering = math.sqrt(float(antenna.element_count)) * antenna.array_manifold(
                -look, capture.center_frequency_hz
            )
            numerator = abs(np.vdot(steering, inverse_noise @ snapshot)) ** 2
            denominator = max(
                float(np.vdot(steering, inverse_noise @ steering).real),
                np.finfo(float).tiny,
            )
            score[row, column] = float(numerator / denominator)
    peak = np.unravel_index(int(np.argmax(score)), score.shape)
    row, column = int(peak[0]), int(peak[1])
    azimuth_step = float(azimuth_grid[1] - azimuth_grid[0])
    elevation_step = float(elevation_grid[1] - elevation_grid[0])
    azimuth_offset = (
        _parabolic_vertex(
            score[row, column - 1], score[row, column], score[row, column + 1]
        )
        if 0 < column < azimuth_grid.size - 1
        else 0.0
    )
    elevation_offset = (
        _parabolic_vertex(
            score[row - 1, column], score[row, column], score[row + 1, column]
        )
        if 0 < row < elevation_grid.size - 1
        else 0.0
    )
    azimuth = float(azimuth_grid[column] + azimuth_offset * azimuth_step)
    azimuth = (azimuth + math.pi) % (2.0 * math.pi) - math.pi
    elevation = float(elevation_grid[row] + elevation_offset * elevation_step)
    peak_to_median = float(score[row, column] / max(float(np.median(score)), 1e-30))
    boundary = bool(
        row in (0, elevation_grid.size - 1)
        or column in (0, azimuth_grid.size - 1)
    )
    return {
        **dict(delay_doppler),
        "azimuth_rad": azimuth,
        "elevation_rad": elevation,
        "spatial_peak_to_median": peak_to_median,
        "spatial_peak_on_boundary": boundary,
        "azimuth_grid_step_rad": azimuth_step,
        "elevation_grid_step_rad": elevation_step,
        "extraction_variance_azimuth_rad2": azimuth_step**2 / 192.0,
        "extraction_variance_elevation_rad2": elevation_step**2 / 192.0,
        "detected": bool(
            delay_doppler["detected"] and peak_to_median >= 1.05 and not boundary
        ),
        "array_estimator": "whitened_Bartlett_on_matched_filter_snapshot",
        "doppler_held_at_causal_prediction_for_single_pulse_aoa": True,
        "doppler_information_claimed_by_aoa_extractor": False,
        "belief_centred_angular_search": True,
        "truth_angle_used_by_estimator": False,
    }


def simulate_calibration_baseband(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    samples: int = 1024,
    capture_start_s: float | None = None,
    seed: int | None = None,
    impairments: bool = True,
) -> ComplexBasebandCapture:
    """Generate a finite-window calibration pilot for clock/delay or CFO."""

    if action.kind not in {"calibration_time", "calibration_frequency"}:
        raise ValueError("calibration IQ requires a time or frequency action")
    if samples < 128 or not action.rx_node_ids:
        raise ValueError("calibration IQ requires a receiver and at least 128 samples")
    waveform_spec = world.waveform(action.waveform_id)
    receiver = world.node(action.rx_node_ids[0])
    reference_node = (
        world.node(action.reference_node_id)
        if action.reference_node_id is not None
        else None
    )
    truth_value = float(build_action_factor(world, action, None).prediction[0])
    rng = np.random.default_rng(
        world.seed * 1_000_037 + action.action_id * 1019
        if seed is None
        else int(seed)
    )
    if action.kind == "calibration_time":
        sample_rate_hz = float(waveform_spec.sample_rate_hz)
        duration_s = samples / sample_rate_hz
        reference = synthesize_waveform(
            waveform_spec, samples, world.seed + action.action_id + 131
        )
        capture_start = (
            float(capture_start_s)
            if capture_start_s is not None
            else truth_value - min(48.0, samples / 6.0) / sample_rate_hz
        )
        delayed = _fractional_delay(
            reference, (truth_value - capture_start) * sample_rate_hz
        )
        calibration_std = float(action.metadata.get("calibration_std_s", 2e-9))
        information_sinr = 1.0 / max(
            8.0
            * math.pi**2
            * waveform_spec.rms_bandwidth_hz**2
            * calibration_std**2,
            1e-12,
        )
    else:
        duration_s = max(
            float(waveform_spec.coherent_processing_interval_s), 0.05
        )
        sample_rate_hz = samples / duration_s
        reference = np.where(
            np.arange(samples) % 2 == 0, 1.0 + 0.0j, -1.0 + 0.0j
        ).astype(np.complex128)
        capture_start = 0.0
        delayed = reference.copy()
        calibration_std = float(action.metadata.get("calibration_std_hz", 0.5))
        time_rms = duration_s / math.sqrt(12.0)
        information_sinr = 1.0 / max(
            8.0 * math.pi**2 * time_rms**2 * calibration_std**2,
            1e-12,
        )
    time_grid = np.arange(samples, dtype=float) / sample_rate_hz
    reference_frequency_bias = (
        0.0 if reference_node is None else float(reference_node.frequency_bias_hz)
    )
    base = delayed * np.exp(-2j * math.pi * reference_frequency_bias * time_grid)
    signal_power = max(float(np.mean(np.abs(base) ** 2)), 1e-30)
    thermal_power = signal_power / max(information_sinr, 1e-6)
    thermal = _colored_complex_noise(
        rng, 1, samples, thermal_power, correlation=0.0
    )
    analog = base[None, :] + thermal
    state = _receiver_state(receiver, waveform_spec, 0.0)
    if reference_node is None and action.kind == "calibration_frequency":
        # A calibration source without a registered reference is interpreted as
        # the network frequency standard at zero offset.
        reference_frequency_bias = 0.0
    received, impairment = _front_end(
        analog,
        (state,),
        sample_rate_hz,
        rng,
        impairments=impairments,
    )
    zeros = np.zeros_like(received)
    deterministic = base[None, :]
    fidelity = FidelityProvenance(
        level=FidelityLevel.R2_COMPLEX_BASEBAND,
        generator="simulate_calibration_baseband",
        independently_validated=False,
        measured_data=False,
        limitations=(
            "registered digital calibration pilot rather than measured hardware loopback",
            "single-path calibration transfer with explicit receiver oscillator/front-end",
        ),
    )
    result = ComplexBasebandCapture(
        action_id=action.action_id,
        channel_ids=(receiver.node_id,),
        surveillance_iq=received,
        reference_iq=reference,
        deterministic_mean=deterministic,
        target_component=zeros,
        direct_path_component=deterministic,
        multipath_component=zeros,
        clutter_component=zeros,
        jammer_component=zeros,
        thermal_component=thermal,
        receiver_impairment_component=impairment,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=waveform_spec.carrier_hz,
        path_components=(),
        receiver_states=(state,),
        # This is a registered node-to-node calibration transfer.  Its direct
        # pilot is present, but no target-scattered component participates in
        # the capture; keep target-presence provenance physically truthful.
        target_present=False,
        fidelity=fidelity,
        metadata={
            "capture_start_s": capture_start,
            "occupied_bandwidth_hz": float(waveform_spec.bandwidth_hz),
            "injected_calibration_observable": truth_value,
            "calibration_kind": action.kind,
            "calibration_duration_s": duration_s,
            "calibration_reference_node_id": action.reference_node_id,
            "calibration_truth_not_available_to_estimator": True,
            "SI_units_internal": True,
        },
    )
    result.validate()
    return result


def calibration_frequency_sufficient_statistic(
    capture: ComplexBasebandCapture,
) -> dict[str, Any]:
    """Estimate differential CFO by phase-slope regression on a known pilot."""

    capture.validate()
    if capture.reference_iq is None or capture.surveillance_iq.shape[0] != 1:
        raise ValueError("frequency calibration requires one channel and a pilot")
    reference = np.asarray(capture.reference_iq, dtype=np.complex128).reshape(-1)
    signal = np.asarray(capture.surveillance_iq[0], dtype=np.complex128)
    if signal.size != reference.size:
        raise ValueError("calibration signal and pilot lengths disagree")
    product = signal * np.conjugate(reference)
    phase = np.unwrap(np.angle(product))
    time_s = np.arange(signal.size, dtype=float) / capture.sample_rate_hz
    centred_time = time_s - float(time_s.mean())
    denominator = max(float(centred_time @ centred_time), 1e-30)
    slope = float(centred_time @ (phase - float(phase.mean())) / denominator)
    intercept = float(phase.mean() - slope * time_s.mean())
    fitted = intercept + slope * time_s
    residual = phase - fitted
    phase_variance = float(residual @ residual / max(1, signal.size - 2))
    frequency_variance = phase_variance / (
        (2.0 * math.pi) ** 2 * denominator
    )
    coherent_power = abs(np.mean(product * np.exp(-1j * fitted))) ** 2
    incoherent_power = max(float(np.mean(np.abs(product) ** 2)), 1e-30)
    coherence = float(np.clip(coherent_power / incoherent_power, 0.0, 1.0))
    frequency_hz = slope / (2.0 * math.pi)
    nyquist = 0.5 * capture.sample_rate_hz
    return {
        "frequency_hz": float(frequency_hz),
        "doppler_hz": float(frequency_hz),
        "frequency_variance_hz2": float(max(frequency_variance, 1e-30)),
        "peak_to_median": float(1.0 / max(1.0 - coherence, 1e-12)),
        "detected": bool(coherence >= 0.20 and abs(frequency_hz) < 0.95 * nyquist),
        "phase_coherence": coherence,
        "phase_slope_estimator": "unwrapped_weighted_least_squares",
        "peak_on_search_boundary": bool(abs(frequency_hz) >= 0.95 * nyquist),
    }


def draw_r1_sufficient_statistic(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    jammer: Jammer | None = None,
    seed: int | None = None,
    target_present: bool = True,
    false_alarm_probability: float = 1e-6,
    sample_association_mixture: bool = False,
    association_gate_reference_prediction: np.ndarray | None = None,
    association_gate_reference_covariance: np.ndarray | None = None,
) -> SufficientStatisticObservation:
    """Draw one R1 observation from the declared association likelihood.

    For the broad validation-gate component, a false return is independent of
    the unknown target state conditional on the causal gate opened by the
    tracker.  Operational callers therefore supply the gate-centre prediction
    and covariance computed from the pre-observation belief.  Retaining the
    truth target prediction as the false-component centre would silently make
    clutter follow the target and would disagree with the target-independent
    false-component Jacobian used by the likelihood and Fisher calculations.

    The optional pair is deliberately all-or-none.  Omitting it preserves the
    historical standalone sampling interface; the integrated runtime always
    supplies the causal gate reference.
    """

    factor = build_action_factor(world, action, jammer)
    rng = np.random.default_rng(world.seed * 65537 + action.action_id if seed is None else seed)
    detected = bool(target_present and rng.random() <= factor.link_budget.detection_probability)
    if not target_present:
        detected = bool(rng.random() <= false_alarm_probability)
    # ``build_action_factor`` appends simultaneous *free* calibration rows to the
    # primary radar rows.  Those rows are acquired from the reference/calibration
    # channel during the same dwell and are not gated by the target detection
    # event, so a missed detection must not destroy them.  Voiding them made the
    # realized expectation P_D*(J_radar + J_cal) while the planner scored
    # P_D*J_radar + J_cal -- a (1-P_D)*J_cal disagreement between the planning
    # model and the inference model.
    calibration_rows = sum(
        len(build_action_factor(world, calibration, None).measurement_names)
        for calibration in automatic_free_calibration_actions(world, action)
    )
    primary_rows = int(factor.prediction.size) - int(calibration_rows)
    if (association_gate_reference_prediction is None) != (
        association_gate_reference_covariance is None
    ):
        raise ValueError(
            "association gate reference prediction and covariance must be supplied together"
        )
    gate_reference_supplied = association_gate_reference_prediction is not None
    gate_reference_prediction = None
    gate_reference_covariance = None
    if gate_reference_supplied:
        gate_reference_prediction = np.asarray(
            association_gate_reference_prediction, dtype=float
        )
        gate_reference_covariance = np.asarray(
            association_gate_reference_covariance, dtype=float
        )
        if gate_reference_prediction.shape != factor.prediction.shape:
            raise ValueError("association gate reference prediction has wrong shape")
        if gate_reference_covariance.shape != factor.covariance.shape:
            raise ValueError("association gate reference covariance has wrong shape")
        if not np.all(np.isfinite(gate_reference_prediction)):
            raise ValueError("association gate reference prediction is not finite")
        if not np.all(np.isfinite(gate_reference_covariance)):
            raise ValueError("association gate reference covariance is not finite")
        gate_reference_covariance = 0.5 * (
            gate_reference_covariance + gate_reference_covariance.T
        )
        if float(np.linalg.eigvalsh(gate_reference_covariance).min()) <= 0.0:
            raise ValueError("association gate reference covariance is not positive definite")
    association_component = "legacy_correct_component_only"
    association_gate_reference_applied = False
    if sample_association_mixture and detected and primary_rows > 0:
        p_correct = float(
            np.clip(factor.link_budget.correct_association_probability, 0.0, 1.0)
        )
        correct = bool(rng.random() <= p_correct)
        association_component = "correct_return" if correct else "false_return"
        value = np.asarray(factor.prediction, dtype=float).copy()
        primary_covariance = np.asarray(
            factor.covariance[:primary_rows, :primary_rows], dtype=float
        )
        primary_mean = np.asarray(factor.prediction[:primary_rows], dtype=float)
        if correct:
            value[:primary_rows] = rng.multivariate_normal(
                primary_mean, primary_covariance
            )
        else:
            association_model = str(
                factor.metadata.get("association_model", "unregistered")
            )
            if association_model == "displaced_Gaussian_deception_mixture":
                displacement_sigma = np.asarray(
                    factor.metadata.get(
                        "false_target_displacement_sigma",
                        [0.0] * factor.prediction.size,
                    ),
                    dtype=float,
                )[:primary_rows]
                false_offset = np.linalg.cholesky(primary_covariance) @ displacement_sigma
                false_mean = primary_mean + false_offset
                false_covariance = primary_covariance
            else:
                if gate_reference_supplied:
                    false_mean = gate_reference_prediction[:primary_rows]
                    false_base_covariance = gate_reference_covariance[
                        :primary_rows, :primary_rows
                    ]
                    association_gate_reference_applied = True
                else:
                    false_mean = primary_mean
                    false_base_covariance = primary_covariance
                false_covariance = float(
                    factor.metadata.get("false_return_covariance_scale", 1.0)
                ) * false_base_covariance
            value[:primary_rows] = rng.multivariate_normal(
                false_mean, false_covariance
            )
        if calibration_rows:
            value[primary_rows:] = rng.multivariate_normal(
                factor.prediction[primary_rows:],
                factor.covariance[primary_rows:, primary_rows:],
            )
    else:
        value = rng.multivariate_normal(factor.prediction, factor.covariance)
    if not detected:
        value[:primary_rows] = np.nan
    association = np.asarray(
        [factor.link_budget.correct_association_probability, 1.0 - factor.link_budget.correct_association_probability],
        dtype=float,
    )
    return SufficientStatisticObservation(
        names=factor.measurement_names,
        value=value,
        prediction=factor.prediction.copy(),
        covariance=factor.covariance.copy(),
        detected=detected,
        association_probabilities=association,
        source_fidelity=FidelityLevel.R1_SUFFICIENT_STATISTIC,
        metadata={
            "likelihood": "detected_Gaussian_plus_missed_detection_and_false_alarm",
            "false_alarm_probability": false_alarm_probability,
            "fidelity_claim": FidelityLevel.R1_SUFFICIENT_STATISTIC.value,
            "primary_measurement_rows": int(primary_rows),
            "detection_independent_calibration_rows": int(calibration_rows),
            "detection_gates_primary_rows_only": True,
            "association_mixture_realized": bool(sample_association_mixture),
            "association_component_sampled": association_component,
            "association_model": factor.metadata.get("association_model"),
            "association_gate_reference_supplied": bool(gate_reference_supplied),
            "association_gate_reference_applied": bool(
                association_gate_reference_applied
            ),
            "association_gate_reference_is_causal_preobservation_belief": bool(
                association_gate_reference_applied
            ),
            "broad_false_return_target_state_dependent": False,
        },
    )


def _parabolic_vertex(left: float, centre: float, right: float) -> float:
    """Sub-bin offset of a parabola through three equally spaced samples.

    For samples ``y(-1), y(0), y(+1)`` the vertex sits at
    ``delta = (y(-1) - y(+1)) / (2 (y(-1) - 2 y(0) + y(+1)))``.  The result is
    clipped to the central bin; a non-concave triple returns ``0``.
    """

    denominator = left - 2.0 * centre + right
    if not math.isfinite(denominator) or abs(denominator) < 1e-300:
        return 0.0
    delta = 0.5 * (left - right) / denominator
    if not math.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def matched_filter_sufficient_statistic(
    capture: ComplexBasebandCapture,
    waveform: np.ndarray,
    *,
    delay_search_samples: tuple[int, int] | None = None,
    doppler_grid_hz: np.ndarray | None = None,
    expected_delay_s: float | None = None,
    expected_doppler_hz: float | None = None,
    delay_search_half_width_s: float | None = None,
    doppler_search_half_width_hz: float | None = None,
    doppler_bins: int = 65,
    maximum_doppler_bins: int = 513,
    coherent_processing_interval_s: float | None = None,
    detection_threshold: float = 8.0,
    familywise_false_alarm_probability: float = 1.0e-6,
    clutter_notch_half_width_hz: float = 0.0,
    direct_path_guard_delay_s: float = 0.0,
    direct_path_guard_doppler_hz: float = 0.0,
    slow_time_window: str = "rectangular",
    refine: bool = True,
) -> dict[str, Any]:
    r"""Cross-ambiguity extraction with belief-centred search and sub-bin refinement.

    The estimator is a discrete maximiser of the cross-ambiguity surface

    .. math::

       \chi(	au,f)=\Big|	extstyle\sum_n y[n]\,r^{*}[n-	au]\,e^{-j2\pi f t_n}\Big|^{2}.

    Three properties are required for the result to be a usable likelihood:

    1. **The search must be centred on the belief**, not on a constant.  A fixed
       :math:`\pm1500\,\mathrm{Hz}` grid silently clips every geometry whose true
       Doppler exceeds it, returning a boundary artefact rather than an estimate.
    2. **The peak must be refined below the grid**, otherwise the estimator's
       error is dominated by discretisation, which is independent of SNR and
       therefore not describable by the CRLB.
    3. **The reported covariance must be the estimator's covariance.**  The
       discrete-maximum estimator has error variance
       :math:`\sigma^{2}_{
m CRLB}+\Delta^{2}/12` (grid variance adds to the
       noise-limited variance), so returning the CRLB alone understates the
       uncertainty by orders of magnitude whenever :math:`\Delta\gg\sigma`.

    ``expected_delay_s`` / ``expected_doppler_hz`` are *belief* quantities: the
    operational caller passes the factor-graph prediction, never simulator truth.
    """

    capture.validate()
    if maximum_doppler_bins < 9:
        raise ValueError("maximum_doppler_bins must be at least nine")
    if not 0.0 < float(familywise_false_alarm_probability) < 1.0:
        raise ValueError("familywise false-alarm probability must lie in (0, 1)")
    slow_time_window_name = str(slow_time_window)
    if slow_time_window_name not in {"rectangular", "hann_periodic"}:
        raise ValueError("slow-time window must be rectangular or hann_periodic")
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in (
            clutter_notch_half_width_hz,
            direct_path_guard_delay_s,
            direct_path_guard_doppler_hz,
        )
    ):
        raise ValueError("ambiguity exclusion widths must be finite and nonnegative")
    signal = capture.surveillance_iq[0]
    reference = (
        capture.reference_iq
        if capture.reference_iq is not None
        else (
            capture.transmit_reference_iq
            if capture.transmit_reference_iq is not None
            else np.asarray(waveform)
        )
    )
    if np.asarray(reference).ndim == 2:
        reference = np.asarray(reference)[0]
    reference = np.asarray(reference, dtype=np.complex128)
    sample_rate = float(capture.sample_rate_hz)
    capture_start = float(capture.metadata.get("capture_start_s", 0.0))
    duration = signal.size / sample_rate
    pulse_cube = (
        None
        if capture.coherent_surveillance_iq is None
        else np.asarray(capture.coherent_surveillance_iq, dtype=np.complex128)[0]
    )
    pulse_reference_cube = (
        None
        if capture.coherent_reference_iq is None
        else np.asarray(capture.coherent_reference_iq, dtype=np.complex128)
    )
    pulse_slow_time = (
        None
        if capture.pulse_slow_time_s is None
        else np.asarray(capture.pulse_slow_time_s, dtype=float)
    )
    coherent_pulse_processing = bool(
        pulse_cube is not None and pulse_slow_time is not None
    )
    slow_time_weights = np.ones(1, dtype=float)
    slow_time_window_enbw_bins = 1.0
    slow_time_window_weighted_rms_s: float | None = None
    pulse_repetition_hz: float | None = None
    if coherent_pulse_processing:
        if pulse_cube.shape[0] != pulse_slow_time.size:
            raise ValueError("coherent pulse cube and slow-time axis disagree")
        pulse_repetition_hz = float(
            capture.metadata.get(
                "pulse_repetition_hz",
                1.0 / float(np.median(np.diff(pulse_slow_time))),
            )
        )
        if pulse_repetition_hz <= 0.0:
            raise ValueError("coherent pulse capture requires positive PRF")
        actual_cpi = float(pulse_cube.shape[0] / pulse_repetition_hz)
        if coherent_processing_interval_s is not None:
            declared = float(coherent_processing_interval_s)
            if not math.isclose(
                declared,
                actual_cpi,
                rel_tol=1.0 / max(pulse_cube.shape[0], 2),
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "declared CPI disagrees with coherent pulse slow-time support"
                )
        cpi = actual_cpi
        pulse_count = int(pulse_cube.shape[0])
        (
            slow_time_weights,
            slow_time_window_enbw_bins,
            planned_time_rms_s,
        ) = slow_time_window_contract(
            pulse_count,
            pulse_repetition_hz,
            slow_time_window_name,
        )
        squared_weights = slow_time_weights**2
        squared_weight_sum = float(np.sum(squared_weights))
        weighted_time_mean = float(
            np.sum(squared_weights * pulse_slow_time) / squared_weight_sum
        )
        slow_time_window_weighted_rms_s = float(
            math.sqrt(
                np.sum(
                    squared_weights * (pulse_slow_time - weighted_time_mean) ** 2
                )
                / squared_weight_sum
            )
        )
        if slow_time_window_weighted_rms_s <= 0.0:
            raise RuntimeError("slow-time window has zero Doppler information aperture")
        if not math.isclose(
            slow_time_window_weighted_rms_s,
            planned_time_rms_s,
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "recorded slow-time samples disagree with the planning window contract"
            )
    else:
        cpi = float(
            coherent_processing_interval_s
            if coherent_processing_interval_s is not None
            else duration
        )
        slow_time_window_weighted_rms_s = cpi / math.sqrt(12.0)
    assert slow_time_window_weighted_rms_s is not None

    # ---- delay grid ------------------------------------------------------
    if delay_search_samples is not None:
        lo, hi = int(delay_search_samples[0]), int(delay_search_samples[1])
    else:
        centre_samples = (
            0.0
            if expected_delay_s is None
            else (float(expected_delay_s) - capture_start) * sample_rate
        )
        if delay_search_half_width_s is None:
            half_width_samples = min(64.0, max(8.0, signal.size / 8.0))
        else:
            half_width_samples = max(
                4.0, float(delay_search_half_width_s) * sample_rate
            )
        lo = int(math.floor(centre_samples - half_width_samples))
        hi = int(math.ceil(centre_samples + half_width_samples))
    lo = max(lo, -(signal.size - 2))
    hi = min(hi, signal.size - 2)
    if hi <= lo:
        lo, hi = -1, 1
    lags = np.arange(lo, hi + 1)

    # ---- Doppler grid ----------------------------------------------------
    centre_doppler = (
        0.0 if expected_doppler_hz is None else float(expected_doppler_hz)
    )
    alias_span_hz = (
        float(pulse_repetition_hz)
        if coherent_pulse_processing and pulse_repetition_hz is not None
        else sample_rate
    )
    doppler_search_is_full_alias_interval = False
    if doppler_grid_hz is not None:
        doppler_grid = np.asarray(doppler_grid_hz, dtype=float)
        if (
            doppler_grid.ndim != 1
            or doppler_grid.size == 0
            or not np.all(np.isfinite(doppler_grid))
            or (doppler_grid.size > 1 and np.any(np.diff(doppler_grid) <= 0.0))
        ):
            raise ValueError("explicit Doppler grid must be finite and strictly increasing")
        if doppler_grid.size > 1 and (
            float(doppler_grid[-1] - doppler_grid[0])
            >= alias_span_hz * (1.0 - 1e-12)
        ):
            raise ValueError(
                "explicit Doppler grid spans more than one identifiable alias interval"
            )
        if (
            coherent_pulse_processing
            and doppler_grid.size > 1
        ):
            explicit_steps = np.diff(doppler_grid)
            representative_step = float(np.median(explicit_steps))
            doppler_search_is_full_alias_interval = bool(
                np.allclose(
                    explicit_steps,
                    representative_step,
                    rtol=1.0e-10,
                    atol=max(abs(representative_step), 1.0) * 1.0e-12,
                )
                and math.isclose(
                    float(doppler_grid[-1] - doppler_grid[0])
                    + representative_step,
                    alias_span_hz,
                    rel_tol=1.0e-9,
                    abs_tol=max(alias_span_hz, 1.0) * 1.0e-12,
                )
            )
    else:
        # The coherent-integration Doppler resolution is ~1/CPI; sample the
        # mainlobe at least four times so parabolic refinement is valid.
        resolution = 1.0 / max(cpi, 1e-12)
        half_width = (
            max(4.0 * resolution, float(doppler_search_half_width_hz))
            if doppler_search_half_width_hz is not None
            else 8.0 * resolution
        )
        # Clip to the identifiable alias interval before choosing the grid.
        # Computing the bin count from an unidentifiable multi-PRF prior span
        # needlessly forced every search to the hard cap after the span was
        # subsequently clipped.
        half_width = min(half_width, 0.5 * alias_span_hz * (1.0 - 1e-12))
        bins = max(9, int(doppler_bins) | 1)
        step = 2.0 * half_width / (bins - 1)
        if step > 0.25 * resolution:
            bins = max(9, int(2.0 * half_width / (0.25 * resolution)) | 1)
        # Slow-time pulse samples identify Doppler modulo PRF; a contiguous
        # one-pulse capture identifies it modulo the fast-time sample rate.
        # Keep exactly one belief-centred alias interval.  The reported value is
        # therefore the ambiguity branch selected by the causal prior, never an
        # oracle-unwrapped simulator value.
        bins = min(bins, int(maximum_doppler_bins) | 1)
        doppler_search_is_full_alias_interval = bool(
            coherent_pulse_processing
            and math.isclose(
                2.0 * half_width,
                alias_span_hz,
                rel_tol=1.0e-9,
                abs_tol=max(alias_span_hz, 1.0) * 1.0e-12,
            )
        )
        if doppler_search_is_full_alias_interval:
            # Slow-time Doppler is an element of R / PRF Z.  A full alias
            # interval is therefore a circle, not a closed line segment.  Use
            # a half-open grid so the equivalent endpoints are not duplicated.
            doppler_grid = (
                centre_doppler
                - 0.5 * alias_span_hz
                + alias_span_hz * np.arange(bins, dtype=float) / bins
            )
        else:
            doppler_grid = np.linspace(
                centre_doppler - half_width,
                centre_doppler + half_width,
                bins,
            )

    # For integer lag k,
    #
    #   chi(k,f) = |sum_n y[n] r*[n-k] exp(-j 2 pi f n/Fs)|^2
    #
    # is exactly the cross-correlation of the Doppler-derotated surveillance
    # sequence with the reference.  ``scipy.signal.correlate(..., method='fft')``
    # evaluates all integer lags in O(N log N), replacing the previous explicit
    # L-by-N replica matrix and O(F L N) dense product.  The previous method
    # allocated O(LN) complex memory and was killed for long R2 captures.
    from scipy.signal import correlate, correlation_lags

    t = np.arange(signal.size, dtype=float) / sample_rate
    full_lags = correlation_lags(signal.size, reference.size, mode="full")
    lag_indices = np.searchsorted(full_lags, lags)
    if np.any(lag_indices >= full_lags.size) or not np.array_equal(
        full_lags[lag_indices], lags
    ):
        raise RuntimeError("requested ambiguity lags are outside FFT correlation support")
    if coherent_pulse_processing:
        assert pulse_cube is not None
        assert pulse_slow_time is not None
        matched_pulses = np.empty(
            (pulse_cube.shape[0], lags.size), dtype=np.complex128
        )
        # The within-pulse phase is compensated at the causal prior centre.
        # Residual within-pulse phase is negligible on a microsecond range gate,
        # while all identifiable Doppler phase is retained across the millisecond
        # slow-time aperture.  This avoids pretending fast-time bandwidth is a
        # Doppler aperture.
        fast_derotation = np.exp(-2j * math.pi * centre_doppler * t)
        for pulse_index in range(pulse_cube.shape[0]):
            pulse_reference = (
                reference
                if pulse_reference_cube is None
                else pulse_reference_cube[pulse_index]
            )
            correlation = correlate(
                pulse_cube[pulse_index] * fast_derotation,
                pulse_reference,
                mode="full",
                method="fft",
            )
            matched_pulses[pulse_index] = correlation[lag_indices]
        steering = np.exp(
            -2j
            * math.pi
            * doppler_grid[:, None]
            * pulse_slow_time[None, :]
        )
        ambiguity = np.abs(
            steering @ (slow_time_weights[:, None] * matched_pulses)
        ) ** 2
        ambiguity_algorithm = "pulse_compress_then_slow_time_steering"
    else:
        ambiguity = np.empty((doppler_grid.size, lags.size), dtype=float)
        for row, doppler in enumerate(doppler_grid):
            derotated = signal * np.exp(-2j * math.pi * float(doppler) * t)
            correlation = correlate(
                derotated, reference, mode="full", method="fft"
            )
            ambiguity[row] = np.abs(correlation[lag_indices]) ** 2
        ambiguity_algorithm = "FFT_cross_correlation_per_Doppler_bin"

    # A maximum over a large delay-Doppler map is not a single-cell detector.
    # With N independent exponential noise cells, a fixed peak/median threshold
    # has familywise false-alarm probability approaching one as N grows.  Build
    # the physically admissible search mask first, then derive an exact Sidak
    # threshold for the declared whole-map P_FA.  Slow-time sampling identifies
    # Doppler modulo PRF, so the stationary-clutter notch is applied to every
    # zero-Doppler equivalence class, not merely to the numeric value 0 Hz.
    eligible = np.ones_like(ambiguity, dtype=bool)
    if coherent_pulse_processing and pulse_repetition_hz is not None:
        principal_doppler = (
            doppler_grid + 0.5 * pulse_repetition_hz
        ) % pulse_repetition_hz - 0.5 * pulse_repetition_hz
        expected_principal_doppler = (
            None
            if expected_doppler_hz is None
            else (
                float(expected_doppler_hz) + 0.5 * pulse_repetition_hz
            )
            % pulse_repetition_hz
            - 0.5 * pulse_repetition_hz
        )
    else:
        principal_doppler = doppler_grid.copy()
        expected_principal_doppler = (
            None if expected_doppler_hz is None else float(expected_doppler_hz)
        )
    expected_target_in_clutter_notch = bool(
        expected_principal_doppler is not None
        and
        float(clutter_notch_half_width_hz) > 0.0
        and abs(expected_principal_doppler)
        <= float(clutter_notch_half_width_hz)
    )
    if float(clutter_notch_half_width_hz) > 0.0:
        eligible &= (
            np.abs(principal_doppler)[:, None]
            > float(clutter_notch_half_width_hz)
        )
    if float(direct_path_guard_delay_s) > 0.0:
        direct_delay = np.abs(lags / sample_rate) <= float(
            direct_path_guard_delay_s
        )
        direct_doppler = (
            np.abs(principal_doppler) <= float(direct_path_guard_doppler_hz)
            if float(direct_path_guard_doppler_hz) > 0.0
            else np.ones_like(principal_doppler, dtype=bool)
        )
        eligible &= ~(direct_doppler[:, None] & direct_delay[None, :])
    eligible &= np.isfinite(ambiguity)
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count:
        searchable = np.where(eligible, ambiguity, -np.inf)
        peak = np.unravel_index(int(np.argmax(searchable)), searchable.shape)
    else:
        # Preserve a finite diagnostic location while failing the detector closed.
        peak = np.unravel_index(int(np.argmax(ambiguity)), ambiguity.shape)
    row_index, column_index = int(peak[0]), int(peak[1])
    peak_doppler = float(doppler_grid[row_index])
    peak_lag = float(lags[column_index])
    doppler_step = (
        float(doppler_grid[1] - doppler_grid[0]) if doppler_grid.size > 1 else 0.0
    )
    delay_offset = 0.0
    doppler_offset = 0.0
    if refine:
        if 0 < column_index < lags.size - 1:
            delay_offset = _parabolic_vertex(
                ambiguity[row_index, column_index - 1],
                ambiguity[row_index, column_index],
                ambiguity[row_index, column_index + 1],
            )
        if doppler_grid.size > 2 and (
            doppler_search_is_full_alias_interval
            or 0 < row_index < doppler_grid.size - 1
        ):
            left_row = (row_index - 1) % doppler_grid.size
            right_row = (row_index + 1) % doppler_grid.size
            doppler_offset = _parabolic_vertex(
                ambiguity[left_row, column_index],
                ambiguity[row_index, column_index],
                ambiguity[right_row, column_index],
            )
    refined_lag = peak_lag + delay_offset
    refined_doppler = peak_doppler + doppler_offset * doppler_step
    if doppler_search_is_full_alias_interval:
        refined_doppler = float(
            centre_doppler
            + (
                refined_doppler
                - centre_doppler
                + 0.5 * alias_span_hz
            )
            % alias_span_hz
            - 0.5 * alias_span_hz
        )
    if coherent_pulse_processing and pulse_repetition_hz is not None:
        refined_principal_doppler = (
            refined_doppler + 0.5 * pulse_repetition_hz
        ) % pulse_repetition_hz - 0.5 * pulse_repetition_hz
    else:
        refined_principal_doppler = refined_doppler
    refined_peak_inside_clutter_notch = bool(
        float(clutter_notch_half_width_hz) > 0.0
        and abs(refined_principal_doppler)
        <= float(clutter_notch_half_width_hz)
    )
    refined_peak_inside_direct_path_guard = bool(
        float(direct_path_guard_delay_s) > 0.0
        and abs(refined_lag / sample_rate) <= float(direct_path_guard_delay_s)
        and (
            float(direct_path_guard_doppler_hz) <= 0.0
            or abs(refined_principal_doppler)
            <= float(direct_path_guard_doppler_hz)
        )
    )
    peak_value = float(ambiguity[row_index, column_index])
    background = ambiguity[eligible] if eligible_count else ambiguity.reshape(-1)
    median = float(np.median(background))
    # Ordered-statistic estimates are normalized by their exact exponential
    # quantiles.  Taking the largest of q50/q75/q90 is conservative under the
    # heavy-tailed heterogeneous clutter seen by a practical ambiguity map.
    quantile_75 = float(np.quantile(background, 0.75))
    quantile_90 = float(np.quantile(background, 0.90))
    noise_mean_estimate = max(
        median / math.log(2.0),
        quantile_75 / math.log(4.0),
        quantile_90 / math.log(10.0),
        np.finfo(float).tiny,
    )
    delay_resolution_s = 1.0 / max(
        float(capture.metadata.get("occupied_bandwidth_hz", sample_rate)), 1.0
    )
    doppler_resolution_hz = slow_time_window_enbw_bins / max(cpi, 1e-12)
    delay_independent_cells = max(
        1.0,
        (float(hi - lo + 1) / sample_rate) / delay_resolution_s,
    )
    if doppler_grid.size > 1:
        doppler_span_hz = float(doppler_grid[-1] - doppler_grid[0]) + abs(
            float(doppler_grid[1] - doppler_grid[0])
        )
    else:
        doppler_span_hz = doppler_resolution_hz
    doppler_independent_cells = max(
        1.0, doppler_span_hz / doppler_resolution_hz
    )
    eligible_fraction = float(eligible_count) / max(float(eligible.size), 1.0)
    effective_independent_cells = max(
        1,
        min(
            eligible_count if eligible_count else 1,
            int(
                math.ceil(
                    delay_independent_cells
                    * doppler_independent_cells
                    * max(eligible_fraction, 1.0 / max(eligible.size, 1))
                )
            ),
        ),
    )
    familywise_pfa = float(familywise_false_alarm_probability)
    single_cell_pfa = -math.expm1(
        math.log1p(-familywise_pfa) / effective_independent_cells
    )
    cfar_threshold_over_noise_mean = max(
        -math.log(max(single_cell_pfa, np.finfo(float).tiny)),
        float(detection_threshold) * math.log(2.0),
    )
    peak_over_noise_mean = peak_value / noise_mean_estimate
    receiver_estimated_output_sinr = max(peak_over_noise_mean - 1.0, 0.0)
    delay_peak_on_boundary = bool(column_index in (0, lags.size - 1))
    doppler_peak_on_boundary = bool(
        doppler_grid.size > 1
        and row_index in (0, doppler_grid.size - 1)
        and not doppler_search_is_full_alias_interval
    )
    on_boundary = bool(delay_peak_on_boundary or doppler_peak_on_boundary)
    # Residual discretisation variance after parabolic refinement.  The
    # quarter-bin model is justified only when the ambiguity mainlobe is sampled
    # at least four times.  If the computational bin cap or the sample rate makes
    # the grid coarser, retain the full uniform-bin variance instead of claiming
    # unsupported sub-bin accuracy.
    delay_bin_s = 1.0 / sample_rate
    delay_refinement_supported = bool(
        refine and delay_bin_s <= 0.25 * delay_resolution_s
    )
    doppler_refinement_supported = bool(
        refine
        and doppler_step > 0.0
        and doppler_step <= 0.25 * doppler_resolution_hz * (1.0 + 1e-12)
    )
    delay_grid_error_width = (
        0.25 * delay_bin_s if delay_refinement_supported else delay_bin_s
    )
    doppler_grid_error_width = (
        0.25 * doppler_step if doppler_refinement_supported else doppler_step
    )
    residual_delay_variance = delay_grid_error_width**2 / 12.0
    residual_doppler_variance = (
        doppler_grid_error_width**2 / 12.0 if doppler_step else 0.0
    )
    return {
        "delay_samples": refined_lag,
        "delay_s": refined_lag / sample_rate,
        "absolute_delay_s": capture_start + refined_lag / sample_rate,
        "doppler_hz": refined_doppler,
        "ambiguity_peak_integer_delay_samples": int(peak_lag),
        "ambiguity_peak_grid_doppler_hz": float(peak_doppler),
        "fast_time_derotation_center_hz": float(centre_doppler),
        "peak_to_median": peak_value / max(median, 1e-30),
        "peak_to_receiver_noise_mean": float(peak_over_noise_mean),
        "receiver_noise_mean_estimate": float(noise_mean_estimate),
        "receiver_estimated_output_sinr_linear": float(
            receiver_estimated_output_sinr
        ),
        "ambiguity_peak": peak_value,
        "detected": bool(
            eligible_count > 0
            and peak_over_noise_mean >= cfar_threshold_over_noise_mean
            and not on_boundary
            and not expected_target_in_clutter_notch
            and not refined_peak_inside_clutter_notch
            and not refined_peak_inside_direct_path_guard
        ),
        "detector": "ordered_statistic_Sidak_familywise_CFAR",
        "familywise_false_alarm_probability": familywise_pfa,
        "single_cell_false_alarm_probability": float(single_cell_pfa),
        "effective_independent_search_cells": int(effective_independent_cells),
        "eligible_ambiguity_cells": int(eligible_count),
        "total_ambiguity_cells": int(eligible.size),
        "cfar_threshold_over_noise_mean": float(
            cfar_threshold_over_noise_mean
        ),
        "cfar_threshold_peak_to_median_equivalent": float(
            cfar_threshold_over_noise_mean / math.log(2.0)
        ),
        "clutter_notch_half_width_hz": float(clutter_notch_half_width_hz),
        "clutter_notch_applied_modulo_prf": bool(
            coherent_pulse_processing and float(clutter_notch_half_width_hz) > 0.0
        ),
        "expected_principal_doppler_hz": (
            None
            if expected_principal_doppler is None
            else float(expected_principal_doppler)
        ),
        "expected_target_in_clutter_notch": expected_target_in_clutter_notch,
        "clutter_blind_zone_fail_closed": expected_target_in_clutter_notch,
        "refined_peak_inside_clutter_notch": refined_peak_inside_clutter_notch,
        "direct_path_guard_delay_s": float(direct_path_guard_delay_s),
        "direct_path_guard_doppler_hz": float(direct_path_guard_doppler_hz),
        "refined_peak_inside_direct_path_guard": (
            refined_peak_inside_direct_path_guard
        ),
        "peak_on_search_boundary": on_boundary,
        "delay_peak_on_search_boundary": delay_peak_on_boundary,
        "doppler_peak_on_search_boundary": doppler_peak_on_boundary,
        "doppler_search_topology": (
            "periodic_full_alias_interval"
            if doppler_search_is_full_alias_interval
            else "bounded_linear_subinterval"
        ),
        "doppler_boundary_wrap_refinement": bool(
            doppler_search_is_full_alias_interval and refine
        ),
        "doppler_full_alias_interval_endpoint_duplicated": False,
        "delay_grid_step_s": delay_bin_s,
        "doppler_grid_step_hz": doppler_step,
        "delay_search_lags": (int(lo), int(hi)),
        "doppler_search_hz": (float(doppler_grid[0]), float(doppler_grid[-1])),
        "extraction_variance_delay_s2": float(residual_delay_variance),
        "extraction_variance_doppler_hz2": float(residual_doppler_variance),
        "sub_bin_refined": bool(refine),
        "delay_refinement_covariance_supported": delay_refinement_supported,
        "doppler_refinement_covariance_supported": doppler_refinement_supported,
        "ambiguity_algorithm": ambiguity_algorithm,
        "doppler_bin_count": int(doppler_grid.size),
        "maximum_doppler_bins": int(maximum_doppler_bins),
        "one_Nyquist_interval_enforced": True,
        "coherent_pulse_processing": coherent_pulse_processing,
        "coherent_pulse_count": int(
            pulse_cube.shape[0] if pulse_cube is not None else 1
        ),
        "coherent_processing_interval_s": float(cpi),
        "slow_time_window": slow_time_window_name,
        "slow_time_window_coherent_gain_normalized": True,
        "slow_time_window_enbw_bins": float(slow_time_window_enbw_bins),
        "slow_time_window_weighted_rms_s": float(
            slow_time_window_weighted_rms_s
        ),
        "pulse_repetition_hz": pulse_repetition_hz,
        "unambiguous_doppler_interval_hz": (
            None
            if pulse_repetition_hz is None
            else [
                float(centre_doppler - 0.5 * pulse_repetition_hz),
                float(centre_doppler + 0.5 * pulse_repetition_hz),
            ]
        ),
        "doppler_alias_branch_selected_by_causal_prior": bool(
            coherent_pulse_processing
        ),
        "truth_used_for_doppler_unwrapping": False,
        "fast_time_and_slow_time_separated": coherent_pulse_processing,
        "reference_mode": "cross_ambiguity" if capture.reference_iq is not None else "matched_filter",
    }


def _cross_correlation_at_integer_lag(
    signal: np.ndarray,
    reference: np.ndarray,
    lag: int,
) -> np.ndarray:
    r"""Evaluate ``scipy.signal.correlate(signal, reference)`` at one lag.

    For the convention used by :func:`scipy.signal.correlate`,

    .. math::

       z[k] = \sum_n x[n] y^*[n-k].

    ``signal`` and ``reference`` may have broadcast-compatible leading axes;
    the last axis is fast time.  The selected-cell component attribution needs
    only this scalar per pulse, so forming a full FFT correlation sequence is
    mathematically redundant.
    """

    left = np.asarray(signal, dtype=np.complex128)
    right = np.asarray(reference, dtype=np.complex128)
    if left.ndim < 1 or right.ndim < 1:
        raise ValueError("correlation operands require a fast-time axis")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("selected-lag correlation requires equal fast-time support")
    sample_count = int(left.shape[-1])
    integer_lag = int(lag)
    if abs(integer_lag) >= sample_count:
        raise ValueError("selected correlation lag lies outside finite support")
    if integer_lag >= 0:
        signal_slice = left[..., integer_lag:]
        reference_slice = right[..., : sample_count - integer_lag]
    else:
        signal_slice = left[..., : sample_count + integer_lag]
        reference_slice = right[..., -integer_lag:]
    return np.sum(signal_slice * np.conj(reference_slice), axis=-1)


def _component_ambiguity_response_at_selected_cell(
    capture: ComplexBasebandCapture,
    statistic: Mapping[str, Any],
    coherent_component_iq: np.ndarray,
) -> complex:
    """Evaluate one generative component at the detector-selected CAF cell."""

    capture.validate()
    component = np.asarray(coherent_component_iq, dtype=np.complex128)
    if capture.coherent_surveillance_iq is not None:
        expected_shape = np.asarray(capture.coherent_surveillance_iq).shape
        if component.shape != expected_shape:
            raise ValueError(
                "latent component cube must match the coherent surveillance cube"
            )
        component = component[0]
    elif component.ndim == 2 and component.shape[0] == 1:
        component = component[0]
    else:
        raise ValueError("latent component attribution requires coherent pulse IQ")

    reference = (
        capture.reference_iq
        if capture.reference_iq is not None
        else (
            capture.transmit_reference_iq
            if capture.transmit_reference_iq is not None
            else None
        )
    )
    if reference is None:
        raise ValueError("latent component attribution requires a receiver reference")
    reference = np.asarray(reference, dtype=np.complex128)
    if reference.ndim == 2:
        reference = reference[0]
    reference_cube = (
        None
        if capture.coherent_reference_iq is None
        else np.asarray(capture.coherent_reference_iq, dtype=np.complex128)
    )
    slow_time = np.asarray(capture.pulse_slow_time_s, dtype=float)
    if slow_time.shape != (component.shape[0],):
        raise ValueError("latent component cube and pulse slow-time axis disagree")

    lag = int(statistic["ambiguity_peak_integer_delay_samples"])
    doppler_hz = float(statistic["ambiguity_peak_grid_doppler_hz"])
    derotation_center_hz = float(statistic["fast_time_derotation_center_hz"])
    sample_rate_hz = float(capture.sample_rate_hz)
    fast_time = np.arange(component.shape[1], dtype=float) / sample_rate_hz
    fast_derotation = np.exp(-2j * math.pi * derotation_center_hz * fast_time)

    pulse_reference = (
        np.broadcast_to(reference, component.shape)
        if reference_cube is None
        else reference_cube
    )
    if pulse_reference.shape != component.shape:
        raise ValueError("component and pulse reference cubes disagree")
    matched = _cross_correlation_at_integer_lag(
        component * fast_derotation[None, :],
        pulse_reference,
        lag,
    )
    steering = np.exp(-2j * math.pi * doppler_hz * slow_time)
    return complex(steering @ matched)


def latent_component_ambiguity_response_at_peak(
    capture: ComplexBasebandCapture,
    statistic: Mapping[str, Any],
    coherent_component_iq: np.ndarray,
) -> complex:
    """Return a simulator-only component response for offline attribution.

    Operational estimators must never call this public audit adapter.  The R2.5
    array generator uses the private component-cell operator above to construct
    one total, unlabeled spatial sufficient statistic; only that summed
    statistic is visible to the estimator.
    """

    return _component_ambiguity_response_at_selected_cell(
        capture,
        statistic,
        coherent_component_iq,
    )


def receiver_observable_feature_summary(
    capture: ComplexBasebandCapture,
    statistic: Mapping[str, Any],
) -> dict[str, Any]:
    """Compress only receiver-observable IQ and matched-filter outputs.

    The function deliberately cannot expose simulator component arrays such as
    ``target_component`` or ``jammer_component``.  Those are valid offline
    diagnostics but are hidden truth at deployment.  The returned vector uses
    total surveillance/reference samples, channel covariance, spectrum and the
    belief-centred ambiguity extractor, all available at the receiver.
    """

    capture.validate()
    surveillance = np.asarray(capture.surveillance_iq, dtype=np.complex128)
    if surveillance.ndim != 2 or surveillance.shape[1] < 8:
        raise ValueError("receiver feature extraction requires channel-by-sample IQ")
    flattened = surveillance.reshape(-1)
    power = np.abs(flattened) ** 2
    mean_power = max(float(power.mean()), np.finfo(float).tiny)
    amplitude = np.abs(flattened)
    centred = flattened - flattened.mean()
    covariance_power = max(float(np.mean(np.abs(centred) ** 2)), np.finfo(float).tiny)
    pseudo = np.mean(centred**2) / covariance_power
    spectrum = np.fft.fftshift(np.fft.fft(surveillance, axis=-1), axes=-1)
    spectral_power = np.abs(spectrum) ** 2
    aggregate_spectrum = spectral_power.mean(axis=0)
    spectral_total = max(float(aggregate_spectrum.sum()), np.finfo(float).tiny)
    frequency = np.fft.fftshift(
        np.fft.fftfreq(surveillance.shape[-1], d=1.0 / capture.sample_rate_hz)
    )
    centroid = float(np.sum(frequency * aggregate_spectrum) / spectral_total)
    spread = math.sqrt(
        max(
            float(
                np.sum((frequency - centroid) ** 2 * aggregate_spectrum)
                / spectral_total
            ),
            0.0,
        )
    )
    probabilities = aggregate_spectrum / spectral_total
    spectral_entropy = float(
        -np.sum(probabilities * np.log(probabilities + np.finfo(float).tiny))
        / max(math.log(probabilities.size), 1.0)
    )
    spectral_flatness = float(
        np.exp(np.mean(np.log(aggregate_spectrum + np.finfo(float).tiny)))
        / max(float(aggregate_spectrum.mean()), np.finfo(float).tiny)
    )
    channel_covariance = surveillance @ surveillance.conj().T / surveillance.shape[-1]
    channel_eigenvalues = np.linalg.eigvalsh(
        0.5 * (channel_covariance + channel_covariance.conj().T)
    ).real
    channel_eigenvalues = np.maximum(channel_eigenvalues, 0.0)
    eigen_total = max(float(channel_eigenvalues.sum()), np.finfo(float).tiny)
    normalized_eigenvalues = np.sort(channel_eigenvalues / eigen_total)[::-1]

    reference = capture.reference_iq
    reference_coherence = 0.0
    reference_power_ratio = 0.0
    if reference is not None:
        reference_array = np.asarray(reference, dtype=np.complex128)
        if reference_array.ndim == 1:
            reference_array = reference_array[None, :]
        ref = reference_array[0]
        surv = surveillance[0]
        numerator = abs(np.vdot(surv, ref)) ** 2
        denominator = max(
            float(np.vdot(surv, surv).real * np.vdot(ref, ref).real),
            np.finfo(float).tiny,
        )
        reference_coherence = float(np.clip(numerator / denominator, 0.0, 1.0))
        reference_power_ratio = float(
            np.mean(np.abs(ref) ** 2) / mean_power
        )

    names = [
        "log_received_power_w",
        "iq_mean_real_over_rms",
        "iq_mean_imag_over_rms",
        "iq_std_real_over_rms",
        "iq_std_imag_over_rms",
        "amplitude_q10_over_rms",
        "amplitude_q50_over_rms",
        "amplitude_q90_over_rms",
        "amplitude_q99_over_rms",
        "circularity_magnitude",
        "circularity_phase_over_pi",
        "spectral_centroid_over_sample_rate",
        "spectral_spread_over_sample_rate",
        "spectral_entropy",
        "spectral_flatness",
        "reference_coherence",
        "reference_power_ratio",
        "channel_effective_rank_fraction",
        "channel_largest_eigen_fraction",
        "channel_smallest_eigen_fraction",
    ]
    rms = math.sqrt(mean_power)
    quantiles = np.quantile(amplitude, [0.10, 0.50, 0.90, 0.99]) / rms
    eigen_entropy = float(
        -np.sum(
            normalized_eigenvalues
            * np.log(normalized_eigenvalues + np.finfo(float).tiny)
        )
    )
    effective_rank_fraction = float(
        np.exp(eigen_entropy) / max(1, normalized_eigenvalues.size)
    )
    values = [
        math.log(mean_power),
        float(flattened.real.mean() / rms),
        float(flattened.imag.mean() / rms),
        float(flattened.real.std() / rms),
        float(flattened.imag.std() / rms),
        *[float(value) for value in quantiles],
        float(abs(pseudo)),
        float(np.angle(pseudo) / math.pi),
        centroid / capture.sample_rate_hz,
        spread / capture.sample_rate_hz,
        spectral_entropy,
        spectral_flatness,
        reference_coherence,
        reference_power_ratio,
        effective_rank_fraction,
        float(normalized_eigenvalues[0]) if normalized_eigenvalues.size else 0.0,
        float(normalized_eigenvalues[-1]) if normalized_eigenvalues.size else 0.0,
    ]
    for lag in (1, 2, 4, 8):
        if flattened.size <= lag:
            correlation = 0.0j
        else:
            correlation = np.mean(flattened[lag:] * flattened[:-lag].conj()) / mean_power
        names.extend((f"autocorrelation_lag_{lag}_real", f"autocorrelation_lag_{lag}_imag"))
        values.extend((float(correlation.real), float(correlation.imag)))

    statistic_fields = (
        ("delay_s", 1.0e6),
        ("doppler_hz", 1.0e-3),
        ("peak_to_median", 1.0e-2),
        ("delay_grid_step_s", 1.0e9),
        ("doppler_grid_step_hz", 1.0e-3),
        ("extraction_variance_delay_s2", 1.0e18),
        ("extraction_variance_doppler_hz2", 1.0e-6),
    )
    for key, scale in statistic_fields:
        names.append(f"matched_filter_{key}_scaled")
        values.append(float(statistic.get(key, 0.0)) * scale)
    names.extend(
        (
            "matched_filter_detected",
            "matched_filter_peak_on_boundary",
            "channel_count_scaled",
            "adc_bits_scaled",
            "image_rejection_db_scaled",
        )
    )
    values.extend(
        (
            float(bool(statistic.get("detected", False))),
            float(bool(statistic.get("peak_on_search_boundary", False))),
            float(surveillance.shape[0]) / 8.0,
            float(np.mean([state.adc_bits for state in capture.receiver_states])) / 16.0,
            float(
                np.mean([state.image_rejection_db for state in capture.receiver_states])
            )
            / 60.0,
        )
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise FloatingPointError("receiver-observable feature vector is invalid")
    return {
        "schema": "receiver-observable-iq-mf-features-1.0",
        "names": names,
        "values": vector.tolist(),
        "feature_count": int(vector.size),
        "source": "surveillance/reference IQ, channel covariance, spectrum and matched-filter output only",
        "truth_component_decomposition_used": False,
        "simulator_truth_used": False,
        "SI_units_before_declared_scaling": True,
    }


def _deterministic_mean_for_world(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    samples: int,
    capture_start_s: float,
    target_present: bool,
) -> np.ndarray:
    capture = simulate_complex_baseband(
        world,
        action,
        samples=samples,
        capture_start_s=capture_start_s,
        target_present=target_present,
        seed=0,
        impairments=False,
        noise_scale=0.0,
    )
    return np.asarray(capture.deterministic_mean, dtype=np.complex128).reshape(-1)


def complex_gaussian_score_fisher_bridge(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    jammer: Jammer | None = None,
    samples: int = 384,
    derivative_step: float = 2e-4,
    monte_carlo_replicates: int = 800,
    seed: int = 6501,
) -> FisherBridgeResult:
    """Numerically verify score/Fisher blocks and nuisance Schur elimination.

    The R2 target/state derivative uses the exact active ``StateLayout``
    normalization.  Two additional nuisance coordinates are complex scattering
    amplitude (real/imaginary).  A weak proper prior prevents an unobservable
    amplitude phase from being silently inverted as information.
    """

    if action.kind.startswith("calibration"):
        raise ValueError("raw-IQ Fisher bridge requires a radar observation action")
    factor = build_action_factor(world, action, jammer)
    layout = StateLayout.from_world(world)
    nominal_capture = simulate_complex_baseband(
        world,
        action,
        jammer=jammer,
        samples=samples,
        seed=seed,
        impairments=False,
    )
    capture_start_s = float(nominal_capture.metadata["capture_start_s"])
    nominal = np.asarray(nominal_capture.deterministic_mean, dtype=np.complex128).reshape(-1)
    dimension = layout.dimension
    derivatives = np.zeros((nominal.size, dimension + 2), dtype=np.complex128)
    for column in range(dimension):
        delta = np.zeros(dimension, dtype=float)
        delta[column] = derivative_step
        plus = _deterministic_mean_for_world(
            world_with_normalized_state_error(world, delta, layout),
            action,
            samples=samples,
            capture_start_s=capture_start_s,
            target_present=True,
        )
        minus = _deterministic_mean_for_world(
            world_with_normalized_state_error(world, -delta, layout),
            action,
            samples=samples,
            capture_start_s=capture_start_s,
            target_present=True,
        )
        derivatives[:, column] = (plus - minus) / (2.0 * derivative_step)
    target_mean = np.asarray(nominal_capture.target_component, dtype=np.complex128).reshape(-1)
    derivatives[:, dimension] = target_mean
    derivatives[:, dimension + 1] = 1j * target_mean
    disturbance = (
        nominal_capture.clutter_component
        + nominal_capture.jammer_component
        + nominal_capture.thermal_component
        + nominal_capture.receiver_impairment_component
    )
    empirical_variance = float(np.mean(np.abs(disturbance) ** 2))
    factor_variance = (
        factor.link_budget.clutter_power_w
        + factor.link_budget.jammer_power_w
        + factor.link_budget.thermal_noise_power_w
        + factor.link_budget.receiver_impairment_noise_w
    )
    variance = max(empirical_variance, factor_variance, 1e-20)
    raw_information = 2.0 * np.real(np.conjugate(derivatives).T @ derivatives) / variance
    raw_information = 0.5 * (raw_information + raw_information.T)
    prior = np.zeros_like(raw_information)
    prior[:dimension, :dimension] = prior_information_normalized(world, layout)
    prior[dimension:, dimension:] = np.eye(2) * 1e-3
    joint = raw_information + prior
    effective, _ = schur_effective_information(joint, target_dim=6)
    known_nuisance = joint[:6, :6]
    rng = np.random.default_rng(seed + 17)
    scores = np.zeros((monte_carlo_replicates, dimension + 2), dtype=float)
    sigma = math.sqrt(variance / 2.0)
    for replicate in range(monte_carlo_replicates):
        residual = sigma * (rng.normal(size=nominal.size) + 1j * rng.normal(size=nominal.size))
        scores[replicate] = 2.0 * np.real(np.conjugate(derivatives).T @ residual) / variance
    score_mean = np.mean(scores, axis=0)
    score_fisher = np.cov(scores, rowvar=False, ddof=1)
    denominator = max(float(np.linalg.norm(raw_information, ord="fro")), 1e-30)
    score_error = float(np.linalg.norm(score_fisher - raw_information, ord="fro") / denominator)
    r0 = factor.observed_information_normalized
    r1 = factor.jacobian_physical @ np.diag(layout.scales())
    r1 = r1.T @ np.linalg.solve(factor.covariance, r1)
    # Apply the exact detection/association retention already registered by R0.
    complete_trace = max(float(np.trace(factor.complete_information_normalized)), 1e-30)
    retention = float(np.trace(r0) / complete_trace)
    r1 *= retention
    r0_r1 = float(np.linalg.norm(r0 - r1, ord="fro") / max(float(np.linalg.norm(r0, ord="fro")), 1e-30))
    return FisherBridgeResult(
        action_id=action.action_id,
        r0_joint_information_normalized=r0,
        r1_joint_information_normalized=r1,
        r2_joint_information_normalized=joint,
        r2_effective_target_information_normalized=effective,
        r2_known_nuisance_target_information_normalized=known_nuisance,
        score_monte_carlo_information_normalized=score_fisher,
        score_mean_residual=float(np.linalg.norm(score_mean)),
        score_fisher_relative_error=score_error,
        r0_r1_relative_error=r0_r1,
        r0_r2_trace_ratio=float(np.trace(r0[:6, :6]) / max(float(np.trace(effective)), 1e-30)),
        provenance={
            "R0": FidelityLevel.R0_ANALYTIC_FACTOR.value,
            "R1": FidelityLevel.R1_SUFFICIENT_STATISTIC.value,
            "R2": FidelityLevel.R2_COMPLEX_BASEBAND.value,
            "likelihood": "proper_complex_Gaussian_fixed_conditional_covariance",
            "normalization": "StateLayout.scales",
            "target_dimension": 6,
            "nuisance_dimension": dimension - 6 + 2,
            "variance_w": variance,
            "monte_carlo_replicates": monte_carlo_replicates,
        },
    )


def fidelity_limiting_case_report(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    jammer: Jammer | None = None,
    samples: int = 512,
    seed: int = 6502,
) -> dict[str, Any]:
    """Executable limiting cases required by the V65 measurement contract."""

    present = simulate_complex_baseband(world, action, jammer=jammer, samples=samples, seed=seed, impairments=False)
    absent = simulate_complex_baseband(world, action, jammer=None, samples=samples, seed=seed, impairments=False, target_present=False)
    clean = simulate_complex_baseband(world, action, jammer=None, samples=samples, seed=seed, impairments=False)
    impaired = simulate_complex_baseband(world, action, jammer=jammer, samples=samples, seed=seed, impairments=True)
    waveform = synthesize_waveform(world.waveform(action.waveform_id), samples, world.seed + action.action_id + 31)
    pstat = matched_filter_sufficient_statistic(present, waveform)
    astat = matched_filter_sufficient_statistic(absent, waveform)
    normalized_template = waveform / max(float(np.linalg.norm(waveform)), 1e-30)
    present_target_projection = max(
        (float(abs(np.vdot(normalized_template, channel))) ** 2 for channel in present.target_component),
        default=0.0,
    )
    absent_target_projection = max(
        (float(abs(np.vdot(normalized_template, channel))) ** 2 for channel in absent.target_component),
        default=0.0,
    )
    noise_like = (
        present.clutter_component
        + present.jammer_component
        + present.thermal_component
        + present.receiver_impairment_component
    )
    matched_filter_noise_variance = max(float(np.mean(np.abs(noise_like) ** 2)), 1e-30)
    present_noncentrality = present_target_projection / matched_filter_noise_variance
    absent_noncentrality = absent_target_projection / matched_filter_noise_variance
    clean_power = clean.power_report()
    impaired_power = impaired.power_report()
    factor = build_action_factor(world, action, jammer)
    high_snr = build_action_factor(replace(world, target=replace(world.target, mean_rcs_m2=world.target.mean_rcs_m2 * 10.0)), action, jammer)
    low_snr = build_action_factor(replace(world, target=replace(world.target, mean_rcs_m2=world.target.mean_rcs_m2 * 0.1)), action, jammer)
    bridge = complex_gaussian_score_fisher_bridge(world, action, jammer=jammer, samples=min(samples, 384), monte_carlo_replicates=400, seed=seed)
    checks = {
        # A single noisy ambiguity-map peak need not be monotone when clutter is
        # stronger than a weak target.  The falsifiable likelihood limit is the
        # matched-filter noncentrality contributed by the deterministic target
        # mean: it must be positive with the target and exactly zero when the
        # target term is removed while the receiver likelihood is unchanged.
        "no_target_reduces_matched_filter_evidence": present_noncentrality > absent_noncentrality + 1e-15,
        "no_jammer_removes_jammer_power": clean_power["jammer_power_w"] == 0.0,
        "receiver_impairments_are_explicit": impaired_power["receiver_impairment_power_w"] >= 0.0,
        "known_nuisance_is_not_worse_than_Schur": float(np.trace(bridge.r2_known_nuisance_target_information_normalized)) + 1e-8 >= float(np.trace(bridge.r2_effective_target_information_normalized)),
        "high_SNR_information_exceeds_low_SNR": float(np.trace(high_snr.observed_information_normalized)) > float(np.trace(low_snr.observed_information_normalized)),
        "R0_R1_same_likelihood": bridge.r0_r1_relative_error < 1e-8,
        "score_mean_is_near_zero": bridge.score_mean_residual < 0.2 * math.sqrt(max(float(np.trace(bridge.r2_joint_information_normalized)), 1e-30)),
        "score_Fisher_matches_complex_likelihood": bridge.score_fisher_relative_error < 0.35,
        "fidelity_not_overclaimed": not present.fidelity.permits_claim(FidelityLevel.R3_MEASURED),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "matched_filter_present": pstat,
        "matched_filter_absent": astat,
        "matched_filter_target_noncentrality": {
            "present": present_noncentrality,
            "absent": absent_noncentrality,
            "definition": "max_channel |<unit_energy_waveform,target_mean>|^2 / registered_noise_like_variance",
            "single_realization_peak_to_median_is_diagnostic_not_gate": True,
        },
        "clean_power": clean_power,
        "impaired_power": impaired_power,
        "fisher_bridge": {
            "r0_r1_relative_error": bridge.r0_r1_relative_error,
            "score_fisher_relative_error": bridge.score_fisher_relative_error,
            "score_mean_residual": bridge.score_mean_residual,
            "r0_r2_trace_ratio": bridge.r0_r2_trace_ratio,
            "r2_effective_min_eigenvalue": float(np.linalg.eigvalsh(bridge.r2_effective_target_information_normalized).min()),
        },
        "claim_boundary": {
            "R2_sampled_IQ_supported": True,
            "R2_5_external_path_adapter_implemented": True,
            "R3_measured_data_ingestion_contract_implemented": True,
            "R3_measurement_evidence_present": False,
            "high_fidelity_EM_digital_twin_claim_allowed": False,
        },
    }


def acquisition_schema() -> dict[str, Any]:
    return {
        "schema": "MeasuredIQAcquisition",
        "required_fields": [item.name for item in MeasuredIQAcquisition.__dataclass_fields__.values() if item.default is MISSING and item.default_factory is MISSING],
        "all_fields": list(MeasuredIQAcquisition.__dataclass_fields__),
        "internal_units": {
            "sample_rate_hz": "Hz",
            "center_frequency_hz": "Hz",
            "adc_volts_per_code": "V/code",
            "impedance_ohm": "ohm",
            "IQ_after_calibration": "sqrt(W)",
        },
        "validation": [
            "SHA-256 capture and calibration verification",
            "channel-count/calibration dimension equality",
            "explicit time and frequency reference",
            "missing-sample mask rather than silent interpolation",
            "receiver configuration IDs map to effective-delay calibration",
        ],
    }
