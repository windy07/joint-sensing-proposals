from __future__ import annotations

"""Typed SI-unit schema for the distributed radar scheduling model.

The records make propagation, receiver, array, timing, resource, and validation
assumptions explicit and serializable. External channel solvers enter through
the same schema without changing the scheduler interface.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

import numpy as np


Vector3 = tuple[float, float, float]
NodeRole = Literal["active", "passive", "hybrid", "illuminator", "calibration"]
CalibrationResourceMode = Literal["disabled", "exclusive_receive_chain", "simultaneous_free"]
TimeReferenceProfile = Literal[
    "legacy_uncorrelated",
    "shared_frequency_and_trigger",
    "disciplined_network",
    "holdover",
    "independent_oscillators",
]
ActionKind = Literal[
    "active_monostatic",
    "active_bistatic",
    "passive_bistatic",
    "tdoa",
    "fdoa",
    "aoa",
    "calibration_time",
    "calibration_frequency",
]
JammerKind = Literal["none", "barrage", "spot", "swept", "drfm_false_target"]
EmitterKind = Literal["communications", "broadcast", "uncooperative_radar", "industrial"]


def _vec3(value: Vector3 | list[float] | np.ndarray, name: str) -> Vector3:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain three finite SI values")
    return tuple(float(x) for x in arr)  # type: ignore[return-value]


@dataclass(frozen=True)
class MaterialSurface:
    relative_permittivity: float = 5.0
    conductivity_s_per_m: float = 0.01
    roughness_rms_m: float = 0.03
    unresolved_diffuse_power_fraction: float = 0.01
    polarization: Literal["horizontal", "vertical"] = "vertical"

    def validate(self) -> None:
        if self.relative_permittivity < 1.0:
            raise ValueError("relative_permittivity must be >= 1")
        if self.conductivity_s_per_m < 0.0 or self.roughness_rms_m < 0.0:
            raise ValueError("surface conductivity and roughness must be nonnegative")
        if not 0.0 <= self.unresolved_diffuse_power_fraction <= 1.0:
            raise ValueError("unresolved diffuse power fraction must be in [0,1]")


@dataclass(frozen=True)
class Atmosphere:
    temperature_k: float = 290.0
    pressure_hpa: float = 1013.25
    water_vapor_density_g_m3: float = 7.5
    specific_attenuation_db_per_km: float = 0.01
    attenuation_source: str = "configured_engineering_value_replaceable_by_ITU_R_P676"

    def validate(self) -> None:
        if self.temperature_k <= 0.0 or self.pressure_hpa <= 0.0:
            raise ValueError("atmospheric temperature and pressure must be positive")
        if self.water_vapor_density_g_m3 < 0.0 or self.specific_attenuation_db_per_km < 0.0:
            raise ValueError("water vapor density and attenuation must be nonnegative")


@dataclass(frozen=True)
class Waveform:
    waveform_id: str
    family: Literal["lfm", "phase_code", "ofdm", "cw"]
    carrier_hz: float
    bandwidth_hz: float
    rms_bandwidth_hz: float
    coherent_processing_interval_s: float
    dwell_s: float
    pulse_repetition_hz: float
    sample_rate_hz: float

    @property
    def coherent_pulse_count(self) -> int:
        return max(1, int(round(self.coherent_processing_interval_s * self.pulse_repetition_hz)))

    @property
    def pulse_width_s(self) -> float:
        """Per-pulse on-air time derived from cumulative coherent dwell."""

        return self.dwell_s / self.coherent_pulse_count

    @property
    def duty_cycle(self) -> float:
        return self.dwell_s / self.coherent_processing_interval_s

    @property
    def matched_filter_processing_gain_linear(self) -> float:
        """Time-bandwidth gain for the coherent on-air observation."""

        return max(1.0, self.bandwidth_hz * self.dwell_s)

    def validate(self) -> None:
        values = [
            self.carrier_hz,
            self.bandwidth_hz,
            self.rms_bandwidth_hz,
            self.coherent_processing_interval_s,
            self.dwell_s,
            self.pulse_repetition_hz,
            self.sample_rate_hz,
        ]
        if any((not np.isfinite(x)) or x <= 0.0 for x in values):
            raise ValueError(f"waveform {self.waveform_id} has nonpositive/nonfinite SI fields")
        if self.rms_bandwidth_hz > self.bandwidth_hz:
            raise ValueError("rms bandwidth cannot exceed occupied bandwidth")
        if self.sample_rate_hz < self.bandwidth_hz:
            raise ValueError("complex-baseband sample rate must cover occupied bandwidth")
        if self.coherent_processing_interval_s < self.dwell_s:
            raise ValueError("coherent processing interval must be at least one dwell")
        if self.duty_cycle > 1.0 + 1e-12 or self.pulse_width_s > 1.0 / self.pulse_repetition_hz + 1e-12:
            raise ValueError("waveform dwell/PRF imply an impossible pulse duty cycle")


@dataclass(frozen=True)
class SensorNode:
    node_id: str
    role: NodeRole
    position_m: Vector3
    velocity_m_s: Vector3 = (0.0, 0.0, 0.0)
    tx_power_w: float = 0.0
    tx_gain_linear: float = 1.0
    rx_gain_linear: float = 1.0
    noise_figure_db: float = 4.0
    array_elements: int = 1
    element_spacing_m: float = 0.0
    element_spacing_wavelengths: float | None = None
    array_rows: int = 0
    array_cols: int = 0
    array_yaw_rad: float = 0.0
    supported_waveform_ids: tuple[str, ...] = ()
    receiver_power_w: float = 5.0
    processing_energy_j_per_complex_sample: float = 2.0e-9
    power_amplifier_efficiency: float = 0.35
    clock_bias_s: float = 0.0
    clock_bias_std_s: float = 20e-9
    clock_drift_s_per_s: float = 0.0
    clock_random_walk_std_s_sqrt_s: float = 0.2e-9
    frequency_bias_hz: float = 0.0
    frequency_bias_std_hz: float = 5.0
    frequency_random_walk_std_hz_sqrt_s: float = 0.1
    adc_bits: int = 14
    adc_full_scale_w: float = 1e-3
    adc_loading_fraction: float = 0.05
    integrated_phase_noise_rad2: float = 1e-4
    timing_jitter_s: float = 1e-12
    calibration_path_bias_s: float = 0.0
    calibration_path_truth_std_s: float = 0.0
    image_rejection_db: float = 40.0
    communication_capacity_mbps: float = 100.0
    latency_s: float = 0.005
    max_energy_j: float = 10.0
    full_duplex_capable: bool = False
    simultaneous_receive_channels: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_m_s", _vec3(self.velocity_m_s, "velocity_m_s"))

    @property
    def resolved_array_shape(self) -> tuple[int, int]:
        """Return the declared rectangular array shape.

        Legacy records without an explicit shape are interpreted as a ULA. New
        generated worlds declare a two-dimensional array for azimuth/elevation
        actions.
        """

        if self.array_rows > 0 or self.array_cols > 0:
            if self.array_rows < 1 or self.array_cols < 1:
                raise ValueError(f"node {self.node_id} must declare both array_rows and array_cols")
            return int(self.array_rows), int(self.array_cols)
        return 1, int(self.array_elements)

    def supports_waveform(self, waveform_id: str) -> bool:
        return not self.supported_waveform_ids or str(waveform_id) in self.supported_waveform_ids

    def validate(self) -> None:
        if self.tx_power_w < 0.0 or self.tx_gain_linear <= 0.0 or self.rx_gain_linear <= 0.0:
            raise ValueError(f"node {self.node_id} has invalid RF power/gain")
        rows, cols = self.resolved_array_shape
        if self.array_elements < 1 or self.element_spacing_m < 0.0 or rows * cols != self.array_elements:
            raise ValueError(f"node {self.node_id} has inconsistent array geometry")
        if self.element_spacing_wavelengths is not None:
            if (
                not math.isfinite(self.element_spacing_wavelengths)
                or self.element_spacing_wavelengths <= 0.0
                or self.element_spacing_m != 0.0
            ):
                raise ValueError(
                    f"node {self.node_id} must choose either metric or wavelength spacing"
                )
        elif self.array_elements > 1 and self.element_spacing_m <= 0.0:
            raise ValueError(
                f"node {self.node_id} has no spacing law for its multi-element array"
            )
        if not math.isfinite(self.array_yaw_rad):
            raise ValueError(f"node {self.node_id} has invalid array orientation")
        if self.receiver_power_w <= 0.0 or self.processing_energy_j_per_complex_sample < 0.0:
            raise ValueError(f"node {self.node_id} has invalid receiver-energy model")
        if not 0.0 < self.power_amplifier_efficiency <= 1.0:
            raise ValueError(f"node {self.node_id} has invalid power-amplifier efficiency")
        if self.clock_bias_std_s <= 0.0 or self.frequency_bias_std_hz <= 0.0:
            raise ValueError(f"node {self.node_id} has invalid oscillator prior")
        if self.clock_random_walk_std_s_sqrt_s < 0.0 or self.frequency_random_walk_std_hz_sqrt_s < 0.0:
            raise ValueError(f"node {self.node_id} has invalid oscillator random-walk intensity")
        if self.adc_bits < 2 or self.adc_full_scale_w <= 0.0 or not 0.0 < self.adc_loading_fraction <= 1.0 or self.integrated_phase_noise_rad2 < 0.0 or self.timing_jitter_s < 0.0 or self.calibration_path_truth_std_s < 0.0 or self.image_rejection_db <= 0.0:
            raise ValueError(f"node {self.node_id} has invalid receiver impairment settings")
        if self.communication_capacity_mbps <= 0.0 or self.latency_s < 0.0 or self.max_energy_j <= 0.0 or self.simultaneous_receive_channels < 1:
            raise ValueError(f"node {self.node_id} has invalid network/resource limits")


@dataclass(frozen=True)
class TargetState:
    target_id: str
    position_m: Vector3
    velocity_m_s: Vector3
    mean_rcs_m2: float
    swerling_case: Literal["0", "I", "II", "III", "IV"] = "I"
    acceleration_std_m_s2: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_m_s", _vec3(self.velocity_m_s, "velocity_m_s"))

    def validate(self) -> None:
        if self.mean_rcs_m2 <= 0.0 or self.acceleration_std_m_s2 < 0.0:
            raise ValueError(f"target {self.target_id} has invalid RCS/dynamics")


@dataclass(frozen=True)
class Jammer:
    jammer_id: str
    kind: JammerKind
    position_m: Vector3
    tx_power_w: float = 0.0
    antenna_gain_linear: float = 1.0
    center_frequency_hz: float = 0.0
    bandwidth_hz: float = 1.0
    duty_cycle: float = 1.0
    false_target_delay_s: float = 0.0
    false_target_doppler_hz: float = 0.0
    deception_count: int = 0
    instantaneous_bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))

    def validate(self) -> None:
        if self.kind == "none":
            return
        if self.tx_power_w <= 0.0 or self.antenna_gain_linear <= 0.0:
            raise ValueError(f"jammer {self.jammer_id} must have positive power and gain")
        if self.center_frequency_hz <= 0.0 or self.bandwidth_hz <= 0.0:
            raise ValueError(f"jammer {self.jammer_id} has invalid spectrum")
        if not 0.0 <= self.duty_cycle <= 1.0 or self.deception_count < 0:
            raise ValueError(f"jammer {self.jammer_id} has invalid duty/deception settings")
        if self.instantaneous_bandwidth_hz is not None:
            if self.instantaneous_bandwidth_hz <= 0.0:
                raise ValueError(f"jammer {self.jammer_id} has invalid instantaneous bandwidth")
            if self.kind == "swept" and self.instantaneous_bandwidth_hz > self.bandwidth_hz:
                raise ValueError(f"swept jammer {self.jammer_id} instantaneous bandwidth exceeds sweep span")


@dataclass(frozen=True)
class SpectrumEmitter:
    emitter_id: str
    kind: EmitterKind
    position_m: Vector3
    tx_power_w: float
    antenna_gain_linear: float
    center_frequency_hz: float
    bandwidth_hz: float
    duty_cycle: float
    modulation_similarity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))

    def validate(self) -> None:
        if not self.emitter_id or self.tx_power_w <= 0.0 or self.antenna_gain_linear <= 0.0:
            raise ValueError("spectrum emitter identity, power, and gain must be valid")
        if self.center_frequency_hz <= 0.0 or self.bandwidth_hz <= 0.0:
            raise ValueError(f"spectrum emitter {self.emitter_id} has invalid spectrum")
        if not 0.0 <= self.duty_cycle <= 1.0 or not 0.0 <= self.modulation_similarity <= 1.0:
            raise ValueError(f"spectrum emitter {self.emitter_id} has invalid duty or modulation similarity")


@dataclass(frozen=True)
class ReceiveChainConfiguration:
    """Configuration-dependent effective receive delay for one physical site.

    The state is deliberately not called a clock offset.  It includes analog
    group delay, ADC/RFDC/DDC and timestamp-pipeline delay for a declared RF
    configuration.  Configurations in the same transfer group share a prior
    component; their residual component is not transferable without evidence.
    """

    configuration_id: str
    node_id: str
    waveform_ids: tuple[str, ...]
    transfer_group_id: str
    rf_chain_id: str
    rfdc_config_id: str
    ddc_config_id: str
    bitstream_id: str
    effective_delay_s: float
    effective_delay_drift_s_per_s: float
    common_delay_std_s: float
    configuration_residual_std_s: float
    delay_random_walk_std_s_sqrt_s: float
    calibration_age_s: float
    temperature_k: float
    temperature_coefficient_s_per_k: float

    @property
    def prior_std_s(self) -> float:
        return math.sqrt(
            self.common_delay_std_s**2
            + self.configuration_residual_std_s**2
            + self.delay_random_walk_std_s_sqrt_s**2 * self.calibration_age_s
        )

    @property
    def transfer_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.transfer_group_id,
            self.rf_chain_id,
            self.rfdc_config_id,
            self.ddc_config_id,
            self.bitstream_id,
        )

    def validate(self, node_ids: set[str], waveform_ids: set[str]) -> None:
        if not self.configuration_id or self.node_id not in node_ids:
            raise ValueError(f"receive configuration {self.configuration_id!r} has an invalid node")
        if not self.waveform_ids or set(self.waveform_ids).difference(waveform_ids):
            raise ValueError(f"receive configuration {self.configuration_id!r} has invalid waveforms")
        if not all((self.transfer_group_id, self.rf_chain_id, self.rfdc_config_id, self.ddc_config_id, self.bitstream_id)):
            raise ValueError(f"receive configuration {self.configuration_id!r} has an incomplete transfer key")
        values = (
            self.effective_delay_s,
            self.effective_delay_drift_s_per_s,
            self.common_delay_std_s,
            self.configuration_residual_std_s,
            self.delay_random_walk_std_s_sqrt_s,
            self.calibration_age_s,
            self.temperature_k,
            self.temperature_coefficient_s_per_k,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"receive configuration {self.configuration_id!r} has non-finite fields")
        if min(
            self.common_delay_std_s,
            self.configuration_residual_std_s,
            self.delay_random_walk_std_s_sqrt_s,
            self.calibration_age_s,
        ) < 0.0 or self.temperature_k <= 0.0:
            raise ValueError(f"receive configuration {self.configuration_id!r} has invalid uncertainty or temperature")


@dataclass(frozen=True)
class RadarAction:
    action_id: int
    kind: ActionKind
    waveform_id: str
    tx_node_id: str | None
    rx_node_ids: tuple[str, ...]
    reference_node_id: str | None = None
    resource_slot: int = 0
    energy_cost_j: float = 0.0
    communication_cost_mb: float = 0.0
    emission_exposure_j: float = 0.0
    allowed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, node_ids: set[str], waveform_ids: set[str]) -> None:
        if self.waveform_id not in waveform_ids:
            raise ValueError(f"action {self.action_id} references unknown waveform")
        referenced = [x for x in [self.tx_node_id, self.reference_node_id, *self.rx_node_ids] if x]
        missing = sorted(set(referenced).difference(node_ids))
        if missing:
            raise ValueError(f"action {self.action_id} references unknown nodes: {missing}")
        if not self.rx_node_ids and not self.kind.startswith("calibration"):
            raise ValueError(f"action {self.action_id} needs at least one receiver")
        if self.kind in {"tdoa", "fdoa"} and len(self.rx_node_ids) != 2:
            raise ValueError(
                f"{self.kind.upper()} action {self.action_id} requires exactly "
                "two receivers because its likelihood and Stein cross-channel "
                "SNR are pairwise"
            )
        if self.resource_slot < 0 or min(self.energy_cost_j, self.communication_cost_mb, self.emission_exposure_j) < 0.0:
            raise ValueError(f"action {self.action_id} has invalid resource costs")


@dataclass(frozen=True)
class PhysicalWorld:
    world_id: str
    seed: int
    family: str
    nodes: tuple[SensorNode, ...]
    waveforms: tuple[Waveform, ...]
    target: TargetState
    actions: tuple[RadarAction, ...]
    jammers: tuple[Jammer, ...]
    surface: MaterialSurface
    atmosphere: Atmosphere
    max_schedule_size: int
    max_total_dwell_s: float
    max_total_energy_j: float
    max_concurrent_active: int
    max_total_coherent_processing_s: float = 1.0
    max_total_emission_j: float = 20.0
    max_total_communication_mb: float = 8.0
    max_decision_latency_s: float = 0.75
    ground_height_m: float = 0.0
    spectrum_emitters: tuple[SpectrumEmitter, ...] = ()
    receive_chain_configurations: tuple[ReceiveChainConfiguration, ...] = ()
    effective_delay_profile: str = "disabled"
    calibration_resource_mode: CalibrationResourceMode = "disabled"
    time_reference_profile: TimeReferenceProfile = "legacy_uncorrelated"
    clock_common_mode_std_s: float = 0.0
    frequency_common_mode_std_hz: float = 0.0
    clock_common_random_walk_std_s_sqrt_s: float = 0.0
    frequency_common_random_walk_std_hz_sqrt_s: float = 0.0
    simulator_fidelity: str = "R2_5_band_averaged_analytic_link_and_reference_channel_model"
    external_channel_provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if (
            not self.world_id
            or self.max_schedule_size <= 0
            or self.max_total_dwell_s <= 0.0
            or self.max_total_energy_j <= 0.0
            or self.max_total_coherent_processing_s <= 0.0
            or self.max_total_emission_j <= 0.0
            or self.max_total_communication_mb <= 0.0
            or self.max_decision_latency_s <= 0.0
            or not math.isfinite(self.ground_height_m)
        ):
            raise ValueError("world identifiers and schedule/resource limits must be valid")
        if self.max_concurrent_active <= 0:
            raise ValueError("max_concurrent_active must be positive")
        self.surface.validate()
        self.atmosphere.validate()
        self.target.validate()
        node_ids = {n.node_id for n in self.nodes}
        waveform_ids = {w.waveform_id for w in self.waveforms}
        if len(node_ids) != len(self.nodes) or len(waveform_ids) != len(self.waveforms):
            raise ValueError("node and waveform identifiers must be unique")
        for node in self.nodes:
            node.validate()
        for waveform in self.waveforms:
            waveform.validate()
        action_ids = {a.action_id for a in self.actions}
        if len(action_ids) != len(self.actions):
            raise ValueError("action identifiers must be unique")
        for action in self.actions:
            action.validate(node_ids, waveform_ids)
        for jammer in self.jammers:
            jammer.validate()
        for emitter in self.spectrum_emitters:
            emitter.validate()
        configuration_ids = {item.configuration_id for item in self.receive_chain_configurations}
        if len(configuration_ids) != len(self.receive_chain_configurations):
            raise ValueError("receive-chain configuration identifiers must be unique")
        configuration_pairs: set[tuple[str, str]] = set()
        for configuration in self.receive_chain_configurations:
            configuration.validate(node_ids, waveform_ids)
            for waveform_id in configuration.waveform_ids:
                pair = (configuration.node_id, waveform_id)
                if pair in configuration_pairs:
                    raise ValueError(f"multiple receive-chain configurations map to {pair}")
                configuration_pairs.add(pair)
        if self.effective_delay_profile == "disabled":
            if self.receive_chain_configurations or self.calibration_resource_mode != "disabled":
                raise ValueError("disabled effective-delay profile cannot carry receive-chain states")
        elif not self.receive_chain_configurations or self.calibration_resource_mode == "disabled":
            raise ValueError("enabled effective-delay profile requires configurations and a resource mode")
        common_reference_parameters = (
            self.clock_common_mode_std_s,
            self.frequency_common_mode_std_hz,
            self.clock_common_random_walk_std_s_sqrt_s,
            self.frequency_common_random_walk_std_hz_sqrt_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in common_reference_parameters):
            raise ValueError("time-reference common-mode uncertainties must be finite and nonnegative")
        if self.time_reference_profile == "legacy_uncorrelated" and any(value != 0.0 for value in common_reference_parameters):
            raise ValueError("legacy uncorrelated time reference cannot carry common-mode covariance")

    def node(self, node_id: str) -> SensorNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def waveform(self, waveform_id: str) -> Waveform:
        for waveform in self.waveforms:
            if waveform.waveform_id == waveform_id:
                return waveform
        raise KeyError(waveform_id)

    def receive_configuration(self, node_id: str, waveform_id: str) -> ReceiveChainConfiguration:
        for configuration in self.receive_chain_configurations:
            if configuration.node_id == node_id and waveform_id in configuration.waveform_ids:
                return configuration
        raise KeyError((node_id, waveform_id))

    @property
    def effective_delay_model_enabled(self) -> bool:
        return self.effective_delay_profile != "disabled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
