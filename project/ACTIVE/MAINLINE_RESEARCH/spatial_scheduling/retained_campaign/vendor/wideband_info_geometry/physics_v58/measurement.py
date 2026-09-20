from __future__ import annotations

"""Radar likelihood, CRLB, detection, and association information model."""

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Any, Iterable, Mapping

import numpy as np

from wideband_info_geometry.small_symmetric_linalg import (
    platform_cholesky,
    platform_matmul,
    platform_norm,
    platform_solve,
    platform_singular_values,
    platform_symmetric_eigh,
)

from .propagation import (
    C0_M_S,
    bistatic_received_power_w,
    distance_m,
    band_averaged_two_ray_power_gain,
    flat_spectrum_intercept_fraction,
    monostatic_received_power_w,
    one_way_interference_power_w,
    spectral_overlap_fraction,
    swept_spectrum_average_intercept_fraction,
    thermal_noise_power_w,
    unit_vector,
    wavelength_m,
)
from .schema import Jammer, PhysicalWorld, RadarAction, SensorNode, TargetState, Waveform
from .antenna_geometry import (
    endpoint_beam_parameters,
    resolved_element_spacing_m,
)
from .effective_delay import automatic_free_calibration_actions, configuration_for


# ``numpy.trapz`` was removed in NumPy 2.0 (NEP 52).  ``numpy.trapezoid`` is the
# NumPy >= 2.0 spelling and ``numpy.trapz`` the NumPy < 2.0 spelling; resolve
# once at import so no scientific quadrature path can raise AttributeError at
# run time on a supported NumPy.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # pragma: no cover - NumPy < 2.0 only
    _trapezoid = np.trapz


# Doppler sign convention (SI, IEEE Std 686 / Skolnik, *Radar Handbook* 3rd ed.
# eq. 1.7; Willis, *Bistatic Radar* 2nd ed. eq. 4.11):
#
#     f_d = -(f_c / c) * d/dt (R_tx + R_rx)
#         = -(f_c / c) * ( u_tx . (v_t - v_tx) + u_rx . (v_t - v_rx) )
#
# with ``u_*`` the unit vector pointing FROM the node TOWARD the target.  A
# CLOSING target shortens the total path (dR/dt < 0) and therefore produces a
# POSITIVE Doppler frequency.  ``_radial_rate_derivative`` returns dR/dt, so
# every Doppler row and prediction multiplies it by this constant.  The
# waveform-tier model in ``real_world_measurement._bistatic_delay_doppler``
# already used this convention; the analytic tier previously used the opposite
# one, which made the two fidelity tiers disagree in sign.
DOPPLER_RANGE_RATE_SIGN = -1.0


@dataclass(frozen=True)
class StateLayout:
    node_ids: tuple[str, ...]
    receive_configuration_ids: tuple[str, ...] = ()
    target_dim: int = 6

    @classmethod
    def from_world(cls, world: PhysicalWorld) -> "StateLayout":
        return cls(
            tuple(sorted(n.node_id for n in world.nodes)),
            tuple(sorted(item.configuration_id for item in world.receive_chain_configurations)),
        )

    @property
    def clock_offset(self) -> int:
        return self.target_dim

    @property
    def frequency_offset(self) -> int:
        return self.target_dim + len(self.node_ids)

    @property
    def effective_delay_offset(self) -> int:
        return self.frequency_offset + len(self.node_ids)

    @property
    def calibration_path_offset(self) -> int:
        return self.effective_delay_offset + len(self.receive_configuration_ids)

    @property
    def dimension(self) -> int:
        path_dimension = len(self.node_ids) if self.receive_configuration_ids else 0
        return self.target_dim + 2 * len(self.node_ids) + len(self.receive_configuration_ids) + path_dimension

    def clock_index(self, node_id: str) -> int:
        return self.clock_offset + self.node_ids.index(node_id)

    def frequency_index(self, node_id: str) -> int:
        return self.frequency_offset + self.node_ids.index(node_id)

    def effective_delay_index(self, configuration_id: str) -> int:
        return self.effective_delay_offset + self.receive_configuration_ids.index(configuration_id)

    def calibration_path_index(self, node_id: str) -> int:
        if not self.receive_configuration_ids:
            raise KeyError("calibration-path state is disabled")
        return self.calibration_path_offset + self.node_ids.index(node_id)

    def scales(self) -> np.ndarray:
        # Normalized coordinates keep Schur solves conditioned while retaining
        # an exact mapping back to SI covariance.
        values = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]
        values += [100e-9] * len(self.node_ids)
        values += [10.0] * len(self.node_ids)
        values += [20e-9] * len(self.receive_configuration_ids)
        if self.receive_configuration_ids:
            values += [20e-9] * len(self.node_ids)
        return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class LinkBudget:
    signal_power_w: float
    thermal_noise_power_w: float
    receiver_impairment_noise_w: float
    clutter_power_w: float
    jammer_power_w: float
    sinr_linear: float
    detection_probability: float
    correct_association_probability: float
    false_target_separation_sigma: float
    rcs_realization_m2: float
    front_end_sinr_linear: float = 0.0
    processing_gain_linear: float = 1.0
    effective_clutter_power_w: float = 0.0
    effective_noise_like_power_w: float = 0.0
    clutter_doppler_leakage: float = 1.0
    ambient_interference_power_w: float = 0.0
    spectrum_occupancy_fraction: float = 0.0
    source_confusion_odds: float = 0.0
    expected_false_returns: float = 0.0
    association_search_cells: int = 1
    reference_sinr_linear: float = math.inf
    reference_signal_purity: float = 1.0
    clutter_patch_area_m2: float = 0.0
    clutter_patch_grazing_rad: float = 0.0
    clutter_patch_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    capture_gate_coverage_probability: float = 1.0
    cfar_effective_independent_cells: int = 1
    cfar_single_cell_false_alarm_probability: float = 1e-6
    planning_target_doppler_hz: float = 0.0
    planning_clutter_notch_half_width_hz: float = 0.0
    expected_target_in_clutter_notch: bool = False
    reference_signal_power_w: float = 0.0
    reference_noise_plus_interference_power_w: float = 0.0
    clutter_fast_time_correlation: float = 0.0
    clutter_color_variance_inflation_upper_bound: float = 1.0
    clutter_fast_time_processing_gain_lower_bound_linear: float = 1.0
    ambient_fast_time_correlation: float = 0.0
    ambient_color_variance_inflation_upper_bound: float = 1.0
    ambient_fast_time_processing_gain_lower_bound_linear: float = 1.0
    effective_ambient_interference_power_w: float = 0.0
    surveillance_output_sinr_linear: float = 0.0
    reference_processing_gain_linear: float = 1.0
    reference_output_sinr_linear: float = math.inf
    reference_incoherent_disturbance_power_w: float = 0.0
    reference_coherent_interference_power_w: float = 0.0
    cross_ambiguity_sinr_model: str = "not_applicable"


@dataclass(frozen=True)
class ClutterEstimate:
    power_w: float
    rcs_m2: float
    area_m2: float
    grazing_rad: float
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class ActionFactor:
    action_id: int
    measurement_names: tuple[str, ...]
    prediction: np.ndarray
    jacobian_physical: np.ndarray
    covariance: np.ndarray
    complete_information_normalized: np.ndarray
    observed_information_normalized: np.ndarray
    missing_information_normalized: np.ndarray
    link_budget: LinkBudget
    metadata: dict[str, Any]
    #: Fisher information *conditional on a declared detection*, i.e. the
    #: expectation of the observed information given ``D = 1``.  It retains the
    #: association-mixture kernel (association ambiguity is still present in a
    #: realized detection) but carries no factor of ``P_D``:
    #:
    #:     J_obs      = P_D * H^T R^{-1/2 T} K R^{-1/2} H  +  J_cal
    #:     J_cond|D=1 =       H^T R^{-1/2 T} K R^{-1/2} H  +  J_cal
    #:
    #: It is built directly rather than by dividing ``J_obs`` by ``P_D``: that
    #: deconvolution is catastrophically ill-conditioned for small ``P_D`` and
    #: is undefined at ``P_D = 0``.
    conditional_detected_information_normalized: np.ndarray | None = None


@dataclass(frozen=True)
class RadarPhaseGeometry:
    """Local Fourier-integral geometry of one target-dependent action.

    Range-bearing actions use ``phi=2*pi*f*tau(x,t)``.  AOA uses the physical
    array-manifold phase ``phi=2*pi*a^T u(x)/lambda``.  Calibration actions
    remain inapplicable rather than being assigned fabricated geometry.
    """

    applicable: bool
    path_count: int
    delay_s: float
    delay_gradient_s_m: np.ndarray
    delay_hessian_s_m2: np.ndarray
    delay_gradient_rate_s_m_s: np.ndarray
    phase_gradient_rad_m: np.ndarray
    phase_hessian_rad_m2: np.ndarray
    aperture_phase_gradient_change_rad_m: np.ndarray
    mixed_phase_rank: int
    mixed_phase_transversality: float
    curvature_rank: int
    curvature_condition_number: float
    normal_evolution_rad: float
    range_normal_resolution_m: float
    aperture_cross_range_resolution_m: float
    phase_nondegenerate: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fio_phase_applicable": self.applicable,
            "fio_path_count": self.path_count,
            "fio_delay_s": self.delay_s,
            "fio_delay_gradient_s_m": self.delay_gradient_s_m.tolist(),
            "fio_delay_hessian_s_m2": self.delay_hessian_s_m2.tolist(),
            "fio_delay_gradient_rate_s_m_s": self.delay_gradient_rate_s_m_s.tolist(),
            "fio_phase_gradient_rad_m": self.phase_gradient_rad_m.tolist(),
            "fio_phase_hessian_rad_m2": self.phase_hessian_rad_m2.tolist(),
            "fio_aperture_phase_gradient_change_rad_m": (
                self.aperture_phase_gradient_change_rad_m.tolist()
            ),
            "fio_mixed_phase_rank": self.mixed_phase_rank,
            "fio_mixed_phase_transversality": self.mixed_phase_transversality,
            "fio_curvature_rank": self.curvature_rank,
            "fio_curvature_condition_number": self.curvature_condition_number,
            "fio_normal_evolution_rad": self.normal_evolution_rad,
            "fio_range_normal_resolution_m": self.range_normal_resolution_m,
            "fio_aperture_cross_range_resolution_m": (
                self.aperture_cross_range_resolution_m
            ),
            "fio_phase_nondegenerate": self.phase_nondegenerate,
            "fio_model": (
                "local_multistatic_delay_oscillatory_integral"
                if math.isfinite(self.range_normal_resolution_m)
                else "local_array_manifold_oscillatory_integral"
            ),
        }


def _range_phase_terms(
    target_position_m: np.ndarray,
    target_velocity_m_s: np.ndarray,
    node: SensorNode,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    displacement = target_position_m - np.asarray(node.position_m, dtype=float)
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-9:
        raise ValueError("target and phase-path node cannot be colocated")
    direction = displacement / distance
    projector = np.eye(3) - np.outer(direction, direction)
    hessian = projector / distance
    relative_velocity = target_velocity_m_s - np.asarray(node.velocity_m_s, dtype=float)
    direction_rate = hessian @ relative_velocity
    return distance, direction, hessian, direction_rate


def _target_delay_path_legs(
    world: PhysicalWorld,
    action: RadarAction,
) -> tuple[tuple[SensorNode, float], ...]:
    if action.kind == "active_monostatic":
        return ((_node(world, action.tx_node_id), 2.0),)
    if action.kind in {"active_bistatic", "passive_bistatic"}:
        if not action.rx_node_ids:
            return ()
        return (
            (_node(world, action.tx_node_id), 1.0),
            (_node(world, action.rx_node_ids[0]), 1.0),
        )
    if action.kind in {"tdoa", "fdoa"} and len(action.rx_node_ids) >= 2:
        return (
            (_node(world, action.rx_node_ids[0]), 1.0),
            (_node(world, action.rx_node_ids[1]), -1.0),
        )
    return ()


def _aoa_phase_geometry(
    world: PhysicalWorld,
    action: RadarAction,
) -> RadarPhaseGeometry:
    """Array-manifold FIO geometry without inventing range resolution."""

    if not action.rx_node_ids:
        raise ValueError("AOA action requires a receiver")
    node = _node(world, action.rx_node_ids[0])
    target_position = np.asarray(world.target.position_m, dtype=float)
    target_velocity = np.asarray(world.target.velocity_m_s, dtype=float)
    node_position = np.asarray(node.position_m, dtype=float)
    node_velocity = np.asarray(node.velocity_m_s, dtype=float)
    displacement = target_position - node_position
    distance = max(platform_norm(displacement), 1e-12)
    look = displacement / distance
    rows, cols = node.resolved_array_shape
    yaw = float(node.array_yaw_rad)
    horizontal = np.asarray(
        [math.cos(yaw), math.sin(yaw), 0.0],
        dtype=float,
    )
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=float)
    spacing = resolved_element_spacing_m(node, world.waveform(action.waveform_id).carrier_hz)
    aperture_axes: list[np.ndarray] = []
    if cols > 1 and spacing > 0.0:
        aperture_axes.append(horizontal)
    if rows > 1 and spacing > 0.0:
        aperture_axes.append(vertical)
    if not aperture_axes:
        aperture_axes.append(horizontal)
    waveform = world.waveform(action.waveform_id)
    wave_number = (
        2.0 * math.pi * waveform.carrier_hz / C0_M_S
    )
    projector = np.eye(3) - np.outer(look, look)
    mixed_columns = np.column_stack(
        [
            wave_number * platform_matmul(projector, axis) / distance
            for axis in aperture_axes
        ]
    )
    mixed_singular_values = platform_singular_values(mixed_columns)
    mixed_scale = max(float(mixed_singular_values[0]), 1e-15)
    mixed_rank = int(
        np.count_nonzero(
            mixed_singular_values > mixed_scale * 1e-8
        )
    )
    column_norms = np.asarray(
        [platform_norm(mixed_columns[:, index]) for index in range(mixed_columns.shape[1])],
        dtype=float,
    )
    normalized_columns = mixed_columns / np.maximum(column_norms[None, :], 1e-15)
    normalized_gram = platform_matmul(
        normalized_columns.T, normalized_columns
    )
    normalized_determinant = float(
        np.prod(platform_symmetric_eigh(normalized_gram)[0])
    )
    mixed_transversality = (
        float(
            np.clip(
                math.sqrt(
                    max(
                        float(
                            normalized_determinant
                        ),
                        0.0,
                    )
                ),
                0.0,
                1.0,
            )
        )
        if normalized_columns.shape[1] == 2
        else 0.0
    )
    principal_axis = aperture_axes[
        int(np.argmax(column_norms))
    ]

    def phase_terms(
        relative_position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_distance = max(
            platform_norm(relative_position),
            1e-12,
        )
        local_look = relative_position / local_distance
        axis_projection = float(platform_matmul(principal_axis, local_look))
        gradient = (
            wave_number
            * (principal_axis - axis_projection * local_look)
            / local_distance
        )
        hessian = (
            wave_number
            * (
                -np.outer(principal_axis, local_look)
                - np.outer(local_look, principal_axis)
                - axis_projection * np.eye(3)
                + 3.0
                * axis_projection
                * np.outer(local_look, local_look)
            )
            / local_distance**2
        )
        return gradient, 0.5 * (hessian + hessian.T)

    phase_gradient, phase_hessian = phase_terms(displacement)
    relative_velocity = target_velocity - node_velocity
    half_displacement = (
        0.5
        * waveform.coherent_processing_interval_s
        * relative_velocity
    )
    plus_gradient, _ = phase_terms(displacement + half_displacement)
    minus_gradient, _ = phase_terms(displacement - half_displacement)
    aperture_change = plus_gradient - minus_gradient
    curvature_eigenvalues = platform_symmetric_eigh(phase_hessian)[0]
    curvature_scale = max(
        float(np.max(np.abs(curvature_eigenvalues))),
        1e-15,
    )
    nonzero_curvature = (
        np.abs(curvature_eigenvalues)
        > curvature_scale * 1e-8
    )
    curvature_rank = int(np.count_nonzero(nonzero_curvature))
    retained_curvature = np.abs(
        curvature_eigenvalues[nonzero_curvature]
    )
    curvature_condition = (
        float(
            np.max(retained_curvature)
            / np.min(retained_curvature)
        )
        if retained_curvature.size
        else math.inf
    )
    gradient_norm = platform_norm(phase_gradient)
    normal = phase_gradient / max(gradient_norm, 1e-15)
    transverse_change = aperture_change - normal * float(
        platform_matmul(normal, aperture_change)
    )
    aperture_extent = spacing * max(
        rows - 1,
        cols - 1,
        1,
    )
    wavelength = C0_M_S / waveform.carrier_hz
    cross_range_resolution = (
        distance
        * wavelength
        / max(aperture_extent, wavelength)
    )
    zero_vector = np.zeros(3, dtype=float)
    zero_matrix = np.zeros((3, 3), dtype=float)
    return RadarPhaseGeometry(
        True,
        1,
        0.0,
        zero_vector,
        zero_matrix,
        zero_vector,
        phase_gradient,
        phase_hessian,
        aperture_change,
        mixed_rank,
        mixed_transversality,
        curvature_rank,
        curvature_condition,
        float(
            platform_norm(transverse_change)
            / max(gradient_norm, 1e-15)
        ),
        math.inf,
        float(cross_range_resolution),
        bool(
            mixed_rank >= 1
            and curvature_rank >= 1
            and np.all(np.isfinite(phase_gradient))
            and np.all(np.isfinite(phase_hessian))
        ),
    )


def radar_phase_geometry(
    world: PhysicalWorld,
    action: RadarAction,
) -> RadarPhaseGeometry:
    """Return the local phase/canonical-relation geometry for an action."""

    if action.kind == "aoa":
        return _aoa_phase_geometry(world, action)
    legs = _target_delay_path_legs(world, action)
    zero_vector = np.zeros(3, dtype=float)
    zero_matrix = np.zeros((3, 3), dtype=float)
    if not legs:
        return RadarPhaseGeometry(
            False,
            0,
            0.0,
            zero_vector,
            zero_matrix,
            zero_vector,
            zero_vector,
            zero_matrix,
            zero_vector,
            0,
            0.0,
            0,
            math.inf,
            0.0,
            math.inf,
            math.inf,
            False,
        )

    target_position = np.asarray(world.target.position_m, dtype=float)
    target_velocity = np.asarray(world.target.velocity_m_s, dtype=float)
    delay = 0.0
    delay_gradient = np.zeros(3, dtype=float)
    delay_hessian = np.zeros((3, 3), dtype=float)
    delay_gradient_rate = np.zeros(3, dtype=float)
    for node, coefficient in legs:
        distance, gradient, hessian, gradient_rate = _range_phase_terms(
            target_position,
            target_velocity,
            node,
        )
        delay += coefficient * distance / C0_M_S
        delay_gradient += coefficient * gradient / C0_M_S
        delay_hessian += coefficient * hessian / C0_M_S
        delay_gradient_rate += coefficient * gradient_rate / C0_M_S

    waveform = world.waveform(action.waveform_id)
    phase_scale = 2.0 * math.pi * float(waveform.carrier_hz)
    phase_gradient = phase_scale * delay_gradient
    phase_hessian = phase_scale * delay_hessian
    aperture_change = (
        phase_scale
        * float(waveform.coherent_processing_interval_s)
        * delay_gradient_rate
    )

    mixed_columns = np.column_stack((phase_gradient, aperture_change))
    normalized_columns: list[np.ndarray] = []
    for column in mixed_columns.T:
        norm = platform_norm(column)
        if norm > 1e-12:
            normalized_columns.append(column / norm)
    if normalized_columns:
        normalized_mixed = np.column_stack(normalized_columns)
        mixed_singular_values = platform_singular_values(normalized_mixed)
        mixed_rank = int(
            np.count_nonzero(
                mixed_singular_values
                > max(float(mixed_singular_values[0]), 1e-15) * 1e-8
            )
        )
    else:
        mixed_singular_values = np.zeros(0, dtype=float)
        mixed_rank = 0
    mixed_transversality = (
        float(
            np.clip(
                math.sqrt(
                    max(
                        0.0,
                        float(
                            np.prod(
                                platform_symmetric_eigh(
                                    platform_matmul(
                                        normalized_mixed.T,
                                        normalized_mixed,
                                    )
                                )[0]
                            )
                        ),
                    )
                ),
                0.0,
                1.0,
            )
        )
        if len(normalized_columns) == 2
        else 0.0
    )

    curvature_eigenvalues = platform_symmetric_eigh(
        0.5 * (phase_hessian + phase_hessian.T)
    )[0]
    curvature_scale = max(float(np.max(np.abs(curvature_eigenvalues))), 1e-15)
    nonzero_curvature = np.abs(curvature_eigenvalues) > curvature_scale * 1e-8
    curvature_rank = int(np.count_nonzero(nonzero_curvature))
    positive_curvature = np.abs(curvature_eigenvalues[nonzero_curvature])
    curvature_condition = (
        float(np.max(positive_curvature) / np.min(positive_curvature))
        if positive_curvature.size
        else math.inf
    )

    gradient_norm = platform_norm(delay_gradient)
    transverse_gradient_rate = delay_gradient_rate.copy()
    if gradient_norm > 1e-15:
        normal = delay_gradient / gradient_norm
        transverse_gradient_rate -= normal * float(
            normal @ delay_gradient_rate
        )
    normal_evolution = float(
        platform_norm(transverse_gradient_rate)
        * waveform.coherent_processing_interval_s
        / max(gradient_norm, 1e-15)
    )
    range_resolution = float(
        1.0
        / max(
            2.0 * waveform.rms_bandwidth_hz * gradient_norm,
            1e-15,
        )
    )
    aperture_rate_norm = platform_norm(delay_gradient_rate)
    aperture_resolution = float(
        1.0
        / max(
            2.0
            * math.pi
            * waveform.carrier_hz
            * waveform.coherent_processing_interval_s
            * aperture_rate_norm,
            1e-15,
        )
    )
    phase_nondegenerate = bool(
        gradient_norm > 1e-15
        and np.all(np.isfinite(phase_gradient))
        and np.all(np.isfinite(phase_hessian))
        and curvature_rank >= 1
    )
    return RadarPhaseGeometry(
        True,
        len(legs),
        float(delay),
        delay_gradient,
        0.5 * (delay_hessian + delay_hessian.T),
        delay_gradient_rate,
        phase_gradient,
        0.5 * (phase_hessian + phase_hessian.T),
        aperture_change,
        mixed_rank,
        mixed_transversality,
        curvature_rank,
        curvature_condition,
        normal_evolution,
        range_resolution,
        aperture_resolution,
        phase_nondegenerate,
    )


def _node(world: PhysicalWorld, node_id: str | None) -> SensorNode:
    if node_id is None:
        raise ValueError("action requires a node id")
    return world.node(node_id)


def _target_vector(target: TargetState) -> np.ndarray:
    return np.asarray([*target.position_m, *target.velocity_m_s], dtype=float)


def _swerling_rcs(target: TargetState, seed: int, fluctuation_samples: int = 1) -> float:
    rng = np.random.default_rng(int(seed))
    mean = float(target.mean_rcs_m2)
    if target.swerling_case == "0":
        return mean
    samples = max(1, int(fluctuation_samples)) if target.swerling_case in {"II", "IV"} else 1
    if target.swerling_case in {"I", "II"}:
        return float(rng.gamma(shape=float(samples), scale=mean / samples))
    return float(rng.gamma(shape=2.0 * samples, scale=mean / (2.0 * samples)))


#: Nose/tail-on to broadside bistatic RCS ratio for the default airframe class.
#: Measured aircraft RCS is minimum along the fuselage axis and maximum near
#: broadside, where the specular flat-plate return of the fuselage side and
#: wing leading edges is visible (Skolnik, *Radar Handbook* 3rd ed. fig. 2.20;
#: Knott, Shaeffer & Tuley, *Radar Cross Section* 2nd ed. ch. 2 and 14).  A
#: 6 dB axial-to-broadside ratio is a conservative small-aircraft value and is
#: overridable per action so the law is never a hidden constant.
AXIAL_TO_BROADSIDE_RCS_RATIO = 0.25


def aspect_rcs_pattern_factor(axial_cosine: float, axial_to_broadside_ratio: float = AXIAL_TO_BROADSIDE_RCS_RATIO) -> float:
    """Return the dimensionless aspect pattern factor of the mean RCS.

    ``axial_cosine`` is ``|cos(theta_a)|`` with ``theta_a`` the angle between
    the airframe axis (taken as the velocity heading) and the bistatic
    bisector.  The factor interpolates between ``axial_to_broadside_ratio`` at
    nose/tail-on (``axial_cosine = 1``) and ``1`` at broadside
    (``axial_cosine = 0``):

        f(c) = r + (1 - r) * (1 - c**2).

    Limits: ``f(0) = 1`` (broadside maximum), ``f(1) = r`` (axial minimum);
    ``r = 1`` recovers an isotropic target.  ``1 - c**2 = sin^2(theta_a)`` is
    the projected-side-area law, the leading term of the physical-optics
    flat-plate return of an elongated body.
    """

    cosine = min(max(abs(float(axial_cosine)), 0.0), 1.0)
    ratio = min(max(float(axial_to_broadside_ratio), 0.0), 1.0)
    return float(ratio + (1.0 - ratio) * (1.0 - cosine**2))


def _aspect_rcs(
    target: TargetState,
    tx: SensorNode,
    rx: SensorNode,
    seed: int,
    fluctuation_samples: int = 1,
    axial_to_broadside_ratio: float = AXIAL_TO_BROADSIDE_RCS_RATIO,
) -> float:
    base = _swerling_rcs(target, seed, fluctuation_samples)
    velocity = np.asarray(target.velocity_m_s, dtype=float)
    if np.linalg.norm(velocity) < 1e-9:
        return base
    heading = velocity / np.linalg.norm(velocity)
    look = unit_vector(target.position_m, tx.position_m) + unit_vector(target.position_m, rx.position_m)
    if np.linalg.norm(look) < 1e-12:
        return base
    look /= np.linalg.norm(look)
    aspect = abs(float(heading @ look))
    return float(base * aspect_rcs_pattern_factor(aspect, axial_to_broadside_ratio))


def _jammer_power(world: PhysicalWorld, waveform: Waveform, rx: SensorNode, jammer: Jammer | None) -> float:
    if jammer is None or jammer.kind == "none":
        return 0.0
    if jammer.kind == "swept":
        instantaneous_bandwidth = jammer.instantaneous_bandwidth_hz or min(jammer.bandwidth_hz, max(1e6, waveform.bandwidth_hz / 4.0))
        overlap = swept_spectrum_average_intercept_fraction(
            waveform.carrier_hz,
            waveform.bandwidth_hz,
            jammer.center_frequency_hz,
            jammer.bandwidth_hz,
            instantaneous_bandwidth,
        ) * jammer.duty_cycle
    else:
        overlap = flat_spectrum_intercept_fraction(
            waveform.carrier_hz,
            waveform.bandwidth_hz,
            jammer.center_frequency_hz,
            jammer.bandwidth_hz,
        ) * jammer.duty_cycle
    return one_way_interference_power_w(
        jammer.tx_power_w,
        jammer.antenna_gain_linear,
        rx,
        jammer.position_m,
        waveform.carrier_hz,
        world.atmosphere,
    ) * overlap


def _ambient_spectrum_interference(world: PhysicalWorld, waveform: Waveform, rx: SensorNode, signal_plus_noise_w: float) -> tuple[float, float, float]:
    powers: list[float] = []
    occupied_terms: list[float] = []
    confusion_odds = 0.0
    for emitter in world.spectrum_emitters:
        captured_power_fraction = flat_spectrum_intercept_fraction(
            waveform.carrier_hz,
            waveform.bandwidth_hz,
            emitter.center_frequency_hz,
            emitter.bandwidth_hz,
        )
        victim_band_overlap = spectral_overlap_fraction(
            waveform.carrier_hz,
            waveform.bandwidth_hz,
            emitter.center_frequency_hz,
            emitter.bandwidth_hz,
        )
        power = one_way_interference_power_w(
            emitter.tx_power_w,
            emitter.antenna_gain_linear,
            rx,
            emitter.position_m,
            waveform.carrier_hz,
            world.atmosphere,
        ) * captured_power_fraction * emitter.duty_cycle
        powers.append(power)
        occupied_terms.append(float(np.clip(victim_band_overlap * emitter.duty_cycle, 0.0, 1.0)))
        confusion_odds += (
            emitter.modulation_similarity
            * victim_band_overlap
            * emitter.duty_cycle
            * power / max(power + signal_plus_noise_w, 1e-300)
        )
    occupancy = 1.0 - float(np.prod([1.0 - value for value in occupied_terms])) if occupied_terms else 0.0
    return float(sum(powers)), float(np.clip(occupancy, 0.0, 1.0)), float(confusion_odds)


def _iso_delay_ground_point(world: PhysicalWorld, tx: SensorNode, rx: SensorNode) -> np.ndarray:
    """Ground point on the target-delay ellipse nearest the target bearing."""

    ground_z = float(world.ground_height_m)
    target = np.asarray(world.target.position_m, dtype=float)
    txp = np.asarray(tx.position_m, dtype=float)
    rxp = np.asarray(rx.position_m, dtype=float)
    center_xy = 0.5 * (txp[:2] + rxp[:2])
    direction = target[:2] - center_xy
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = target[:2] - txp[:2]
        norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.asarray([1.0, 0.0])
    else:
        direction /= norm
    target_path = float(np.linalg.norm(target - txp) + np.linalg.norm(target - rxp))

    def point(distance: float) -> np.ndarray:
        xy = center_xy + float(distance) * direction
        return np.asarray([xy[0], xy[1], ground_z], dtype=float)

    def residual(distance: float) -> float:
        candidate = point(distance)
        return float(np.linalg.norm(candidate - txp) + np.linalg.norm(candidate - rxp) - target_path)

    projected = max(0.0, float((target[:2] - center_xy) @ direction))
    low = 0.0
    if residual(projected) <= 0.0:
        low = projected
        high = max(projected + 100.0, 250.0)
        while residual(high) < 0.0 and high < 50_000.0:
            high *= 1.6
    else:
        high = projected
        while high > 1.0 and residual(0.5 * high) > 0.0:
            high *= 0.5
        low = 0.0
    if residual(low) * residual(high) > 0.0:
        return np.asarray([target[0], target[1], ground_z], dtype=float)
    for _ in range(64):
        middle = 0.5 * (low + high)
        if residual(middle) <= 0.0:
            low = middle
        else:
            high = middle
    return point(0.5 * (low + high))


def _clutter_power(
    world: PhysicalWorld,
    action: RadarAction,
    waveform: Waveform,
    tx: SensorNode,
    rx: SensorNode,
) -> ClutterEstimate:
    """Resolution-cell surface clutter at the target-delay ground intersection."""

    sigma0 = max(0.0, float(action.metadata.get("clutter_sigma0", 1e-3)))
    ground_point = _iso_delay_ground_point(world, tx, rx)
    if sigma0 == 0.0:
        return ClutterEstimate(0.0, 0.0, 0.0, 0.0, tuple(float(v) for v in ground_point))
    txp = np.asarray(tx.position_m, dtype=float)
    rxp = np.asarray(rx.position_m, dtype=float)
    rt = float(np.linalg.norm(ground_point - txp))
    rr = float(np.linalg.norm(ground_point - rxp))
    u_tx = (ground_point - txp) / max(rt, 1e-12)
    u_rx = (ground_point - rxp) / max(rr, 1e-12)
    ground_path_gradient = float(np.linalg.norm((u_tx + u_rx)[:2]))
    path_resolution = C0_M_S / waveform.bandwidth_hz
    delay_normal_width = path_resolution / max(ground_path_gradient, 0.08)
    beam_az = endpoint_beam_parameters(
        action.metadata, rx.node_id
    )[0]
    cross_range_width = max(path_resolution, math.sqrt(rt * rr) * beam_az)
    area = float(np.clip(delay_normal_width * cross_range_width, path_resolution**2, 5.0e6))
    grazing_tx = math.asin(float(np.clip((txp[2] - world.ground_height_m) / max(rt, 1e-12), -1.0, 1.0)))
    grazing_rx = math.asin(float(np.clip((rxp[2] - world.ground_height_m) / max(rr, 1e-12), -1.0, 1.0)))
    grazing = 0.5 * (abs(grazing_tx) + abs(grazing_rx))
    shape = max(0.1, float(action.metadata.get("clutter_texture_shape", 4.0)))
    texture_rng = np.random.default_rng(world.seed * 1_000_003 + action.action_id * 1009 + sum(ord(c) for c in rx.node_id))
    texture = float(texture_rng.gamma(shape=shape, scale=1.0 / shape))
    clutter_rcs = sigma0 * texture * area
    if action.kind == "active_monostatic" and tx.node_id == rx.node_id:
        power = monostatic_received_power_w(
            tx, ground_point, waveform.carrier_hz, clutter_rcs, 1.0,
            world.surface, world.atmosphere, waveform.bandwidth_hz, False,
            ground_height_m=float(world.ground_height_m),
        )
    else:
        power = bistatic_received_power_w(
            tx, rx, ground_point, waveform.carrier_hz, clutter_rcs, 1.0,
            world.surface, world.atmosphere, waveform.bandwidth_hz, False,
            ground_height_m=float(world.ground_height_m),
        )
    return ClutterEstimate(float(power), float(clutter_rcs), area, grazing, tuple(float(v) for v in ground_point))

def _direct_path_range_rate(tx: SensorNode, rx: SensorNode) -> float:
    """Time derivative of the transmitter-to-receiver direct-path range."""

    u_tx_rx = unit_vector(tx.position_m, rx.position_m)
    relative_velocity = (
        np.asarray(rx.velocity_m_s, dtype=float)
        - np.asarray(tx.velocity_m_s, dtype=float)
    )
    return float(u_tx_rx @ relative_velocity)


def _target_doppler_hz(
    world: PhysicalWorld,
    waveform: Waveform,
    tx: SensorNode,
    rx: SensorNode,
    *,
    subtract_direct_reference: bool = False,
) -> float:
    target_velocity = np.asarray(world.target.velocity_m_s, dtype=float)
    u_tx = unit_vector(tx.position_m, world.target.position_m)
    u_rx = unit_vector(rx.position_m, world.target.position_m)
    tx_rate = float(u_tx @ (target_velocity - np.asarray(tx.velocity_m_s, dtype=float)))
    rx_rate = float(u_rx @ (target_velocity - np.asarray(rx.velocity_m_s, dtype=float)))
    path_rate = tx_rate + rx_rate
    if subtract_direct_reference:
        path_rate -= _direct_path_range_rate(tx, rx)
    return DOPPLER_RANGE_RATE_SIGN * waveform.carrier_hz / C0_M_S * path_rate


def _clutter_doppler_leakage(world: PhysicalWorld, action: RadarAction, waveform: Waveform, tx: SensorNode, rx: SensorNode) -> float:
    """Power leakage of zero-Doppler clutter into the target Doppler cell.

    A rectangular coherent aperture gives ``sinc(fd*T)^2``.  A Gaussian
    moving-clutter spectrum and an explicit sidelobe/STAP floor prevent an
    unrealistically perfect notch.  These are observable processing
    assumptions, not post-hoc utility bonuses.
    """

    target_doppler = abs(
        _target_doppler_hz(
            world,
            waveform,
            tx,
            rx,
            subtract_direct_reference=action.kind == "passive_bistatic",
        )
    )
    rectangular = float(np.sinc(target_doppler * waveform.coherent_processing_interval_s) ** 2)
    spread_hz = max(0.0, float(action.metadata.get("clutter_doppler_spread_hz", 8.0)))
    moving_clutter = math.exp(-0.5 * (target_doppler / max(spread_hz, 1e-12)) ** 2) if spread_hz > 0.0 else 0.0
    floor_db = max(0.0, float(action.metadata.get("clutter_rejection_floor_db", 22.0)))
    floor = 10.0 ** (-floor_db / 10.0)
    return float(np.clip(max(rectangular, moving_clutter, floor), 0.0, 1.0))


def ar1_variance_inflation_upper_bound(correlation: float) -> float:
    r"""Return a rigorous matched-filter variance bound for AR(1) noise.

    For the Toeplitz correlation matrix

    .. math:: R_{ij}=\rho^{|i-j|},

    the spectral norm is bounded by the maximum absolute row sum,

    .. math:: \lambda_{\max}(R)\leq(1+|\rho|)/(1-|\rho|).

    Consequently, for every finite matched-filter vector ``s``,
    ``s^H R s`` is no larger than this factor times ``s^H s``.  The bound is
    deliberately waveform independent and conservative; it prevents a colored
    diffuse disturbance from receiving the white-noise processing gain without
    inventing a fitted correction.
    """

    rho = abs(float(correlation))
    if not math.isfinite(rho) or rho >= 1.0:
        raise ValueError("AR(1) correlation magnitude must be finite and below one")
    return float((1.0 + rho) / max(1.0 - rho, 1.0e-12))


def colored_disturbance_processing_gain_lower_bound(
    nominal_processing_gain_linear: float,
    correlation: float,
) -> tuple[float, float]:
    """Return ``(gain lower bound, variance inflation upper bound)``.

    The nominal gain is the declared coherent time-bandwidth gain.  Dividing it
    by the AR(1) variance-inflation bound gives a conservative gain for a
    stochastic colored disturbance at one matched-filter cell.
    """

    nominal = max(1.0, float(nominal_processing_gain_linear))
    inflation = ar1_variance_inflation_upper_bound(correlation)
    return float(max(1.0, nominal / inflation)), float(inflation)


def ambient_interference_fast_time_correlation(
    waveform: Waveform,
    spectrum_occupancy_fraction: float,
) -> float:
    """Return the shared analytic/IQ AR(1) correlation of ambient RF energy."""

    occupancy = float(np.clip(spectrum_occupancy_fraction, 0.0, 1.0))
    equivalent_bandwidth_hz = max(
        waveform.bandwidth_hz / 64.0,
        occupancy * waveform.bandwidth_hz,
    )
    return float(
        np.clip(
            math.exp(
                -2.0
                * math.pi
                * equivalent_bandwidth_hz
                / waveform.sample_rate_hz
            ),
            0.0,
            0.98,
        )
    )


def probability_of_detection(sinr_linear: float, false_alarm_probability: float = 1e-6) -> float:
    """Coherent single-cell detector probability using noncentral chi-square."""

    snr = max(0.0, float(sinr_linear))
    pfa = min(max(float(false_alarm_probability), 1e-12), 0.2)
    threshold = -2.0 * math.log(pfa)
    try:
        from scipy.stats import ncx2

        return float(np.clip(ncx2.sf(threshold, df=2, nc=2.0 * snr), 0.0, 1.0))
    except Exception:
        # Smooth fallback with the same limiting behavior.  Scientific runs
        # record whether scipy or the fallback was used.
        return float(np.clip(math.exp(-threshold / (2.0 * (1.0 + snr))), 0.0, 1.0))


@lru_cache(maxsize=2048)
def mixture_information_retention(correct_probability_rounded: float, separation_sigma_rounded: float) -> float:
    """Observed scalar location information of a two-Gaussian association mixture.

    The true component mean is the parameter; the false component is fixed.
    Numerical quadrature evaluates E[(d log p(y|mu)/d mu)^2] at mu=0 with
    unit variance.  It equals one for an unambiguous observation and never
    exceeds the complete-data information.
    """

    parallel, _ = mixture_information_coefficients(correct_probability_rounded, separation_sigma_rounded)
    return parallel


@lru_cache(maxsize=2048)
def variance_mixture_information_retention(
    correct_probability_rounded: float,
    false_variance_ratio_rounded: float,
    dimension: int,
) -> float:
    """Location Fisher retention for a Gaussian target / broad false-return mixture.

    The target component has identity covariance in whitened coordinates. The
    false-return component is centered on the validation gate with isotropic
    covariance ``false_variance_ratio * I``. Radial quadrature evaluates the
    exact mixture Fisher coefficient for one location coordinate.
    """

    p = float(np.clip(correct_probability_rounded, 1e-8, 1.0))
    variance_ratio = max(1.0, float(false_variance_ratio_rounded))
    d = max(1, int(dimension))
    if p >= 1.0 - 1e-12:
        return 1.0
    surface_area = 2.0 * math.pi ** (0.5 * d) / math.gamma(0.5 * d)
    upper = 12.0 * math.sqrt(variance_ratio)
    radius = np.linspace(0.0, upper, 8193)
    phi_true = (2.0 * math.pi) ** (-0.5 * d) * np.exp(-0.5 * radius**2)
    phi_false = (2.0 * math.pi * variance_ratio) ** (-0.5 * d) * np.exp(-0.5 * radius**2 / variance_ratio)
    mixture = p * phi_true + (1.0 - p) * phi_false
    integrand = (
        surface_area
        * radius ** (d - 1)
        * (radius**2 / d)
        * (p * phi_true) ** 2
        / np.maximum(mixture, 1e-300)
    )
    value = float(_trapezoid(integrand, radius))
    return float(np.clip(value, 0.0, 1.0))


@lru_cache(maxsize=2048)
def mixture_information_coefficients(correct_probability_rounded: float, separation_sigma_rounded: float) -> tuple[float, float]:
    """Parallel/orthogonal observed-information coefficients of a Gaussian mixture.

    In whitened coordinates the false-target displacement defines one special
    direction.  The Fisher information along it is generally different from
    every orthogonal measurement direction, so a scalar retention factor is
    insufficient for joint delay-Doppler observations.
    """

    p = min(max(float(correct_probability_rounded), 1e-6), 1.0)
    delta = min(max(abs(float(separation_sigma_rounded)), 0.0), 12.0)
    y = np.linspace(-12.0, 12.0 + delta, 4097)
    phi0 = np.exp(-0.5 * y**2) / math.sqrt(2.0 * math.pi)
    phif = np.exp(-0.5 * (y - delta) ** 2) / math.sqrt(2.0 * math.pi)
    mix = p * phi0 + (1.0 - p) * phif
    posterior_square_density = (p * phi0) ** 2 / np.maximum(mix, 1e-300)
    parallel = float(_trapezoid(posterior_square_density * y**2, y))
    orthogonal = float(_trapezoid(posterior_square_density, y))
    return float(np.clip(parallel, 0.0, 1.0)), float(np.clip(orthogonal, 0.0, 1.0))


def _reference_channel_quality(
    world: PhysicalWorld,
    waveform: Waveform,
    tx: SensorNode,
    rx: SensorNode,
    jammer: Jammer | None,
) -> tuple[float, float, float, float]:
    """Direct-path reference-channel SINR and signal purity for passive sensing."""

    gain, _ = band_averaged_two_ray_power_gain(
        tx.position_m, rx.position_m, waveform.carrier_hz, waveform.bandwidth_hz,
        world.surface, world.atmosphere,
        ground_height_m=float(world.ground_height_m),
    )
    direct_signal = tx.tx_power_w * tx.tx_gain_linear * rx.rx_gain_linear * gain
    noise = thermal_noise_power_w(waveform.bandwidth_hz, rx.noise_figure_db, world.atmosphere)
    jam = _jammer_power(world, waveform, rx, jammer)
    ambient, _, _ = _ambient_spectrum_interference(world, waveform, rx, direct_signal + noise)
    denominator = max(noise + jam + ambient, 1e-300)
    sinr = direct_signal / denominator
    purity = direct_signal / max(direct_signal + denominator, 1e-300)
    return (
        float(sinr),
        float(np.clip(purity, 0.0, 1.0)),
        float(direct_signal),
        float(denominator),
    )


def _azimuth_elevation_jacobian(
    node_position_m: np.ndarray,
    target_position_m: np.ndarray,
) -> np.ndarray:
    displacement = np.asarray(target_position_m, dtype=float) - np.asarray(
        node_position_m,
        dtype=float,
    )
    x, y, z = (float(value) for value in displacement)
    horizontal_squared = x * x + y * y
    range_squared = horizontal_squared + z * z
    if horizontal_squared <= 1e-12 or range_squared <= 1e-12:
        raise ValueError("beam-angle linearization requires nondegenerate geometry")
    horizontal = math.sqrt(horizontal_squared)
    return np.asarray(
        [
            [-y / horizontal_squared, x / horizontal_squared, 0.0],
            [
                -z * x / (horizontal * range_squared),
                -z * y / (horizontal * range_squared),
                horizontal / range_squared,
            ],
        ],
        dtype=float,
    )


def expected_commanded_beam_pointing_power_gain(
    world: PhysicalWorld,
    action: RadarAction,
    tx: SensorNode,
    rx: SensorNode,
    position_covariance_m2: np.ndarray,
) -> float:
    r"""Linearized posterior expectation of the commanded mainlobe gain.

    With position error ``e ~ N(0,P)`` and local angle error ``J e``, the
    Gaussian beam product is ``exp(-e' Q e)``.  Its exact Gaussian expectation
    under the local angular linearization is

    ``E[g] = det(I + 2 P^(1/2) Q P^(1/2))^(-1/2)``.

    Transmit and receive terms are accumulated in the same Cartesian quadratic
    form, preserving their correlation through the common target-position
    error.  The sidelobe floor is a conservative lower bound to the expectation
    of the clipped pattern.
    """

    covariance = np.asarray(position_covariance_m2, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("position covariance for beam expectation must be finite 3x3")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = platform_symmetric_eigh(covariance)
    tolerance = 1e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("position covariance for beam expectation is indefinite")
    covariance_root = platform_matmul(
        platform_matmul(
            eigenvectors,
            np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))),
        ),
        eigenvectors.T,
    )
    target = np.asarray(world.target.position_m, dtype=float)
    aim = np.asarray(
        action.metadata.get("commanded_beam_aim_position_m", target),
        dtype=float,
    ).reshape(3)
    if not np.all(np.isfinite(aim)):
        raise ValueError("commanded beam aim must contain finite SI coordinates")
    endpoints = [rx]
    if action.kind.startswith("active_"):
        endpoints.append(tx)
    quadratic = np.zeros((3, 3), dtype=float)
    linear = np.zeros(3, dtype=float)
    constant = 0.0
    peak_scale_product = 1.0
    floor_product = 1.0
    for endpoint in endpoints:
        azimuth_width, elevation_width, floor, peak_scale = endpoint_beam_parameters(
            action.metadata, endpoint.node_id
        )
        angle_weight = np.diag(
            [
                4.0 * math.log(2.0) / azimuth_width**2,
                4.0 * math.log(2.0) / elevation_width**2,
            ]
        )
        peak_scale_product *= peak_scale
        floor_product *= floor
        node_position = np.asarray(endpoint.position_m, dtype=float)
        jacobian = _azimuth_elevation_jacobian(
            node_position,
            target,
        )
        actual_vector = target - node_position
        aimed_vector = aim - node_position
        actual_horizontal = math.hypot(
            float(actual_vector[0]),
            float(actual_vector[1]),
        )
        aimed_horizontal = math.hypot(
            float(aimed_vector[0]),
            float(aimed_vector[1]),
        )
        if min(
            platform_norm(actual_vector),
            platform_norm(aimed_vector),
            actual_horizontal,
            aimed_horizontal,
        ) <= 0.0:
            return peak_scale_product * floor_product
        angular_mean_error = np.asarray(
            [
                (
                    math.atan2(float(actual_vector[1]), float(actual_vector[0]))
                    - math.atan2(float(aimed_vector[1]), float(aimed_vector[0]))
                    + math.pi
                )
                % (2.0 * math.pi)
                - math.pi,
                math.atan2(float(actual_vector[2]), actual_horizontal)
                - math.atan2(float(aimed_vector[2]), aimed_horizontal),
            ],
            dtype=float,
        )
        quadratic += platform_matmul(
            platform_matmul(jacobian.T, angle_weight), jacobian
        )
        linear += platform_matmul(
            platform_matmul(jacobian.T, angle_weight), angular_mean_error
        )
        constant += float(
            platform_matmul(
                platform_matmul(angular_mean_error, angle_weight),
                angular_mean_error,
            )
        )
    moment_matrix = np.eye(3) + 2.0 * platform_matmul(
        platform_matmul(covariance_root, quadratic), covariance_root
    )
    moment_matrix = 0.5 * (moment_matrix + moment_matrix.T)
    moment_eigenvalues, _ = platform_symmetric_eigh(moment_matrix)
    if float(moment_eigenvalues.min()) <= 0.0:
        raise ValueError("beam expectation moment matrix is not positive definite")
    logdet = float(np.log(moment_eigenvalues).sum())
    if not math.isfinite(logdet):
        raise ValueError("beam expectation moment matrix has non-finite log determinant")
    transformed_linear = platform_matmul(covariance_root, linear)
    exponent = -constant + 2.0 * float(
        platform_matmul(
            transformed_linear,
            platform_solve(moment_matrix, transformed_linear),
        )
    )
    mainlobe_expectation = math.exp(-0.5 * float(logdet) + exponent)
    retained = peak_scale_product * max(mainlobe_expectation, floor_product)
    return float(np.clip(retained, 0.0, 1.0))


def commanded_beam_pointing_power_gain(
    world: PhysicalWorld,
    action: RadarAction,
    tx: SensorNode,
    rx: SensorNode,
) -> float:
    """Power gain retained when a commanded beam misses the true target.

    The action command carries the controller's graph-derived aim point.  The
    simulator evaluates that same command at hidden truth.  A separable Gaussian
    mainlobe uses the declared full 3-dB azimuth/elevation widths, with an
    explicit sidelobe floor.  Active links pay transmit and receive pointing
    loss; externally illuminated passive links pay receive loss only.
    """

    aim_value = action.metadata.get("commanded_beam_aim_position_m")
    if aim_value is None or action.kind.startswith("calibration"):
        return 1.0
    aim = np.asarray(aim_value, dtype=float).reshape(3)
    target = np.asarray(world.target.position_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(aim)):
        raise ValueError("commanded beam aim must contain finite SI coordinates")
    def endpoint_gain(node: SensorNode) -> float:
        azimuth_width, elevation_width, floor, peak_scale = endpoint_beam_parameters(
            action.metadata, node.node_id
        )
        aimed = aim - np.asarray(node.position_m, dtype=float)
        actual = target - np.asarray(node.position_m, dtype=float)
        aimed_norm = platform_norm(aimed)
        actual_norm = platform_norm(actual)
        if min(aimed_norm, actual_norm) <= 0.0:
            return peak_scale * floor
        aimed_azimuth = math.atan2(float(aimed[1]), float(aimed[0]))
        actual_azimuth = math.atan2(float(actual[1]), float(actual[0]))
        azimuth_error = (
            actual_azimuth - aimed_azimuth + math.pi
        ) % (2.0 * math.pi) - math.pi
        aimed_elevation = math.asin(float(np.clip(aimed[2] / aimed_norm, -1.0, 1.0)))
        actual_elevation = math.asin(float(np.clip(actual[2] / actual_norm, -1.0, 1.0)))
        elevation_error = actual_elevation - aimed_elevation
        mainlobe = math.exp(
            -4.0
            * math.log(2.0)
            * (
                (azimuth_error / azimuth_width) ** 2
                + (elevation_error / elevation_width) ** 2
            )
        )
        return float(np.clip(peak_scale * max(mainlobe, floor), 0.0, 1.0))

    receive_gain = endpoint_gain(rx)
    transmit_gain = endpoint_gain(tx) if action.kind.startswith("active_") else 1.0
    return float(np.clip(transmit_gain * receive_gain, 0.0, 1.0))


def _planning_target_doppler_hz(
    world: PhysicalWorld,
    action: RadarAction,
    tx: SensorNode,
    rx: SensorNode,
    waveform: Waveform,
) -> float:
    """Causal predicted Doppler used by the clutter-blind-zone contract."""

    position = np.asarray(world.target.position_m, dtype=float)
    velocity = np.asarray(world.target.velocity_m_s, dtype=float)
    tx_direction = unit_vector(tx.position_m, position)
    rx_direction = unit_vector(rx.position_m, position)
    path_rate = float(
        tx_direction
        @ (velocity - np.asarray(tx.velocity_m_s, dtype=float))
        + rx_direction
        @ (velocity - np.asarray(rx.velocity_m_s, dtype=float))
    )
    if action.kind == "passive_bistatic":
        direct_direction = unit_vector(tx.position_m, rx.position_m)
        path_rate -= float(
            direct_direction
            @ (
                np.asarray(rx.velocity_m_s, dtype=float)
                - np.asarray(tx.velocity_m_s, dtype=float)
            )
        )
    frequency_term = (
        float(rx.frequency_bias_hz) - float(tx.frequency_bias_hz)
        if action.kind == "active_bistatic"
        else 0.0
    )
    return float(
        DOPPLER_RANGE_RATE_SIGN
        * waveform.carrier_hz
        / C0_M_S
        * path_rate
        + frequency_term
    )


def _dynamic_reference_channel_powers(
    action: RadarAction,
    coherent_signal_power_w: float,
    disturbance_power_w: float,
) -> tuple[float, float, float, float]:
    r"""Apply the observable passive-reference quality to one link budget.

    ``dynamic_reference_quality`` is the fraction :math:`q\in[0,1]` of the
    direct-path reference power that remains coherent with the registered
    illuminator after reference-channel contamination and conditioning.  The
    complementary power is not discarded: it becomes decorrelated reference
    disturbance.  Consequently

    .. math::

       P_{r,\mathrm{coh}}(q) &= qP_r,\\
       P_{r,\mathrm{dist}}(q) &= N_r+I_r+(1-q)P_r,\\
       \gamma_r(q) &= P_{r,\mathrm{coh}}(q)/P_{r,\mathrm{dist}}(q).

    This preserves total reference-channel input power, is exactly backward
    compatible at ``q=1``, and drives the usable reference SINR to zero as
    ``q`` tends to zero.  Active actions ignore the field even though the
    dynamic world stores it in every action's metadata.
    """

    signal = max(float(coherent_signal_power_w), 0.0)
    disturbance = max(float(disturbance_power_w), 0.0)
    if action.kind not in {"passive_bistatic", "tdoa", "fdoa", "aoa"}:
        total = signal + disturbance
        purity = signal / max(total, 1e-300) if total > 0.0 else 1.0
        sinr = signal / max(disturbance, 1e-300)
        return signal, disturbance, float(sinr), float(purity)
    quality = float(action.metadata.get("dynamic_reference_quality", 1.0))
    if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
        raise ValueError("dynamic reference quality must be finite and in [0, 1]")
    coherent = quality * signal
    contaminated = disturbance + (1.0 - quality) * signal
    sinr = coherent / max(contaminated, 1e-300)
    purity = coherent / max(coherent + contaminated, 1e-300)
    return float(coherent), float(contaminated), float(sinr), float(purity)


def noisy_reference_cross_ambiguity_sinr(
    surveillance_output_sinr_linear: float,
    reference_input_sinr_linear: float,
    reference_processing_gain_linear: float,
) -> float:
    r"""Return the output SINR of a noisy-reference cross-ambiguity cell.

    Let the surveillance and reference samples contain the same deterministic
    unit-energy sequence with pre-processing SINRs ``gamma_s`` and
    ``gamma_r``. For ``G`` independent time-bandwidth cells, the coherent
    cross-correlation statistic has

    .. math::

       \Gamma_{\rm cc}
       = \frac{G\gamma_s\gamma_r}
                    {\gamma_s+\gamma_r+1}.

    The three denominator terms arise from surveillance-noise times
    reference-signal, surveillance-signal times reference-noise, and the
    noise-noise cross product. The analytic surveillance budget is already
    stored after its matched-filter gain, ``Gamma_s = G gamma_s``. Therefore

    .. math::

       \Gamma_{\rm cc}^{-1}
       = \Gamma_s^{-1}
       + (G\gamma_r)^{-1}
       + (\Gamma_s\gamma_r)^{-1}.

    Mixing ``Gamma_s`` with an unprocessed ``gamma_r`` in the symmetric Stein
    expression drops the reference integration gain. Applying that symmetric
    expression to two separately amplified SINRs instead suppresses the
    noise-noise term by an extra factor of ``G`` and is optimistic. The caller
    must pass a conservative effective gain for colored or coherent reference
    disturbance.
    """

    surveillance = float(surveillance_output_sinr_linear)
    reference = float(reference_input_sinr_linear)
    gain = float(reference_processing_gain_linear)
    if any(math.isnan(value) for value in (surveillance, reference, gain)):
        raise ValueError("cross-ambiguity SINRs and gain cannot be NaN")
    if surveillance < 0.0 or reference < 0.0:
        raise ValueError("cross-ambiguity SINRs must be nonnegative")
    if not math.isfinite(gain) or gain < 1.0:
        raise ValueError(
            "reference processing gain must be finite and at least one"
        )
    if surveillance == 0.0 or reference == 0.0:
        return 0.0
    if math.isinf(reference):
        return surveillance
    inverse = 1.0 / surveillance if math.isfinite(surveillance) else 0.0
    inverse += 1.0 / (gain * reference)
    if math.isfinite(surveillance):
        inverse += 1.0 / (surveillance * reference)
    return float(1.0 / max(inverse, 1e-300))


def _single_link_budget(world: PhysicalWorld, action: RadarAction, jammer: Jammer | None, rx: SensorNode) -> LinkBudget:
    waveform = world.waveform(action.waveform_id)
    tx = _node(world, action.tx_node_id or action.reference_node_id)
    if not tx.supports_waveform(waveform.waveform_id):
        raise ValueError(f"transmitter {tx.node_id} does not support waveform {waveform.waveform_id}")
    rcs_overrides = world.external_channel_provenance.get("rcs_m2_by_action_id", {})
    fluctuation_samples = waveform.coherent_pulse_count if world.target.swerling_case in {"II", "IV"} else 1
    axial_ratio = float(action.metadata.get("axial_to_broadside_rcs_ratio", AXIAL_TO_BROADSIDE_RCS_RATIO))
    scattering_contract = str(
        world.external_channel_provenance.get(
            "truth_swerling_texture_contract", "legacy_action_fixed_texture"
        )
    )
    scattering_epoch = int(
        world.external_channel_provenance.get("truth_swerling_texture_epoch", 0)
    )
    if (
        scattering_contract
        in {
            "scan_indexed_common_texture_for_slow_fluctuation_swerling_I_III",
            "scan_indexed_carrier_group_texture_for_slow_fluctuation_swerling_I_III",
        }
        and world.target.swerling_case in {"I", "III"}
    ):
        carrier_group = (
            int(round(float(waveform.carrier_hz) / 1.0e6))
            if scattering_contract
            == "scan_indexed_carrier_group_texture_for_slow_fluctuation_swerling_I_III"
            else 0
        )
        scattering_seed = (
            (int(world.seed) + 1) * 1_000_003
            + (scattering_epoch + 1) * 65_537
            + (1 if world.target.swerling_case == "I" else 3) * 7_919
            + carrier_group * 104_729
        )
    else:
        scattering_seed = int(world.seed) * 1_000_003 + int(action.action_id) * 97
    rcs = float(
        rcs_overrides.get(
            str(action.action_id),
            rcs_overrides.get(
                action.action_id,
                _aspect_rcs(
                    world.target,
                    tx,
                    rx,
                    scattering_seed,
                    fluctuation_samples,
                    axial_ratio,
                ),
            ),
        )
    )
    loss = max(1.0, float(action.metadata.get("system_loss_linear", 2.0)))
    if action.kind == "active_monostatic" and tx.node_id == rx.node_id:
        signal = monostatic_received_power_w(
            tx, world.target.position_m, waveform.carrier_hz, rcs, loss,
            world.surface, world.atmosphere, waveform.bandwidth_hz,
            ground_height_m=float(world.ground_height_m),
        )
    else:
        signal = bistatic_received_power_w(
            tx, rx, world.target.position_m, waveform.carrier_hz, rcs, loss,
            world.surface, world.atmosphere, waveform.bandwidth_hz,
            ground_height_m=float(world.ground_height_m),
        )
    pointing_gain = commanded_beam_pointing_power_gain(world, action, tx, rx)
    planning_gain_by_receiver = action.metadata.get(
        "planning_expected_beam_pointing_power_gain_by_receiver",
        {},
    )
    if planning_gain_by_receiver and not isinstance(planning_gain_by_receiver, Mapping):
        raise ValueError("planning beam-gain map must be a receiver mapping")
    planning_gain_available = bool(
        isinstance(planning_gain_by_receiver, Mapping)
        and (
            rx.node_id in planning_gain_by_receiver
            or str(rx.node_id) in planning_gain_by_receiver
        )
    )
    planning_gain = float(
        planning_gain_by_receiver.get(
            rx.node_id,
            planning_gain_by_receiver.get(str(rx.node_id), 1.0),
        )
        if planning_gain_available
        else 1.0
    )
    if not 0.0 <= planning_gain <= 1.0:
        raise ValueError("planning expected beam gain must lie in [0, 1]")
    received_overrides = world.external_channel_provenance.get("received_power_w_by_action_id", {})
    override_keys = [f"{action.action_id}:{rx.node_id}", str(action.action_id), action.action_id]
    received_override_applied = False
    for key in override_keys:
        if key in received_overrides:
            signal = float(received_overrides[key])
            received_override_applied = True
            break
    override_includes_pointing = bool(
        world.external_channel_provenance.get(
            "received_power_override_includes_commanded_beam_pointing_gain",
            False,
        )
    )
    # The two gains describe mutually exclusive epistemic situations.  In the
    # controller world, ``planning_gain`` is already E[g(e)] over the graph
    # posterior and includes the widened-beam peak-scale loss.  In the hidden
    # truth world that map is not copied, so ``pointing_gain`` evaluates the
    # frozen command at the realized target.  Multiplying both would count the
    # beamforming loss twice in planning while truth paid it once.
    applied_beam_gain = planning_gain if planning_gain_available else pointing_gain
    if not (received_override_applied and override_includes_pointing):
        signal *= applied_beam_gain
    clutter_estimate = _clutter_power(world, action, waveform, tx, rx)
    clutter = clutter_estimate.power_w
    jam = _jammer_power(world, waveform, rx, jammer)
    noise = thermal_noise_power_w(waveform.bandwidth_hz, rx.noise_figure_db, world.atmosphere)
    ambient, occupancy, source_confusion = _ambient_spectrum_interference(world, waveform, rx, signal + noise)
    input_power = signal + noise + clutter + jam + ambient
    adc_sqnr = 10.0 ** ((6.02 * rx.adc_bits + 1.76) / 10.0)
    quantization_noise = input_power / max(rx.adc_loading_fraction * adc_sqnr, 1.0)
    # The IQ front end applies random phase and conjugate-image leakage to the
    # complete ADC input, not only to the desired echo.  For phi~N(0,sigma^2),
    # E|exp(j phi)-1|^2 = 2(1-exp(-sigma^2/2)); this exact factor avoids the
    # small-angle approximation and makes the analytic impairment MSE match
    # the finite-window complex-baseband realization.
    phase_error_factor = 2.0 * (-math.expm1(-0.5 * rx.integrated_phase_noise_rad2))
    image_power_ratio = 10.0 ** (-rx.image_rejection_db / 10.0)
    phase_noise_error = input_power * phase_error_factor
    image_error = input_power * image_power_ratio
    impairment_noise = quantization_noise + phase_noise_error + image_error
    # Only the target-origin part is a coherent target distortion after the
    # matched filter.  Noise-like input terms retain their stochastic status
    # and must not be promoted to coherent target leakage.
    target_phase_noise_leakage = signal * phase_error_factor
    target_image_noise = signal * image_power_ratio
    front_end_sinr = signal / max(noise + impairment_noise + clutter + jam + ambient, 1e-300)
    processing_gain = waveform.matched_filter_processing_gain_linear
    clutter_fast_time_correlation = float(
        action.metadata.get("clutter_fast_time_correlation", 0.82)
    )
    (
        clutter_fast_time_processing_gain,
        clutter_color_variance_inflation,
    ) = colored_disturbance_processing_gain_lower_bound(
        processing_gain,
        clutter_fast_time_correlation,
    )
    ambient_fast_time_correlation = ambient_interference_fast_time_correlation(
        waveform,
        occupancy,
    )
    (
        ambient_fast_time_processing_gain,
        ambient_color_variance_inflation,
    ) = colored_disturbance_processing_gain_lower_bound(
        processing_gain,
        ambient_fast_time_correlation,
    )
    clutter_leakage = _clutter_doppler_leakage(world, action, waveform, tx, rx)
    # ``clutter`` is the pre-processing power of a diffuse random resolution
    # cell.  The R2 generator realizes it as fast-time AR(1) clutter, so it is
    # first reduced by the conservative colored-noise matched-filter gain and
    # only then by the slow-time target-Doppler leakage.  Treating the input
    # clutter power as a coherent target-cell tone understated AOA detection
    # probability by several orders of magnitude in the R18p4 confirmation
    # campaign.
    effective_clutter = (
        clutter
        * clutter_leakage
        / clutter_fast_time_processing_gain
    )
    noise_like_jammer = jam if jammer is None or jammer.kind != "drfm_false_target" else 0.0
    coherent_false_target = jam if jammer is not None and jammer.kind == "drfm_false_target" else 0.0
    effective_ambient = ambient / ambient_fast_time_processing_gain
    effective_noise_like = (
        (noise + quantization_noise + noise_like_jammer) / processing_gain
        + effective_ambient
    )
    coherent_impairment = target_phase_noise_leakage + target_image_noise + effective_clutter + coherent_false_target
    surveillance_sinr = signal / max(effective_noise_like + coherent_impairment, 1e-300)
    reference_sinr = math.inf
    reference_purity = 1.0
    reference_signal_power = 0.0
    reference_noise_plus_interference_power = 0.0
    reference_processing_gain = 1.0
    reference_output_sinr = math.inf
    reference_incoherent_disturbance = 0.0
    reference_coherent_interference = 0.0
    cross_ambiguity_sinr_model = "not_applicable"
    if action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"}:
        (
            reference_sinr,
            reference_purity,
            reference_signal_power,
            reference_noise_plus_interference_power,
        ) = _reference_channel_quality(world, waveform, tx, rx, jammer)
        (
            reference_signal_power,
            reference_noise_plus_interference_power,
            reference_sinr,
            reference_purity,
        ) = _dynamic_reference_channel_powers(
            action,
            reference_signal_power,
            reference_noise_plus_interference_power,
        )
        # The dynamic-quality complement is declared decorrelated reference
        # disturbance and is reduced by cross-correlation processing. A DRFM
        # component is structured and receives no such gain. Matching the R2
        # IQ decomposition gives the equivalent lower-bound gain
        #
        #   G_ref = D_total / (D_incoherent/G + D_coherent).
        #
        # Both channels then enter the cross-ambiguity equation at one common
        # processing point.
        reference_coherent_interference = (
            min(reference_noise_plus_interference_power, jam)
            if jammer is not None and jammer.kind == "drfm_false_target"
            else 0.0
        )
        reference_incoherent_disturbance = max(
            reference_noise_plus_interference_power
            - reference_coherent_interference,
            0.0,
        )
        output_equivalent_reference_disturbance = (
            reference_incoherent_disturbance / processing_gain
            + reference_coherent_interference
        )
        if reference_noise_plus_interference_power > 0.0:
            reference_processing_gain = max(
                1.0,
                reference_noise_plus_interference_power
                / max(output_equivalent_reference_disturbance, 1e-300),
            )
        reference_output_sinr = reference_signal_power / max(
            output_equivalent_reference_disturbance,
            1e-300,
        )
        sinr = noisy_reference_cross_ambiguity_sinr(
            surveillance_sinr,
            reference_sinr,
            reference_processing_gain,
        )
        cross_ambiguity_sinr_model = (
            "noisy_reference_cross_correlation_same_processing_domain_v1"
        )
    else:
        sinr = surveillance_sinr
    familywise_pfa = float(action.metadata.get("false_alarm_probability", 1e-6))
    cfar_cells_by_receiver = action.metadata.get(
        "planning_cfar_effective_independent_cells_by_receiver", {}
    )
    if cfar_cells_by_receiver and not isinstance(cfar_cells_by_receiver, Mapping):
        raise ValueError("planning CFAR cells must be a receiver mapping")
    cfar_cells = max(
        1,
        int(
            cfar_cells_by_receiver.get(
                rx.node_id, cfar_cells_by_receiver.get(str(rx.node_id), 1)
            )
            if isinstance(cfar_cells_by_receiver, Mapping)
            else 1
        ),
    )
    single_cell_pfa = -math.expm1(
        math.log1p(-familywise_pfa) / cfar_cells
    )
    gate_coverage_by_receiver = action.metadata.get(
        "planning_capture_gate_coverage_probability_by_receiver", {}
    )
    if gate_coverage_by_receiver and not isinstance(
        gate_coverage_by_receiver, Mapping
    ):
        raise ValueError("planning capture coverage must be a receiver mapping")
    gate_coverage = float(
        gate_coverage_by_receiver.get(
            rx.node_id, gate_coverage_by_receiver.get(str(rx.node_id), 1.0)
        )
        if isinstance(gate_coverage_by_receiver, Mapping)
        else 1.0
    )
    if not 0.0 <= gate_coverage <= 1.0:
        raise ValueError("planning capture coverage must lie in [0, 1]")
    planning_target_doppler = _planning_target_doppler_hz(
        world, action, tx, rx, waveform
    )
    planning_notch_half_width = max(
        0.0,
        float(action.metadata.get("planning_clutter_notch_half_width_hz", 0.0)),
    )
    principal_planning_doppler = (
        planning_target_doppler + 0.5 * waveform.pulse_repetition_hz
    ) % waveform.pulse_repetition_hz - 0.5 * waveform.pulse_repetition_hz
    target_in_clutter_notch = bool(
        action.kind
        in {"active_monostatic", "active_bistatic", "passive_bistatic"}
        and planning_notch_half_width > 0.0
        and abs(principal_planning_doppler) <= planning_notch_half_width
    )
    pd = (
        0.0
        if target_in_clutter_notch
        else gate_coverage * probability_of_detection(sinr, single_cell_pfa)
    )
    search_cells = max(1, int(action.metadata.get("association_search_cells", 32)))
    clutter_cell_sinr = effective_clutter / max(
        effective_noise_like + target_phase_noise_leakage + target_image_noise,
        1e-300,
    )
    clutter_false_probability = probability_of_detection(
        clutter_cell_sinr, single_cell_pfa
    )
    clutter_fraction = effective_clutter / max(effective_clutter + signal, 1e-300)
    expected_clutter_false = search_cells * clutter_false_probability * clutter_fraction
    expected_noise_false = search_cells * familywise_pfa
    expected_source_false = max(0.0, source_confusion)
    false_count = int(jammer.deception_count if jammer and jammer.kind == "drfm_false_target" else 0)
    false_power_ratio = jam / max(jam + signal, 1e-300) if false_count else 0.0
    expected_deception_false = false_count * false_power_ratio
    expected_false = expected_noise_false + expected_clutter_false + expected_source_false + expected_deception_false
    p_correct = pd / max(pd + expected_false, 1e-300)
    gate_sigma = max(0.5, float(action.metadata.get("association_gate_sigma", 4.0)))
    if jammer and jammer.kind == "drfm_false_target" and abs(jammer.false_target_delay_s) > 0.0:
        nominal_delay_std = math.sqrt(delay_variance_s2(waveform, max(sinr, 1e-9)))
        separation = abs(jammer.false_target_delay_s) / max(nominal_delay_std, 1e-15)
    else:
        dimension = 2 if action.kind in {"active_monostatic", "active_bistatic", "passive_bistatic", "aoa"} else 1
        separation = gate_sigma / max((1.0 + expected_false) ** (1.0 / dimension), 1.0)
    return LinkBudget(
        signal, noise, impairment_noise, clutter, jam, sinr, pd,
        float(np.clip(p_correct, 0.0, 1.0)), separation, rcs,
        front_end_sinr, processing_gain, effective_clutter,
        effective_noise_like, clutter_leakage, ambient, occupancy,
        source_confusion, expected_false, search_cells, reference_sinr,
        reference_purity, clutter_estimate.area_m2, clutter_estimate.grazing_rad,
        clutter_estimate.position_m, gate_coverage, cfar_cells, single_cell_pfa,
        planning_target_doppler, planning_notch_half_width,
        target_in_clutter_notch,
        reference_signal_power,
        reference_noise_plus_interference_power,
        clutter_fast_time_correlation,
        clutter_color_variance_inflation,
        clutter_fast_time_processing_gain,
        ambient_fast_time_correlation,
        ambient_color_variance_inflation,
        ambient_fast_time_processing_gain,
        effective_ambient,
        surveillance_output_sinr_linear=surveillance_sinr,
        reference_processing_gain_linear=reference_processing_gain,
        reference_output_sinr_linear=reference_output_sinr,
        reference_incoherent_disturbance_power_w=(
            reference_incoherent_disturbance
        ),
        reference_coherent_interference_power_w=(
            reference_coherent_interference
        ),
        cross_ambiguity_sinr_model=cross_ambiguity_sinr_model,
    )


def receiver_link_budget(
    world: PhysicalWorld,
    action: RadarAction,
    jammer: Jammer | None,
    receiver_node_id: str,
) -> LinkBudget:
    """Return the authoritative link budget for one action/receiver channel.

    R2 IQ generation and the analytic R0/R1 likelihood must share the same
    received-power realization. Recomputing a second scattering amplitude in
    the waveform simulator created differences of three to five orders of
    magnitude, so this bridge deliberately delegates to the exact scalar
    budget used by :func:`build_action_factor`.
    """

    if receiver_node_id not in action.rx_node_ids:
        raise ValueError(
            f"receiver {receiver_node_id!r} is outside action {action.action_id}"
        )
    return _single_link_budget(
        world,
        action,
        jammer,
        _node(world, receiver_node_id),
    )

def _link_budget(world: PhysicalWorld, action: RadarAction, jammer: Jammer | None) -> LinkBudget:
    receivers = [_node(world, node_id) for node_id in action.rx_node_ids]
    if not receivers:
        raise ValueError(f"action {action.action_id} has no receiver for link budget")
    budgets = [_single_link_budget(world, action, jammer, receiver) for receiver in receivers]
    if action.kind not in {"tdoa", "fdoa"} or len(budgets) < 2:
        return budgets[0]
    # Stein (IEEE T-ASSP 29(3):588-599, 1981, eq. 6) gives the effective
    # signal-to-noise ratio of the generalized cross-correlation of two noisy
    # channels with individual ratios gamma_1, gamma_2 as
    #     1/gamma_eff = 1/gamma_1 + 1/gamma_2 + 1/(gamma_1 * gamma_2).
    # The trailing product term is the noise x noise cross-product and dominates
    # when both channels are weak; omitting it made TDOA/FDOA optimistic at low
    # SNR and disagreed with the passive reference-channel branch of
    # ``_single_link_budget``, which already used the full Stein expression.
    inverse_sinr = sum(1.0 / max(b.sinr_linear, 1e-30) for b in budgets)
    inverse_sinr += float(np.prod([1.0 / max(b.sinr_linear, 1e-30) for b in budgets]))
    effective_sinr = 1.0 / inverse_sinr
    total_clutter = sum(b.clutter_power_w for b in budgets)
    effective_clutter = sum(b.effective_clutter_power_w for b in budgets)
    aggregate_clutter_processing_gain = min(
        b.clutter_fast_time_processing_gain_lower_bound_linear
        for b in budgets
    )
    # Preserve the single-link decomposition after TDOA/FDOA aggregation:
    #
    #   C_eff = C_input * L_slow / G_fast.
    #
    # The previous aggregate divided ``effective_clutter`` only by input
    # clutter and therefore stored the combined fast/slow suppression in a
    # field whose contract is slow-time Doppler leakage.  Multiplying by the
    # conservative common gain produces a clutter-power-weighted equivalent
    # leakage in [0, 1] and leaves every reported term independently auditable.
    equivalent_clutter_leakage = (
        effective_clutter
        * aggregate_clutter_processing_gain
        / max(total_clutter, 1e-300)
        if total_clutter > 0.0
        else 0.0
    )
    finite_reference = [b.reference_sinr_linear for b in budgets if math.isfinite(b.reference_sinr_linear)]
    finite_reference_output = [
        b.reference_output_sinr_linear
        for b in budgets
        if math.isfinite(b.reference_output_sinr_linear)
    ]
    return LinkBudget(
        sum(b.signal_power_w for b in budgets),
        sum(b.thermal_noise_power_w for b in budgets),
        sum(b.receiver_impairment_noise_w for b in budgets),
        total_clutter,
        sum(b.jammer_power_w for b in budgets),
        effective_sinr,
        float(np.prod([b.detection_probability for b in budgets])),
        float(np.prod([b.correct_association_probability for b in budgets])),
        min(b.false_target_separation_sigma for b in budgets),
        float(np.mean([b.rcs_realization_m2 for b in budgets])),
        1.0 / sum(1.0 / max(b.front_end_sinr_linear, 1e-30) for b in budgets),
        min(b.processing_gain_linear for b in budgets),
        effective_clutter,
        sum(b.effective_noise_like_power_w for b in budgets),
        equivalent_clutter_leakage,
        sum(b.ambient_interference_power_w for b in budgets),
        1.0 - float(np.prod([1.0 - b.spectrum_occupancy_fraction for b in budgets])),
        sum(b.source_confusion_odds for b in budgets),
        sum(b.expected_false_returns for b in budgets),
        sum(b.association_search_cells for b in budgets),
        min(finite_reference) if finite_reference else math.inf,
        float(np.prod([b.reference_signal_purity for b in budgets])),
        sum(b.clutter_patch_area_m2 for b in budgets),
        float(np.mean([b.clutter_patch_grazing_rad for b in budgets])),
        budgets[0].clutter_patch_position_m,
        float(np.prod([b.capture_gate_coverage_probability for b in budgets])),
        int(np.prod([b.cfar_effective_independent_cells for b in budgets])),
        float(np.prod([b.cfar_single_cell_false_alarm_probability for b in budgets])),
        float(np.mean([b.planning_target_doppler_hz for b in budgets])),
        max(b.planning_clutter_notch_half_width_hz for b in budgets),
        any(b.expected_target_in_clutter_notch for b in budgets),
        sum(b.reference_signal_power_w for b in budgets),
        sum(b.reference_noise_plus_interference_power_w for b in budgets),
        max(b.clutter_fast_time_correlation for b in budgets),
        max(b.clutter_color_variance_inflation_upper_bound for b in budgets),
        aggregate_clutter_processing_gain,
        max(b.ambient_fast_time_correlation for b in budgets),
        max(b.ambient_color_variance_inflation_upper_bound for b in budgets),
        min(
            b.ambient_fast_time_processing_gain_lower_bound_linear
            for b in budgets
        ),
        sum(b.effective_ambient_interference_power_w for b in budgets),
        surveillance_output_sinr_linear=min(
            b.surveillance_output_sinr_linear for b in budgets
        ),
        reference_processing_gain_linear=min(
            b.reference_processing_gain_linear for b in budgets
        ),
        reference_output_sinr_linear=(
            min(finite_reference_output) if finite_reference_output else math.inf
        ),
        reference_incoherent_disturbance_power_w=sum(
            b.reference_incoherent_disturbance_power_w for b in budgets
        ),
        reference_coherent_interference_power_w=sum(
            b.reference_coherent_interference_power_w for b in budgets
        ),
        cross_ambiguity_sinr_model=(
            "multi_receiver_aggregate_of_"
            "noisy_reference_cross_correlation_same_processing_domain_v1"
        ),
    )

def delay_variance_s2(waveform: Waveform, sinr_linear: float) -> float:
    info = 8.0 * math.pi**2 * waveform.rms_bandwidth_hz**2 * max(float(sinr_linear), 1e-12)
    return 1.0 / info


def slow_time_window_contract(
    pulse_count: int,
    pulse_repetition_hz: float,
    slow_time_window: str = "rectangular",
) -> tuple[np.ndarray, float, float]:
    """Return coherent-gain-normalized weights, ENBW, and weighted time RMS.

    The Doppler Fisher information of a windowed pulse train with unknown
    complex amplitude depends on the second central moment under ``w[m]^2``.
    The same weights and moments are therefore shared by planning and the R2
    extractor; otherwise exact reranking would value a different instrument
    from the one that generates the committed observation.
    """

    count = int(pulse_count)
    prf = float(pulse_repetition_hz)
    name = str(slow_time_window)
    if count <= 0 or not math.isfinite(prf) or prf <= 0.0:
        raise ValueError("slow-time window requires positive pulse count and PRF")
    if name not in {"rectangular", "hann_periodic"}:
        raise ValueError("slow-time window must be rectangular or hann_periodic")
    if name == "hann_periodic":
        if count < 3:
            raise ValueError(
                "periodic Hann slow-time processing requires at least three pulses"
            )
        raw = np.hanning(count + 1)[:-1]
    else:
        raw = np.ones(count, dtype=float)
    coherent_gain = float(np.sum(raw))
    if not math.isfinite(coherent_gain) or coherent_gain <= 0.0:
        raise RuntimeError("slow-time window has no coherent gain")
    weights = np.asarray(raw * (count / coherent_gain), dtype=float)
    enbw_bins = float(
        count
        * np.sum(weights**2)
        / max(float(np.sum(weights)) ** 2, np.finfo(float).tiny)
    )
    if count == 1:
        time_rms_s = 1.0 / (prf * math.sqrt(12.0))
    else:
        slow_time_s = np.arange(count, dtype=float) / prf
        squared = weights**2
        squared_sum = float(np.sum(squared))
        mean_s = float(np.sum(squared * slow_time_s) / squared_sum)
        time_rms_s = float(
            math.sqrt(
                np.sum(squared * (slow_time_s - mean_s) ** 2) / squared_sum
            )
        )
    if not math.isfinite(time_rms_s) or time_rms_s <= 0.0:
        raise RuntimeError("slow-time window has zero Doppler information aperture")
    return weights, enbw_bins, time_rms_s


def doppler_variance_hz2(
    waveform: Waveform,
    sinr_linear: float,
    slow_time_window: str = "rectangular",
) -> float:
    _, _, time_rms = slow_time_window_contract(
        waveform.coherent_pulse_count,
        waveform.pulse_repetition_hz,
        slow_time_window,
    )
    info = 8.0 * math.pi**2 * time_rms**2 * max(float(sinr_linear), 1e-12)
    return 1.0 / info


def angle_covariance_rad2(
    node: SensorNode,
    waveform: Waveform,
    sinr_linear: float,
    azimuth_rad: float,
    elevation_rad: float,
) -> np.ndarray:
    """Azimuth/elevation CRLB for a centered vertical rectangular array."""

    rows, cols = node.resolved_array_shape
    spacing = resolved_element_spacing_m(node, waveform.carrier_hz)
    if rows < 2 or cols < 2 or spacing <= 0.0:
        # A one-dimensional or zero-aperture array cannot identify both
        # azimuth and elevation.  Use an almost-uninformative finite covariance
        # for numerical transport; generated V60+ AOA actions require a planar
        # array and are rejected by the physical gate otherwise.
        return np.eye(2) * (math.pi**2 / 3.0)
    horizontal_second_moment = rows * spacing**2 * cols * (cols**2 - 1) / 12.0
    vertical_second_moment = cols * spacing**2 * rows * (rows**2 - 1) / 12.0
    yaw = float(node.array_yaw_rad)
    horizontal_axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    vertical_axis = np.asarray([0.0, 0.0, 1.0])
    az = float(azimuth_rad)
    el = float(elevation_rad)
    du_daz = np.asarray([-math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), 0.0])
    du_del = np.asarray([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])

    def aperture_quadratic(left: np.ndarray, right: np.ndarray) -> float:
        return (
            horizontal_second_moment
            * float(platform_matmul(left, horizontal_axis))
            * float(platform_matmul(right, horizontal_axis))
            + vertical_second_moment
            * float(platform_matmul(left, vertical_axis))
            * float(platform_matmul(right, vertical_axis))
        )

    k = 2.0 * math.pi / wavelength_m(waveform.carrier_hz)
    scale = 2.0 * max(float(sinr_linear), 1e-12) * k**2
    fim = scale * np.asarray([
        [aperture_quadratic(du_daz, du_daz), aperture_quadratic(du_daz, du_del)],
        [aperture_quadratic(du_del, du_daz), aperture_quadratic(du_del, du_del)],
    ])
    fim = 0.5 * (fim + fim.T)
    values, vectors = platform_symmetric_eigh(fim)
    # Numerical regularization must not create a hidden 30-degree information
    # floor.  Retain the physical SINR scaling and floor only at a relative
    # machine-level eigenvalue so denied AOA actions become uninformative.
    numerical_floor = max(1e-18, float(np.max(values)) * 1e-12)
    covariance = platform_matmul(
        vectors * (1.0 / np.maximum(values, numerical_floor)), vectors.T
    )
    return 0.5 * (covariance + covariance.T)


def angle_variance_rad2(node: SensorNode, waveform: Waveform, sinr_linear: float, angle_rad: float = 0.0) -> float:
    """Compatibility wrapper returning the azimuth variance at zero elevation."""

    return float(angle_covariance_rad2(node, waveform, sinr_linear, angle_rad, 0.0)[0, 0])

def _range_unit_and_derivative(position: np.ndarray, sensor: SensorNode) -> tuple[float, np.ndarray]:
    r = distance_m(position, sensor.position_m)
    return r, unit_vector(sensor.position_m, position)


def _radial_rate_derivative(position: np.ndarray, velocity: np.ndarray, sensor: SensorNode) -> tuple[float, np.ndarray, np.ndarray]:
    delta = position - np.asarray(sensor.position_m, dtype=float)
    r = float(np.linalg.norm(delta))
    u = delta / r
    du_dp = (np.eye(3) - np.outer(u, u)) / r
    relative_velocity = velocity - np.asarray(sensor.velocity_m_s, dtype=float)
    return float(u @ relative_velocity), du_dp @ relative_velocity, u


def _measurement_model(world: PhysicalWorld, action: RadarAction, layout: StateLayout) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    waveform = world.waveform(action.waveform_id)
    p = np.asarray(world.target.position_m, dtype=float)
    v = np.asarray(world.target.velocity_m_s, dtype=float)
    h_rows: list[np.ndarray] = []
    values: list[float] = []
    names: list[str] = []

    def row() -> np.ndarray:
        return np.zeros(layout.dimension, dtype=float)

    def add_receive_chain_delay(h: np.ndarray, node_id: str, coefficient: float) -> float:
        configuration = configuration_for(world, node_id, action.waveform_id)
        if configuration is None:
            return 0.0
        h[layout.effective_delay_index(configuration.configuration_id)] += float(coefficient)
        return float(coefficient) * configuration.effective_delay_s

    def add_calibration_path_truth(h: np.ndarray, node_id: str, coefficient: float) -> float:
        if not world.effective_delay_model_enabled:
            return 0.0
        h[layout.calibration_path_index(node_id)] += float(coefficient)
        return float(coefficient) * world.node(node_id).calibration_path_bias_s

    if action.kind == "active_monostatic":
        tx = _node(world, action.tx_node_id)
        r, u = _range_unit_and_derivative(p, tx)
        h = row(); h[:3] = 2.0 * u / C0_M_S
        receive_delay = add_receive_chain_delay(h, tx.node_id, 1.0)
        values.append(2.0 * r / C0_M_S + receive_delay); h_rows.append(h); names.append("monostatic_delay_s")
        radial, dp, dv = _radial_rate_derivative(p, v, tx)
        doppler_scale = DOPPLER_RANGE_RATE_SIGN * 2.0 * waveform.carrier_hz / C0_M_S
        hd = row(); hd[:3] = doppler_scale * dp; hd[3:6] = doppler_scale * dv
        values.append(doppler_scale * radial); h_rows.append(hd); names.append("monostatic_doppler_hz")
    elif action.kind in {"active_bistatic", "passive_bistatic"}:
        tx = _node(world, action.tx_node_id or action.reference_node_id)
        rx = _node(world, action.rx_node_ids[0])
        rt, ut = _range_unit_and_derivative(p, tx); rr, ur = _range_unit_and_derivative(p, rx)
        h = row(); h[:3] = (ut + ur) / C0_M_S
        if action.kind == "active_bistatic":
            # Accumulate rather than assign: if a degenerate action names the
            # same node as transmitter and receiver the clock bias cancels
            # exactly (d z / d b = +1 - 1 = 0).  Assignment would leave -1 and
            # fabricate clock information that the physics does not contain.
            h[layout.clock_index(rx.node_id)] += 1.0; h[layout.clock_index(tx.node_id)] += -1.0
        direct_reference = distance_m(tx.position_m, rx.position_m) if action.kind == "passive_bistatic" else 0.0
        clock_term = rx.clock_bias_s - tx.clock_bias_s if action.kind == "active_bistatic" else 0.0
        if action.kind == "active_bistatic":
            clock_term += add_receive_chain_delay(h, rx.node_id, 1.0)
        values.append((rt + rr - direct_reference) / C0_M_S + clock_term); h_rows.append(h); names.append("pbr_excess_delay_s" if action.kind == "passive_bistatic" else "bistatic_delay_s")
        radial_tx, dpt, dvt = _radial_rate_derivative(p, v, tx); radial_rx, dpr, dvr = _radial_rate_derivative(p, v, rx)
        doppler_scale = DOPPLER_RANGE_RATE_SIGN * waveform.carrier_hz / C0_M_S
        hd = row(); hd[:3] = doppler_scale * (dpt + dpr); hd[3:6] = doppler_scale * (dvt + dvr)
        if action.kind == "active_bistatic":
            hd[layout.frequency_index(rx.node_id)] += 1.0; hd[layout.frequency_index(tx.node_id)] += -1.0
        frequency_term = rx.frequency_bias_hz - tx.frequency_bias_hz if action.kind == "active_bistatic" else 0.0
        path_rate = radial_tx + radial_rx
        if action.kind == "passive_bistatic":
            # The direct-path reference removes both oscillator offset and the
            # geometric Doppler of a moving illuminator/receiver baseline.
            path_rate -= _direct_path_range_rate(tx, rx)
        fd = doppler_scale * path_rate + frequency_term
        values.append(float(fd)); h_rows.append(hd); names.append("bistatic_doppler_hz")
    elif action.kind == "tdoa":
        if len(action.rx_node_ids) < 2:
            raise ValueError("TDOA action requires two receivers")
        a = _node(world, action.rx_node_ids[0]); b = _node(world, action.rx_node_ids[1])
        ra, ua = _range_unit_and_derivative(p, a); rb, ub = _range_unit_and_derivative(p, b)
        h = row(); h[:3] = (ua - ub) / C0_M_S
        h[layout.clock_index(a.node_id)] += 1.0; h[layout.clock_index(b.node_id)] += -1.0
        chain_term = add_receive_chain_delay(h, a.node_id, 1.0) + add_receive_chain_delay(h, b.node_id, -1.0)
        values.append((ra - rb) / C0_M_S + a.clock_bias_s - b.clock_bias_s + chain_term); h_rows.append(h); names.append("tdoa_s")
    elif action.kind == "fdoa":
        if len(action.rx_node_ids) < 2:
            raise ValueError("FDOA action requires two receivers")
        a = _node(world, action.rx_node_ids[0]); b = _node(world, action.rx_node_ids[1])
        va, dpa, dva = _radial_rate_derivative(p, v, a); vb, dpb, dvb = _radial_rate_derivative(p, v, b)
        doppler_scale = DOPPLER_RANGE_RATE_SIGN * waveform.carrier_hz / C0_M_S
        h = row(); h[:3] = doppler_scale * (dpa - dpb); h[3:6] = doppler_scale * (dva - dvb)
        h[layout.frequency_index(a.node_id)] += 1.0; h[layout.frequency_index(b.node_id)] += -1.0
        values.append(doppler_scale * (va - vb) + a.frequency_bias_hz - b.frequency_bias_hz); h_rows.append(h); names.append("fdoa_hz")
    elif action.kind == "aoa":
        rx = _node(world, action.rx_node_ids[0])
        d = p - np.asarray(rx.position_m, dtype=float); rho2 = float(d[0] ** 2 + d[1] ** 2); rho = math.sqrt(rho2); r2 = float(d @ d)
        haz = row(); haz[:3] = [-d[1] / rho2, d[0] / rho2, 0.0]
        hel = row(); hel[:3] = [-d[0] * d[2] / (r2 * rho), -d[1] * d[2] / (r2 * rho), rho / r2]
        values.extend([math.atan2(d[1], d[0]), math.atan2(d[2], rho)]); h_rows.extend([haz, hel]); names.extend(["azimuth_rad", "elevation_rad"])
    elif action.kind == "calibration_time":
        node = _node(world, action.rx_node_ids[0] if action.rx_node_ids else action.tx_node_id)
        h = row(); h[layout.clock_index(node.node_id)] += 1.0
        reference = _node(world, action.reference_node_id) if action.reference_node_id else None
        value = node.clock_bias_s + add_receive_chain_delay(h, node.node_id, 1.0) + add_calibration_path_truth(h, node.node_id, 1.0)
        name = "effective_delay_calibration_s" if world.effective_delay_model_enabled else "clock_calibration_s"
        if reference is not None and reference.node_id != node.node_id:
            h[layout.clock_index(reference.node_id)] += -1.0
            value -= reference.clock_bias_s
            value += add_receive_chain_delay(h, reference.node_id, -1.0)
            value += add_calibration_path_truth(h, reference.node_id, -1.0)
            name = "differential_effective_delay_calibration_s" if world.effective_delay_model_enabled else "differential_clock_calibration_s"
        values.append(value); h_rows.append(h); names.append(name)
    elif action.kind == "calibration_frequency":
        node = _node(world, action.rx_node_ids[0] if action.rx_node_ids else action.tx_node_id)
        h = row(); h[layout.frequency_index(node.node_id)] += 1.0
        reference = _node(world, action.reference_node_id) if action.reference_node_id else None
        value = node.frequency_bias_hz
        name = "frequency_calibration_hz"
        if reference is not None and reference.node_id != node.node_id:
            h[layout.frequency_index(reference.node_id)] += -1.0
            value -= reference.frequency_bias_hz
            name = "differential_frequency_calibration_hz"
        values.append(value); h_rows.append(h); names.append(name)
    else:
        raise ValueError(f"unsupported V58 action kind {action.kind!r}")
    return tuple(names), np.asarray(values, dtype=float), np.vstack(h_rows)


def drfm_false_target_measurement_offset(
    world: PhysicalWorld,
    action: RadarAction,
    jammer: Jammer,
    layout: StateLayout | None = None,
) -> np.ndarray:
    """Measurement-consistent offset induced by a DRFM false target.

    The configured delay and Doppler offsets are mapped to the minimum-norm
    target position and velocity displacement for the selected geometry, then
    evaluated through the nonlinear measurement function. This gives TDOA,
    FDOA, and angle actions offsets consistent with one displaced target state.
    """

    if jammer.kind != "drfm_false_target":
        raise ValueError("DRFM false-target offset requires a drfm_false_target jammer")
    layout = layout or StateLayout.from_world(world)
    names, nominal, jacobian = _measurement_model(world, action, layout)
    position_delta = np.zeros(3, dtype=float)
    velocity_delta = np.zeros(3, dtype=float)
    delay_rows = [index for index, name in enumerate(names) if "delay" in name or name == "tdoa_s"]
    doppler_rows = [index for index, name in enumerate(names) if "doppler" in name or name == "fdoa_hz"]
    if delay_rows and abs(jammer.false_target_delay_s) > 0.0:
        gradient = jacobian[delay_rows[0], :3]
        denominator = float(gradient @ gradient)
        if denominator > 1e-30:
            position_delta = gradient * float(jammer.false_target_delay_s) / denominator
    elif abs(jammer.false_target_delay_s) > 0.0:
        sensor_positions = [np.asarray(world.node(node_id).position_m, dtype=float) for node_id in action.rx_node_ids]
        if action.tx_node_id is not None:
            sensor_positions.append(np.asarray(world.node(action.tx_node_id).position_m, dtype=float))
        centroid = np.mean(sensor_positions, axis=0) if sensor_positions else np.zeros(3)
        radial = np.asarray(world.target.position_m, dtype=float) - centroid
        radial /= max(float(np.linalg.norm(radial)), 1e-12)
        position_delta = radial * (0.5 * C0_M_S * float(jammer.false_target_delay_s))
    if doppler_rows and abs(jammer.false_target_doppler_hz) > 0.0:
        gradient = jacobian[doppler_rows[0], 3:6]
        denominator = float(gradient @ gradient)
        if denominator > 1e-30:
            velocity_delta = gradient * float(jammer.false_target_doppler_hz) / denominator
    position_norm = float(np.linalg.norm(position_delta))
    if position_norm > 3000.0:
        position_delta *= 3000.0 / position_norm
    velocity_norm = float(np.linalg.norm(velocity_delta))
    if velocity_norm > 150.0:
        velocity_delta *= 150.0 / velocity_norm
    shifted_target = replace(
        world.target,
        position_m=tuple(np.asarray(world.target.position_m, dtype=float) + position_delta),
        velocity_m_s=tuple(np.asarray(world.target.velocity_m_s, dtype=float) + velocity_delta),
    )
    shifted_world = replace(world, target=shifted_target)
    shifted_names, shifted, _ = _measurement_model(shifted_world, action, layout)
    if shifted_names != names:
        raise RuntimeError("false-target measurement model changed dimension")
    offset = np.asarray(shifted - nominal, dtype=float)
    for index, name in enumerate(names):
        if name in {"azimuth_rad", "elevation_rad"}:
            offset[index] = (offset[index] + math.pi) % (2.0 * math.pi) - math.pi
    return offset


def build_action_factor(
    world: PhysicalWorld,
    action: RadarAction,
    jammer: Jammer | None = None,
    *,
    measurement_covariance_override: np.ndarray | None = None,
) -> ActionFactor:
    """Build the likelihood factor of one action.

    ``measurement_covariance_override`` replaces the analytic CRLB covariance of
    the *primary* measurement rows (not the appended free-calibration rows) with
    an externally determined measurement covariance.  It exists so that a higher
    fidelity extractor -- whose error is not purely noise limited -- is fused with
    its own likelihood rather than with the analytic bound.  Every downstream
    quantity (whitened Jacobian, complete / observed / conditional / missing
    information) is recomputed from the substituted covariance, so the factor
    stays internally consistent.
    """

    world.validate()
    layout = StateLayout.from_world(world)
    names, prediction, h = _measurement_model(world, action, layout)
    waveform = world.waveform(action.waveform_id)
    if action.kind.startswith("calibration"):
        # Calibration accuracy is represented by the explicit measurement
        # covariance below, not by an artificial RF link with infinite SINR.
        # Keeping every link-budget field finite also prevents the proposal
        # network from learning a non-physical ``inf == calibration`` shortcut.
        link = LinkBudget(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, world.target.mean_rcs_m2)
        if action.kind == "calibration_time":
            variances = [float(action.metadata.get("calibration_std_s", 2e-9)) ** 2]
        else:
            variances = [float(action.metadata.get("calibration_std_hz", 0.5)) ** 2]
        retention = 1.0
    else:
        link = _link_budget(world, action, jammer)
        variances = []
        if action.kind == "aoa" and names == ("azimuth_rad", "elevation_rad"):
            rx = _node(world, action.rx_node_ids[0])
            covariance = angle_covariance_rad2(rx, waveform, link.sinr_linear, prediction[0], prediction[1])
        else:
            for name in names:
                receiver_jitter = sum(world.node(node_id).timing_jitter_s**2 for node_id in action.rx_node_ids)
                if "delay" in name or name == "tdoa_s":
                    variances.append(delay_variance_s2(waveform, link.sinr_linear) + receiver_jitter)
                elif "doppler" in name or name == "fdoa_hz":
                    variances.append(
                        doppler_variance_hz2(
                            waveform,
                            link.sinr_linear,
                            str(
                                action.metadata.get(
                                    "planning_slow_time_window", "rectangular"
                                )
                            ),
                        )
                    )
                else:
                    raise ValueError(f"unhandled measurement covariance for {name}")
            covariance = np.diag(np.maximum(np.asarray(variances, dtype=float), 1e-30))
    pointing_gains = (
        [1.0]
        if action.kind.startswith("calibration")
        else [
            commanded_beam_pointing_power_gain(
                world,
                action,
                _node(world, action.tx_node_id or action.reference_node_id),
                _node(world, node_id),
            )
            for node_id in action.rx_node_ids
        ]
    )
    if action.kind.startswith("calibration"):
        covariance = np.diag(np.maximum(np.asarray(variances, dtype=float), 1e-30))
    if measurement_covariance_override is not None:
        override = np.asarray(measurement_covariance_override, dtype=float)
        override = 0.5 * (override + override.T)
        if override.shape != covariance.shape:
            raise ValueError(
                "measurement covariance override must match the primary "
                f"measurement dimension {covariance.shape}"
            )
        if (
            not np.all(np.isfinite(override))
            or float(platform_symmetric_eigh(override)[0].min()) <= 0.0
        ):
            raise ValueError("measurement covariance override must be finite and positive definite")
        covariance = override
    scales = layout.scales()
    h_norm = h * scales[None, :]
    covariance_cholesky = platform_cholesky(covariance)
    standard_deviation = np.sqrt(np.diag(covariance))
    h_white = platform_solve(covariance_cholesky, h_norm)
    complete = platform_matmul(h_white.T, h_white)
    association_parallel = 1.0
    association_orthogonal = 1.0
    false_return_covariance_scale = 1.0
    false_displacement_sigma = np.zeros(len(names), dtype=float)
    if action.kind.startswith("calibration"):
        observed = complete.copy()
        # A calibration measurement is not gated by target detection, so the
        # detection-conditional information is the complete-data information.
        conditional_detected = complete.copy()
        retention = 1.0
    else:
        false_offset = np.zeros(len(names), dtype=float)
        known_association_control = bool(
            action.metadata.get("known_association_control", False)
        )
        conditional_on_detection_control = bool(
            action.metadata.get("conditional_on_target_detection_control", False)
        )
        if known_association_control:
            kernel = np.eye(len(names), dtype=float)
            association_model = "known_association_control"
            association_parallel = association_orthogonal = 1.0
        elif jammer is not None and jammer.kind == "drfm_false_target":
            false_offset = drfm_false_target_measurement_offset(world, action, jammer, layout)
            false_displacement_sigma = platform_solve(
                covariance_cholesky, false_offset
            )
            separation = platform_norm(false_displacement_sigma)
            association_parallel, association_orthogonal = mixture_information_coefficients(
                round(link.correct_association_probability, 6), round(separation, 4)
            )
            if separation > 1e-12:
                direction = false_displacement_sigma / separation
                kernel = association_orthogonal * np.eye(len(names)) + (association_parallel - association_orthogonal) * np.outer(direction, direction)
            else:
                kernel = association_parallel * np.eye(len(names))
            association_model = "displaced_Gaussian_deception_mixture"
        else:
            gate_sigma = max(0.5, float(action.metadata.get("association_gate_sigma", 4.0)))
            false_variance_ratio = 1.0 + gate_sigma**2 / 3.0
            false_return_covariance_scale = false_variance_ratio
            coefficient = variance_mixture_information_retention(
                round(link.correct_association_probability, 6), round(false_variance_ratio, 4), len(names)
            )
            association_parallel = association_orthogonal = coefficient
            kernel = coefficient * np.eye(len(names))
            association_model = "broad_validation_gate_false_return_mixture"
        # ``kernel`` is the association-mixture information kernel in whitened
        # measurement coordinates.  Both of its coefficients are clipped to
        # [0, 1], hence 0 <= K <= I and therefore J_cond|D=1 <= J_complete in
        # the Loewner order, exactly and without any numerical projection.
        conditional_detected = h_white.T @ kernel @ h_white
        observed = (
            conditional_detected.copy()
            if conditional_on_detection_control
            else link.detection_probability * conditional_detected
        )
        retention = float(np.trace(observed) / max(float(np.trace(complete)), 1e-300))
    missing = complete - observed
    for matrix in (complete, observed, missing, conditional_detected):
        matrix[:] = 0.5 * (matrix + matrix.T)
    automatic_calibrations = automatic_free_calibration_actions(world, action)
    if automatic_calibrations:
        calibration_factors = [build_action_factor(world, calibration, None) for calibration in automatic_calibrations]
        original_dimension = covariance.shape[0]
        total_dimension = original_dimension + sum(factor.covariance.shape[0] for factor in calibration_factors)
        combined_covariance = np.zeros((total_dimension, total_dimension), dtype=float)
        combined_covariance[:original_dimension, :original_dimension] = covariance
        cursor = original_dimension
        for factor in calibration_factors:
            width = factor.covariance.shape[0]
            combined_covariance[cursor : cursor + width, cursor : cursor + width] = factor.covariance
            cursor += width
            complete += factor.complete_information_normalized
            observed += factor.observed_information_normalized
            missing += factor.missing_information_normalized
            # Simultaneous free calibration is delivered whether or not the
            # radar return is detected, so it enters the detection-conditional
            # information at full weight.
            conditional_detected += factor.conditional_detected_information_normalized
        names = tuple([*names, *(name for factor in calibration_factors for name in factor.measurement_names)])
        prediction = np.concatenate([prediction, *(factor.prediction for factor in calibration_factors)])
        h = np.vstack([h, *(factor.jacobian_physical for factor in calibration_factors)])
        covariance = combined_covariance
        false_displacement_sigma = np.concatenate(
            [false_displacement_sigma, np.zeros(total_dimension - original_dimension, dtype=float)]
        )
        for matrix in (complete, observed, missing, conditional_detected):
            matrix[:] = 0.5 * (matrix + matrix.T)
    return ActionFactor(
        action.action_id,
        names,
        prediction,
        h,
        covariance,
        complete,
        observed,
        missing,
        link,
        {
            "association_information_retention": retention,
            "known_association_control": bool(
                action.metadata.get("known_association_control", False)
            ),
            "conditional_on_target_detection_control": bool(
                action.metadata.get(
                    "conditional_on_target_detection_control", False
                )
            ),
            "association_model": "calibration_direct" if action.kind.startswith("calibration") else association_model,
            "association_parallel_information_coefficient": association_parallel,
            "association_orthogonal_information_coefficient": association_orthogonal,
            "false_target_displacement_sigma": false_displacement_sigma.tolist(),
            "false_return_covariance_scale": false_return_covariance_scale,
            "detection_information_retention": link.detection_probability,
            "front_end_sinr_linear": link.front_end_sinr_linear,
            "matched_filter_processing_gain_linear": link.processing_gain_linear,
            "effective_clutter_power_w": link.effective_clutter_power_w,
            "effective_noise_like_power_w": link.effective_noise_like_power_w,
            "clutter_doppler_leakage": link.clutter_doppler_leakage,
            "clutter_fast_time_correlation": (
                link.clutter_fast_time_correlation
            ),
            "clutter_color_variance_inflation_upper_bound": (
                link.clutter_color_variance_inflation_upper_bound
            ),
            "clutter_fast_time_processing_gain_lower_bound_linear": (
                link.clutter_fast_time_processing_gain_lower_bound_linear
            ),
            "ambient_interference_power_w": link.ambient_interference_power_w,
            "ambient_fast_time_correlation": (
                link.ambient_fast_time_correlation
            ),
            "ambient_color_variance_inflation_upper_bound": (
                link.ambient_color_variance_inflation_upper_bound
            ),
            "ambient_fast_time_processing_gain_lower_bound_linear": (
                link.ambient_fast_time_processing_gain_lower_bound_linear
            ),
            "effective_ambient_interference_power_w": (
                link.effective_ambient_interference_power_w
            ),
            "spectrum_occupancy_fraction": link.spectrum_occupancy_fraction,
            "source_confusion_odds": link.source_confusion_odds,
            "expected_false_returns": link.expected_false_returns,
            "association_search_cells": link.association_search_cells,
            "reference_sinr_linear": link.reference_sinr_linear,
            "reference_sinr_processing_domain": "receiver_input",
            "reference_processing_gain_linear": (
                link.reference_processing_gain_linear
            ),
            "reference_output_sinr_linear": (
                link.reference_output_sinr_linear
            ),
            "surveillance_output_sinr_linear": (
                link.surveillance_output_sinr_linear
            ),
            "reference_incoherent_disturbance_power_w": (
                link.reference_incoherent_disturbance_power_w
            ),
            "reference_coherent_interference_power_w": (
                link.reference_coherent_interference_power_w
            ),
            "cross_ambiguity_sinr_model": link.cross_ambiguity_sinr_model,
            "reference_signal_purity": link.reference_signal_purity,
            "clutter_patch_area_m2": link.clutter_patch_area_m2,
            "clutter_patch_grazing_rad": link.clutter_patch_grazing_rad,
            "clutter_patch_position_m": list(link.clutter_patch_position_m),
            "capture_gate_coverage_probability": (
                link.capture_gate_coverage_probability
            ),
            "cfar_effective_independent_cells": (
                link.cfar_effective_independent_cells
            ),
            "cfar_single_cell_false_alarm_probability": (
                link.cfar_single_cell_false_alarm_probability
            ),
            "planning_target_doppler_hz": link.planning_target_doppler_hz,
            "planning_clutter_notch_half_width_hz": (
                link.planning_clutter_notch_half_width_hz
            ),
            "expected_target_in_clutter_notch": (
                link.expected_target_in_clutter_notch
            ),
            "clutter_blind_zone_detection_retention": (
                0.0 if link.expected_target_in_clutter_notch else 1.0
            ),
            "automatic_simultaneous_free_calibration_count": len(automatic_calibrations),
            "scheduling_question_exists": world.calibration_resource_mode == "exclusive_receive_chain",
            "model_level": (
                "externally_supplied_extractor_covariance_likelihood"
                if measurement_covariance_override is not None
                else "band_averaged_link_budget_ground_clutter_reference_channel_likelihood"
            ),
            "measurement_covariance_override_applied": measurement_covariance_override is not None,
            "external_received_power_override_pointing_semantics": (
                "already_includes_commanded_beam_pointing_gain"
                if world.external_channel_provenance.get(
                    "received_power_override_includes_commanded_beam_pointing_gain",
                    False,
                )
                else "pre_commanded_beam_pointing_gain"
            ),
            "planning_expected_beam_pointing_power_gain_by_receiver": dict(
                action.metadata.get(
                    "planning_expected_beam_pointing_power_gain_by_receiver",
                    {},
                )
            ),
            "planning_beam_gain_integrates_graph_position_covariance": bool(
                action.metadata.get(
                    "planning_beam_gain_integrates_graph_position_covariance",
                    False,
                )
            ),
            "commanded_beam_pointing_power_gain_by_receiver": pointing_gains,
            "commanded_beam_aim_source": action.metadata.get(
                "commanded_beam_aim_source",
                "legacy_unspecified_peak_gain",
            ),
            "commanded_beam_uses_truth": action.metadata.get(
                "commanded_beam_uses_truth",
                False,
            ),
            "conditional_detected_information_available": True,
            "measurement_units": ["s" if "delay" in n or n == "tdoa_s" or "clock" in n else "Hz" if "doppler" in n or n == "fdoa_hz" or "frequency" in n else "rad" for n in names],
            **radar_phase_geometry(world, action).to_metadata(),
        },
        conditional_detected,
    )


def world_propagated_to_time(world: PhysicalWorld, time_s: float) -> PhysicalWorld:
    """Propagate nominal target motion and deterministic oscillator drift."""

    time = float(time_s)
    target = replace(
        world.target,
        position_m=tuple(np.asarray(world.target.position_m, dtype=float) + time * np.asarray(world.target.velocity_m_s, dtype=float)),
    )
    nodes = tuple(replace(node, clock_bias_s=node.clock_bias_s + time * node.clock_drift_s_per_s) for node in world.nodes)
    configurations = tuple(
        replace(
            configuration,
            effective_delay_s=(
                configuration.effective_delay_s
                + time * configuration.effective_delay_drift_s_per_s
            ),
            calibration_age_s=configuration.calibration_age_s + max(time, 0.0),
        )
        for configuration in world.receive_chain_configurations
    )
    return replace(world, target=target, nodes=nodes, receive_chain_configurations=configurations)


def world_with_normalized_state_error(world: PhysicalWorld, normalized_error: np.ndarray, layout: StateLayout | None = None) -> PhysicalWorld:
    """Apply a normalized target/nuisance perturbation to a world record."""

    layout = layout or StateLayout.from_world(world)
    error = np.asarray(normalized_error, dtype=float)
    if error.shape != (layout.dimension,):
        raise ValueError("normalized state error has the wrong dimension")
    physical = layout.scales() * error
    target = replace(
        world.target,
        position_m=tuple(np.asarray(world.target.position_m, dtype=float) + physical[:3]),
        velocity_m_s=tuple(np.asarray(world.target.velocity_m_s, dtype=float) + physical[3:6]),
    )
    nodes = []
    for node in world.nodes:
        nodes.append(replace(
            node,
            clock_bias_s=node.clock_bias_s + physical[layout.clock_index(node.node_id)],
            frequency_bias_hz=node.frequency_bias_hz + physical[layout.frequency_index(node.node_id)],
            calibration_path_bias_s=(
                node.calibration_path_bias_s + physical[layout.calibration_path_index(node.node_id)]
                if layout.receive_configuration_ids
                else node.calibration_path_bias_s
            ),
        ))
    configurations = tuple(
        replace(
            configuration,
            effective_delay_s=(
                configuration.effective_delay_s
                + physical[layout.effective_delay_index(configuration.configuration_id)]
            ),
        )
        for configuration in world.receive_chain_configurations
    )
    return replace(world, target=target, nodes=tuple(nodes), receive_chain_configurations=configurations)


def normalized_world_state_error(
    reference: PhysicalWorld,
    realization: PhysicalWorld,
    layout: StateLayout | None = None,
) -> np.ndarray:
    """Return ``realization - reference`` in the registered normalized state.

    This is the inverse operation of :func:`world_with_normalized_state_error`
    for the target kinematics and registered receiver nuisance variables.  It is
    a simulator-side operation: controller-facing records must never receive the
    realized vector because it contains truth.
    """

    state_layout = layout or StateLayout.from_world(reference)
    realization_layout = StateLayout.from_world(realization)
    if realization_layout.node_ids != state_layout.node_ids:
        raise ValueError("reference and realization node registries disagree")
    if (
        realization_layout.receive_configuration_ids
        != state_layout.receive_configuration_ids
    ):
        raise ValueError(
            "reference and realization receive-configuration registries disagree"
        )

    physical = np.zeros(state_layout.dimension, dtype=float)
    physical[:3] = (
        np.asarray(realization.target.position_m, dtype=float)
        - np.asarray(reference.target.position_m, dtype=float)
    )
    physical[3:6] = (
        np.asarray(realization.target.velocity_m_s, dtype=float)
        - np.asarray(reference.target.velocity_m_s, dtype=float)
    )
    for node_id in state_layout.node_ids:
        left = reference.node(node_id)
        right = realization.node(node_id)
        physical[state_layout.clock_index(node_id)] = (
            right.clock_bias_s - left.clock_bias_s
        )
        physical[state_layout.frequency_index(node_id)] = (
            right.frequency_bias_hz - left.frequency_bias_hz
        )
        if state_layout.receive_configuration_ids:
            physical[state_layout.calibration_path_index(node_id)] = (
                right.calibration_path_bias_s - left.calibration_path_bias_s
            )

    left_configurations = {
        item.configuration_id: item
        for item in reference.receive_chain_configurations
    }
    right_configurations = {
        item.configuration_id: item
        for item in realization.receive_chain_configurations
    }
    for configuration_id in state_layout.receive_configuration_ids:
        physical[state_layout.effective_delay_index(configuration_id)] = (
            right_configurations[configuration_id].effective_delay_s
            - left_configurations[configuration_id].effective_delay_s
        )
    return physical / state_layout.scales()


def nonlinear_measurement_delta(world: PhysicalWorld, action: RadarAction, normalized_error: np.ndarray, layout: StateLayout | None = None) -> np.ndarray:
    """Exact measurement-function displacement for a perturbed local state."""

    layout = layout or StateLayout.from_world(world)
    names, nominal, _ = _measurement_model(world, action, layout)
    perturbed = world_with_normalized_state_error(world, normalized_error, layout)
    _, shifted, _ = _measurement_model(perturbed, action, layout)
    delta = np.asarray(shifted - nominal, dtype=float)
    for index, name in enumerate(names):
        if name in {"azimuth_rad", "elevation_rad"}:
            delta[index] = (delta[index] + math.pi) % (2.0 * math.pi) - math.pi
    return delta


def false_return_normalized_jacobian(factor: ActionFactor, layout: StateLayout) -> np.ndarray:
    """False-return Jacobian retaining receiver clock/frequency dependence."""

    h = factor.jacobian_physical * layout.scales()[None, :]
    result = h.copy()
    result[:, : layout.target_dim] = 0.0
    return result


def finite_difference_target_jacobian(world: PhysicalWorld, action: RadarAction, relative_step: float = 1e-5) -> np.ndarray:
    """Independent central-difference check of the six target-state columns."""

    layout = StateLayout.from_world(world)
    x0 = _target_vector(world.target)
    rows = len(_measurement_model(world, action, layout)[0])
    jac = np.zeros((rows, 6), dtype=float)
    for i in range(6):
        step = max(abs(x0[i]), 1.0) * relative_step
        xp = x0.copy(); xm = x0.copy(); xp[i] += step; xm[i] -= step
        tp = TargetState(world.target.target_id, tuple(xp[:3]), tuple(xp[3:]), world.target.mean_rcs_m2, world.target.swerling_case, world.target.acceleration_std_m_s2)
        tm = TargetState(world.target.target_id, tuple(xm[:3]), tuple(xm[3:]), world.target.mean_rcs_m2, world.target.swerling_case, world.target.acceleration_std_m_s2)
        wp = replace(world, target=tp)
        wm = replace(world, target=tm)
        zp = _measurement_model(wp, action, layout)[1]; zm = _measurement_model(wm, action, layout)[1]
        jac[:, i] = (zp - zm) / (2.0 * step)
    return jac
