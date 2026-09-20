from __future__ import annotations

"""Truth evolution and leakage-free controller views for the unified runtime.

The dynamic episode remains the authoritative simulator state.  The controller
never receives that state directly.  It receives only:

* observable node/spectrum/topology/resource fields;
* a deterministic nominal world propagated to the current epoch; and
* the current normalized error estimate from the authoritative factor graph.

Target truth, oscillator truth, receive-chain truth, hidden adversary mode and
future events are excluded from the controller view by construction.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_episode import (
    EpisodeConfig,
    EpisodeState,
    ManeuverMode,
    TargetDynamicState,
    ThreatMode,
    initial_episode_state,
    transition_episode_state,
    episode_state_to_world,
)
from wideband_info_geometry.physics_v58.measurement import (
    StateLayout,
    expected_commanded_beam_pointing_power_gain,
    slow_time_window_contract,
    world_propagated_to_time,
    world_with_normalized_state_error,
)
from wideband_info_geometry.physics_v58.information import prior_covariance_physical
from wideband_info_geometry.physics_v58.schema import PhysicalWorld
from wideband_info_geometry.physics_v58.propagation import C0_M_S
from wideband_info_geometry.physics_v58.antenna_geometry import (
    broadened_beam_peak_scale,
    endpoint_beam_parameters,
)

from .belief import GraphBeliefEngine
from .contracts import OperationalNetworkState, RuntimeConfig


@dataclass(frozen=True)
class WorldViews:
    epoch: int
    truth_world: PhysicalWorld
    operational_reference_world: PhysicalWorld
    controller_world: PhysicalWorld
    controller_state: EpisodeState
    leakage_audit: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Structural truth-isolation contract
# ---------------------------------------------------------------------------
#
# ``EpisodeState`` is the simulator's hidden state.  The controller projection
# is only sound if *every* field is explicitly classified.  A field is
# OBSERVABLE when an operational system could measure or already owns it
# (its own resource ledger, its own past commands, measured spectrum occupancy,
# reported node health, the realized link topology).  Every other field is
# HIDDEN and must be overwritten with a declared, truth-independent constant or
# with a belief-derived quantity.
#
# The constants below are the declared sanitized values.  ``structural_leakage_
# audit`` re-reads the projection and fails closed if any HIDDEN field deviates
# from its declared value, so a future edit cannot reopen an oracle channel
# without also breaking the audit.

CONTROLLER_PROJECTION_VERSION = "unified_runtime_controller_projection_v4"

_TARGET_SENSING_ACTION_KINDS = frozenset(
    {
        "active_monostatic",
        "active_bistatic",
        "passive_bistatic",
        "tdoa",
        "fdoa",
        "aoa",
    }
)

#: Target fields that are hidden scattering / semi-Markov mode truth.  The
#: controller's dynamic model is the same constant-velocity model the
#: authoritative factor graph integrates, so the declared maneuver mode is
#: CONSTANT_VELOCITY with no residual mode lifetime (a nonzero
#: ``mode_duration_steps`` would disclose the epoch of the next true manoeuvre).
HIDDEN_TARGET_FIELDS: dict[str, Any] = {
    "maneuver_mode": ManeuverMode.CONSTANT_VELOCITY,
    "mode_age_steps": 0,
    "mode_duration_steps": 1,
    "maneuver_severity": 0.0,
    "maneuver_source": "sanitized_controller_projection",
    "aspect_state_rad": 0.0,
    "frequency_scattering_slope": 0.0,
    "acceleration_m_s2": None,  # checked as an all-zero vector
    "existence_probability": 1.0,
    "exists": True,
}

#: Adversary fields that are hidden threat truth.  ``mode`` is separately
#: forced to QUIET.  The declared projection asserts *no* adversary knowledge.
HIDDEN_ADVERSARY_FIELDS: dict[str, Any] = {
    "mode_age_steps": 0,
    "mode_duration_steps": 1,
    "energy_remaining_j": 0.0,
    "deception_budget": 0,
    "response_latency_steps": 0,
    "pending_response": None,
    "pending_response_age_steps": 0,
    "detected_radar_band_hz": None,
    "action_reactive": False,
    "internal_belief_active_probability": 0.5,
}

#: Episode-level fields that are hidden simulator bookkeeping.
HIDDEN_EPISODE_FIELDS: dict[str, Any] = {
    "events_active": (),
    "events_started": (),
    "adversary_action_history": (),
    "transition_log_probability": 0.0,
}


class DynamicWorldEngine:
    """Own the only simulator truth state and produce typed non-oracle views."""

    def __init__(
        self,
        base_world: PhysicalWorld,
        runtime_config: RuntimeConfig,
        episode_config: EpisodeConfig,
    ) -> None:
        base_world.validate()
        runtime_config.validate()
        episode_config.validate()
        if abs(float(episode_config.dt_s) - float(runtime_config.dt_s)) > 1e-12:
            raise ValueError("runtime and episode dt_s must agree")
        self.runtime_config = runtime_config
        self.episode_config = episode_config
        self.layout = StateLayout.from_world(base_world)
        prior_covariance = prior_covariance_physical(base_world, self.layout)
        prior_covariance = 0.5 * (prior_covariance + prior_covariance.T)
        prior_cholesky = np.linalg.cholesky(prior_covariance)
        prior_rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(base_world.seed), int(runtime_config.seed), 0x70A11CE]
            )
        )
        reference_minus_truth_physical = (
            prior_cholesky
            @ prior_rng.standard_normal(self.layout.dimension)
        )
        reference_minus_truth_normalized = (
            reference_minus_truth_physical / self.layout.scales()
        )
        condition_profile = str(
            base_world.external_channel_provenance.get(
                "experiment_condition_profile", "native"
            )
        )
        if condition_profile == "controlled_gaussian_localization":
            # A posterior-calibration experiment must sample hidden truth from
            # the registered controller prior.  Sampling the prior centre around
            # a fixed truth has the same marginal error distribution, but it is
            # not conditionally equivalent once the action policy adapts to that
            # centre.  Keep the supplied world as the controller-known prior
            # centre and draw simulator truth from it instead.
            controller_reference = base_world
            self.base_world = world_with_normalized_state_error(
                base_world,
                -reference_minus_truth_normalized,
                self.layout,
            )
            self._prior_sampling_contract = (
                "truth_drawn_conditionally_from_registered_controller_prior"
            )
        else:
            self.base_world = base_world
            controller_reference = world_with_normalized_state_error(
                base_world,
                reference_minus_truth_normalized,
                self.layout,
            )
            self._prior_sampling_contract = (
                "controller_reference_drawn_around_scenario_truth"
            )
        self.state = initial_episode_state(
            self.base_world,
            episode_config,
            seed=runtime_config.seed,
        )
        scattering_prior = dict(base_world.external_channel_provenance)
        prior_rcs_m2 = float(scattering_prior.get("target_rcs_prior_median_m2", 0.8))
        prior_swerling_case = str(
            scattering_prior.get("target_swerling_prior_case", "I")
        )
        if prior_rcs_m2 <= 0.0 or prior_swerling_case not in {
            "0",
            "I",
            "II",
            "III",
            "IV",
        }:
            raise ValueError("registered target scattering prior is invalid")
        self._controller_reference_base_world = replace(
            controller_reference,
            target=replace(
                controller_reference.target,
                mean_rcs_m2=prior_rcs_m2,
                swerling_case=prior_swerling_case,
            ),
            external_channel_provenance={
                **scattering_prior,
                "controller_target_scattering_source": (
                    "registered_target_class_prior_not_realized_truth"
                ),
            },
        )
        self._controller_reference_minus_truth_physical = np.asarray(
            reference_minus_truth_physical,
            dtype=float,
        )

    def _point_actions_from_controller_estimate(
        self,
        world: PhysicalWorld,
        position_covariance_m2: np.ndarray,
        *,
        search_covariance_source: str = (
            "authoritative_dynamic_factor_graph_posterior_covariance"
        ),
    ) -> PhysicalWorld:
        """Attach causal beam and fast-time acquisition commands.

        Both commands are functions only of the graph posterior and declared
        receiver hardware.  The simulator later evaluates the frozen command at
        hidden truth.  A broad acquisition beam conserves first-order mainlobe
        solid angle by paying a peak-gain penalty; a wide range gate consumes
        actual fast-time samples and incurs the corresponding whole-map CFAR
        multiplicity penalty in planning.
        """

        aim = tuple(float(value) for value in world.target.position_m)
        target = np.asarray(aim, dtype=float)
        covariance = np.asarray(position_covariance_m2, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("beam/capture command requires finite 3x3 position covariance")

        def angle_covariance(node_position: np.ndarray) -> np.ndarray:
            displacement = target - node_position
            x, y, z = (float(value) for value in displacement)
            horizontal_squared = x * x + y * y
            range_squared = horizontal_squared + z * z
            if min(horizontal_squared, range_squared) <= 1e-12:
                return np.eye(2) * math.pi**2
            horizontal = math.sqrt(horizontal_squared)
            jacobian = np.asarray(
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
            result = jacobian @ covariance @ jacobian.T
            return 0.5 * (result + result.T)

        def path_delay_std_s(tx_position: np.ndarray, rx_position: np.ndarray) -> float:
            tx_vector = target - tx_position
            rx_vector = target - rx_position
            tx_norm = max(float(np.linalg.norm(tx_vector)), 1e-12)
            rx_norm = max(float(np.linalg.norm(rx_vector)), 1e-12)
            gradient = (tx_vector / tx_norm + rx_vector / rx_norm) / C0_M_S
            variance = float(gradient @ covariance @ gradient)
            return math.sqrt(max(variance, 0.0))

        def next_power_of_two(value: int) -> int:
            return 1 << max(0, int(value - 1)).bit_length()

        def capture_contract(
            action: Any,
        ) -> tuple[int, dict[str, float], int, dict[str, float]]:
            waveform = world.waveform(action.waveform_id)
            tx = world.node(action.tx_node_id or action.reference_node_id)
            delay_std_by_receiver = {
                receiver_id: path_delay_std_s(
                    np.asarray(tx.position_m, dtype=float),
                    np.asarray(world.node(receiver_id).position_m, dtype=float),
                )
                for receiver_id in action.rx_node_ids
            }
            pulse_samples = (
                0
                if waveform.family == "cw" or waveform.duty_cycle >= 1.0 - 1e-12
                else int(math.ceil(waveform.pulse_width_s * waveform.sample_rate_hz))
            )
            resolution_guard_samples = int(
                math.ceil(
                    self.runtime_config.r2_delay_search_minimum_cells
                    * waveform.sample_rate_hz
                    / max(waveform.bandwidth_hz, 1.0)
                )
            )
            required = self.runtime_config.r2_samples
            if self.runtime_config.r2_adaptive_fast_time_enabled:
                for standard_deviation in delay_std_by_receiver.values():
                    margin = int(
                        math.ceil(
                            self.runtime_config.r2_capture_delay_sigmas
                            * standard_deviation
                            * waveform.sample_rate_hz
                        )
                    )
                    required = max(
                        required,
                        pulse_samples + 2 * (margin + resolution_guard_samples),
                    )
                required = next_power_of_two(required)
            samples = min(required, self.runtime_config.r2_max_samples)
            available_margin_s = max(
                0.0,
                (samples - pulse_samples) / (2.0 * waveform.sample_rate_hz),
            )
            coverage = {
                receiver_id: (
                    1.0
                    if standard_deviation <= 1e-15
                    else float(
                        np.clip(
                            math.erf(
                                available_margin_s
                                / (math.sqrt(2.0) * standard_deviation)
                            ),
                            0.0,
                            1.0,
                        )
                    )
                )
                for receiver_id, standard_deviation in delay_std_by_receiver.items()
            }
            delay_cells = max(
                1.0,
                2.0
                * waveform.bandwidth_hz
                * min(
                    available_margin_s,
                    self.runtime_config.r2_delay_search_sigmas
                    * max(delay_std_by_receiver.values(), default=0.0),
                ),
            )
            doppler_cells = max(1, waveform.coherent_pulse_count)
            independent_cells = max(1, int(math.ceil(delay_cells * doppler_cells)))
            return (
                int(samples),
                coverage,
                independent_cells,
                delay_std_by_receiver,
            )

        actions = []
        for action in world.actions:
            if action.kind not in _TARGET_SENSING_ACTION_KINDS:
                actions.append(action)
                continue
            tx = world.node(action.tx_node_id or action.reference_node_id)
            endpoint_ids = list(dict.fromkeys(action.rx_node_ids))
            if action.kind.startswith("active_"):
                endpoint_ids = list(dict.fromkeys([*endpoint_ids, tx.node_id]))
            commanded_azimuth: dict[str, float] = {}
            commanded_elevation: dict[str, float] = {}
            peak_scale: dict[str, float] = {}
            angular_coverage: dict[str, float] = {}
            for endpoint_id in endpoint_ids:
                endpoint = world.node(endpoint_id)
                native_az, native_el, _, _ = endpoint_beam_parameters(
                    action.metadata, endpoint_id
                )
                angle_cov = angle_covariance(
                    np.asarray(endpoint.position_m, dtype=float)
                )
                az_std = math.sqrt(max(float(angle_cov[0, 0]), 0.0))
                el_std = math.sqrt(max(float(angle_cov[1, 1]), 0.0))
                desired_az = max(
                    native_az,
                    2.0 * self.runtime_config.r2_beam_coverage_sigmas * az_std,
                )
                desired_el = max(
                    native_el,
                    2.0 * self.runtime_config.r2_beam_coverage_sigmas * el_std,
                )
                command_az = min(
                    desired_az,
                    self.runtime_config.r2_max_commanded_beamwidth_rad,
                )
                command_el = min(
                    desired_el,
                    self.runtime_config.r2_max_commanded_beamwidth_rad,
                )
                commanded_azimuth[endpoint_id] = command_az
                commanded_elevation[endpoint_id] = command_el
                peak_scale[endpoint_id] = broadened_beam_peak_scale(
                    native_az, native_el, command_az, command_el
                )
                az_coverage = (
                    1.0
                    if az_std <= 1e-15
                    else math.erf(
                        0.5 * command_az / (math.sqrt(2.0) * az_std)
                    )
                )
                el_coverage = (
                    1.0
                    if el_std <= 1e-15
                    else math.erf(
                        0.5 * command_el / (math.sqrt(2.0) * el_std)
                    )
                )
                angular_coverage[endpoint_id] = float(
                    np.clip(az_coverage * el_coverage, 0.0, 1.0)
                )
            (
                samples,
                range_gate_coverage,
                independent_cells,
                delay_std_by_receiver,
            ) = capture_contract(action)
            waveform = world.waveform(action.waveform_id)
            (
                _,
                planning_slow_time_window_enbw_bins,
                planning_slow_time_window_weighted_rms_s,
            ) = slow_time_window_contract(
                waveform.coherent_pulse_count,
                waveform.pulse_repetition_hz,
                self.runtime_config.r2_slow_time_window,
            )
            planning_clutter_notch_half_width_hz = max(
                self.runtime_config.r2_clutter_notch_spread_multiples
                * max(
                    0.0,
                    float(action.metadata.get("clutter_doppler_spread_hz", 3.0)),
                ),
                self.runtime_config.r2_clutter_notch_resolution_cells
                * planning_slow_time_window_enbw_bins
                / max(waveform.coherent_processing_interval_s, 1e-30),
            )
            declared_processing_samples = int(
                round(waveform.sample_rate_hz * waveform.dwell_s)
            )
            commanded_processing_samples = int(
                samples * waveform.coherent_pulse_count
            )
            extra_processing_samples = max(
                0,
                commanded_processing_samples - declared_processing_samples,
            )
            incremental_processing_energy_by_node = {
                receiver_id: float(
                    extra_processing_samples
                    * world.node(receiver_id).processing_energy_j_per_complex_sample
                )
                for receiver_id in action.rx_node_ids
            }
            base_energy_by_node = {
                str(node_id): float(value)
                for node_id, value in dict(
                    action.metadata.get("energy_j_by_node", {})
                ).items()
            }
            adjusted_energy_by_node = dict(base_energy_by_node)
            for node_id, increment in incremental_processing_energy_by_node.items():
                adjusted_energy_by_node[node_id] = (
                    adjusted_energy_by_node.get(node_id, 0.0) + increment
                )
            incremental_processing_energy_j = float(
                sum(incremental_processing_energy_by_node.values())
            )
            command_metadata = {
                **dict(action.metadata),
                "commanded_beam_aim_position_m": aim,
                "commanded_beam_aim_source": (
                    "authoritative_dynamic_factor_graph_posterior_mean"
                ),
                "commanded_beam_uses_truth": False,
                "commanded_search_covariance_source": str(
                    search_covariance_source
                ),
                "commanded_search_covariance_trace_m2": float(
                    np.trace(covariance)
                ),
                "commanded_search_covariance_is_parallel_posterior": False,
                "commanded_search_covariance_uses_truth": False,
                "commanded_beam_azimuth_width_rad_by_node": commanded_azimuth,
                "commanded_beam_elevation_width_rad_by_node": commanded_elevation,
                "commanded_beam_peak_gain_scale_by_node": peak_scale,
                "commanded_beam_angular_coverage_probability_by_node": (
                    angular_coverage
                ),
                "commanded_beam_mode": (
                    "posterior_coverage_broadened"
                    if any(value < 1.0 - 1e-12 for value in peak_scale.values())
                    else "native_tracking"
                ),
                "commanded_capture_fast_time_samples": samples,
                "commanded_capture_source": (
                    f"{search_covariance_source}_delay_covariance_and_"
                    "declared_receiver_limit"
                ),
                "commanded_capture_processing_samples_per_receiver": (
                    commanded_processing_samples
                ),
                "commanded_capture_incremental_processing_energy_j_by_node": (
                    incremental_processing_energy_by_node
                ),
                "commanded_capture_incremental_processing_energy_j": (
                    incremental_processing_energy_j
                ),
                "commanded_capture_resource_accounting": (
                    "receiver_energy_and_world_transition_updated"
                ),
                "energy_j_by_node": adjusted_energy_by_node,
                "planning_capture_gate_coverage_probability_by_receiver": (
                    range_gate_coverage
                ),
                "planning_delay_standard_deviation_s_by_receiver": {
                    receiver_id: float(standard_deviation)
                    for receiver_id, standard_deviation in (
                        delay_std_by_receiver.items()
                    )
                },
                "planning_cfar_effective_independent_cells_by_receiver": {
                    receiver_id: independent_cells
                    for receiver_id in action.rx_node_ids
                },
                "planning_clutter_notch_half_width_hz": (
                    planning_clutter_notch_half_width_hz
                ),
                "planning_clutter_blind_zone_uses_causal_predicted_doppler": True,
                "planning_slow_time_window": (
                    self.runtime_config.r2_slow_time_window
                ),
                "planning_slow_time_window_enbw_bins": float(
                    planning_slow_time_window_enbw_bins
                ),
                "planning_slow_time_window_weighted_rms_s": float(
                    planning_slow_time_window_weighted_rms_s
                ),
                "planning_doppler_information_uses_same_window_as_R2_extractor": True,
            }
            commanded_action = replace(
                action,
                energy_cost_j=(
                    float(action.energy_cost_j)
                    + incremental_processing_energy_j
                ),
                metadata=command_metadata,
            )
            expected_gain_by_receiver = {
                receiver_id: expected_commanded_beam_pointing_power_gain(
                    world,
                    commanded_action,
                    tx,
                    world.node(receiver_id),
                    position_covariance_m2,
                )
                for receiver_id in action.rx_node_ids
            }
            actions.append(
                replace(
                    commanded_action,
                    metadata={
                        **command_metadata,
                        "planning_expected_beam_pointing_power_gain_by_receiver": (
                            expected_gain_by_receiver
                        ),
                        "planning_beam_gain_integrates_graph_position_covariance": True,
                        "planning_beam_gain_linearization": (
                            "azimuth_elevation_jacobian_gaussian_quadratic_moment"
                        ),
                    },
                )
            )
        return replace(world, actions=tuple(actions))

    @property
    def epoch(self) -> int:
        return int(self.state.epoch)

    def truth_world(self) -> PhysicalWorld:
        """Return the simulator-only physical world at the current epoch."""

        world = episode_state_to_world(self.base_world, self.state)
        if self.runtime_config.scan_correlated_swerling_truth_enabled:
            world = replace(
                world,
                external_channel_provenance={
                    **dict(world.external_channel_provenance),
                    "truth_swerling_texture_contract": (
                        "scan_indexed_carrier_group_texture_for_slow_"
                        "fluctuation_swerling_I_III"
                    ),
                    "truth_swerling_texture_epoch": int(self.state.epoch),
                    "truth_swerling_texture_hidden_from_controller": True,
                    "truth_swerling_texture_action_id_used_for_slow_fluctuation": (
                        False
                    ),
                    "truth_swerling_texture_correlation_scope": (
                        "perfect_within_declared_carrier_independent_across_"
                        "declared_carriers_proxy"
                    ),
                    "truth_swerling_texture_is_full_frequency_correlation_model": (
                        False
                    ),
                },
            )
            world.validate()
        return world

    def operational_reference_world(self) -> PhysicalWorld:
        """Return a nominal world with only observable operational changes.

        Node health, receiver availability, spectrum occupancy and reference
        quality determine which actions exist operationally.  Hidden target,
        clock, frequency, effective-delay and jammer truth are replaced by the
        deterministic nominal propagation before the graph estimate is applied.
        """

        projected = self.truth_world()
        nominal = world_propagated_to_time(
            self._controller_reference_base_world,
            self.state.time_s,
        )
        nominal_nodes = {node.node_id: node for node in nominal.nodes}
        nodes = tuple(
            replace(
                operational_node,
                clock_bias_s=nominal_nodes[operational_node.node_id].clock_bias_s,
                clock_drift_s_per_s=nominal_nodes[
                    operational_node.node_id
                ].clock_drift_s_per_s,
                frequency_bias_hz=nominal_nodes[
                    operational_node.node_id
                ].frequency_bias_hz,
                calibration_path_bias_s=nominal_nodes[
                    operational_node.node_id
                ].calibration_path_bias_s,
            )
            for operational_node in projected.nodes
        )
        nominal_provenance = dict(nominal.external_channel_provenance)
        channel_override_keys = (
            "rcs_m2_by_action_id",
            "received_power_w_by_action_id",
            "received_power_override_includes_commanded_beam_pointing_gain",
        )
        controller_channel_prior_accessible = bool(
            nominal_provenance.get(
                "controller_accessible_channel_prior",
                False,
            )
        )
        excluded_channel_overrides = []
        if not controller_channel_prior_accessible:
            for key in channel_override_keys:
                if key in nominal_provenance:
                    excluded_channel_overrides.append(key)
                    nominal_provenance.pop(key, None)
        provenance = {
            **nominal_provenance,
            "dynamic_episode_id": self.state.episode_id,
            "dynamic_epoch": self.state.epoch,
            "controller_truth_fields_excluded": [
                "target_state",
                "node_clock_bias",
                "node_frequency_bias",
                "receive_chain_effective_delay",
                "hidden_adversary_mode",
                "future_events",
            ],
            "observable_operational_fields_retained": [
                "node_health",
                "receiver_transmitter_availability",
                "communication_topology",
                "spectrum_occupancy",
                "reference_quality",
                "campaign_resources",
            ],
            "controller_prior_reference_source": (
                "independent_draw_from_registered_joint_prior_covariance"
            ),
            "controller_prior_sampling_contract": self._prior_sampling_contract,
            "controller_prior_reference_realized_truth_offset_exposed": False,
            "controller_accessible_channel_prior": (
                controller_channel_prior_accessible
            ),
            "controller_external_channel_truth_overrides_excluded": sorted(
                excluded_channel_overrides
            ),
        }
        observable_calibration_ages = {
            str(configuration_id): float(age_s)
            for configuration_id, age_s in (
                self.state.receive_chain_calibration_age_s_by_configuration.items()
            )
        }
        reference_configurations = tuple(
            replace(
                configuration,
                calibration_age_s=observable_calibration_ages.get(
                    configuration.configuration_id,
                    configuration.calibration_age_s,
                ),
            )
            for configuration in nominal.receive_chain_configurations
        )
        reference = replace(
            projected,
            target=nominal.target,
            nodes=nodes,
            receive_chain_configurations=reference_configurations,
            jammers=(),
            external_channel_provenance=provenance,
        )
        reference.validate()
        return reference

    def operational_network_state(self) -> OperationalNetworkState:
        """Return only the epoch and observable communication topology."""

        view = OperationalNetworkState(
            epoch=int(self.state.epoch),
            communication_edges=tuple(
                tuple(sorted((str(left), str(right))))
                for left, right in self.state.communication_edges
            ),
            node_packet_delay_steps={
                node.node_id: int(node.packet_delay_steps)
                for node in self.state.nodes
            },
            node_queue_megabytes={
                node.node_id: float(node.queue_megabytes)
                for node in self.state.nodes
            },
            node_compute_load_fraction={
                node.node_id: float(node.compute_load_fraction)
                for node in self.state.nodes
            },
            node_packet_loss_probability={
                node.node_id: float(node.packet_loss_probability)
                for node in self.state.nodes
            },
        )
        view.validate()
        return view

    def controller_world(
        self,
        belief: GraphBeliefEngine,
        reference_world: PhysicalWorld | None = None,
        *,
        position_search_covariance_m2: np.ndarray | None = None,
        position_search_covariance_source: str | None = None,
    ) -> PhysicalWorld:
        reference = reference_world or self.operational_reference_world()
        estimate = belief.current_state_vector()
        world = world_with_normalized_state_error(reference, estimate, self.layout)
        posterior_covariance = np.asarray(
            belief.record().position_covariance_m2, dtype=float
        )
        search_covariance = posterior_covariance
        covariance_source = (
            "authoritative_dynamic_factor_graph_posterior_covariance"
        )
        if position_search_covariance_m2 is not None:
            search_covariance = np.asarray(
                position_search_covariance_m2, dtype=float
            )
            if search_covariance.shape != (3, 3) or not np.all(
                np.isfinite(search_covariance)
            ):
                raise ValueError(
                    "controller position search covariance must be finite 3x3"
                )
            search_covariance = 0.5 * (
                search_covariance + search_covariance.T
            )
            dominance = np.linalg.eigvalsh(
                search_covariance - posterior_covariance
            )
            tolerance = max(
                float(np.max(np.abs(search_covariance))), 1.0
            ) * 1.0e-10
            if float(np.min(dominance)) < -tolerance:
                raise ValueError(
                    "controller search covariance must dominate the authoritative "
                    "posterior covariance"
                )
            covariance_source = str(
                position_search_covariance_source
                or "registered_causal_search_covariance_override"
            )
        world = self._point_actions_from_controller_estimate(
            world,
            search_covariance,
            search_covariance_source=covariance_source,
        )
        world = replace(
            world,
            external_channel_provenance={
                **dict(world.external_channel_provenance),
                "state_source": "authoritative_DynamicFactorGraph_posterior",
                "independent_controller_covariance": False,
                "oracle_target_or_nuisance_state_used": False,
                "action_command_search_covariance_source": covariance_source,
                "action_command_search_covariance_is_authoritative_posterior": (
                    position_search_covariance_m2 is None
                ),
                "action_command_search_covariance_is_parallel_posterior": False,
                "action_command_search_covariance_uses_truth": False,
            },
        )
        world.validate()
        return world

    def controller_state(self, controller_world: PhysicalWorld) -> EpisodeState:
        """Return the observable control-state projection used by the shield.

        The safety shield requires campaign resources and current action
        availability.  It does not require target or adversary truth.  Those
        fields are replaced here so an implementation change cannot
        accidentally make them available to candidate generation or scoring.
        """

        primary = self.state.primary_target
        # Hidden target truth removed:
        #   * ``maneuver_mode``/``mode_age_steps``/``mode_duration_steps`` encode
        #     the semi-Markov manoeuvre regime *and the epoch of the next true
        #     manoeuvre*, i.e. future world state;
        #   * ``mean_rcs_m2``/``aspect_state_rad``/``frequency_scattering_slope``
        #     are hidden scattering truth that drives SINR, P_D and therefore the
        #     information value of every candidate action.
        # The controller keeps only the registered target-class scattering
        # prior carried by its reference world, never the realized truth RCS or
        # Swerling case.
        sanitized_target = replace(
            primary,
            target_id=controller_world.target.target_id,
            position_m=np.asarray(controller_world.target.position_m, dtype=float),
            velocity_m_s=np.asarray(controller_world.target.velocity_m_s, dtype=float),
            acceleration_m_s2=np.zeros(3),
            existence_probability=1.0,
            exists=True,
            maneuver_mode=HIDDEN_TARGET_FIELDS["maneuver_mode"],
            mode_age_steps=HIDDEN_TARGET_FIELDS["mode_age_steps"],
            mode_duration_steps=HIDDEN_TARGET_FIELDS["mode_duration_steps"],
            maneuver_severity=HIDDEN_TARGET_FIELDS["maneuver_severity"],
            maneuver_source=HIDDEN_TARGET_FIELDS["maneuver_source"],
            mean_rcs_m2=float(controller_world.target.mean_rcs_m2),
            aspect_state_rad=HIDDEN_TARGET_FIELDS["aspect_state_rad"],
            frequency_scattering_slope=HIDDEN_TARGET_FIELDS["frequency_scattering_slope"],
            age_steps=int(self.state.epoch),
        )
        controller_nodes = {node.node_id: node for node in controller_world.nodes}
        sanitized_nodes = tuple(
            replace(
                node_state,
                clock_bias_s=controller_nodes[node_state.node_id].clock_bias_s,
                clock_drift_s_per_s=controller_nodes[
                    node_state.node_id
                ].clock_drift_s_per_s,
                cfo_hz=controller_nodes[node_state.node_id].frequency_bias_hz,
            )
            for node_state in self.state.nodes
        )
        sanitized_effective_delays = {
            configuration.configuration_id: float(configuration.effective_delay_s)
            for configuration in controller_world.receive_chain_configurations
        }
        sanitized_calibration_ages = {
            configuration.configuration_id: float(configuration.calibration_age_s)
            for configuration in controller_world.receive_chain_configurations
        }
        sanitized_calibration_paths = {
            node.node_id: float(node.calibration_path_bias_s)
            for node in controller_world.nodes
        }
        # Hidden adversary truth removed: internal energy, deception budget,
        # response latency, adaptation memory and mode lifetime are all
        # unobservable threat state.  The declared projection is "no adversary
        # knowledge"; robustness to the adversary enters only through the
        # explicit scenario bank used by the exact reranker.
        sanitized_adversary = replace(
            self.state.adversary,
            mode=ThreatMode.QUIET,
            adaptation_memory={},
            **HIDDEN_ADVERSARY_FIELDS,
        )
        # ``endogeneity_memory`` is simulator regime memory rather than a
        # measured spectrum quantity; occupancy, illuminator availability,
        # reference quality, clutter regime and regime age are observable.
        sanitized_spectrum = replace(
            self.state.spectrum,
            endogeneity_memory={},
        )
        result = replace(
            self.state,
            targets=(sanitized_target,),
            adversary=sanitized_adversary,
            nodes=sanitized_nodes,
            spectrum=sanitized_spectrum,
            effective_delay_s_by_configuration=sanitized_effective_delays,
            receive_chain_calibration_age_s_by_configuration=(
                sanitized_calibration_ages
            ),
            calibration_path_bias_s_by_node=sanitized_calibration_paths,
            association_context={
                "information_retention": 1.0,
                "source": "controller_projection_no_oracle_association",
            },
            generator_version=CONTROLLER_PROJECTION_VERSION,
            **HIDDEN_EPISODE_FIELDS,
        )
        result.validate()
        return result

    def structural_leakage_audit(
        self,
        controller_state: EpisodeState,
        controller_world: PhysicalWorld,
    ) -> dict[str, Any]:
        """Field-level proof that no hidden simulator state reached the controller.

        Every hidden field is compared against its *declared* sanitized value
        rather than against truth.  Comparing against truth alone is unsound:
        a hidden field can coincide with its sanitized constant by accident, and
        a newly added hidden field would silently pass.  Requiring the declared
        value makes the contract total over the fields enumerated above.
        """

        violations: list[str] = []
        target = controller_state.primary_target
        for name, expected in HIDDEN_TARGET_FIELDS.items():
            value = getattr(target, name)
            if name == "acceleration_m_s2":
                if float(np.max(np.abs(np.asarray(value, dtype=float)))) != 0.0:
                    violations.append(f"target.{name}")
                continue
            if value != expected:
                violations.append(f"target.{name}")
        if float(target.mean_rcs_m2) != float(controller_world.target.mean_rcs_m2):
            violations.append("target.mean_rcs_m2")
        adversary = controller_state.adversary
        if adversary.mode is not ThreatMode.QUIET:
            violations.append("adversary.mode")
        if dict(adversary.adaptation_memory):
            violations.append("adversary.adaptation_memory")
        for name, expected in HIDDEN_ADVERSARY_FIELDS.items():
            if getattr(adversary, name) != expected:
                violations.append(f"adversary.{name}")
        for name, expected in HIDDEN_EPISODE_FIELDS.items():
            if getattr(controller_state, name) != expected:
                violations.append(f"episode.{name}")
        if dict(controller_state.spectrum.endogeneity_memory):
            violations.append("spectrum.endogeneity_memory")
        if len(controller_state.targets) != 1:
            violations.append("episode.targets_multiplicity")
        if controller_state.generator_version != CONTROLLER_PROJECTION_VERSION:
            violations.append("episode.generator_version")
        if controller_world.jammers:
            violations.append("world.jammers")
        if not controller_world.external_channel_provenance.get(
            "controller_accessible_channel_prior",
            False,
        ):
            for key in (
                "rcs_m2_by_action_id",
                "received_power_w_by_action_id",
                "received_power_override_includes_commanded_beam_pointing_gain",
            ):
                if key in controller_world.external_channel_provenance:
                    violations.append(f"world.external_channel_provenance.{key}")
        expected_effective_delays = {
            item.configuration_id: float(item.effective_delay_s)
            for item in controller_world.receive_chain_configurations
        }
        if dict(controller_state.effective_delay_s_by_configuration) != expected_effective_delays:
            violations.append("episode.effective_delay_s_by_configuration")
        expected_calibration_ages = {
            item.configuration_id: float(item.calibration_age_s)
            for item in controller_world.receive_chain_configurations
        }
        if (
            dict(
                controller_state.receive_chain_calibration_age_s_by_configuration
            )
            != expected_calibration_ages
        ):
            violations.append(
                "episode.receive_chain_calibration_age_s_by_configuration"
            )
        expected_calibration_paths = {
            node.node_id: float(node.calibration_path_bias_s)
            for node in controller_world.nodes
        }
        if dict(controller_state.calibration_path_bias_s_by_node) != expected_calibration_paths:
            violations.append("episode.calibration_path_bias_s_by_node")
        # The projection must also be *usable*: the derived PhysicalWorld that
        # the safety shield builds from it may not carry oracle scattering.
        projected = episode_state_to_world(controller_world, controller_state)
        if float(projected.target.mean_rcs_m2) != float(controller_world.target.mean_rcs_m2):
            violations.append("projected_world.target.mean_rcs_m2")
        if projected.target.swerling_case != controller_world.target.swerling_case:
            violations.append("projected_world.target.swerling_case")
        return {
            "declared_projection_version": CONTROLLER_PROJECTION_VERSION,
            "hidden_field_violations": tuple(violations),
            "hidden_fields_checked": (
                len(HIDDEN_TARGET_FIELDS)
                + len(HIDDEN_ADVERSARY_FIELDS)
                + len(HIDDEN_EPISODE_FIELDS)
                + 8
            ),
            "status": "passed" if not violations else "failed_closed",
        }

    def views(
        self,
        belief: GraphBeliefEngine,
        *,
        position_search_covariance_m2: np.ndarray | None = None,
        position_search_covariance_source: str | None = None,
    ) -> WorldViews:
        truth = self.truth_world()
        reference = self.operational_reference_world()
        controller = self.controller_world(
            belief,
            reference,
            position_search_covariance_m2=position_search_covariance_m2,
            position_search_covariance_source=(
                position_search_covariance_source
            ),
        )
        controller_state = self.controller_state(controller)
        truth_target = np.asarray(truth.target.position_m, dtype=float)
        controller_target = np.asarray(controller.target.position_m, dtype=float)
        provenance = dict(controller.external_channel_provenance)
        structural = self.structural_leakage_audit(controller_state, controller)
        audit = {
            "structural_hidden_field_audit": structural,
            "controller_hidden_target_mode_and_scattering_sanitized": (
                structural["status"] == "passed"
            ),
            "truth_object_identity_shared_with_controller": truth is controller,
            "truth_target_read_by_decision_engine": False,
            "truth_jammer_read_by_decision_engine": False,
            "truth_clock_frequency_delay_read_by_decision_engine": False,
            "controller_world_graph_source_verified": (
                provenance.get("state_source")
                == "authoritative_DynamicFactorGraph_posterior"
            ),
            "controller_world_independent_covariance_forbidden": (
                provenance.get("independent_controller_covariance") is False
            ),
            "controller_world_oracle_state_forbidden": (
                provenance.get("oracle_target_or_nuisance_state_used") is False
            ),
            "controller_hidden_jammer_excluded": len(controller.jammers) == 0,
            "controller_hidden_adversary_sanitized": (
                controller_state.adversary.mode is ThreatMode.QUIET
                and not controller_state.events_active
                and not controller_state.events_started
            ),
            "controller_target_error_norm_m_for_offline_audit": float(
                np.linalg.norm(controller_target - truth_target)
            ),
            "controller_target_error_vector_m_for_offline_audit": (
                controller_target - truth_target
            ).tolist(),
            "controller_jammer_count": len(controller.jammers),
            "truth_jammer_count": len(truth.jammers),
            "controller_target_scattering_source": provenance.get(
                "controller_target_scattering_source"
            ),
            "controller_target_rcs_m2_for_offline_audit": float(
                controller.target.mean_rcs_m2
            ),
            "truth_target_rcs_m2_for_offline_audit": float(
                truth.target.mean_rcs_m2
            ),
            "controller_target_swerling_case_for_offline_audit": (
                controller.target.swerling_case
            ),
            "truth_target_swerling_case_for_offline_audit": (
                truth.target.swerling_case
            ),
            "controller_scattering_uses_registered_class_prior": (
                provenance.get("controller_target_scattering_source")
                == "registered_target_class_prior_not_realized_truth"
            ),
            "controller_external_channel_truth_overrides_absent_or_registered": (
                bool(provenance.get("controller_accessible_channel_prior", False))
                or not any(
                    key in provenance
                    for key in (
                        "rcs_m2_by_action_id",
                        "received_power_w_by_action_id",
                        "received_power_override_includes_commanded_beam_pointing_gain",
                    )
                )
            ),
            "controller_state_generator": controller_state.generator_version,
            "controller_prior_reference_is_truth_object": (
                self._controller_reference_base_world is self.base_world
            ),
            "controller_prior_reference_draw_nonzero_for_offline_audit": bool(
                np.linalg.norm(self._controller_reference_minus_truth_physical) > 0.0
            ),
            "controller_prior_reference_covariance_source": (
                "prior_covariance_physical"
            ),
            "controller_prior_sampling_contract": self._prior_sampling_contract,
            "controller_beam_commands_from_graph_mean": all(
                action.metadata.get("commanded_beam_aim_source")
                == "authoritative_dynamic_factor_graph_posterior_mean"
                and action.metadata.get("commanded_beam_uses_truth") is False
                for action in controller.actions
                if action.kind in _TARGET_SENSING_ACTION_KINDS
            ),
            "controller_planning_beam_gain_integrates_graph_covariance": all(
                action.metadata.get(
                    "planning_beam_gain_integrates_graph_position_covariance"
                )
                is True
                and bool(
                    action.metadata.get(
                        "planning_expected_beam_pointing_power_gain_by_receiver"
                    )
                )
                for action in controller.actions
                if action.kind in _TARGET_SENSING_ACTION_KINDS
            ),
        }
        return WorldViews(
            epoch=self.epoch,
            truth_world=truth,
            operational_reference_world=reference,
            controller_world=controller,
            controller_state=controller_state,
            leakage_audit=audit,
        )

    def transition(
        self,
        action_ids: Sequence[int],
        *,
        compute_cost_s: float,
        committed_actions: Sequence[Any] | None = None,
    ) -> EpisodeState:
        transition_world = self.base_world
        if committed_actions is not None:
            committed_by_id = {
                int(action.action_id): action for action in committed_actions
            }
            missing = sorted(set(int(value) for value in action_ids) - committed_by_id.keys())
            if missing:
                raise ValueError(
                    f"committed action resource records are missing ids {missing}"
                )
            transition_world = replace(
                self.base_world,
                actions=tuple(
                    replace(
                        action,
                        energy_cost_j=float(
                            committed_by_id.get(action.action_id, action).energy_cost_j
                        ),
                        communication_cost_mb=float(
                            committed_by_id.get(
                                action.action_id, action
                            ).communication_cost_mb
                        ),
                        emission_exposure_j=float(
                            committed_by_id.get(
                                action.action_id, action
                            ).emission_exposure_j
                        ),
                        metadata={
                            **dict(action.metadata),
                            **{
                                key: value
                                for key, value in dict(
                                    committed_by_id.get(
                                        action.action_id, action
                                    ).metadata
                                ).items()
                                if key == "energy_j_by_node"
                                or key.startswith("commanded_capture_")
                            },
                        },
                    )
                    for action in self.base_world.actions
                ),
            )
        next_state = transition_episode_state(
            transition_world,
            self.state,
            action_ids,
            config=self.episode_config,
            seed=self.runtime_config.seed * 1_000_003 + self.state.epoch * 65_537,
            action_reactive_adversary=True,
            compute_cost_s=max(float(compute_cost_s), 0.0),
        )
        self.state = next_state
        return next_state

    @staticmethod
    def resource_snapshot(state: EpisodeState) -> dict[str, float]:
        values = asdict(state.resources)
        return {
            str(key): float(value)
            for key, value in values.items()
            if isinstance(value, (int, float, np.floating))
        }
