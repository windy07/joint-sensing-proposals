from __future__ import annotations

"""Configuration-aware receive-chain delay mechanisms for the large world model.

This module refines the abstract clock nuisance into concrete clock, RF-chain,
configuration-transfer, calibration-path, drift, and resource mechanisms.  It
contains no reward shaping: all planning value is produced downstream by the
measurement likelihood and nuisance-marginal information geometry.
"""

from dataclasses import replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from .schema import PhysicalWorld, RadarAction, ReceiveChainConfiguration, SensorNode, Waveform
from .time_reference import time_reference_world_summary


EFFECTIVE_DELAY_PROFILES: dict[str, dict[str, float | str]] = {
    "nominal": {
        "common_delay_std_s": 0.30e-9,
        "configuration_residual_std_s": 0.08e-9,
        "delay_random_walk_std_s_sqrt_s": 0.015e-9,
        "calibration_age_s": 5.0,
        "calibration_path_truth_std_s": 0.08e-9,
        "raw_iq_delay_std_s": 0.12e-9,
        "drift_std_s_per_s": 0.035e-9,
        "calibration_resource_mode": "exclusive_receive_chain",
    },
    "aged": {
        "common_delay_std_s": 0.45e-9,
        "configuration_residual_std_s": 0.15e-9,
        "delay_random_walk_std_s_sqrt_s": 0.08e-9,
        "calibration_age_s": 180.0,
        "calibration_path_truth_std_s": 0.12e-9,
        "raw_iq_delay_std_s": 0.15e-9,
        "drift_std_s_per_s": 0.09e-9,
        "calibration_resource_mode": "exclusive_receive_chain",
    },
    "configuration_mismatch": {
        "common_delay_std_s": 0.25e-9,
        "configuration_residual_std_s": 1.20e-9,
        "delay_random_walk_std_s_sqrt_s": 0.02e-9,
        "calibration_age_s": 30.0,
        "calibration_path_truth_std_s": 0.12e-9,
        "raw_iq_delay_std_s": 0.15e-9,
        "drift_std_s_per_s": 0.05e-9,
        "calibration_resource_mode": "exclusive_receive_chain",
    },
    "path_uncertain": {
        "common_delay_std_s": 0.30e-9,
        "configuration_residual_std_s": 0.10e-9,
        "delay_random_walk_std_s_sqrt_s": 0.02e-9,
        "calibration_age_s": 15.0,
        "calibration_path_truth_std_s": 1.00e-9,
        "raw_iq_delay_std_s": 0.12e-9,
        "drift_std_s_per_s": 0.04e-9,
        "calibration_resource_mode": "exclusive_receive_chain",
    },
    "joint_stress": {
        "common_delay_std_s": 0.60e-9,
        "configuration_residual_std_s": 0.90e-9,
        "delay_random_walk_std_s_sqrt_s": 0.10e-9,
        "calibration_age_s": 240.0,
        "calibration_path_truth_std_s": 1.20e-9,
        "raw_iq_delay_std_s": 0.25e-9,
        "drift_std_s_per_s": 0.12e-9,
        "calibration_resource_mode": "exclusive_receive_chain",
    },
    "simultaneous_free": {
        "common_delay_std_s": 0.30e-9,
        "configuration_residual_std_s": 0.08e-9,
        "delay_random_walk_std_s_sqrt_s": 0.015e-9,
        "calibration_age_s": 5.0,
        "calibration_path_truth_std_s": 0.08e-9,
        "raw_iq_delay_std_s": 0.12e-9,
        "drift_std_s_per_s": 0.035e-9,
        "calibration_resource_mode": "simultaneous_free",
    },
}


def profile_parameters(profile: str) -> dict[str, float | str]:
    if profile not in EFFECTIVE_DELAY_PROFILES:
        raise ValueError(f"unknown effective-delay profile {profile!r}")
    return dict(EFFECTIVE_DELAY_PROFILES[profile])


def waveform_frontend_group(waveform: Waveform) -> str:
    return "low_frontend" if waveform.carrier_hz < 4.0e9 else "high_frontend"


def build_effective_delay_components(
    nodes: Sequence[SensorNode],
    waveforms: Sequence[Waveform],
    *,
    seed: int,
    profile: str,
) -> tuple[tuple[SensorNode, ...], tuple[ReceiveChainConfiguration, ...], str]:
    """Create deterministic concrete states without changing world geometry."""

    params = profile_parameters(profile)
    rng = np.random.default_rng(int(seed) * 1_299_709 + 63)
    path_std = float(params["calibration_path_truth_std_s"])
    updated_nodes = tuple(
        replace(
            node,
            calibration_path_bias_s=float(rng.normal(0.0, max(path_std, 1e-15))),
            calibration_path_truth_std_s=path_std,
        )
        for node in nodes
    )
    groups: dict[str, list[Waveform]] = {}
    for waveform in waveforms:
        groups.setdefault(waveform_frontend_group(waveform), []).append(waveform)
    configurations: list[ReceiveChainConfiguration] = []
    for node in updated_nodes:
        node_base_delay = float(rng.normal(18.0e-9, 4.0e-9))
        for group_id, group_waveforms in sorted(groups.items()):
            group_offset = float(rng.normal(0.0, 1.2e-9))
            drift = float(rng.normal(0.0, float(params["drift_std_s_per_s"])))
            temperature_coefficient = float(rng.normal(4.0e-12, 1.0e-12))
            temperature_k = float(rng.uniform(278.0, 318.0))
            configuration_id = f"{node.node_id}:{group_id}"
            configurations.append(
                ReceiveChainConfiguration(
                    configuration_id=configuration_id,
                    node_id=node.node_id,
                    waveform_ids=tuple(waveform.waveform_id for waveform in group_waveforms),
                    transfer_group_id=f"{node.node_id}:shared_receive_chain",
                    rf_chain_id=f"{node.node_id}:{group_id}:rf",
                    rfdc_config_id=f"rfdc_{group_id}",
                    ddc_config_id=f"ddc_{group_id}",
                    bitstream_id="wideband_scheduler_bitstream_r1",
                    effective_delay_s=node_base_delay + group_offset + temperature_coefficient * (temperature_k - 298.15),
                    effective_delay_drift_s_per_s=drift,
                    common_delay_std_s=float(params["common_delay_std_s"]),
                    configuration_residual_std_s=float(params["configuration_residual_std_s"]),
                    delay_random_walk_std_s_sqrt_s=float(params["delay_random_walk_std_s_sqrt_s"]),
                    calibration_age_s=float(params["calibration_age_s"]),
                    temperature_k=temperature_k,
                    temperature_coefficient_s_per_k=temperature_coefficient,
                )
            )
    return updated_nodes, tuple(configurations), str(params["calibration_resource_mode"])


def configuration_for(world: PhysicalWorld, node_id: str, waveform_id: str) -> ReceiveChainConfiguration | None:
    if not world.effective_delay_model_enabled:
        return None
    return world.receive_configuration(node_id, waveform_id)


def action_receive_configurations(world: PhysicalWorld, action: RadarAction) -> tuple[ReceiveChainConfiguration, ...]:
    if not world.effective_delay_model_enabled:
        return ()
    node_ids = list(action.rx_node_ids)
    if action.kind.startswith("calibration") and action.reference_node_id:
        node_ids.append(action.reference_node_id)
    unique: dict[str, ReceiveChainConfiguration] = {}
    for node_id in node_ids:
        configuration = world.receive_configuration(node_id, action.waveform_id)
        unique[configuration.configuration_id] = configuration
    return tuple(unique[key] for key in sorted(unique))


def annotate_actions_with_effective_delay(
    actions: Sequence[RadarAction],
    nodes: Sequence[SensorNode],
    configurations: Sequence[ReceiveChainConfiguration],
    *,
    profile: str,
) -> tuple[RadarAction, ...]:
    params = profile_parameters(profile)
    by_pair = {
        (configuration.node_id, waveform_id): configuration
        for configuration in configurations
        for waveform_id in configuration.waveform_ids
    }
    node_by_id = {node.node_id: node for node in nodes}
    annotated: list[RadarAction] = []
    for action in actions:
        node_ids = list(action.rx_node_ids)
        if action.kind.startswith("calibration") and action.reference_node_id:
            node_ids.append(action.reference_node_id)
        selected = [by_pair[(node_id, action.waveform_id)] for node_id in node_ids]
        metadata = dict(action.metadata)
        metadata.update(
            {
                "effective_delay_realism_enabled": True,
                "effective_delay_profile": profile,
                "receive_configuration_ids": sorted({item.configuration_id for item in selected}),
                "receive_transfer_group_ids": sorted({item.transfer_group_id for item in selected}),
                "receive_chain_delay_prior_std_s_mean": float(np.mean([item.prior_std_s for item in selected])) if selected else 0.0,
                "configuration_residual_std_s_mean": float(np.mean([item.configuration_residual_std_s for item in selected])) if selected else 0.0,
                "calibration_age_s_max": max((item.calibration_age_s for item in selected), default=0.0),
                "calibration_path_truth_std_s_mean": float(np.mean([node_by_id[node_id].calibration_path_truth_std_s for node_id in node_ids])) if node_ids else 0.0,
                "calibration_resource_mode": str(params["calibration_resource_mode"]),
                "raw_iq_delay_std_s": float(params["raw_iq_delay_std_s"]),
            }
        )
        if action.kind == "calibration_time":
            metadata["calibration_std_s"] = float(params["raw_iq_delay_std_s"])
        allowed = action.allowed
        if str(params["calibration_resource_mode"]) == "simultaneous_free" and action.kind.startswith("calibration"):
            allowed = False
            metadata["disabled_reason"] = "free_simultaneous_calibration_is_attached_to_sensing_factors"
        annotated.append(replace(action, metadata=metadata, allowed=allowed))
    return tuple(annotated)


def automatic_free_calibration_actions(world: PhysicalWorld, sensing_action: RadarAction) -> tuple[RadarAction, ...]:
    """Return zero-resource differential calibration factors for the null model."""

    if (
        world.calibration_resource_mode != "simultaneous_free"
        or sensing_action.kind.startswith("calibration")
    ):
        return ()
    reference_nodes = [node for node in world.nodes if node.role == "calibration"]
    reference = reference_nodes[0] if reference_nodes else world.nodes[0]
    results: list[RadarAction] = []
    for ordinal, node_id in enumerate(sorted(set(sensing_action.rx_node_ids))):
        if node_id == reference.node_id:
            continue
        metadata = dict(sensing_action.metadata)
        metadata.update(
            {
                "calibration_std_s": float(metadata.get("raw_iq_delay_std_s", 0.12e-9)),
                "automatic_simultaneous_free_calibration": True,
                "energy_j_by_node": {},
                "communication_mb_by_node": {},
            }
        )
        results.append(
            RadarAction(
                action_id=-(1_000_000 + sensing_action.action_id * 100 + ordinal),
                kind="calibration_time",
                waveform_id=sensing_action.waveform_id,
                tx_node_id=None,
                rx_node_ids=(node_id,),
                reference_node_id=reference.node_id,
                resource_slot=sensing_action.resource_slot,
                energy_cost_j=0.0,
                communication_cost_mb=0.0,
                emission_exposure_j=0.0,
                allowed=True,
                metadata=metadata,
            )
        )
    return tuple(results)


def profile_for_world(
    profiles: Sequence[str],
    *,
    seed: int,
    family: str,
    world_index: int,
) -> str:
    if not profiles:
        return "disabled"
    for profile in profiles:
        profile_parameters(str(profile))
    offset = int(seed) + int(world_index) * 17 + sum(ord(character) for character in family)
    return str(profiles[offset % len(profiles)])


def effective_delay_world_summary(world: PhysicalWorld) -> dict[str, Any]:
    if not world.effective_delay_model_enabled:
        return {
            "enabled": False,
            "profile": "disabled",
            "configuration_count": 0,
            "calibration_resource_mode": "disabled",
        }
    configurations = world.receive_chain_configurations
    nodes = {node.node_id: node for node in world.nodes}
    return {
        "enabled": True,
        "profile": world.effective_delay_profile,
        "configuration_count": len(configurations),
        "transfer_group_count": len({item.transfer_group_id for item in configurations}),
        "prior_std_s_mean": float(np.mean([item.prior_std_s for item in configurations])),
        "configuration_residual_std_s_mean": float(np.mean([item.configuration_residual_std_s for item in configurations])),
        "calibration_age_s_max": max(item.calibration_age_s for item in configurations),
        "calibration_path_truth_std_s_mean": float(np.mean([nodes[item.node_id].calibration_path_truth_std_s for item in configurations])),
        "calibration_resource_mode": world.calibration_resource_mode,
        "scheduling_question_exists": world.calibration_resource_mode == "exclusive_receive_chain",
        "state_interpretation": "clock_plus_configuration_effective_receive_delay_plus_persistent_calibration_path_truth",
    }


def schedule_effective_delay_features(world: PhysicalWorld, actions: Iterable[RadarAction]) -> dict[str, float | int]:
    selected = list(actions)
    if not world.effective_delay_model_enabled:
        return {
            "effective_delay_configuration_coverage": 0.0,
            "uncalibrated_configuration_count": 0,
            "calibration_age_weighted_coverage": 0.0,
            "same_slot_calibration_sensing_conflict_count": 0,
            "configuration_transfer_mismatch_count": 0,
        }
    sensed: set[str] = set()
    calibrated: set[str] = set()
    by_id = {item.configuration_id: item for item in world.receive_chain_configurations}
    slot_roles: dict[tuple[int, str], set[str]] = {}
    for action in selected:
        configurations = action_receive_configurations(world, action)
        role = "calibration" if action.kind.startswith("calibration") else "sensing"
        for configuration in configurations:
            slot_roles.setdefault((action.resource_slot, configuration.configuration_id), set()).add(role)
            if role == "calibration":
                calibrated.add(configuration.configuration_id)
            else:
                sensed.add(configuration.configuration_id)
    covered = sensed.intersection(calibrated)
    transfer_mismatch = 0
    calibrated_groups = {by_id[item].transfer_group_id for item in calibrated}
    for item in sensed - calibrated:
        if by_id[item].transfer_group_id in calibrated_groups:
            transfer_mismatch += 1
    age_total = sum(by_id[item].calibration_age_s for item in sensed)
    covered_age = sum(by_id[item].calibration_age_s for item in covered)
    return {
        "effective_delay_configuration_coverage": len(covered) / max(1, len(sensed)),
        "uncalibrated_configuration_count": len(sensed - calibrated),
        "calibration_age_weighted_coverage": covered_age / max(age_total, 1e-30),
        "same_slot_calibration_sensing_conflict_count": sum(roles == {"calibration", "sensing"} for roles in slot_roles.values()),
        "configuration_transfer_mismatch_count": transfer_mismatch,
    }


def abstract_concrete_contract(world: PhysicalWorld) -> dict[str, Any]:
    """Machine-readable map from physical records to the abstract state model."""

    summary = effective_delay_world_summary(world)
    time_reference = time_reference_world_summary(world)
    return {
        "concrete_world": {
            "raw_objects": ["sensor_node", "waveform", "receive_chain_configuration", "radar_action"],
            "effective_delay_profile": world.effective_delay_profile,
            "configuration_count": summary["configuration_count"],
            "time_reference_profile": time_reference["profile"],
        },
        "abstract_state": {
            "blocks": ["target_position_velocity", "clock_bias", "frequency_bias", "configuration_effective_delay", "calibration_path_truth"],
            "task_measurement_dependency": "target_geometry_plus_clock_difference_plus_receive_configuration_delay",
            "calibration_dependency": "clock_difference_plus_configuration_delay_difference_plus_persistent_path_truth_difference",
            "epoch_gauge": "common_transmit_epoch_removed_by_reference_or_TDOA_contrast",
            "clock_prior": "correlated_common_epoch_plus_receiver_relative_residual",
        },
        "inference": {
            "joint_information": "J_prior_plus_sum_H_transpose_R_inverse_H",
            "task_information": "J_xx_minus_J_xeta_J_etaeta_inverse_J_etax",
            "configuration_transfer": "non_diagonal_prior_covariance_with_residual_floor",
        },
        "decision": {
            "exclusive_mode": world.calibration_resource_mode == "exclusive_receive_chain",
            "simultaneous_free_null": world.calibration_resource_mode == "simultaneous_free",
            "artificial_calibration_bonus": False,
        },
    }
