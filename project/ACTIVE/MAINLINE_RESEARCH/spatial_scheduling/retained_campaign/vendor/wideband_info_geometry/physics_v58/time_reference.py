from __future__ import annotations

"""Correlated time/frequency-reference mechanisms for network radar worlds.

The node fields describe receiver-to-receiver residual uncertainty.  Common
reference uncertainty is stored once on the world and enters the full prior
and process covariance as a rank-one component.  This distinction is needed:
a shared trigger may have a large absolute epoch error while preserving a much
smaller relative TDOA error, and shared 10 MHz does not remove RF-chain delay.
"""

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .schema import SensorNode


TIME_REFERENCE_PROFILES: dict[str, dict[str, float]] = {
    "shared_frequency_and_trigger": {
        "clock_residual_std_s": 0.25e-9,
        "clock_common_mode_std_s": 8.0e-9,
        "clock_residual_random_walk_std_s_sqrt_s": 0.010e-9,
        "clock_common_random_walk_std_s_sqrt_s": 0.020e-9,
        "clock_drift_std_s_per_s": 0.020e-9,
        "frequency_residual_std_hz": 0.030,
        "frequency_common_mode_std_hz": 0.50,
        "frequency_residual_random_walk_std_hz_sqrt_s": 0.002,
        "frequency_common_random_walk_std_hz_sqrt_s": 0.010,
    },
    "disciplined_network": {
        "clock_residual_std_s": 2.0e-9,
        "clock_common_mode_std_s": 8.0e-9,
        "clock_residual_random_walk_std_s_sqrt_s": 0.080e-9,
        "clock_common_random_walk_std_s_sqrt_s": 0.050e-9,
        "clock_drift_std_s_per_s": 0.080e-9,
        "frequency_residual_std_hz": 0.20,
        "frequency_common_mode_std_hz": 0.50,
        "frequency_residual_random_walk_std_hz_sqrt_s": 0.020,
        "frequency_common_random_walk_std_hz_sqrt_s": 0.010,
    },
    "holdover": {
        "clock_residual_std_s": 25.0e-9,
        "clock_common_mode_std_s": 8.0e-9,
        "clock_residual_random_walk_std_s_sqrt_s": 0.40e-9,
        "clock_common_random_walk_std_s_sqrt_s": 0.10e-9,
        "clock_drift_std_s_per_s": 0.30e-9,
        "frequency_residual_std_hz": 1.5,
        "frequency_common_mode_std_hz": 0.50,
        "frequency_residual_random_walk_std_hz_sqrt_s": 0.080,
        "frequency_common_random_walk_std_hz_sqrt_s": 0.020,
    },
    "independent_oscillators": {
        "clock_residual_std_s": 80.0e-9,
        "clock_common_mode_std_s": 0.0,
        "clock_residual_random_walk_std_s_sqrt_s": 1.0e-9,
        "clock_common_random_walk_std_s_sqrt_s": 0.0,
        "clock_drift_std_s_per_s": 0.80e-9,
        "frequency_residual_std_hz": 6.0,
        "frequency_common_mode_std_hz": 0.0,
        "frequency_residual_random_walk_std_hz_sqrt_s": 0.20,
        "frequency_common_random_walk_std_hz_sqrt_s": 0.0,
    },
}


def time_reference_parameters(profile: str) -> dict[str, float]:
    if profile == "legacy_uncorrelated":
        return {
            "clock_common_mode_std_s": 0.0,
            "frequency_common_mode_std_hz": 0.0,
            "clock_common_random_walk_std_s_sqrt_s": 0.0,
            "frequency_common_random_walk_std_hz_sqrt_s": 0.0,
        }
    if profile not in TIME_REFERENCE_PROFILES:
        raise ValueError(f"unknown time-reference profile {profile!r}")
    return dict(TIME_REFERENCE_PROFILES[profile])


def apply_time_reference_profile(
    nodes: Sequence[SensorNode],
    *,
    seed: int,
    profile: str,
) -> tuple[tuple[SensorNode, ...], dict[str, float]]:
    """Instantiate common and receiver-residual oscillator states."""

    params = time_reference_parameters(profile)
    if profile == "legacy_uncorrelated":
        return tuple(nodes), params
    rng = np.random.default_rng(int(seed) * 1_000_033 + 71)
    common_clock = float(rng.normal(0.0, params["clock_common_mode_std_s"]))
    common_frequency = float(rng.normal(0.0, params["frequency_common_mode_std_hz"]))
    updated = tuple(
        replace(
            node,
            clock_bias_s=common_clock + float(rng.normal(0.0, params["clock_residual_std_s"])),
            clock_bias_std_s=params["clock_residual_std_s"],
            clock_drift_s_per_s=float(rng.normal(0.0, params["clock_drift_std_s_per_s"])),
            clock_random_walk_std_s_sqrt_s=params["clock_residual_random_walk_std_s_sqrt_s"],
            frequency_bias_hz=common_frequency + float(rng.normal(0.0, params["frequency_residual_std_hz"])),
            frequency_bias_std_hz=params["frequency_residual_std_hz"],
            frequency_random_walk_std_hz_sqrt_s=params["frequency_residual_random_walk_std_hz_sqrt_s"],
        )
        for node in nodes
    )
    return updated, params


def time_reference_profile_for_world(
    profiles: Sequence[str],
    *,
    seed: int,
    family: str,
    world_index: int,
) -> str:
    if not profiles:
        return "legacy_uncorrelated"
    for profile in profiles:
        time_reference_parameters(str(profile))
    offset = int(seed) * 3 + int(world_index) * 19 + sum(ord(character) for character in family)
    return str(profiles[offset % len(profiles)])


def time_reference_world_summary(world: Any) -> dict[str, float | str | bool]:
    clocks = [node.clock_bias_std_s for node in world.nodes]
    frequencies = [node.frequency_bias_std_hz for node in world.nodes]
    return {
        "profile": str(world.time_reference_profile),
        "relative_clock_residual_std_s_mean": float(np.mean(clocks)),
        "absolute_clock_std_s_mean": float(
            np.mean(np.sqrt(np.square(clocks) + world.clock_common_mode_std_s**2))
        ),
        "relative_frequency_residual_std_hz_mean": float(np.mean(frequencies)),
        "absolute_frequency_std_hz_mean": float(
            np.mean(np.sqrt(np.square(frequencies) + world.frequency_common_mode_std_hz**2))
        ),
        "clock_common_mode_std_s": float(world.clock_common_mode_std_s),
        "frequency_common_mode_std_hz": float(world.frequency_common_mode_std_hz),
        "common_epoch_is_not_relative_tdoa_uncertainty": True,
        "shared_reference_is_not_receive_chain_calibration": True,
    }
