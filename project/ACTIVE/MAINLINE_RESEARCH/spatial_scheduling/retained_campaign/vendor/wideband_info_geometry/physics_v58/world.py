from __future__ import annotations

"""Deterministic SI-unit world and action-pool generation for the V60 model."""

import hashlib
from itertools import combinations
import math
from typing import Any

import numpy as np

from .schema import (
    Atmosphere,
    Jammer,
    MaterialSurface,
    PhysicalWorld,
    RadarAction,
    SensorNode,
    SpectrumEmitter,
    TargetState,
    Waveform,
)
from .antenna_geometry import array_peak_gain_linear, native_action_beam_metadata
from .effective_delay import (
    annotate_actions_with_effective_delay,
    build_effective_delay_components,
    waveform_frontend_group,
)
from .time_reference import apply_time_reference_profile


FAMILY_PARAMETERS: dict[str, dict[str, float]] = {
    "nominal_clear": {"altitude_m": 420.0, "clutter_sigma0": 2e-4, "roughness_m": 0.015, "clock_scale": 1.0, "attenuation_db_km": 0.008, "clutter_doppler_spread_hz": 4.0, "clutter_rejection_floor_db": 30.0},
    "suburban_low_altitude": {"altitude_m": 90.0, "clutter_sigma0": 2e-3, "roughness_m": 0.035, "clock_scale": 1.4, "attenuation_db_km": 0.010, "clutter_doppler_spread_hz": 10.0, "clutter_rejection_floor_db": 24.0},
    "urban_canyon": {"altitude_m": 75.0, "clutter_sigma0": 8e-3, "roughness_m": 0.080, "clock_scale": 2.5, "attenuation_db_km": 0.012, "clutter_doppler_spread_hz": 20.0, "clutter_rejection_floor_db": 18.0},
    "coastal_clutter": {"altitude_m": 140.0, "clutter_sigma0": 1.5e-2, "roughness_m": 0.050, "clock_scale": 1.8, "attenuation_db_km": 0.014, "clutter_doppler_spread_hz": 15.0, "clutter_rejection_floor_db": 20.0},
    "jammed_corridor": {"altitude_m": 160.0, "clutter_sigma0": 4e-3, "roughness_m": 0.040, "clock_scale": 2.0, "attenuation_db_km": 0.010, "clutter_doppler_spread_hz": 10.0, "clutter_rejection_floor_db": 24.0},
    "reference_contaminated_pbr": {"altitude_m": 110.0, "clutter_sigma0": 5e-3, "roughness_m": 0.045, "clock_scale": 2.2, "attenuation_db_km": 0.011, "clutter_doppler_spread_hz": 12.0, "clutter_rejection_floor_db": 22.0},
}

WORLD_RANDOM_STREAM_VERSION = (
    "v75_part2_physical_pulse_cube_and_action_compatibility_v1"
)
WORLD_GENERATOR_VERSION = (
    "v75_part2_balanced_nonredundant_action_catalog_v2"
)


def _component_seed(
    seed: int,
    family: str,
    world_index: int,
    component: str,
) -> int:
    """Derive collision-resistant deterministic streams for world components.

    Arithmetic combinations such as ``seed + world_index`` alias adjacent
    worlds. Domain-separated hashing keeps scene, node, timing, jammer and
    spectrum draws independent while preserving exact reproducibility.
    """

    payload = (
        f"{WORLD_RANDOM_STREAM_VERSION}|{int(seed)}|{str(family)}|"
        f"{int(world_index)}|{str(component)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _family(family: str) -> dict[str, float]:
    if family not in FAMILY_PARAMETERS:
        raise ValueError(f"unknown V60 world family {family!r}")
    return FAMILY_PARAMETERS[family]


def _waveforms() -> tuple[Waveform, ...]:
    # ``dwell_s`` is cumulative on-air time over the CPI, not the CPI itself.
    # Pulse widths are chosen below the minimum-range round trip of the
    # kilometre-scale scenarios; the pure CW channel is reserved for Doppler or
    # angle observables that do not claim delay/range identifiability.
    return (
        Waveform("L_lfm", "lfm", 1.30e9, 12e6, 12e6 / math.sqrt(12.0), 0.080, 72 * 2.0e-6, 900.0, 30e6),
        Waveform("S_phase", "phase_code", 2.80e9, 24e6, 7.5e6, 0.060, 78 * 1.5e-6, 1300.0, 60e6),
        Waveform("C_ofdm", "ofdm", 5.60e9, 80e6, 22e6, 0.045, 81 * 5.0e-6, 1800.0, 180e6),
        Waveform("X_lfm", "lfm", 9.60e9, 160e6, 160e6 / math.sqrt(12.0), 0.035, 84 * 0.5e-6, 2400.0, 360e6),
        Waveform("L_cw", "cw", 1.30e9, 100e3, 25e3, 0.160, 0.160, 500.0, 500e3),
    )


def _nodes(seed: int, family: str, target_xy: tuple[float, float], count: int) -> tuple[SensorNode, ...]:
    """Generate a heterogeneous network with explicit planar arrays and RF bands.

    Each logical node represents a site with colocated band-specific RF chains.
    Illuminator sites are assigned disjoint waveform groups when the network
    contains more than one such site; a reduced smoke-test network uses one
    multiband source so every passive action remains defined.
    """

    rng = np.random.default_rng(int(seed))
    params = _family(family)
    count = max(8, int(count))
    role_pattern = [
        "hybrid", "passive", "passive", "active", "illuminator",
        "calibration", "passive", "active", "illuminator", "hybrid",
    ]
    roles = [role_pattern[i % len(role_pattern)] for i in range(count)]
    illuminator_indices = [i for i, role in enumerate(roles) if role == "illuminator"]
    waveform_ids = tuple(w.waveform_id for w in _waveforms())
    low_band_ids = ("L_lfm", "L_cw", "S_phase")
    high_band_ids = ("C_ofdm", "X_lfm")
    nodes: list[SensorNode] = []
    illuminator_ordinal = 0
    for i, role in enumerate(roles):
        radius = 1800.0 + 350.0 * (i % 3) + rng.uniform(-120.0, 120.0)
        theta = 2.0 * math.pi * i / count + rng.uniform(-0.12, 0.12)
        height = 22.0 if role in {"passive", "calibration"} else 35.0 if role in {"active", "hybrid"} else 80.0
        position = (target_xy[0] + radius * math.cos(theta), target_xy[1] + radius * math.sin(theta), height + rng.uniform(-4.0, 4.0))
        clock_scale = params["clock_scale"]
        tx_power = 0.0
        if role in {"active", "hybrid"}:
            tx_power = 1200.0 if role == "active" else 650.0
        elif role == "illuminator":
            tx_power = 5000.0
        # Sensing sites use a cost-bounded 8x8 frontend.  Each multi-band site
        # contains colocated, band-specific half-wavelength arrays; it is not
        # one impossible fixed-spacing aperture reused from L through X band.
        if role in {"active", "passive", "hybrid"}:
            array_n, array_rows, array_cols = 64, 8, 8
        else:
            array_n, array_rows, array_cols = 8, 2, 4
        sensing_array_gain = array_peak_gain_linear(
            array_n, element_gain_db=5.0, aperture_efficiency=0.70
        )
        if role == "illuminator":
            if len(illuminator_indices) == 1:
                supported = waveform_ids
            else:
                supported = low_band_ids if illuminator_ordinal % 2 == 0 else high_band_ids
            illuminator_ordinal += 1
        elif role in {"active", "hybrid"}:
            supported = waveform_ids
        else:
            supported = ()
        nodes.append(SensorNode(
            node_id=f"n{i}",
            role=role,  # type: ignore[arg-type]
            position_m=position,
            tx_power_w=tx_power,
            tx_gain_linear=(
                sensing_array_gain
                if role in {"active", "hybrid"}
                else 10.0 ** (12.0 / 10.0)
                if role == "illuminator"
                else 1.0
            ),
            rx_gain_linear=(
                sensing_array_gain
                if role in {"active", "passive", "hybrid"}
                else 10.0 ** (12.0 / 10.0)
            ),
            noise_figure_db=4.0 + 0.4 * (i % 3),
            array_elements=array_n,
            array_rows=array_rows,
            array_cols=array_cols,
            array_yaw_rad=float(theta + math.pi / 2.0),
            element_spacing_m=0.0,
            element_spacing_wavelengths=0.5,
            supported_waveform_ids=supported,
            receiver_power_w=11.0 if role in {"active", "hybrid"} else 7.0,
            processing_energy_j_per_complex_sample=2.0e-9,
            power_amplifier_efficiency=0.38 if role in {"active", "hybrid"} else 0.45,
            clock_bias_s=float(rng.normal(0.0, 15e-9 * clock_scale)),
            clock_bias_std_s=25e-9 * clock_scale,
            clock_drift_s_per_s=float(rng.normal(0.0, 0.4e-9 * clock_scale)),
            clock_random_walk_std_s_sqrt_s=0.20e-9 * clock_scale,
            frequency_bias_hz=float(rng.normal(0.0, 3.0 * clock_scale)),
            frequency_bias_std_hz=5.0 * clock_scale,
            frequency_random_walk_std_hz_sqrt_s=0.15 * clock_scale,
            adc_bits=14 if role in {"active", "hybrid"} else 12,
            adc_full_scale_w=1e-3,
            adc_loading_fraction=0.05 if role in {"active", "hybrid"} else 0.035,
            integrated_phase_noise_rad2=1e-4 * clock_scale,
            timing_jitter_s=0.8e-12 * clock_scale,
            image_rejection_db=38.0 if role in {"passive", "hybrid"} else 42.0,
            communication_capacity_mbps=1000.0 if role != "illuminator" else 500.0,
            latency_s=0.004 + 0.002 * (i % 3),
            max_energy_j=42.0 if role in {"active", "hybrid"} else 10.0,
            full_duplex_capable=False,
            simultaneous_receive_channels=2 if role in {"passive", "hybrid"} else 1,
        ))
    return tuple(nodes)

def _jammer_bank(seed: int, family: str, center: tuple[float, float]) -> tuple[Jammer, ...]:
    if family not in {"jammed_corridor", "reference_contaminated_pbr", "urban_canyon"}:
        return ()
    rng = np.random.default_rng(int(seed))
    pos = (center[0] + 2900.0 + rng.uniform(-200.0, 200.0), center[1] - 2100.0 + rng.uniform(-200.0, 200.0), 95.0)
    if family == "jammed_corridor":
        return (
            Jammer("j_barrage", "barrage", pos, 1400.0, 8.0, 5.60e9, 300e6, 1.0),
            Jammer("j_spot", "spot", pos, 950.0, 10.0, 9.60e9, 160e6, 1.0),
            Jammer("j_drfm", "drfm_false_target", pos, 320.0, 12.0, 2.80e9, 30e6, 1.0, 0.8e-6, 120.0, 4),
        )
    if family == "reference_contaminated_pbr":
        return (Jammer("j_reference_mimic", "drfm_false_target", pos, 180.0, 8.0, 1.30e9, 15e6, 1.0, 1.4e-6, 65.0, 3),)
    return (Jammer("j_urban_swept", "swept", pos, 600.0, 6.0, 5.60e9, 500e6, 0.30, instantaneous_bandwidth_hz=20e6),)


def _spectrum_emitters(seed: int, family: str, center: tuple[float, float], waveforms: tuple[Waveform, ...]) -> tuple[SpectrumEmitter, ...]:
    rng = np.random.default_rng(int(seed))
    count = {
        "nominal_clear": 1,
        "suburban_low_altitude": 2,
        "urban_canyon": 5,
        "coastal_clutter": 3,
        "jammed_corridor": 3,
        "reference_contaminated_pbr": 4,
    }[family]
    kinds = ("communications", "broadcast", "uncooperative_radar", "industrial")
    emitters: list[SpectrumEmitter] = []
    family_offset = sum(ord(char) for char in family)
    for index in range(count):
        waveform = waveforms[(index + family_offset) % len(waveforms)]
        kind = kinds[(index + seed) % len(kinds)]
        radius = rng.uniform(1000.0, 5500.0)
        angle = rng.uniform(-math.pi, math.pi)
        position = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), rng.uniform(15.0, 180.0))
        bandwidth_scale = {"communications": 0.30, "broadcast": 0.80, "uncooperative_radar": 0.12, "industrial": 0.05}[kind]
        bandwidth = max(100e3, waveform.bandwidth_hz * bandwidth_scale)
        frequency_offset = rng.uniform(-0.35, 0.35) * waveform.bandwidth_hz
        power = {"communications": 80.0, "broadcast": 3000.0, "uncooperative_radar": 900.0, "industrial": 30.0}[kind]
        gain = {"communications": 2.0, "broadcast": 6.0, "uncooperative_radar": 12.0, "industrial": 1.5}[kind]
        duty = {"communications": 0.65, "broadcast": 1.0, "uncooperative_radar": 0.18, "industrial": 0.35}[kind]
        similarity = {"communications": 0.25, "broadcast": 0.45, "uncooperative_radar": 0.90, "industrial": 0.10}[kind]
        emitters.append(SpectrumEmitter(
            emitter_id=f"e{index}_{kind}",
            kind=kind,  # type: ignore[arg-type]
            position_m=position,
            tx_power_w=power * float(rng.uniform(0.75, 1.25)),
            antenna_gain_linear=gain,
            center_frequency_hz=waveform.carrier_hz + frequency_offset,
            bandwidth_hz=bandwidth,
            duty_cycle=duty,
            modulation_similarity=similarity,
        ))
    return tuple(emitters)


#: Measurement-vector cardinality per action kind.  Delay+Doppler families
#: return two scalars; TDOA/FDOA and calibration return one; AOA returns an
#: azimuth/elevation pair.  Communication load is proportional to this count,
#: so it must be derived from one table rather than re-typed at each call site.
ACTION_MEASUREMENT_COUNT: dict[str, int] = {
    "active_monostatic": 2,
    "active_bistatic": 2,
    "passive_bistatic": 2,
    "aoa": 2,
    "tdoa": 1,
    "fdoa": 1,
    "calibration_time": 1,
    "calibration_frequency": 1,
}


def derive_action_resources(
    nodes: tuple[SensorNode, ...],
    waveform: Waveform,
    kind: str,
    tx_node_id: str | None,
    rx_node_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Derive an action's SI resource costs from primitive node/waveform state.

    Single source of truth for the energy, emission and communication a sensing
    action consumes:

        E_tx      = P_tx * T_dwell / eta_pa                      [J]  (active only)
        E_rx(n)   = P_rx(n) * T_cpi
                    + e_sample(n) * f_s * T_dwell                [J]  per receiver
        E_total   = E_tx + sum_n E_rx(n)                         [J]
        C         = |rx| * (0.035 + 0.006 * m
                            + 0.002 * sqrt(B / 1 MHz))           [MB]
        Emission  = P_tx * T_dwell                               [J]  radiated

    with ``m`` the measurement-vector cardinality of the action kind.

    V68: extracted from ``_action_pool``.  A counterfactual world that changes a
    primitive (a waveform carrier, a node power, a receiver's sampling rate)
    must regenerate these derived costs with exactly the formulas the generator
    used, otherwise the factual and counterfactual arms silently differ in
    resource accounting and the paired comparison is no longer matched.  Having
    two copies of the formula is precisely how that happens, so there is now
    one.
    """

    by_id = {node.node_id: node for node in nodes}
    transmitter = by_id.get(tx_node_id) if tx_node_id is not None else None
    radiated_energy_j = 0.0
    electrical_tx_energy_j = 0.0
    if transmitter is not None and str(kind).startswith("active_"):
        radiated_energy_j = transmitter.tx_power_w * waveform.dwell_s
        electrical_tx_energy_j = radiated_energy_j / max(
            transmitter.power_amplifier_efficiency, 1e-6
        )
    receiver_energy_by_node: dict[str, float] = {}
    for node_id in rx_node_ids:
        receiver = by_id[node_id]
        receiver_energy_by_node[node_id] = (
            receiver.receiver_power_w * waveform.coherent_processing_interval_s
            + receiver.processing_energy_j_per_complex_sample
            * waveform.sample_rate_hz
            * waveform.dwell_s
        )
    measurement_count = ACTION_MEASUREMENT_COUNT.get(str(kind), 1)
    communication_mb = len(rx_node_ids) * (
        0.035
        + 0.006 * measurement_count
        + 0.002 * math.sqrt(waveform.bandwidth_hz / 1e6)
    )
    energy_by_node: dict[str, float] = dict(receiver_energy_by_node)
    if tx_node_id is not None and electrical_tx_energy_j > 0.0:
        energy_by_node[tx_node_id] = (
            energy_by_node.get(tx_node_id, 0.0) + electrical_tx_energy_j
        )
    return {
        "energy_cost_j": float(electrical_tx_energy_j + sum(receiver_energy_by_node.values())),
        "communication_cost_mb": float(communication_mb),
        "emission_exposure_j": float(radiated_energy_j),
        "energy_j_by_node": energy_by_node,
        "communication_mb_by_node": {
            node_id: communication_mb / max(1, len(rx_node_ids))
            for node_id in rx_node_ids
        },
        "measurement_count": int(measurement_count),
    }


def _action_pool(
    nodes: tuple[SensorNode, ...],
    waveforms: tuple[Waveform, ...],
    family: str,
    actions_per_world: int,
    target: TargetState,
) -> tuple[RadarAction, ...]:
    """Construct deterministic, band-compatible sensing and calibration actions."""

    active = [n for n in nodes if n.role in {"active", "hybrid"}]
    passive = [n for n in nodes if n.role in {"passive", "hybrid"}]
    illuminators = [n for n in nodes if n.role == "illuminator"]
    calibration = [n for n in nodes if n.role == "calibration"]
    params = _family(family)
    kinds = [
        "active_monostatic", "active_bistatic", "passive_bistatic", "tdoa",
        "fdoa", "aoa", "calibration_time", "calibration_frequency",
    ]
    kind_offsets = {kind: index for index, kind in enumerate(kinds)}
    waveform_by_id = {w.waveform_id: w for w in waveforms}
    active_pairs = [(node, waveform) for node in active for waveform in waveforms if node.supports_waveform(waveform.waveform_id)]
    passive_pairs = [(node, waveform) for node in illuminators for waveform in waveforms if node.supports_waveform(waveform.waveform_id)]
    if not active_pairs or not passive_pairs:
        raise ValueError("world requires at least one compatible active and passive illumination pair")

    kind_count = {
        kind: sum(
            1
            for action_index in range(int(actions_per_world))
            if kinds[action_index % len(kinds)] == kind
        )
        for kind in kinds
    }

    def balanced_choice(candidates: list[Any], kind: str, ordinal: int) -> Any:
        """Sample a kind-specific catalogue without the old modulo aliasing.

        The previous expression used the global action index both before and
        after division by the number of action kinds.  For 32 actions it mapped
        all four monostatic entries onto the same transmitter and waveform and
        concentrated every other sensing family on one receiver pair.  Evenly
        spaced catalogue indices retain deterministic generation while exposing
        the controller to genuinely different geometries and RF endpoints.
        """

        if not candidates:
            raise ValueError(f"no physically compatible candidates for {kind}")
        count = max(kind_count[kind], 1)
        base_index = min(
            len(candidates) - 1,
            int(math.floor(int(ordinal) * len(candidates) / count)),
        )
        index = (base_index + kind_offsets[kind]) % len(candidates)
        return candidates[index]

    actions: list[RadarAction] = []
    for i in range(int(actions_per_world)):
        kind = kinds[i % len(kinds)]
        tx_id: str | None = None
        reference_id: str | None = None
        rx_ids: tuple[str, ...]
        external_illumination = False
        kind_ordinal = i // len(kinds)
        if kind == "active_monostatic":
            candidates = [
                item
                for item in active_pairs
                if item[1].family in {"lfm", "phase_code"}
            ]
            tx, waveform = balanced_choice(
                candidates, kind, kind_ordinal
            )
            tx_id = tx.node_id
            rx_ids = (tx.node_id,)
        elif kind == "active_bistatic":
            tx = active[kind_ordinal % len(active)]
            compatible_waveforms = [
                waveform
                for waveform in waveforms
                if waveform.family != "cw"
                and tx.supports_waveform(waveform.waveform_id)
            ]
            waveform = compatible_waveforms[
                (kind_ordinal + kind_offsets[kind])
                % len(compatible_waveforms)
            ]
            compatible_receivers = [
                rx
                for rx in passive
                if rx.node_id != tx.node_id
                and rx.supports_waveform(waveform.waveform_id)
            ]
            rx = compatible_receivers[
                (2 * kind_ordinal + 1) % len(compatible_receivers)
            ]
            tx_id = tx.node_id
            rx_ids = (rx.node_id,)
        elif kind == "passive_bistatic":
            candidates = [
                (tx, waveform, rx)
                for rx in passive
                for tx, waveform in passive_pairs
                if waveform.family != "cw"
                if rx.node_id != tx.node_id
                and rx.supports_waveform(waveform.waveform_id)
            ]
            tx, waveform, rx = balanced_choice(
                candidates, kind, kind_ordinal
            )
            tx_id = tx.node_id
            reference_id = rx.node_id  # colocated direct-path reference channel
            rx_ids = (rx.node_id,)
            external_illumination = True
        elif kind in {"tdoa", "fdoa"}:
            candidates = [
                (tx, waveform, a, b)
                for a, b in combinations(passive, 2)
                for tx, waveform in passive_pairs
                if kind == "fdoa" or waveform.family != "cw"
                if a.node_id != tx.node_id
                and b.node_id != tx.node_id
                and a.supports_waveform(waveform.waveform_id)
                and b.supports_waveform(waveform.waveform_id)
            ]
            tx, waveform, a, b = balanced_choice(
                candidates, kind, kind_ordinal
            )
            rx_ids = (a.node_id, b.node_id)
            tx_id = tx.node_id
            external_illumination = True
        elif kind == "aoa":
            candidates = [
                (tx, waveform, rx)
                for tx, waveform in passive_pairs
                for rx in passive
                if rx.node_id != tx.node_id
                and rx.supports_waveform(waveform.waveform_id)
            ]
            tx, waveform, rx = balanced_choice(
                candidates, kind, kind_ordinal
            )
            rx_ids = (rx.node_id,)
            tx_id = tx.node_id
            reference_id = rx.node_id
            external_illumination = True
        else:
            if not calibration:
                raise ValueError("world requires a calibration reference node")
            cal_target = passive[kind_ordinal % len(passive)]
            reference = calibration[
                (kind_ordinal // max(1, len(passive))) % len(calibration)
            ]
            frontend_groups = sorted(
                {waveform_frontend_group(waveform) for waveform in waveforms}
            )
            frontend_group = frontend_groups[
                (kind_ordinal + kind_offsets[kind]) % len(frontend_groups)
            ]
            compatible_group_waveforms = [
                waveform
                for waveform in waveforms
                if waveform_frontend_group(waveform) == frontend_group
                and cal_target.supports_waveform(waveform.waveform_id)
            ]
            if not compatible_group_waveforms:
                raise ValueError(
                    f"node {cal_target.node_id} has no calibration waveform "
                    f"for {frontend_group}"
                )
            waveform = compatible_group_waveforms[
                (kind_ordinal // len(frontend_groups))
                % len(compatible_group_waveforms)
            ]
            reference_id = reference.node_id
            rx_ids = (cal_target.node_id,)

        tx = next((node for node in nodes if node.node_id == tx_id), None)
        resources = derive_action_resources(nodes, waveform, kind, tx_id, rx_ids)
        active_rf_energy = resources["emission_exposure_j"]
        energy = resources["energy_cost_j"]
        communication_mb = resources["communication_cost_mb"]
        energy_by_node = resources["energy_j_by_node"]

        beam_metadata = native_action_beam_metadata(
            nodes, waveform, kind, tx_id, rx_ids
        )
        metadata: dict[str, Any] = {
            "clutter_sigma0": params["clutter_sigma0"],
            **beam_metadata,
            "system_loss_linear": 2.2,
            "false_alarm_probability": 1e-6,
            "association_search_cells": 64 if kind in {"active_monostatic", "active_bistatic", "passive_bistatic"} else 32 if kind in {"tdoa", "fdoa"} else 25,
            "association_gate_sigma": 4.0,
            "clutter_texture_shape": 0.7 if family == "coastal_clutter" else 1.2 if family in {"urban_canyon", "reference_contaminated_pbr"} else 4.0,
            "clutter_doppler_spread_hz": params["clutter_doppler_spread_hz"],
            "clutter_rejection_floor_db": params["clutter_rejection_floor_db"],
            # Shared by the analytic target-cell projection and the R2 IQ
            # generator.  It represents range-correlated diffuse clutter at
            # the receiver input, not a coherent target-cell reflector.
            "clutter_fast_time_correlation": 0.82,
            "calibration_std_s": 2.0e-9,
            "calibration_std_hz": 0.4,
            "frequency_interval_hz": [waveform.carrier_hz - waveform.bandwidth_hz / 2.0, waveform.carrier_hz + waveform.bandwidth_hz / 2.0],
            "energy_j_by_node": energy_by_node,
            "communication_mb_by_node": resources["communication_mb_by_node"],
            "external_illumination": external_illumination,
            "reference_channel_node_id": reference_id,
            "illuminator_waveform_compatible": tx.supports_waveform(waveform.waveform_id) if tx is not None else True,
            "delay_identifiable_from_waveform": bool(
                waveform.family != "cw"
                or kind
                not in {
                    "active_monostatic",
                    "active_bistatic",
                    "passive_bistatic",
                    "tdoa",
                }
            ),
            "pulse_width_s": float(waveform.pulse_width_s),
            "duplexer_recovery_s": float(
                0.5e-6 if kind == "active_monostatic" else 0.0
            ),
            "direct_path_cancellation_db": float(
                100.0
                if kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"}
                else 60.0 if kind == "active_bistatic" else 0.0
            ),
            "direct_path_cancellation_model": (
                "reference_channel_adaptive_projection"
                if kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"}
                else "known_waveform_leakage_cancellation"
                if kind == "active_bistatic"
                else "duplexer_blanking_and_range_gate"
                if kind == "active_monostatic"
                else "not_applicable"
            ),
            "minimum_instrumented_range_m": float(
                0.5
                * 299_792_458.0
                * (waveform.pulse_width_s + 0.5e-6)
                if kind == "active_monostatic"
                else 0.0
            ),
            "target_range_at_generation_m": float(
                np.linalg.norm(
                    np.asarray(target.position_m, dtype=float)
                    - np.asarray(tx.position_m, dtype=float)
                )
                if kind == "active_monostatic" and tx is not None
                else 0.0
            ),
        }
        if (
            kind == "active_monostatic"
            and metadata["target_range_at_generation_m"]
            <= metadata["minimum_instrumented_range_m"]
        ):
            raise ValueError(
                "generated monostatic action places the target inside the "
                "pulse-width and duplexer-recovery blind range"
            )
        actions.append(RadarAction(
            action_id=i,
            kind=kind,  # type: ignore[arg-type]
            waveform_id=waveform.waveform_id,
            tx_node_id=tx_id,
            rx_node_ids=rx_ids,
            reference_node_id=reference_id,
            resource_slot=(i // 4) % 10,
            energy_cost_j=float(energy),
            communication_cost_mb=float(communication_mb),
            emission_exposure_j=float(active_rf_energy),
            allowed=True,
            metadata=metadata,
        ))
    return tuple(actions)

def generate_physical_world(
    seed: int,
    family: str,
    world_index: int,
    *,
    actions_per_world: int = 32,
    node_count: int = 8,
    budget: int = 5,
    effective_delay_profile: str = "disabled",
    time_reference_profile: str = "legacy_uncorrelated",
) -> PhysicalWorld:
    params = _family(family)
    rng = np.random.default_rng(
        _component_seed(seed, family, world_index, "scene")
    )
    center = (float(rng.uniform(-500.0, 500.0)), float(rng.uniform(-500.0, 500.0)))
    position = (center[0] + rng.uniform(-120.0, 120.0), center[1] + rng.uniform(-120.0, 120.0), params["altitude_m"] + rng.uniform(-12.0, 12.0))
    speed = 38.0 + rng.uniform(-8.0, 12.0); heading = rng.uniform(-math.pi, math.pi)
    velocity = (speed * math.cos(heading), speed * math.sin(heading), rng.uniform(-2.0, 2.0))
    swerling_cases = ("I", "II", "III", "IV")
    swerling_rng = np.random.default_rng(
        _component_seed(seed, family, world_index, "target_swerling_case")
    )
    swerling_index = int(swerling_rng.integers(0, len(swerling_cases)))
    target = TargetState(
        f"t{seed}_{world_index}",
        position,
        velocity,
        mean_rcs_m2=float(np.exp(rng.normal(math.log(0.8), 0.35))),
        swerling_case=swerling_cases[swerling_index],  # type: ignore[arg-type]
    )
    nodes = _nodes(
        _component_seed(seed, family, world_index, "nodes"),
        family,
        center,
        node_count,
    )
    nodes, time_reference = apply_time_reference_profile(
        nodes,
        seed=_component_seed(seed, family, world_index, "time_reference"),
        profile=time_reference_profile,
    )
    waveforms = _waveforms()
    actions = _action_pool(nodes, waveforms, family, actions_per_world, target)
    receive_chain_configurations = ()
    calibration_resource_mode = "disabled"
    if effective_delay_profile != "disabled":
        nodes, receive_chain_configurations, calibration_resource_mode = build_effective_delay_components(
            nodes,
            waveforms,
            seed=_component_seed(seed, family, world_index, "effective_delay"),
            profile=effective_delay_profile,
        )
        actions = annotate_actions_with_effective_delay(
            actions,
            nodes,
            receive_chain_configurations,
            profile=effective_delay_profile,
        )
    world = PhysicalWorld(
        world_id=f"v60_s{seed}_{family}_w{world_index}",
        seed=int(seed),
        family=family,
        nodes=nodes,
        waveforms=waveforms,
        target=target,
        actions=actions,
        jammers=_jammer_bank(
            _component_seed(seed, family, world_index, "jammers"),
            family,
            center,
        ),
        surface=MaterialSurface(
            relative_permittivity=15.0 if family == "coastal_clutter" else 5.0 if family != "urban_canyon" else 7.0,
            conductivity_s_per_m=3.5 if family == "coastal_clutter" else 0.02,
            roughness_rms_m=params["roughness_m"],
            polarization="vertical",
        ),
        atmosphere=Atmosphere(specific_attenuation_db_per_km=params["attenuation_db_km"]),
        max_schedule_size=int(budget),
        max_total_dwell_s=0.045,
        max_total_energy_j=50.0,
        max_concurrent_active=2,
        max_total_coherent_processing_s=0.60,
        max_total_emission_j=18.0,
        max_total_communication_mb=4.0,
        max_decision_latency_s=0.70,
        ground_height_m=0.0,
        spectrum_emitters=_spectrum_emitters(
            _component_seed(seed, family, world_index, "spectrum_emitters"),
            family,
            center,
            waveforms,
        ),
        receive_chain_configurations=receive_chain_configurations,
        effective_delay_profile=effective_delay_profile,
        calibration_resource_mode=calibration_resource_mode,  # type: ignore[arg-type]
        time_reference_profile=time_reference_profile,  # type: ignore[arg-type]
        clock_common_mode_std_s=float(time_reference["clock_common_mode_std_s"]),
        frequency_common_mode_std_hz=float(time_reference["frequency_common_mode_std_hz"]),
        clock_common_random_walk_std_s_sqrt_s=float(
            time_reference["clock_common_random_walk_std_s_sqrt_s"]
        ),
        frequency_common_random_walk_std_hz_sqrt_s=float(
            time_reference["frequency_common_random_walk_std_hz_sqrt_s"]
        ),
        simulator_fidelity=(
            "R2_5_band_averaged_analytic_link_effective_delay_and_reference_channel_model"
            if effective_delay_profile != "disabled"
            else "R2_5_band_averaged_analytic_link_and_reference_channel_model"
        ),
        external_channel_provenance={
            "position_prior_std_m": 180.0 if family == "nominal_clear" else 260.0,
            "velocity_prior_std_m_s": 28.0,
            "position_prior_std_m_xyz": [
                180.0 if family == "nominal_clear" else 260.0,
                180.0 if family == "nominal_clear" else 260.0,
                {
                    "nominal_clear": 80.0,
                    "suburban_low_altitude": 20.0,
                    "urban_canyon": 18.0,
                    "coastal_clutter": 30.0,
                    "jammed_corridor": 35.0,
                    "reference_contaminated_pbr": 25.0,
                }[family],
            ],
            "velocity_prior_std_m_s_xyz": [28.0, 28.0, 8.0],
            "target_rcs_prior_median_m2": 0.8,
            "target_rcs_prior_log_std": 0.35,
            "target_swerling_prior_case": "I",
            "target_scattering_prior_source": (
                "registered_low_altitude_air_target_class_not_realized_truth"
            ),
            "target_prior_anisotropy_source": (
                "registered_surveillance_volume_and_low_altitude_track_class"
            ),
            "propagation_tier": "band_averaged_two_ray_with_ground_clutter_and_external_solver_bridge",
            "atmospheric_model_source": "frequency_resolved_engineering_attenuation_or_ITU_R_P676",
            "rf_site_interpretation": "colocated_band_specific_frontends",
            "array_frontend_interpretation": (
                "8x8_sensing_UPA_per_band_with_half_wavelength_spacing"
            ),
            "array_element_gain_db": 5.0,
            "array_aperture_efficiency": 0.70,
            "beamwidth_source": (
                "uniform_vertical_rectangular_array_aperture_not_hard_coded"
            ),
            "fidelity_tier": "R2.5 analytic propagation and likelihood proxy",
            "measurement_or_full_wave_validation_present": False,
            "effective_delay_realism_enabled": effective_delay_profile != "disabled",
            "effective_delay_profile": effective_delay_profile,
            "effective_delay_evidence_tier": "synthetic_configuration_aware_factor_model",
            "time_reference_profile": time_reference_profile,
            "time_reference_common_and_residual_covariance_separated": True,
        },
    )
    world.validate()
    return world
