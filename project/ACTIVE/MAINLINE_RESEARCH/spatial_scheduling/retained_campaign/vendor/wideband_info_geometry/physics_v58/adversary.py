from __future__ import annotations

"""Electronic-attack scenario construction for robust V58 planning."""

import math

from .schema import Jammer, PhysicalWorld


def canonical_adversary_scenarios(world: PhysicalWorld) -> list[Jammer | None]:
    """Return nominal plus physically parameterized jammer tactics.

    Existing world jammers are retained.  If no jammer bank was supplied, a
    deterministic bank is placed outside the sensor perimeter.  This bank is a
    threat model, not ground truth about a specific real jammer.
    """

    cx = sum(n.position_m[0] for n in world.nodes) / max(1, len(world.nodes))
    cy = sum(n.position_m[1] for n in world.nodes) / max(1, len(world.nodes))
    waveforms = sorted(world.waveforms, key=lambda waveform: waveform.carrier_hz)
    mid_waveform = waveforms[len(waveforms) // 2]
    low_edge = min(w.carrier_hz - w.bandwidth_hz / 2.0 for w in waveforms)
    high_edge = max(w.carrier_hz + w.bandwidth_hz / 2.0 for w in waveforms)
    sweep_center = 0.5 * (low_edge + high_edge)
    sweep_span = high_edge - low_edge
    position = (cx + 2500.0, cy + 1400.0, 120.0)
    global_sweep = Jammer(
        "frequency_agile_swept_global",
        "swept",
        position,
        1100.0,
        8.0,
        sweep_center,
        sweep_span,
        0.35,
        instantaneous_bandwidth_hz=20e6,
    )
    if world.jammers:
        return [None, *world.jammers, global_sweep]
    return [
        None,
        Jammer("barrage_default", "barrage", position, 1600.0, 8.0, sweep_center, sweep_span, 1.0),
        Jammer("spot_default", "spot", position, 800.0, 10.0, mid_waveform.carrier_hz, max(1e6, mid_waveform.bandwidth_hz / 4.0), 1.0),
        global_sweep,
        Jammer("drfm_default", "drfm_false_target", position, 250.0, 12.0, mid_waveform.carrier_hz, max(1e6, mid_waveform.bandwidth_hz / 4.0), 1.0, 1.2e-6, 140.0, 3),
    ]


def jammer_budget_watt_seconds(jammer: Jammer, horizon_s: float) -> float:
    if jammer.kind == "none":
        return 0.0
    return float(jammer.tx_power_w * jammer.duty_cycle * max(float(horizon_s), 0.0))


def adversary_diversity_report(scenarios: list[Jammer | None]) -> dict[str, object]:
    real = [x for x in scenarios if x is not None and x.kind != "none"]
    return {
        "scenario_count_including_nominal": len(scenarios),
        "jammer_kinds": sorted({x.kind for x in real}),
        "has_noise_like_attack": any(x.kind in {"barrage", "spot", "swept"} for x in real),
        "has_deceptive_attack": any(x.kind == "drfm_false_target" for x in real),
        "finite_parameters": all(math.isfinite(x.tx_power_w) and math.isfinite(x.bandwidth_hz) for x in real),
        "claim_boundary": "finite_tactic_bank_not_universal_electronic_warfare_model",
    }
