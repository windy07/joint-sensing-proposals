from __future__ import annotations

"""Representative finite-window complex-baseband IQ generation for V58."""

from dataclasses import dataclass
import math

import numpy as np

from wideband_info_geometry.signal_sim.finite_window_channel import causal_fractional_delay_window

from .measurement import ActionFactor, build_action_factor
from .schema import Jammer, PhysicalWorld, RadarAction, SensorNode, Waveform


@dataclass(frozen=True)
class IQRealization:
    reference: np.ndarray
    target_component: np.ndarray
    clutter_component: np.ndarray
    jammer_component: np.ndarray
    ambient_interference_component: np.ndarray
    thermal_noise_component: np.ndarray
    target_after_deterministic_lo: np.ndarray
    effective_disturbance_component: np.ndarray
    receiver_impairment_error: np.ndarray
    received: np.ndarray
    sample_rate_hz: float
    absolute_delay_s: float
    capture_start_sample: float
    inserted_delay_samples: float
    doppler_hz: float
    predicted_signal_power_w: float
    predicted_noise_plus_interference_power_w: float
    predicted_receiver_impairment_power_w: float
    predicted_ambient_interference_power_w: float
    agc_voltage_gain: float
    adc_clipping_fraction: float
    provenance: str


def synthesize_waveform(waveform: Waveform, n_samples: int, seed: int) -> np.ndarray:
    if n_samples <= 16:
        raise ValueError("IQ waveform requires more than 16 samples")
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float) / waveform.sample_rate_hz
    duration = n_samples / waveform.sample_rate_hz
    if waveform.family == "lfm":
        slope = waveform.bandwidth_hz / max(duration, 1e-12)
        signal = np.exp(1j * math.pi * slope * (t - duration / 2.0) ** 2)
    elif waveform.family == "phase_code":
        # Generate biphase chips at the declared chip rate (chip rate ~= occupied
        # bandwidth) rather than stretching a fixed 31-chip code across the whole
        # window.  The stretched form only realised a small fraction of the
        # declared RMS bandwidth, breaking R0<->R2 delay-Fisher consistency; the
        # multi-fidelity bridge (fidelity_bridge.py) checks this explicitly.
        samples_per_chip = max(1.0, waveform.sample_rate_hz / max(waveform.bandwidth_hz, 1.0))
        n_chips = int(np.ceil(n_samples / samples_per_chip)) + 1
        chips = rng.choice([-1.0, 1.0], size=n_chips)
        indices = np.minimum(n_chips - 1, (np.arange(n_samples) / samples_per_chip).astype(int))
        signal = chips[indices].astype(complex)
    elif waveform.family == "ofdm":
        carriers = np.arange(-16, 17)
        symbols = rng.choice([1.0, -1.0, 1j, -1j], size=carriers.size)
        spacing = waveform.bandwidth_hz / max(1, carriers.size)
        signal = sum(symbol * np.exp(2j * math.pi * carrier * spacing * t) for symbol, carrier in zip(symbols, carriers))
    else:
        signal = np.ones(n_samples, dtype=complex)
    power = float(np.mean(np.abs(signal) ** 2))
    return np.asarray(signal / math.sqrt(max(power, 1e-30)), dtype=np.complex128)


def _measurement_delay_and_doppler(factor: ActionFactor) -> tuple[float, float]:
    delay = 0.0; doppler = 0.0
    for name, value in zip(factor.measurement_names, factor.prediction):
        if "delay" in name or name == "tdoa_s":
            delay = float(value)
        elif "doppler" in name or name == "fdoa_hz":
            doppler = float(value)
    return delay, doppler


def _receiver_frontend(
    analog: np.ndarray,
    target: np.ndarray,
    *,
    sample_rate_hz: float,
    receiver: SensorNode,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    rx = receiver
    t = np.arange(analog.size, dtype=float) / float(sample_rate_hz)
    deterministic_lo = np.exp(2j * math.pi * float(rx.frequency_bias_hz) * t)
    phase_noise = rng.normal(0.0, math.sqrt(float(rx.integrated_phase_noise_rad2)), size=analog.size)
    random_phase = np.exp(1j * phase_noise)
    image_amplitude = 10.0 ** (-float(rx.image_rejection_db) / 20.0)
    image_phase = np.exp(1j * rng.uniform(-math.pi, math.pi))
    impaired_analog = (analog + image_amplitude * image_phase * np.conjugate(analog)) * deterministic_lo * random_phase

    input_power = max(float(np.mean(np.abs(impaired_analog) ** 2)), 1e-300)
    target_adc_power = float(rx.adc_loading_fraction) * float(rx.adc_full_scale_w)
    gain = math.sqrt(target_adc_power / input_power)
    adc = impaired_analog * gain
    rail = math.sqrt(float(rx.adc_full_scale_w) / 2.0)
    clipped_real = np.clip(np.real(adc), -rail, rail)
    clipped_imag = np.clip(np.imag(adc), -rail, rail)
    clipping_fraction = float(np.mean((np.abs(np.real(adc)) > rail) | (np.abs(np.imag(adc)) > rail)))
    levels = float(2 ** int(rx.adc_bits))
    step = 2.0 * rail / max(levels - 1.0, 1.0)
    quantized = np.round(clipped_real / step) * step + 1j * np.round(clipped_imag / step) * step
    received = quantized / gain
    deterministic_analog = analog * deterministic_lo
    target_reference = target * deterministic_lo
    impairment_error = received - deterministic_analog
    effective_disturbance = received - target_reference
    return received, target_reference, effective_disturbance, impairment_error, gain, clipping_fraction


def simulate_action_iq(
    world: PhysicalWorld,
    action: RadarAction,
    *,
    jammer: Jammer | None = None,
    n_samples: int = 4096,
    inserted_delay_samples: float | None = None,
    seed: int | None = None,
) -> IQRealization:
    if action.kind.startswith("calibration"):
        raise ValueError("calibration actions are scalar reference measurements, not radar IQ echoes")
    factor = build_action_factor(world, action, jammer)
    waveform = world.waveform(action.waveform_id)
    rng = np.random.default_rng(world.seed * 104729 + action.action_id * 1009 if seed is None else seed)
    reference = synthesize_waveform(waveform, n_samples, world.seed + action.action_id)
    absolute_delay_s, doppler_hz = _measurement_delay_and_doppler(factor)
    absolute_delay_samples = absolute_delay_s * waveform.sample_rate_hz
    if inserted_delay_samples is None:
        range_gate_lead = min(64.0, max(8.0, n_samples / 8.0))
        capture_start_sample = math.floor(absolute_delay_samples) - range_gate_lead
        in_window_delay_samples = absolute_delay_samples - capture_start_sample
    else:
        in_window_delay_samples = float(inserted_delay_samples)
        capture_start_sample = absolute_delay_samples - in_window_delay_samples
    if not 0.0 <= in_window_delay_samples < n_samples:
        raise ValueError("range-gated echo delay must fall inside the finite IQ capture")
    delayed = causal_fractional_delay_window(reference, absolute_delay_samples, n_samples, capture_start_sample=capture_start_sample)
    t = np.arange(n_samples, dtype=float) / waveform.sample_rate_hz
    target_unit = delayed * np.exp(2j * math.pi * doppler_hz * t)
    target_unit /= math.sqrt(max(float(np.mean(np.abs(target_unit) ** 2)), 1e-30))
    target = math.sqrt(max(factor.link_budget.signal_power_w, 0.0)) * target_unit
    # The compound-Gaussian texture is already sampled deterministically in
    # the action link budget.  Re-sampling it here would make IQ and likelihood
    # refer to different worlds and double-count clutter uncertainty.
    clutter_variance = factor.link_budget.clutter_power_w
    clutter = math.sqrt(max(clutter_variance, 0.0) / 2.0) * (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples))
    jammer_component = np.zeros(n_samples, dtype=np.complex128)
    if jammer is not None and factor.link_budget.jammer_power_w > 0.0:
        if jammer.kind == "drfm_false_target":
            false_delay = absolute_delay_samples + jammer.false_target_delay_s * waveform.sample_rate_hz
            false = causal_fractional_delay_window(reference, false_delay, n_samples, capture_start_sample=capture_start_sample) * np.exp(2j * math.pi * jammer.false_target_doppler_hz * t)
            false /= math.sqrt(max(float(np.mean(np.abs(false) ** 2)), 1e-30))
            jammer_component = math.sqrt(factor.link_budget.jammer_power_w) * false
        else:
            jammer_component = math.sqrt(factor.link_budget.jammer_power_w / 2.0) * (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples))
    ambient_interference_power = factor.link_budget.ambient_interference_power_w
    ambient_interference = math.sqrt(max(ambient_interference_power, 0.0) / 2.0) * (
        rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)
    )
    thermal_noise_power = factor.link_budget.thermal_noise_power_w
    noise = math.sqrt(max(thermal_noise_power, 0.0) / 2.0) * (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples))
    analog = target + clutter + jammer_component + ambient_interference + noise
    rx = world.node(action.rx_node_ids[0])
    received, target_reference, effective_disturbance, impairment_error, agc_gain, clipping_fraction = _receiver_frontend(
        analog,
        target,
        sample_rate_hz=waveform.sample_rate_hz,
        receiver=rx,
        rng=rng,
    )
    return IQRealization(
        reference=reference,
        target_component=target,
        clutter_component=clutter,
        jammer_component=jammer_component,
        ambient_interference_component=ambient_interference,
        thermal_noise_component=noise,
        target_after_deterministic_lo=target_reference,
        effective_disturbance_component=effective_disturbance,
        receiver_impairment_error=impairment_error,
        received=received,
        sample_rate_hz=waveform.sample_rate_hz,
        absolute_delay_s=absolute_delay_s,
        capture_start_sample=float(capture_start_sample),
        inserted_delay_samples=float(in_window_delay_samples),
        doppler_hz=doppler_hz,
        predicted_signal_power_w=factor.link_budget.signal_power_w,
        predicted_noise_plus_interference_power_w=(
            thermal_noise_power
            + factor.link_budget.receiver_impairment_noise_w
            + factor.link_budget.clutter_power_w
            + factor.link_budget.jammer_power_w
            + ambient_interference_power
        ),
        predicted_receiver_impairment_power_w=factor.link_budget.receiver_impairment_noise_w,
        predicted_ambient_interference_power_w=ambient_interference_power,
        agc_voltage_gain=agc_gain,
        adc_clipping_fraction=clipping_fraction,
        provenance="finite_window_range_gated_complex_baseband_from_V58_link_budget_not_field_IQ",
    )


def iq_power_report(realization: IQRealization) -> dict[str, float | str | bool]:
    power = lambda x: float(np.mean(np.abs(x) ** 2))
    signal = power(realization.target_after_deterministic_lo)
    disturbance = power(realization.effective_disturbance_component)
    impairment = power(realization.receiver_impairment_error)
    ambient_interference = power(realization.ambient_interference_component)
    return {
        "empirical_signal_power_w": signal,
        "predicted_signal_power_w": realization.predicted_signal_power_w,
        "empirical_disturbance_power_w": disturbance,
        "predicted_disturbance_power_w": realization.predicted_noise_plus_interference_power_w,
        "empirical_receiver_impairment_power_w": impairment,
        "predicted_receiver_impairment_power_w": realization.predicted_receiver_impairment_power_w,
        "empirical_ambient_interference_power_w": ambient_interference,
        "predicted_ambient_interference_power_w": realization.predicted_ambient_interference_power_w,
        "agc_voltage_gain": realization.agc_voltage_gain,
        "adc_clipping_fraction": realization.adc_clipping_fraction,
        "empirical_sinr_linear": signal / max(disturbance, 1e-300),
        "finite_samples": bool(np.isfinite(realization.received).all()),
        "provenance": realization.provenance,
    }
