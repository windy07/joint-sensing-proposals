from __future__ import annotations

"""Propagation and link-budget equations for the V60 analytic fidelity tier."""

import cmath
import math
from typing import Iterable

import numpy as np

from wideband_info_geometry.small_symmetric_linalg import (
    platform_gauss_legendre,
    platform_matmul,
)

from .schema import Atmosphere, MaterialSurface, SensorNode

C0_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380_649e-23
VACUUM_PERMITTIVITY_F_M = 8.854_187_8128e-12


def distance_m(a: Iterable[float], b: Iterable[float]) -> float:
    d = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("propagation endpoints must be distinct and finite")
    return d


def unit_vector(source: Iterable[float], destination: Iterable[float]) -> np.ndarray:
    delta = np.asarray(destination, dtype=float) - np.asarray(source, dtype=float)
    return delta / distance_m(source, destination)


def wavelength_m(carrier_hz: float) -> float:
    if carrier_hz <= 0.0:
        raise ValueError("carrier_hz must be positive")
    return C0_M_S / float(carrier_hz)


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 10.0)


def linear_to_db(value: float, floor: float = 1e-300) -> float:
    return 10.0 * math.log10(max(float(value), floor))


def atmospheric_specific_attenuation_db_per_km(carrier_hz: float | None, atmosphere: Atmosphere) -> float:
    if atmosphere.attenuation_source == "ITU_R_P676_exact":
        if carrier_hz is None or carrier_hz <= 0.0:
            raise ValueError("ITU-R P.676 attenuation requires a positive carrier frequency")
        try:
            from itur.models.itu676 import gamma_exact  # type: ignore
        except Exception as exc:
            raise RuntimeError("attenuation_source=ITU_R_P676_exact requires the optional 'itur' package") from exc
        quantity = gamma_exact(carrier_hz / 1e9, atmosphere.pressure_hpa, atmosphere.water_vapor_density_g_m3, atmosphere.temperature_k)
        value = float(getattr(quantity, "value", quantity))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("ITU-R P.676 returned invalid specific attenuation")
        return value
    return float(atmosphere.specific_attenuation_db_per_km)


def atmospheric_power_factor(distance: float, atmosphere: Atmosphere, carrier_hz: float | None = None) -> float:
    loss_db = atmospheric_specific_attenuation_db_per_km(carrier_hz, atmosphere) * float(distance) / 1000.0
    return db_to_linear(-loss_db)


def free_space_power_gain(distance: float, carrier_hz: float, atmosphere: Atmosphere | None = None) -> float:
    lam = wavelength_m(carrier_hz)
    gain = (lam / (4.0 * math.pi * float(distance))) ** 2
    if atmosphere is not None:
        gain *= atmospheric_power_factor(distance, atmosphere, carrier_hz)
    return float(gain)


def complex_relative_permittivity(surface: MaterialSurface, carrier_hz: float) -> complex:
    omega = 2.0 * math.pi * float(carrier_hz)
    return complex(surface.relative_permittivity, -surface.conductivity_s_per_m / (omega * VACUUM_PERMITTIVITY_F_M))


def specular_roughness_field_factor(
    grazing_sine: float,
    carrier_hz: float,
    roughness_rms_m: float,
) -> float:
    """Ament specular-scattering *field* attenuation factor.

    Reference: W. S. Ament, "Toward a theory of reflection by a rough surface",
    Proc. IRE 41(1):142-146, 1953; Beckmann & Spizzichino, *The Scattering of
    Electromagnetic Waves from Rough Surfaces*, 1963, ch. 5; Barton, *Radar
    Equations for Modern Radar*, 2013, eq. (6.19).

    With Rayleigh roughness phase ``Delta_phi = 4*pi*sigma*sin(psi)/lambda``
    (``psi`` the grazing angle, ``sigma`` the surface height RMS), a Gaussian
    height distribution gives the coherent *field* factor

        rho_s = E[exp(j*Delta_phi)] = exp(-Delta_phi**2 / 2)
              = exp(-8*pi**2 * (sigma*sin(psi)/lambda)**2).

    The corresponding *power* factor is ``rho_s**2``.  Returning ``rho_s`` here
    is required because the caller multiplies it into the complex Fresnel field
    coefficient and only later squares the summed field.

    Units: ``grazing_sine`` dimensionless in [0,1]; ``carrier_hz`` Hz;
    ``roughness_rms_m`` m.  Returns a dimensionless factor in (0, 1].
    Limits: ``sigma -> 0`` or ``psi -> 0`` give 1 (smooth / grazing specular);
    ``sigma/lambda -> inf`` gives 0 (fully diffuse).
    """

    sine = min(max(float(grazing_sine), 0.0), 1.0)
    sigma = max(float(roughness_rms_m), 0.0)
    if sigma == 0.0 or sine == 0.0:
        return 1.0
    rayleigh_phase = 4.0 * math.pi * sigma * sine / wavelength_m(carrier_hz)
    return float(math.exp(-0.5 * rayleigh_phase**2))


def fresnel_reflection_coefficient(
    incidence_from_normal_rad: float,
    carrier_hz: float,
    surface: MaterialSurface,
) -> complex:
    """Return the flat-interface electric-field reflection coefficient.

    ``incidence_from_normal_rad`` is measured from the surface normal, so
    ``cos(theta) = sin(psi)`` with ``psi`` the grazing angle.  The Ament
    roughness factor is applied to the *field*, not to the power (see
    :func:`specular_roughness_field_factor`).
    """

    theta = min(max(float(incidence_from_normal_rad), 0.0), math.pi / 2.0 - 1e-9)
    eps = complex_relative_permittivity(surface, carrier_hz)
    sin2 = math.sin(theta) ** 2
    root = cmath.sqrt(eps - sin2)
    c = math.cos(theta)
    if surface.polarization == "horizontal":
        gamma = (c - root) / (c + root)
    else:
        gamma = (eps * c - root) / (eps * c + root)
    roughness = specular_roughness_field_factor(c, carrier_hz, surface.roughness_rms_m)
    return gamma * roughness


def two_ray_complex_gain(
    source_m: Iterable[float],
    receiver_m: Iterable[float],
    carrier_hz: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    ground_height_m: float = 0.0,
) -> complex:
    """Coherent direct plus one specular ground-reflection field gain.

    The image method is valid for a locally planar surface at height
    ``ground_height_m``.  It is an analytic R2.5 mechanism, not a replacement
    for urban ray tracing or full-wave EM.

    Validity domain: flat locally-planar reflecting plane ``z = ground_height_m``;
    both endpoints strictly above it; no earth curvature or divergence factor
    (valid while the horizontal separation stays well inside the 4/3-earth
    horizon, i.e. ``d << sqrt(2*k_e*R_e*h)``); single specular bounce only.
    """

    src = np.asarray(source_m, dtype=float)
    rx = np.asarray(receiver_m, dtype=float)
    ground = float(ground_height_m)
    direct_r = distance_m(src, rx)
    image = src.copy()
    image[2] = 2.0 * ground - image[2]
    reflected_r = distance_m(image, rx)
    horizontal = float(np.linalg.norm(src[:2] - rx[:2]))
    height_sum = (src[2] - ground) + (rx[2] - ground)
    incidence = math.atan2(max(horizontal, 1e-12), max(height_sum, 1e-12))
    gamma = fresnel_reflection_coefficient(incidence, carrier_hz, surface)
    k = 2.0 * math.pi / wavelength_m(carrier_hz)
    direct_amp = math.sqrt(free_space_power_gain(direct_r, carrier_hz, atmosphere))
    reflected_amp = math.sqrt(free_space_power_gain(reflected_r, carrier_hz, atmosphere))
    return direct_amp * cmath.exp(-1j * k * direct_r) + gamma * reflected_amp * cmath.exp(-1j * k * reflected_r)


def two_ray_power_gain(
    source_m: Iterable[float],
    receiver_m: Iterable[float],
    carrier_hz: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    ground_height_m: float = 0.0,
) -> float:
    return float(
        abs(
            two_ray_complex_gain(
                source_m, receiver_m, carrier_hz, surface, atmosphere, ground_height_m
            )
        )
        ** 2
    )


def effective_two_ray_power_gain(
    source_m: Iterable[float],
    receiver_m: Iterable[float],
    carrier_hz: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    ground_height_m: float = 0.0,
) -> float:
    """Coherent two-ray gain plus an explicit unresolved diffuse component.

    The diffuse term replaces the old hidden floor/cap.  It is an incoherent
    power fraction relative to the direct path and is therefore auditable in
    the world schema instead of silently clipping a physical fade.
    """

    direct = free_space_power_gain(distance_m(source_m, receiver_m), carrier_hz, atmosphere)
    coherent = two_ray_power_gain(
        source_m, receiver_m, carrier_hz, surface, atmosphere, ground_height_m
    )
    return float(coherent + surface.unresolved_diffuse_power_fraction * direct)


def band_averaged_two_ray_power_gain(
    source_m: Iterable[float],
    receiver_m: Iterable[float],
    carrier_hz: float,
    bandwidth_hz: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    *,
    frequency_samples: int = 24,
    ground_height_m: float = 0.0,
) -> tuple[float, float]:
    """Return band-averaged two-ray and direct-path power gains.

    Wideband matched filtering integrates energy over frequency. Gauss-Legendre
    quadrature averages coherent two-ray power across the occupied spectrum and
    retains the narrowband limit.
    """

    bandwidth = max(0.0, float(bandwidth_hz))
    if bandwidth <= max(1.0, 1e-6 * float(carrier_hz)):
        direct = free_space_power_gain(distance_m(source_m, receiver_m), carrier_hz, atmosphere)
        return (
            effective_two_ray_power_gain(
                source_m, receiver_m, carrier_hz, surface, atmosphere, ground_height_m
            ),
            direct,
        )
    count = max(8, int(frequency_samples))
    nodes, weights = platform_gauss_legendre(count)
    frequencies = carrier_hz + 0.5 * bandwidth * nodes
    if float(np.min(frequencies)) <= 0.0:
        raise ValueError("occupied band must remain at positive frequency")
    two_ray = np.asarray([
        effective_two_ray_power_gain(
            source_m, receiver_m, float(frequency), surface, atmosphere, ground_height_m
        )
        for frequency in frequencies
    ])
    propagation_distance = distance_m(source_m, receiver_m)
    direct = np.asarray([
        free_space_power_gain(propagation_distance, float(frequency), atmosphere)
        for frequency in frequencies
    ])
    # Gauss-Legendre weights integrate over [-1,1]; division by two returns
    # the average over the occupied rectangular spectrum.
    return (
        float(0.5 * platform_matmul(weights, two_ray)),
        float(0.5 * platform_matmul(weights, direct)),
    )


def thermal_noise_power_w(
    bandwidth_hz: float,
    noise_figure_db: float,
    atmosphere: Atmosphere,
) -> float:
    if bandwidth_hz <= 0.0:
        raise ValueError("noise bandwidth must be positive")
    return BOLTZMANN_J_K * atmosphere.temperature_k * float(bandwidth_hz) * db_to_linear(noise_figure_db)


def monostatic_received_power_w(
    tx: SensorNode,
    target_position_m: Iterable[float],
    carrier_hz: float,
    rcs_m2: float,
    system_loss_linear: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    bandwidth_hz: float = 0.0,
    apply_two_ray: bool = True,
    ground_height_m: float = 0.0,
) -> float:
    if tx.tx_power_w <= 0.0 or rcs_m2 <= 0.0 or system_loss_linear < 1.0:
        return 0.0
    r = distance_m(tx.position_m, target_position_m)
    lam = wavelength_m(carrier_hz)
    base = tx.tx_power_w * tx.tx_gain_linear * tx.rx_gain_linear * lam**2 * float(rcs_m2)
    base /= (4.0 * math.pi) ** 3 * r**4 * system_loss_linear
    if apply_two_ray:
        two_ray_gain, direct_gain = band_averaged_two_ray_power_gain(
            tx.position_m,
            target_position_m,
            carrier_hz,
            bandwidth_hz,
            surface,
            atmosphere,
            ground_height_m=ground_height_m,
        )
        path_factor = two_ray_gain / max(direct_gain, 1e-300)
    else:
        path_factor = 1.0
    return float(base * atmospheric_power_factor(2.0 * r, atmosphere, carrier_hz) * path_factor**2)


def bistatic_received_power_w(
    tx: SensorNode,
    rx: SensorNode,
    target_position_m: Iterable[float],
    carrier_hz: float,
    bistatic_rcs_m2: float,
    system_loss_linear: float,
    surface: MaterialSurface,
    atmosphere: Atmosphere,
    bandwidth_hz: float = 0.0,
    apply_two_ray: bool = True,
    ground_height_m: float = 0.0,
) -> float:
    if tx.tx_power_w <= 0.0 or bistatic_rcs_m2 <= 0.0 or system_loss_linear < 1.0:
        return 0.0
    rt = distance_m(tx.position_m, target_position_m)
    rr = distance_m(rx.position_m, target_position_m)
    lam = wavelength_m(carrier_hz)
    power = tx.tx_power_w * tx.tx_gain_linear * rx.rx_gain_linear * lam**2 * float(bistatic_rcs_m2)
    power /= (4.0 * math.pi) ** 3 * rt**2 * rr**2 * system_loss_linear
    if apply_two_ray:
        tx_two_ray, tx_direct = band_averaged_two_ray_power_gain(
            tx.position_m,
            target_position_m,
            carrier_hz,
            bandwidth_hz,
            surface,
            atmosphere,
            ground_height_m=ground_height_m,
        )
        rx_two_ray, rx_direct = band_averaged_two_ray_power_gain(
            target_position_m,
            rx.position_m,
            carrier_hz,
            bandwidth_hz,
            surface,
            atmosphere,
            ground_height_m=ground_height_m,
        )
        tx_factor = tx_two_ray / max(tx_direct, 1e-300)
        rx_factor = rx_two_ray / max(rx_direct, 1e-300)
    else:
        tx_factor = 1.0
        rx_factor = 1.0
    return float(power * atmospheric_power_factor(rt + rr, atmosphere, carrier_hz) * tx_factor * rx_factor)


def one_way_interference_power_w(
    tx_power_w: float,
    tx_gain_linear: float,
    rx: SensorNode,
    source_position_m: Iterable[float],
    carrier_hz: float,
    atmosphere: Atmosphere,
) -> float:
    if tx_power_w <= 0.0:
        return 0.0
    r = distance_m(source_position_m, rx.position_m)
    return float(tx_power_w * tx_gain_linear * rx.rx_gain_linear * free_space_power_gain(r, carrier_hz, atmosphere))


def spectral_overlap_fraction(center_a_hz: float, bandwidth_a_hz: float, center_b_hz: float, bandwidth_b_hz: float) -> float:
    """Fraction of receiver A's occupied band overlapped by source B."""

    a0, a1 = center_a_hz - bandwidth_a_hz / 2.0, center_a_hz + bandwidth_a_hz / 2.0
    b0, b1 = center_b_hz - bandwidth_b_hz / 2.0, center_b_hz + bandwidth_b_hz / 2.0
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return float(overlap / max(float(bandwidth_a_hz), 1e-30))


def flat_spectrum_intercept_fraction(receiver_center_hz: float, receiver_bandwidth_hz: float, source_center_hz: float, source_bandwidth_hz: float) -> float:
    """Fraction of a flat-spectrum source's total power entering a receiver.

    Source transmit power is integrated over ``source_bandwidth_hz``.  Power
    conservation therefore requires overlap/source-bandwidth, not
    overlap/receiver-bandwidth.
    """

    r0, r1 = receiver_center_hz - receiver_bandwidth_hz / 2.0, receiver_center_hz + receiver_bandwidth_hz / 2.0
    s0, s1 = source_center_hz - source_bandwidth_hz / 2.0, source_center_hz + source_bandwidth_hz / 2.0
    overlap = max(0.0, min(r1, s1) - max(r0, s0))
    return float(np.clip(overlap / max(float(source_bandwidth_hz), 1e-30), 0.0, 1.0))


def swept_spectrum_average_intercept_fraction(
    receiver_center_hz: float,
    receiver_bandwidth_hz: float,
    sweep_center_hz: float,
    sweep_span_hz: float,
    instantaneous_bandwidth_hz: float,
    *,
    quadrature_points: int = 257,
) -> float:
    """Time-average captured power of a uniformly swept narrowband source."""

    span = float(sweep_span_hz)
    instant = float(instantaneous_bandwidth_hz)
    if min(receiver_bandwidth_hz, span, instant) <= 0.0 or instant > span:
        raise ValueError("receiver, sweep, and instantaneous bandwidths must be positive with instantaneous <= span")
    half_travel = max(0.0, 0.5 * (span - instant))
    if half_travel == 0.0:
        return flat_spectrum_intercept_fraction(receiver_center_hz, receiver_bandwidth_hz, sweep_center_hz, instant)
    centers = np.linspace(sweep_center_hz - half_travel, sweep_center_hz + half_travel, max(3, int(quadrature_points)))
    captured = [flat_spectrum_intercept_fraction(receiver_center_hz, receiver_bandwidth_hz, float(center), instant) for center in centers]
    return float(np.mean(captured))
