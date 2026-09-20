from __future__ import annotations

"""Shared, unit-bearing geometry for the generated rectangular arrays.

The operational model uses a vertical rectangular aperture at each site.  The
horizontal array axis is declared by ``SensorNode.array_yaw_rad`` and the
vertical axis is the local +z direction.  A zero metric spacing together with
``element_spacing_wavelengths`` denotes a colocated, band-specific frontend;
this is required for the generated multi-band sites and is not an invitation to
reinterpret one physical antenna at every carrier.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .propagation import wavelength_m
from .schema import SensorNode, Waveform


def resolved_element_spacing_m(node: SensorNode, carrier_hz: float) -> float:
    """Return the physical spacing of the frontend used at ``carrier_hz``."""

    if node.element_spacing_wavelengths is not None:
        return float(node.element_spacing_wavelengths) * wavelength_m(carrier_hz)
    if node.element_spacing_m > 0.0:
        return float(node.element_spacing_m)
    if node.array_elements == 1:
        return 0.0
    raise ValueError(
        f"node {node.node_id} has a multi-element array without a spacing law"
    )


def rectangular_array_hpbw_rad(
    node: SensorNode,
    carrier_hz: float,
) -> tuple[float, float]:
    r"""Approximate broadside full 3-dB widths of a uniform rectangular array.

    For a uniformly weighted broadside aperture, the full half-power width of
    one separable dimension is approximately ``0.886 lambda / (N d)``.  The
    expression is accurate in the electrically large regime and is clipped to
    a hemisphere when the aperture is too small for the asymptotic formula.
    """

    rows, cols = node.resolved_array_shape
    spacing = resolved_element_spacing_m(node, carrier_hz)
    wavelength = wavelength_m(carrier_hz)

    def width(count: int) -> float:
        if count <= 1 or spacing <= 0.0:
            return math.pi
        return float(
            np.clip(0.886 * wavelength / (count * spacing), math.radians(0.5), math.pi)
        )

    return width(cols), width(rows)


def vertical_rectangular_element_positions_m(
    node: SensorNode,
    carrier_hz: float,
) -> np.ndarray:
    """Element coordinates for the vertical aperture in world axes."""

    rows, cols = node.resolved_array_shape
    spacing = resolved_element_spacing_m(node, carrier_hz)
    horizontal = np.asarray(
        [math.cos(node.array_yaw_rad), math.sin(node.array_yaw_rad), 0.0],
        dtype=float,
    )
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=float)
    points: list[np.ndarray] = []
    for index in range(node.array_elements):
        row, col = divmod(index, cols)
        horizontal_offset = (col - (cols - 1) / 2.0) * spacing
        vertical_offset = (row - (rows - 1) / 2.0) * spacing
        points.append(horizontal_offset * horizontal + vertical_offset * vertical)
    return np.asarray(points, dtype=float)


def inward_horizontal_boresight(node: SensorNode) -> np.ndarray:
    """Return the broadside unit vector for the declared vertical aperture."""

    # z cross horizontal points to the inward normal for generated ring sites.
    vector = np.asarray(
        [-math.sin(node.array_yaw_rad), math.cos(node.array_yaw_rad), 0.0],
        dtype=float,
    )
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def array_peak_gain_linear(
    elements: int,
    *,
    element_gain_db: float = 5.0,
    aperture_efficiency: float = 0.70,
) -> float:
    """Peak gain from element gain, coherent element count and efficiency."""

    if elements < 1 or not 0.0 < aperture_efficiency <= 1.0:
        raise ValueError("array gain requires positive elements and valid efficiency")
    return float(
        aperture_efficiency
        * int(elements)
        * 10.0 ** (float(element_gain_db) / 10.0)
    )


def native_action_beam_metadata(
    nodes: Sequence[SensorNode],
    waveform: Waveform,
    kind: str,
    tx_node_id: str | None,
    rx_node_ids: Sequence[str],
) -> dict[str, Any]:
    """Build endpoint-specific native beam widths for one sensing action."""

    node_by_id = {node.node_id: node for node in nodes}
    endpoint_ids = list(dict.fromkeys(str(item) for item in rx_node_ids))
    if kind.startswith("active_") and tx_node_id is not None:
        endpoint_ids.append(str(tx_node_id))
        endpoint_ids = list(dict.fromkeys(endpoint_ids))
    azimuth: dict[str, float] = {}
    elevation: dict[str, float] = {}
    for node_id in endpoint_ids:
        if node_id not in node_by_id:
            raise ValueError(f"beam endpoint {node_id!r} is absent from the network")
        az, el = rectangular_array_hpbw_rad(
            node_by_id[node_id], waveform.carrier_hz
        )
        azimuth[node_id] = float(az)
        elevation[node_id] = float(el)
    scalar_az = min(azimuth.values(), default=math.pi)
    scalar_el = min(elevation.values(), default=math.pi)
    return {
        "beamwidth_rad": scalar_az,
        "beam_azimuth_width_rad": scalar_az,
        "beam_elevation_width_rad": scalar_el,
        "native_beam_azimuth_width_rad_by_node": azimuth,
        "native_beam_elevation_width_rad_by_node": elevation,
        "native_beam_model": (
            "uniform_vertical_rectangular_array_0.886_lambda_over_Nd_HPBW"
        ),
    }


def endpoint_beam_parameters(
    metadata: Mapping[str, Any],
    node_id: str,
) -> tuple[float, float, float, float]:
    """Resolve commanded width, sidelobe floor and peak-gain scale."""

    commanded_azimuth = metadata.get(
        "commanded_beam_azimuth_width_rad_by_node", {}
    )
    commanded_elevation = metadata.get(
        "commanded_beam_elevation_width_rad_by_node", {}
    )
    native_azimuth = metadata.get("native_beam_azimuth_width_rad_by_node", {})
    native_elevation = metadata.get(
        "native_beam_elevation_width_rad_by_node", {}
    )
    peak_scale = metadata.get("commanded_beam_peak_gain_scale_by_node", {})
    for value, label in (
        (commanded_azimuth, "commanded azimuth widths"),
        (commanded_elevation, "commanded elevation widths"),
        (native_azimuth, "native azimuth widths"),
        (native_elevation, "native elevation widths"),
        (peak_scale, "commanded peak scales"),
    ):
        if value and not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an endpoint mapping")

    def mapped(mapping: Any, fallback: float) -> float:
        if isinstance(mapping, Mapping):
            return float(mapping.get(node_id, mapping.get(str(node_id), fallback)))
        return fallback

    fallback_az = float(
        metadata.get("beam_azimuth_width_rad", metadata.get("beamwidth_rad", math.pi))
    )
    fallback_el = float(metadata.get("beam_elevation_width_rad", fallback_az))
    az = mapped(commanded_azimuth, mapped(native_azimuth, fallback_az))
    el = mapped(commanded_elevation, mapped(native_elevation, fallback_el))
    scale = mapped(peak_scale, 1.0)
    floor = float(metadata.get("beam_sidelobe_floor_linear", 1e-4))
    if not math.radians(0.1) <= az <= math.pi:
        raise ValueError("beam azimuth width is outside the physical domain")
    if not math.radians(0.1) <= el <= math.pi:
        raise ValueError("beam elevation width is outside the physical domain")
    if not 0.0 <= floor <= 1.0 or not 0.0 < scale <= 1.0:
        raise ValueError("beam floor/peak scale is outside [0, 1]")
    return az, el, floor, scale


def broadened_beam_peak_scale(
    native_azimuth_rad: float,
    native_elevation_rad: float,
    commanded_azimuth_rad: float,
    commanded_elevation_rad: float,
) -> float:
    """Conserve first-order Gaussian mainlobe solid angle when broadening."""

    native_product = max(native_azimuth_rad * native_elevation_rad, 1e-30)
    commanded_product = max(
        commanded_azimuth_rad * commanded_elevation_rad, native_product
    )
    return float(np.clip(native_product / commanded_product, 1e-6, 1.0))
