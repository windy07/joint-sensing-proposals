"""First-principles radar planning physics introduced in V58.

The package intentionally separates physical measurement generation from
decision preferences.  Physical modules produce likelihoods and information
matrices in SI units; planners apply explicit, auditable policy weights.
"""

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

__all__ = [
    "Atmosphere",
    "Jammer",
    "MaterialSurface",
    "PhysicalWorld",
    "RadarAction",
    "SensorNode",
    "SpectrumEmitter",
    "TargetState",
    "Waveform",
]
