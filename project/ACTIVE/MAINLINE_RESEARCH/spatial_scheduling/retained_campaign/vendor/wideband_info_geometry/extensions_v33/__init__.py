"""V33 advanced theory-extension gates.

These modules are deliberately small, deterministic bridges from the V33 audit
items to executable factors and acceptance checks.  They do not replace the
existing v31 server pipeline; they add unit/development evidence for the
advanced structures that were previously outside the validated core.
"""

from __future__ import annotations

__all__ = [
    "se3_pose_factors",
    "u1_gauge_phase",
    "heisenberg_delay_doppler",
    "spd_hpd_geometry",
    "grassmann_wideband_array",
    "rfs_jpda_multitarget",
    "multi_epoch_factor_graph",
    "multiband_spectrum_iq",
    "ew_adversarial_bank",
    "pinn_neural_operator_to_fim",
    "factor_graph_extension_driver",
]
