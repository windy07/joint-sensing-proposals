from __future__ import annotations

"""Observation adapters from R0/R1/R2 physics into graph factors."""

from dataclasses import replace
from itertools import combinations, product
import math
from typing import Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.antenna_geometry import (
    rectangular_array_hpbw_rad,
)
from wideband_info_geometry.physics_v58.dynamic_factor_graph import (
    DynamicFactor,
    FactorKind,
    RobustKernel,
    VariableKey,
    VariableKind,
)
from wideband_info_geometry.physics_v58.effective_delay import (
    automatic_free_calibration_actions,
)
from wideband_info_geometry.physics_v58.iq import synthesize_waveform
from wideband_info_geometry.physics_v58.measurement import (
    StateLayout,
    angle_covariance_rad2,
    build_action_factor,
    delay_variance_s2,
    false_return_normalized_jacobian,
    slow_time_window_contract,
    world_with_normalized_state_error,
)
from wideband_info_geometry.physics_v58.real_world_measurement import (
    antenna_from_node,
    array_angle_sufficient_statistic,
    calibration_frequency_sufficient_statistic,
    draw_controlled_local_linear_gaussian_sufficient_statistic,
    draw_r1_sufficient_statistic,
    latent_component_ambiguity_response_at_peak,
    matched_filter_sufficient_statistic,
    receiver_channel_capture,
    receiver_observable_delay_doppler,
    receiver_observable_feature_summary,
    pulse_compressed_array_angle_sufficient_statistic,
    simulate_calibration_baseband,
    simulate_complex_baseband,
)
from wideband_info_geometry.physics_v58.schema import Jammer, PhysicalWorld, RadarAction

from .availability import DistributedAvailabilityManager
from .belief import GraphBeliefEngine, make_operational_action_factor
from .contracts import (
    FactorInformationMode,
    LikelihoodUpdateMode,
    ObservationRecord,
    RuntimeConfig,
    RuntimeFidelity,
)
from .likelihood_v71 import (
    make_operational_mixture_factor,
    mixture_contract_report,
)


_R2_SUPPORTED_KINDS = {
    "active_monostatic",
    "active_bistatic",
    "passive_bistatic",
    "tdoa",
    "fdoa",
    "aoa",
    "calibration_time",
    "calibration_frequency",
}


def _action_keyed_observation_seed(
    master_seed: int,
    epoch: int,
    action_id: int,
) -> int:
    """Derive a CRN seed independent of schedule order and sibling actions."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(epoch), int(action_id), 0x0B5E7A71]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _offline_target_origin_classification(
    truth_mahalanobis2: float | None,
    truth_gate_threshold: float,
    target_origin_response_fraction: float | None,
) -> dict[str, bool]:
    """Classify a selected peak using simulator truth for offline audit only."""

    near_truth = bool(
        truth_mahalanobis2 is not None
        and math.isfinite(float(truth_mahalanobis2))
        and float(truth_mahalanobis2) <= float(truth_gate_threshold)
    )
    target_origin_dominant = bool(
        target_origin_response_fraction is not None
        and math.isfinite(float(target_origin_response_fraction))
        and float(target_origin_response_fraction) > 0.5
    )
    return {
        "near_truth": near_truth,
        "target_origin_dominant": target_origin_dominant,
        "target_associated": bool(near_truth and target_origin_dominant),
        "near_truth_but_nontarget_dominated": bool(
            near_truth and not target_origin_dominant
        ),
    }


def _offline_difference_target_origin_audit(
    *,
    measurement_name: str,
    observed_difference: float | None,
    truth_difference: float,
    variance: float,
    raw_detected: bool,
    ambiguity_period: float | None,
    per_receiver_target_origin_response_fractions: Sequence[float | None],
) -> dict[str, object]:
    r"""Classify a TDOA/FDOA peak without leaking truth to the estimator.

    A difference factor is target-origin only when its selected difference is
    statistically compatible with the truth *and* every receiver leg used to
    form that difference is target-dominated.  The latter requirement prevents
    one clutter/noise peak from being hidden by subtraction against a genuine
    target peak.  All fields returned here are simulator-only audit metadata.
    """

    if measurement_name not in {"tdoa_s", "fdoa_hz"}:
        raise ValueError("difference target-origin audit requires TDOA or FDOA")
    if not math.isfinite(float(truth_difference)):
        raise ValueError("difference truth must be finite")
    if not math.isfinite(float(variance)) or float(variance) <= 0.0:
        raise ValueError("difference variance must be finite and positive")
    if len(per_receiver_target_origin_response_fractions) < 2:
        raise ValueError("difference target-origin audit requires at least two legs")

    residual: float | None = None
    mahalanobis2: float | None = None
    if raw_detected:
        if observed_difference is None or not math.isfinite(float(observed_difference)):
            raise ValueError("a detected difference requires a finite observation")
        residual = float(
            _wrap_residual(
                (measurement_name,),
                np.asarray([float(observed_difference) - float(truth_difference)]),
                (ambiguity_period,),
            )[0]
        )
        mahalanobis2 = residual * residual / float(variance)

    finite_fractions = [
        float(value)
        for value in per_receiver_target_origin_response_fractions
        if value is not None and math.isfinite(float(value))
    ]
    all_legs_attributed = len(finite_fractions) == len(
        per_receiver_target_origin_response_fractions
    )
    weakest_leg_fraction = min(finite_fractions) if all_legs_attributed else None
    # chi2.ppf(0.99, df=1), fixed explicitly to avoid a SciPy runtime dependency.
    truth_gate_threshold = 6.6348966010212145
    classification = _offline_target_origin_classification(
        mahalanobis2,
        truth_gate_threshold,
        weakest_leg_fraction,
    )
    target_associated = bool(raw_detected and classification["target_associated"])
    return {
        "simulator_truth_measurement_vector": [float(truth_difference)],
        "simulator_truth_raw_measurement_residual": (
            None if residual is None else [residual]
        ),
        "simulator_truth_target_association_mahalanobis2": mahalanobis2,
        "simulator_truth_target_association_threshold_chi2_1d_99pct": (
            truth_gate_threshold
        ),
        "simulator_truth_per_receiver_target_origin_response_fraction": [
            None if value is None else float(value)
            for value in per_receiver_target_origin_response_fractions
        ],
        "simulator_truth_target_origin_response_fraction": weakest_leg_fraction,
        "simulator_truth_target_origin_dominant_at_selected_cell": bool(
            classification["target_origin_dominant"]
        ),
        "simulator_truth_raw_target_peak_within_99pct_gate": bool(
            classification["near_truth"]
        ),
        "simulator_truth_near_truth_peak_but_nontarget_dominated": bool(
            classification["near_truth_but_nontarget_dominated"]
        ),
        "simulator_truth_target_association_accepted": target_associated,
        "simulator_truth_false_alarm_like_accepted": bool(
            raw_detected and not target_associated
        ),
        "simulator_truth_target_peak_rejected_by_prior_gate": False,
        "simulator_truth_measurement_modality": measurement_name.removesuffix(
            "_s" if measurement_name == "tdoa_s" else "_hz"
        ),
    }


def loewner_upper_envelope(
    first_covariance: np.ndarray,
    second_covariance: np.ndarray,
) -> np.ndarray:
    """Return a symmetric covariance that dominates both inputs in PSD order."""

    first = np.asarray(first_covariance, dtype=float)
    second = np.asarray(second_covariance, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("Loewner covariance envelope requires equal square matrices")
    difference = 0.5 * ((first - second) + (first - second).T)
    values, vectors = np.linalg.eigh(difference)
    positive_part = (vectors * np.maximum(values, 0.0)) @ vectors.T
    envelope = second + positive_part
    return 0.5 * (envelope + envelope.T)


# Retain the historical private name for source compatibility with the
# acquisition-physics tests and any frozen downstream analysis imports.
_loewner_upper_envelope = loewner_upper_envelope


def _commanded_capture_samples(
    config: RuntimeConfig,
    action: RadarAction,
) -> int:
    """Resolve the causal fast-time receiver command for one action."""

    samples = int(
        action.metadata.get("commanded_capture_fast_time_samples", config.r2_samples)
    )
    if samples < config.r2_samples or samples > config.r2_max_samples:
        raise ValueError(
            "commanded R2 capture samples violate the declared receiver limits"
        )
    return samples


def _belief_centered_capture_start_s(
    predicted_delay_s: float,
    waveform: object,
    samples: int,
) -> float:
    """Center the predicted return pulse inside the commanded fast-time gate."""

    sample_rate_hz = float(getattr(waveform, "sample_rate_hz"))
    family = str(getattr(waveform, "family"))
    duty_cycle = float(getattr(waveform, "duty_cycle"))
    pulse_width_s = (
        0.0
        if family == "cw" or duty_cycle >= 1.0 - 1e-12
        else float(getattr(waveform, "pulse_width_s"))
    )
    free_gate_s = max(float(samples) / sample_rate_hz - pulse_width_s, 0.0)
    return float(predicted_delay_s) - 0.5 * free_gate_s


def _coherent_capture_support(
    capture: object,
    waveform: object,
) -> dict[str, float | int | bool]:
    """Return the actual fast/slow-time support of an R2 capture.

    The analytic link budget is normalized to the waveform's complete on-air
    dwell.  An R2 capture may retain only a range-gated fraction of each pulse,
    but it contains every coherent pulse.  Its information fraction is thus
    ``B * captured_on_air_dwell / (B * declared_dwell)``; Doppler curvature is
    governed by the energy-weighted slow-time second moment, not by the
    microsecond fast-time range window.
    """

    metadata = getattr(capture, "metadata")
    sample_rate = float(getattr(capture, "sample_rate_hz"))
    fast_samples = int(np.asarray(getattr(capture, "surveillance_iq")).shape[-1])
    fast_window_s = fast_samples / sample_rate
    slow_time_value = getattr(capture, "pulse_slow_time_s", None)
    if slow_time_value is None:
        slow_time = np.asarray([0.0], dtype=float)
        pulse_count = 1
        actual_cpi_s = fast_window_s
        time_rms_s = fast_window_s / math.sqrt(12.0)
        coherent_pulse_processing = False
    else:
        slow_time = np.asarray(slow_time_value, dtype=float)
        pulse_count = int(slow_time.size)
        prf_hz = float(metadata["pulse_repetition_hz"])
        actual_cpi_s = pulse_count / prf_hz
        centred = slow_time - float(np.mean(slow_time))
        time_rms_s = math.sqrt(max(float(np.mean(centred**2)), 1e-30))
        coherent_pulse_processing = True
    captured_dwell_s = float(
        metadata.get(
            "captured_on_air_dwell_s",
            pulse_count
            * min(fast_window_s, float(getattr(waveform, "pulse_width_s"))),
        )
    )
    declared_dwell_s = float(getattr(waveform, "dwell_s"))
    # A fully retained pulse passed through a fractional-delay interpolator can
    # differ from unit energy by a few ulps.  Normalize that numerical boundary
    # to the declared on-air dwell so changing unused zero-padding cannot make
    # a complete capture appear non-monotone.  Material truncation remains
    # represented by its measured retained-energy fraction.
    if math.isclose(
        captured_dwell_s,
        declared_dwell_s,
        rel_tol=1e-6,
        abs_tol=np.finfo(float).eps * max(declared_dwell_s, 1.0),
    ):
        captured_dwell_s = declared_dwell_s
    else:
        captured_dwell_s = min(captured_dwell_s, declared_dwell_s)
    analytic_gain = max(
        1.0,
        float(getattr(waveform, "bandwidth_hz"))
        * declared_dwell_s,
    )
    capture_gain = max(
        1.0,
        float(getattr(waveform, "bandwidth_hz")) * captured_dwell_s,
    )
    return {
        "fast_window_s": float(fast_window_s),
        "pulse_count": pulse_count,
        "actual_cpi_s": float(actual_cpi_s),
        "time_rms_s": float(time_rms_s),
        "captured_dwell_s": float(captured_dwell_s),
        "analytic_gain": float(analytic_gain),
        "capture_gain": float(capture_gain),
        "gain_fraction": float(min(1.0, capture_gain / analytic_gain)),
        "coherent_pulse_processing": coherent_pulse_processing,
    }


def _r2_detector_parameters(
    config: RuntimeConfig,
    action: RadarAction,
    waveform: object,
    coherent_processing_interval_s: float,
) -> dict[str, float]:
    """Return receiver-configured CFAR and ambiguity-exclusion parameters.

    The false-alarm probability and clutter spread are action/receiver design
    quantities.  No simulator target state or component decomposition enters
    this calculation.  A pulse-Doppler notch creates a declared blind zone
    around every PRF-equivalent zero-Doppler ridge; passive reference leakage is
    additionally guarded around zero excess delay.
    """

    cpi = max(float(coherent_processing_interval_s), 1e-30)
    bandwidth = max(float(getattr(waveform, "bandwidth_hz")), 1.0)
    pfa = float(action.metadata.get("false_alarm_probability", 1.0e-6))
    if not 0.0 < pfa < 1.0:
        raise ValueError("action false-alarm probability must lie in (0, 1)")
    clutter_spread = max(
        0.0, float(action.metadata.get("clutter_doppler_spread_hz", 3.0))
    )
    _, slow_time_window_enbw_bins, _ = slow_time_window_contract(
        int(getattr(waveform, "coherent_pulse_count")),
        float(getattr(waveform, "pulse_repetition_hz")),
        config.r2_slow_time_window,
    )
    notch = max(
        config.r2_clutter_notch_spread_multiples * clutter_spread,
        config.r2_clutter_notch_resolution_cells
        * slow_time_window_enbw_bins
        / cpi,
    )
    passive_reference = bool(
        action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"}
        or action.reference_node_id is not None
    )
    return {
        "familywise_false_alarm_probability": pfa,
        "clutter_notch_half_width_hz": float(notch),
        "direct_path_guard_delay_s": (
            config.r2_direct_path_guard_delay_cells / bandwidth
            if passive_reference
            else 0.0
        ),
        "direct_path_guard_doppler_hz": float(notch if passive_reference else 0.0),
    }


def _r2_delay_doppler_false_peak_density(
    statistic: Mapping[str, object],
) -> dict[str, object]:
    r"""Normalize the exchangeable CFAR-null peak-location density.

    Conditional on a homogeneous noise-only ambiguity map and on one eligible
    cell attaining the maximum, permutation symmetry makes the selected cell
    uniform over the eligible set. The sub-bin extractor represents each cell
    by a rectangle of width ``delta_tau * delta_f``. Therefore

    .. math::

       p_F(z\mid D=1,\mathcal G)
       = \left(N_{\rm eligible}\,\Delta\tau\,\Delta f\right)^{-1}

    on the frozen receiver search support and zero outside it. Accepted peaks
    on the outer boundary are already rejected, so the local score of this
    piecewise-constant component is exactly zero. Structured clutter violates
    exchangeability; that mismatch remains an explicit robustness boundary.
    """

    eligible_cells = int(statistic.get("eligible_ambiguity_cells", 0))
    total_cells = int(statistic.get("total_ambiguity_cells", 0))
    delay_step_s = float(statistic.get("delay_grid_step_s", 0.0))
    doppler_step_hz = float(statistic.get("doppler_grid_step_hz", 0.0))
    if eligible_cells <= 0 or total_cells < eligible_cells:
        raise ValueError("R2 false-peak support has an invalid eligible-cell count")
    if (
        not math.isfinite(delay_step_s)
        or delay_step_s <= 0.0
        or not math.isfinite(doppler_step_hz)
        or doppler_step_hz <= 0.0
    ):
        raise ValueError("R2 false-peak cell measure must be finite and positive")
    cell_measure = delay_step_s * doppler_step_hz
    support_measure = float(eligible_cells) * cell_measure
    if not math.isfinite(support_measure) or support_measure <= 0.0:
        raise ValueError("R2 false-peak support measure must be finite and positive")
    return {
        "association_model_override": (
            "R2_exchangeable_Cfar_validation_gate_false_peak_mixture"
        ),
        "false_component_density_model": (
            "uniform_over_eligible_delay_doppler_cells_with_subbin_cell_measure"
        ),
        "false_component_log_density": float(-math.log(support_measure)),
        "false_component_state_independent": True,
        "false_component_support_measure_s_hz": support_measure,
        "false_component_cell_measure_s_hz": cell_measure,
        "false_component_eligible_cell_count": eligible_cells,
        "false_component_total_cell_count": total_cells,
        "false_component_support_uses_receiver_observable_search_only": True,
        "false_component_homogeneous_exchangeability_assumption": True,
        "false_component_structured_clutter_exactness_claimed": False,
        "false_component_local_score_zero_away_from_gate_boundaries": True,
    }


def _wrap_residual(
    names: Sequence[str],
    residual: np.ndarray,
    ambiguity_periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    r"""Return the principal residual on every registered quotient space.

    Azimuth and elevation live on :math:`S^1` and are therefore wrapped with
    period :math:`2\pi`.  Coherently sampled pulse-Doppler observations live on
    :math:`\mathbb{R}/(f_r\mathbb{Z})`, where ``f_r`` is the pulse repetition
    frequency.  Their integer ambiguity is an observation property and is
    supplied explicitly through ``ambiguity_periods``.  All other rows remain
    ordinary Euclidean measurements.

    The derivative of the principal representative equals the derivative of
    the unwrapped residual away from the measure-zero branch boundaries, so the
    existing local Jacobian is valid inside each ambiguity branch.
    """

    value = np.asarray(residual, dtype=float).copy()
    if value.shape != (len(names),):
        raise ValueError("residual and measurement-name dimensions disagree")
    if ambiguity_periods is None:
        periods: tuple[float | None, ...] = (None,) * len(names)
    else:
        periods = tuple(ambiguity_periods)
        if len(periods) != len(names):
            raise ValueError("ambiguity periods and measurement names disagree")
    for index, name in enumerate(names):
        if name in {"azimuth_rad", "elevation_rad"}:
            value[index] = (value[index] + math.pi) % (2.0 * math.pi) - math.pi
        period = periods[index]
        if period is None:
            continue
        period_value = float(period)
        if not math.isfinite(period_value) or period_value <= 0.0:
            raise ValueError("measurement ambiguity periods must be finite and positive")
        value[index] = (
            (value[index] + 0.5 * period_value) % period_value
            - 0.5 * period_value
        )
    return value


def _finalize_offline_association_metadata(
    metadata: dict[str, object],
    *,
    detection_accepted: bool,
) -> dict[str, object]:
    """Align truth-audit association labels with the final extractor decision."""

    finalized = dict(metadata)
    raw_target_associated = bool(
        finalized.get("simulator_truth_target_association_accepted", False)
    )
    raw_false_alarm_like = bool(
        finalized.get("simulator_truth_false_alarm_like_accepted", False)
    )
    finalized["simulator_truth_target_associated_before_consistency_gate"] = (
        raw_target_associated
    )
    finalized["simulator_truth_false_alarm_like_before_consistency_gate"] = (
        raw_false_alarm_like
    )
    finalized["r2_consistency_gate_accepted"] = bool(detection_accepted)
    finalized["simulator_truth_target_association_accepted"] = bool(
        detection_accepted and raw_target_associated
    )
    finalized["simulator_truth_false_alarm_like_accepted"] = bool(
        detection_accepted and raw_false_alarm_like
    )
    if "simulator_truth_target_peak_rejected_by_prior_gate" in finalized:
        finalized["simulator_truth_target_peak_rejected_by_prior_gate"] = bool(
            finalized["simulator_truth_target_peak_rejected_by_prior_gate"]
            or (raw_target_associated and not detection_accepted)
        )
    return finalized


class ObservationEngine:
    """Generate realized observations without exposing truth to the controller."""

    def __init__(
        self,
        config: RuntimeConfig,
        availability: DistributedAvailabilityManager,
    ) -> None:
        config.validate()
        self.config = config
        self.availability = availability
        self._capture_dispatches = ()

    def take_capture_dispatches(self) -> tuple:
        """Runtime capture side only; not a controller input or a wire format."""
        result = self._capture_dispatches
        self._capture_dispatches = ()
        return result

    def relinearize_mixture_factor(
        self,
        belief_world: PhysicalWorld,
        record: ObservationRecord,
        *,
        initial_state_normalized: np.ndarray,
        current_state_normalized: np.ndarray,
        current_covariance_normalized: np.ndarray,
        score_step_scale: float = 1.0,
        allow_empty_information_factor: bool = False,
    ) -> tuple[DynamicFactor | None, dict[str, object]]:
        """Rebuild the local physical likelihood at the current graph iterate.

        ``belief_world`` is the physical world represented by
        ``initial_state_normalized``.  The graph state itself is expressed
        relative to the immutable operational reference world, so the physical
        displacement for this local iteration is ``current - initial``.  The
        previous implementation changed only the canonical score while keeping
        the initial prediction, Jacobian, and analytic covariance frozen.  That
        is not a nonlinear relinearization and can make a wrong range/Doppler
        basin increasingly overconfident.

        This method now reconstructs the current physical world, evaluates the
        measurement function and Jacobian there, and preserves the covariance
        delivered by the actual observation extractor.  The live graph remains
        the sole posterior authority; only the factor with the same provenance
        and identifier is atomically replaced.
        """

        if record.factor_id is None or record.residual is None:
            raise ValueError("mixture relinearization requires a realized factor")
        reference_action = next(
            item for item in belief_world.actions if item.action_id == record.action_id
        )
        layout = StateLayout.from_world(belief_world)
        initial = np.asarray(initial_state_normalized, dtype=float)
        current = np.asarray(current_state_normalized, dtype=float)
        if initial.shape != (layout.dimension,) or current.shape != (layout.dimension,):
            raise ValueError("mixture relinearization state has the wrong dimension")
        delta = current - initial
        iterated_world = world_with_normalized_state_error(
            belief_world,
            delta,
            layout,
        )
        action = next(
            item
            for item in iterated_world.actions
            if item.action_id == record.action_id
        )
        observation_covariance = np.asarray(record.covariance, dtype=float)
        reference_factor = build_action_factor(
            belief_world,
            reference_action,
            None,
            measurement_covariance_override=observation_covariance,
        )
        captured_priors_raw = record.metadata.get(
            "association_component_priors_at_capture"
        )
        captured_priors_source = "recorded_causal_pre_observation_prior"
        if captured_priors_raw is None:
            contract = record.metadata.get("mixture_likelihood_contract")
            if isinstance(contract, Mapping):
                captured_priors_raw = contract.get(
                    "association_component_priors"
                )
                captured_priors_source = "legacy_recorded_mixture_contract"
        if captured_priors_raw is None:
            captured_correct = float(
                np.clip(
                    reference_factor.link_budget.correct_association_probability,
                    0.0,
                    1.0,
                )
            )
            captured_priors_raw = (captured_correct, 1.0 - captured_correct)
            captured_priors_source = "legacy_causal_reference_factor_reconstruction"
        captured_priors = np.asarray(captured_priors_raw, dtype=float).reshape(-1)
        if (
            captured_priors.shape != (2,)
            or not np.all(np.isfinite(captured_priors))
            or np.any(captured_priors < 0.0)
            or float(np.sum(captured_priors)) <= 0.0
        ):
            raise ValueError("captured association component priors are invalid")
        captured_priors = captured_priors / float(np.sum(captured_priors))
        action_factor = build_action_factor(
            iterated_world,
            action,
            None,
            measurement_covariance_override=observation_covariance,
        )
        if record.value is None:
            raise ValueError("mixture relinearization requires the realized measurement")
        if tuple(record.measurement_names) != tuple(action_factor.measurement_names):
            raise ValueError("measurement schema changed during nonlinear relinearization")
        ambiguity_periods = tuple(
            record.metadata.get(
                "measurement_ambiguity_periods",
                (None,) * len(action_factor.measurement_names),
            )
        )
        residual = _wrap_residual(
            action_factor.measurement_names,
            np.asarray(record.value, dtype=float) - action_factor.prediction,
            ambiguity_periods,
        )
        primary_rows = int(
            record.metadata.get(
                "primary_measurement_rows",
                len(action_factor.measurement_names),
            )
        )
        if not 0 < primary_rows <= len(action_factor.measurement_names):
            raise ValueError("primary measurement row count is invalid")
        false_reference = record.metadata.get(
            "false_component_reference_residual"
        )
        if false_reference is None:
            displacement_sigma = np.asarray(
                reference_factor.metadata.get(
                    "false_target_displacement_sigma",
                    [0.0] * len(reference_factor.measurement_names),
                ),
                dtype=float,
            )[:primary_rows]
            primary_covariance = np.asarray(
                reference_factor.covariance[:primary_rows, :primary_rows],
                dtype=float,
            )
            false_reference = _wrap_residual(
                action_factor.measurement_names[:primary_rows],
                (
                    np.asarray(record.residual, dtype=float)[:primary_rows]
                    - np.linalg.cholesky(primary_covariance) @ displacement_sigma
                ),
                ambiguity_periods[:primary_rows],
            )
        false_component_state_independent = bool(
            record.metadata.get("false_component_state_independent", False)
        )
        if false_component_state_independent:
            false_residual = np.asarray(false_reference, dtype=float)[:primary_rows]
        else:
            h_false = false_return_normalized_jacobian(reference_factor, layout)
            false_residual = _wrap_residual(
                action_factor.measurement_names[:primary_rows],
                (
                    np.asarray(false_reference, dtype=float)[:primary_rows]
                    - h_false[:primary_rows] @ delta
                ),
                ambiguity_periods[:primary_rows],
            )
        factor, result = make_operational_mixture_factor(
            iterated_world,
            action,
            action_factor,
            epoch=record.observation_epoch,
            arrival_epoch=record.arrival_epoch,
            participants=self.availability.factor_participants(action),
            linearization_state_normalized=current,
            measurement_residual=residual,
            primary_rows=primary_rows,
            provenance_id=record.provenance_id,
            factor_id=record.factor_id,
            fidelity_level=record.used_fidelity,
            update_mode=self.config.likelihood_update_mode,
            sandwich_temperature=self.config.sandwich_temperature,
            score_step_scale=score_step_scale,
            robust_kernel=RobustKernel.CAUCHY,
            eigenvalue_floor=self.config.information_eigenvalue_floor,
            communication_snapshot_epoch=record.observation_epoch,
            predictive_state_covariance_normalized=np.asarray(
                current_covariance_normalized, dtype=float
            ),
            false_component_residual=false_residual,
            false_component_log_density=record.metadata.get(
                "false_component_log_density"
            ),
            false_component_state_independent=false_component_state_independent,
            false_component_density_model=record.metadata.get(
                "false_component_density_model"
            ),
            association_model_override=record.metadata.get(
                "association_model_override"
            ),
            association_component_priors=captured_priors,
        )
        empty_information_factor = factor is None
        empty_factor_materialized = False
        if factor is None and not allow_empty_information_factor:
            score = np.asarray(result.state_score_normalized, dtype=float)
            louis = np.asarray(
                result.louis_observed_information_normalized, dtype=float
            )
            certified_state_independent_false_peak = bool(
                result.responsibilities[1] >= 1.0 - 1.0e-15
                and not result.parameter_dependent_components[1]
                and not np.any(score != 0.0)
                and not np.any(louis != 0.0)
                and result.louis_negative_eigenvalue_mass == 0.0
            )
            if not certified_state_independent_false_peak:
                raise RuntimeError(
                    "mixture relinearization produced zero PSD curvature "
                    "without a certified zero-score state-independent false peak"
                )
            nuisance_dimension = layout.dimension - layout.target_dim
            factor = DynamicFactor(
                factor_id=str(record.factor_id),
                kind=(
                    FactorKind.CALIBRATION
                    if action.kind.startswith("calibration")
                    else FactorKind.ASSOCIATION
                ),
                observation_epoch=int(record.observation_epoch),
                arrival_epoch=int(record.arrival_epoch),
                variables=(
                    VariableKey(
                        VariableKind.TARGET,
                        "primary",
                        int(record.observation_epoch),
                    ),
                    VariableKey(
                        VariableKind.NUISANCE,
                        "network",
                        int(record.observation_epoch),
                    ),
                ),
                jacobians=(
                    np.zeros((1, layout.target_dim), dtype=float),
                    np.zeros((1, nuisance_dimension), dtype=float),
                ),
                measurement=np.zeros(1, dtype=float),
                covariance=np.eye(1, dtype=float),
                participants=self.availability.factor_participants(action),
                provenance_ids=(record.provenance_id,),
                robust_kernel=RobustKernel.NONE,
                association_probabilities=(1.0,),
                missing=True,
                fidelity_level=record.used_fidelity,
                metadata={
                    "action_id": int(action.action_id),
                    "action_kind": action.kind,
                    "information_mode": (
                        FactorInformationMode.CONDITIONAL_DETECTED.value
                    ),
                    "factor_semantics": (
                        "inactive_zero_information_state_independent_false_peak"
                    ),
                    "zero_local_mixture_information_tombstone": True,
                    "zero_information_tombstone_contributes_to_graph": False,
                    "zero_information_tombstone_preserves_factor_identity": True,
                    "association_responsibilities": list(
                        result.responsibilities
                    ),
                    "association_component_priors": list(
                        result.component_priors
                    ),
                    "association_component_priors_frozen_at_capture": True,
                    "mixture_log_likelihood": result.log_likelihood,
                    "communication_snapshot_epoch": int(
                        record.observation_epoch
                    ),
                },
            )
            empty_factor_materialized = True
        reference_h_normalized = (
            reference_factor.jacobian_physical * layout.scales()[None, :]
        )
        exact_prediction_shift = _wrap_residual(
            action_factor.measurement_names,
            action_factor.prediction - reference_factor.prediction,
            ambiguity_periods,
        )
        frozen_prediction_shift = reference_h_normalized @ delta
        if factor is not None:
            factor = replace(
                factor,
                metadata={
                    **dict(factor.metadata),
                    "nonlinear_physical_relinearization": True,
                    "relinearization_covariance_source": (
                        "realized_observation_extractor_covariance"
                    ),
                    "relinearization_truth_state_used": False,
                    "relinearization_preserves_factor_id_and_provenance": True,
                    "association_component_priors_frozen_at_capture": True,
                    "association_component_priors_source": (
                        captured_priors_source
                    ),
                    "periodic_measurement_residual_applied": any(
                        period is not None for period in ambiguity_periods
                    ),
                },
            )
        return factor, {
            "factor_id": record.factor_id,
            "responsibilities": list(result.responsibilities),
            "log_likelihood": result.log_likelihood,
            "residual_norm": float(np.linalg.norm(residual)),
            "score_step_scale": float(score_step_scale),
            "nonlinear_physical_relinearization": True,
            "truth_state_used": False,
            "linearization_displacement_norm": float(np.linalg.norm(delta)),
            "exact_prediction_shift_norm": float(
                np.linalg.norm(exact_prediction_shift)
            ),
            "frozen_prediction_shift_error_norm": float(
                np.linalg.norm(exact_prediction_shift - frozen_prediction_shift)
            ),
            "measurement_covariance_source": (
                "realized_observation_extractor_covariance"
            ),
            "measurement_ambiguity_periods": list(ambiguity_periods),
            "periodic_measurement_residual_applied": any(
                period is not None for period in ambiguity_periods
            ),
            "empty_information_factor": empty_information_factor,
            "empty_information_factor_materialized_as_inactive_tombstone": (
                empty_factor_materialized
            ),
            "association_component_priors_at_capture": (
                captured_priors.tolist()
            ),
            "association_component_priors_source": captured_priors_source,
            "state_score_norm": float(
                np.linalg.norm(result.state_score_normalized)
            ),
            "measurement_covariance_preserved": bool(
                np.allclose(
                    action_factor.covariance,
                    observation_covariance,
                    rtol=0.0,
                    atol=0.0,
                )
            ),
            "contract": mixture_contract_report(result),
        }

    def select_nonlinear_acquisition_hypothesis(
        self,
        belief_world: PhysicalWorld,
        records: Sequence[ObservationRecord],
        *,
        initial_state_normalized: np.ndarray,
        prior_covariance_normalized: np.ndarray,
        sigma_override: float | None = None,
    ) -> tuple[np.ndarray, tuple[DynamicFactor, ...], dict[str, object]]:
        r"""Select one causal acquisition basin before local graph iteration.

        For the current position mean :math:`m_p` and covariance :math:`P_p`,
        the registered lattice is

        .. math::

            p_j = m_p + P_p^{1/2} u_j,\qquad
            u_j \in \{-\gamma,0,+\gamma\}^3.

        Each point is ranked by a position-conditioned, nuisance-marginalized
        predictive mixture log likelihood minus its Gaussian prior penalty.
        When registered, the strongest coarse basins are then refined by a
        deterministic bounded derivative-free search in the same whitened
        coordinates. Position uncertainty is removed from the scoring
        covariance because it is represented explicitly by the candidate
        coordinate; velocity and nuisance uncertainty remain marginalized.
        This prevents position uncertainty from being counted once in the
        candidate set and again in the likelihood.

        The procedure is a bounded deterministic multi-start method, not an
        exact global posterior enumeration.  Candidate evaluation is read-only;
        only factors for the single selected basin are returned for atomic
        replacement in the authoritative graph.
        """

        layout = StateLayout.from_world(belief_world)
        initial = np.asarray(initial_state_normalized, dtype=float)
        covariance = np.asarray(prior_covariance_normalized, dtype=float)
        if initial.shape != (layout.dimension,):
            raise ValueError("nonlinear acquisition state has the wrong dimension")
        if covariance.shape != (layout.dimension, layout.dimension):
            raise ValueError(
                "nonlinear acquisition covariance has the wrong dimension"
            )
        active_records = tuple(
            record
            for record in records
            if record.factor_id is not None
            and record.value is not None
            and record.residual is not None
            and record.metadata.get("mixture_likelihood_contract") is not None
        )
        base_audit: dict[str, object] = {
            "enabled": bool(self.config.nonlinear_acquisition_enabled),
            "truth_state_used": False,
            "future_observation_used": False,
            "candidate_evaluation_mutated_world": False,
            "candidate_evaluation_mutated_graph": False,
            "single_authoritative_posterior_preserved": True,
            "global_mode_enumeration_claimed": False,
            "registered_hypothesis_family": (
                "three_axis_prior_covariance_lattice_with_optional_"
                "bounded_continuous_refinement"
            ),
            "active_observation_count": len(active_records),
            "continuous_refinement_enabled": bool(
                self.config.nonlinear_acquisition_refinement_enabled
            ),
            "continuous_refinement_is_exact_global_search": False,
            "latent_assignment_refinement_enabled": bool(
                self.config.nonlinear_acquisition_latent_assignment_refinement_enabled
            ),
            "latent_assignment_candidates_are_proposal_only": True,
            "full_marginalized_mixture_posterior_remains_selection_authority": True,
        }
        if not self.config.nonlinear_acquisition_enabled:
            return initial.copy(), (), {
                **base_audit,
                "status": "disabled",
                "reason": "runtime_configuration_disabled",
            }
        if len(active_records) < 2:
            return initial.copy(), (), {
                **base_audit,
                "status": "not_identifiable",
                "reason": "fewer_than_two_arrived_detected_observations",
            }

        position_covariance = 0.5 * (
            covariance[:3, :3] + covariance[:3, :3].T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)
        tolerance = max(
            float(np.max(np.abs(eigenvalues))) * 1.0e-12,
            np.finfo(float).tiny,
        )
        positive = np.maximum(eigenvalues, 0.0)
        position_rank = int(np.count_nonzero(positive > tolerance))
        if position_rank < 2:
            return initial.copy(), (), {
                **base_audit,
                "status": "not_identifiable",
                "reason": "position_prior_support_rank_below_two",
                "position_prior_rank": position_rank,
            }
        position_root = eigenvectors @ np.diag(np.sqrt(positive))

        reference_jacobians = []
        for record in active_records:
            action = next(
                item
                for item in belief_world.actions
                if item.action_id == record.action_id
            )
            factor = build_action_factor(
                belief_world,
                action,
                None,
                measurement_covariance_override=np.asarray(
                    record.covariance, dtype=float
                ),
            )
            reference_jacobians.append(
                factor.jacobian_physical[:, :3]
                * layout.scales()[None, :3]
            )
        stacked_position_jacobian = np.vstack(reference_jacobians)
        observable_position_rank = int(
            np.linalg.matrix_rank(stacked_position_jacobian)
        )
        if observable_position_rank < 2:
            return initial.copy(), (), {
                **base_audit,
                "status": "not_identifiable",
                "reason": "arrived_measurement_geometry_rank_below_two",
                "position_prior_rank": position_rank,
                "observable_position_rank": observable_position_rank,
            }

        sigma = float(
            self.config.nonlinear_acquisition_sigma
            if sigma_override is None
            else sigma_override
        )
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "nonlinear acquisition sigma override must be finite and positive"
            )
        coordinates = list(product((-sigma, 0.0, sigma), repeat=3))
        coordinates.sort(
            key=lambda value: (
                float(np.linalg.norm(value)),
                tuple(float(item) for item in value),
            )
        )
        coordinates = coordinates[
            : int(self.config.nonlinear_acquisition_max_hypotheses)
        ]
        if (0.0, 0.0, 0.0) not in coordinates:
            coordinates[-1] = (0.0, 0.0, 0.0)
            coordinates.sort(
                key=lambda value: (
                    float(np.linalg.norm(value)),
                    tuple(float(item) for item in value),
                )
            )

        # The lattice explicitly resolves position.  Retain uncertainty in
        # velocity, clocks, CFOs, path biases and receiver delays only.
        scoring_covariance = 0.5 * (covariance + covariance.T)
        scoring_covariance[:3, :] = 0.0
        scoring_covariance[:, :3] = 0.0
        position_precision = np.linalg.pinv(position_covariance, rcond=1.0e-12)
        rows: list[dict[str, object]] = []
        candidate_states: dict[int, np.ndarray] = {}

        def evaluate_coordinate(
            coordinate: Sequence[float],
            *,
            hypothesis_index: int,
            source: str,
        ) -> tuple[dict[str, object], np.ndarray]:
            coordinate_array = np.asarray(coordinate, dtype=float)
            if coordinate_array.shape != (3,) or not np.all(
                np.isfinite(coordinate_array)
            ):
                raise ValueError(
                    "nonlinear acquisition coordinate must contain three "
                    "finite values"
                )
            if np.any(np.abs(coordinate_array) > sigma + 1.0e-10):
                raise ValueError(
                    "nonlinear acquisition coordinate left its registered "
                    "prior-whitened support"
                )
            position_delta = position_root @ coordinate_array
            candidate = initial.copy()
            candidate[:3] += position_delta
            candidate_world = world_with_normalized_state_error(
                belief_world,
                candidate - initial,
                layout,
            )
            physical_position = np.asarray(
                candidate_world.target.position_m, dtype=float
            )
            if (
                physical_position[2]
                < self.config.nonlinear_acquisition_minimum_altitude_m
            ):
                return {
                    "hypothesis_index": hypothesis_index,
                    "candidate_id": f"acquisition-{hypothesis_index:04d}",
                    "source": source,
                    "coordinate": coordinate_array.tolist(),
                    "position_m": physical_position.tolist(),
                    "valid": False,
                    "reason": "below_registered_altitude_support",
                    "log_posterior_objective": -math.inf,
                }, candidate
            record_log_likelihoods: dict[str, float] = {}
            record_component_diagnostics: dict[str, dict[str, object]] = {}
            valid = True
            error_text = None
            for record in active_records:
                try:
                    _, audit = self.relinearize_mixture_factor(
                        belief_world,
                        record,
                        initial_state_normalized=initial,
                        current_state_normalized=candidate,
                        current_covariance_normalized=scoring_covariance,
                        score_step_scale=1.0,
                        allow_empty_information_factor=True,
                    )
                    record_log_likelihoods[str(record.factor_id)] = float(
                        audit["log_likelihood"]
                    )
                    contract = audit.get("contract", {})
                    component_logs = tuple(
                        float(value)
                        for value in contract.get(
                            "component_log_likelihoods", ()
                        )
                    )
                    responsibilities = tuple(
                        float(value)
                        for value in audit.get("responsibilities", ())
                    )
                    component_priors = tuple(
                        float(value)
                        for value in contract.get(
                            "association_component_priors", ()
                        )
                    )
                    if len(component_logs) != 2:
                        raise RuntimeError(
                            "nonlinear acquisition requires the registered "
                            "two-component association likelihood"
                        )
                    record_component_diagnostics[str(record.factor_id)] = {
                        "target_component_log_joint_score": component_logs[0],
                        "false_component_log_joint_score": component_logs[1],
                        "component_log_values_include_association_priors": True,
                        "target_over_false_log_joint_component_odds": (
                            component_logs[0] - component_logs[1]
                        ),
                        "association_responsibilities": list(responsibilities),
                        "association_component_priors": list(component_priors),
                    }
                except Exception as error:
                    valid = False
                    error_text = f"{type(error).__name__}: {error}"
                    break
            prior_penalty = float(
                0.5
                * position_delta
                @ position_precision
                @ position_delta
            )
            likelihood = float(sum(record_log_likelihoods.values()))
            target_component_log_joint_score = float(
                sum(
                    row["target_component_log_joint_score"]
                    for row in record_component_diagnostics.values()
                )
            )
            false_component_log_joint_score = float(
                sum(
                    row["false_component_log_joint_score"]
                    for row in record_component_diagnostics.values()
                )
            )
            objective = likelihood - prior_penalty if valid else -math.inf
            return {
                "hypothesis_index": hypothesis_index,
                "candidate_id": f"acquisition-{hypothesis_index:04d}",
                "source": source,
                "coordinate": coordinate_array.tolist(),
                "position_m": physical_position.tolist(),
                "valid": valid,
                "record_log_likelihoods": record_log_likelihoods,
                "record_component_diagnostics": record_component_diagnostics,
                "predictive_log_likelihood": likelihood,
                "aggregate_target_component_log_joint_score": (
                    target_component_log_joint_score
                ),
                "aggregate_false_component_log_joint_score": (
                    false_component_log_joint_score
                ),
                "aggregate_target_over_false_log_joint_component_odds": (
                    target_component_log_joint_score
                    - false_component_log_joint_score
                ),
                "Gaussian_position_prior_penalty": prior_penalty,
                "log_posterior_objective": objective,
                "error": error_text,
            }, candidate

        for hypothesis_index, coordinate in enumerate(coordinates):
            row, candidate = evaluate_coordinate(
                coordinate,
                hypothesis_index=hypothesis_index,
                source="coarse_prior_covariance_lattice",
            )
            rows.append(row)
            candidate_states[hypothesis_index] = candidate

        valid_coarse_rows = [row for row in rows if bool(row.get("valid"))]
        central_rows = [
            row
            for row in valid_coarse_rows
            if np.allclose(row["coordinate"], (0.0, 0.0, 0.0))
        ]
        if not central_rows:
            raise RuntimeError(
                "nonlinear acquisition could not evaluate the causal prior mean"
            )
        central = central_rows[0]
        best_coarse = max(
            valid_coarse_rows,
            key=lambda row: (
                float(row["log_posterior_objective"]),
                -float(np.linalg.norm(row["coordinate"])),
                -int(row["hypothesis_index"]),
            ),
        )

        refined_rows: list[dict[str, object]] = []
        refinement_runs: list[dict[str, object]] = []
        latent_assignment_definitions: list[dict[str, object]] = []
        latent_assignment_refinement_runs: list[dict[str, object]] = []
        refinement_evaluation_count = 0
        refinement_invalid_evaluation_count = 0
        if self.config.nonlinear_acquisition_refinement_enabled:
            try:
                from scipy.optimize import minimize
            except ImportError as error:
                raise RuntimeError(
                    "continuous nonlinear acquisition refinement requires scipy"
                ) from error

            ranked_starts = sorted(
                valid_coarse_rows,
                key=lambda row: (
                    -float(row["log_posterior_objective"]),
                    float(np.linalg.norm(row["coordinate"])),
                    int(row["hypothesis_index"]),
                ),
            )[: self.config.nonlinear_acquisition_refinement_start_count]
            bounds = [(-sigma, sigma)] * 3
            for start_rank, start_row in enumerate(ranked_starts):
                local_best_row: dict[str, object] | None = None
                local_best_state: np.ndarray | None = None
                local_evaluation_count = 0
                local_invalid_evaluation_count = 0

                def negative_log_posterior(coordinate: np.ndarray) -> float:
                    nonlocal refinement_evaluation_count
                    nonlocal refinement_invalid_evaluation_count
                    nonlocal local_best_row
                    nonlocal local_best_state
                    nonlocal local_evaluation_count
                    nonlocal local_invalid_evaluation_count

                    refinement_evaluation_count += 1
                    local_evaluation_count += 1
                    evaluated_row, evaluated_state = evaluate_coordinate(
                        coordinate,
                        hypothesis_index=-1,
                        source="bounded_continuous_refinement_evaluation",
                    )
                    if not bool(evaluated_row.get("valid")):
                        refinement_invalid_evaluation_count += 1
                        local_invalid_evaluation_count += 1
                        return 1.0e100
                    if (
                        local_best_row is None
                        or float(evaluated_row["log_posterior_objective"])
                        > float(local_best_row["log_posterior_objective"])
                    ):
                        local_best_row = evaluated_row
                        local_best_state = evaluated_state
                    return -float(evaluated_row["log_posterior_objective"])

                result = minimize(
                    negative_log_posterior,
                    np.asarray(start_row["coordinate"], dtype=float),
                    method="Powell",
                    bounds=bounds,
                    options={
                        "maxfev": int(
                            self.config.nonlinear_acquisition_refinement_max_evaluations_per_start
                        ),
                        "xtol": float(
                            self.config.nonlinear_acquisition_refinement_coordinate_tolerance
                        ),
                        "ftol": float(
                            self.config.nonlinear_acquisition_refinement_objective_tolerance
                        ),
                        "disp": False,
                    },
                )
                if local_best_row is not None and local_best_state is not None:
                    refined_index = len(rows) + len(refined_rows)
                    retained_row = {
                        **local_best_row,
                        "hypothesis_index": refined_index,
                        "candidate_id": f"acquisition-{refined_index:04d}",
                        "source": "bounded_continuous_refinement",
                        "coarse_start_rank": start_rank,
                        "coarse_start_hypothesis_index": int(
                            start_row["hypothesis_index"]
                        ),
                    }
                    refined_rows.append(retained_row)
                    candidate_states[refined_index] = local_best_state
                refinement_runs.append(
                    {
                        "start_rank": start_rank,
                        "coarse_start_hypothesis_index": int(
                            start_row["hypothesis_index"]
                        ),
                        "coarse_start_coordinate": list(start_row["coordinate"]),
                        "coarse_start_log_posterior_objective": float(
                            start_row["log_posterior_objective"]
                        ),
                        "optimizer": "scipy_Powell_bounded",
                        "optimizer_success": bool(result.success),
                        "optimizer_status": int(result.status),
                        "optimizer_message": str(result.message),
                        "optimizer_function_evaluations": int(result.nfev),
                        "optimizer_iterations": int(getattr(result, "nit", 0)),
                        "evaluated_coordinate_count": local_evaluation_count,
                        "invalid_coordinate_count": local_invalid_evaluation_count,
                        "optimizer_reported_coordinate": np.asarray(
                            result.x, dtype=float
                        ).tolist(),
                        "retained_coordinate": (
                            None
                            if local_best_row is None
                            else list(local_best_row["coordinate"])
                        ),
                        "retained_log_posterior_objective": (
                            None
                            if local_best_row is None
                            else float(
                                local_best_row["log_posterior_objective"]
                            )
                        ),
                    }
                )

            if (
                self.config.nonlinear_acquisition_latent_assignment_refinement_enabled
            ):
                factor_ids = tuple(str(record.factor_id) for record in active_records)
                raw_assignments: list[
                    tuple[int, tuple[str, ...], tuple[str, ...]]
                ] = []
                for target_count in range(2, len(active_records) + 1):
                    for target_indices in combinations(
                        range(len(active_records)), target_count
                    ):
                        target_index_set = set(target_indices)
                        target_factor_ids = tuple(
                            factor_ids[index] for index in target_indices
                        )
                        false_factor_ids = tuple(
                            factor_ids[index]
                            for index in range(len(active_records))
                            if index not in target_index_set
                        )
                        assignment_rank = int(
                            np.linalg.matrix_rank(
                                np.vstack(
                                    [
                                        reference_jacobians[index]
                                        for index in target_indices
                                    ]
                                )
                            )
                        )
                        if assignment_rank >= 2:
                            raw_assignments.append(
                                (
                                    assignment_rank,
                                    target_factor_ids,
                                    false_factor_ids,
                                )
                            )
                raw_assignments.sort(
                    key=lambda item: (
                        -int(item[0]),
                        -len(item[1]),
                        item[1],
                    )
                )
                assignments = raw_assignments[
                    : self.config.nonlinear_acquisition_max_latent_assignments
                ]

                def assignment_objective(
                    row: Mapping[str, object],
                    target_factor_ids: frozenset[str],
                ) -> float:
                    diagnostics = row.get("record_component_diagnostics", {})
                    if not isinstance(diagnostics, Mapping):
                        raise RuntimeError(
                            "latent-assignment acquisition diagnostics are missing"
                        )
                    component_score = 0.0
                    for factor_id in factor_ids:
                        factor_diagnostic = diagnostics.get(factor_id)
                        if not isinstance(factor_diagnostic, Mapping):
                            raise RuntimeError(
                                "latent-assignment acquisition factor diagnostics "
                                "are incomplete"
                            )
                        key = (
                            "target_component_log_joint_score"
                            if factor_id in target_factor_ids
                            else "false_component_log_joint_score"
                        )
                        component_score += float(factor_diagnostic[key])
                    return component_score - float(
                        row["Gaussian_position_prior_penalty"]
                    )

                for assignment_index, (
                    assignment_rank,
                    target_ids,
                    false_ids,
                ) in enumerate(assignments):
                    target_id_set = frozenset(target_ids)
                    latent_assignment_definitions.append(
                        {
                            "assignment_index": assignment_index,
                            "target_component_factor_ids": list(target_ids),
                            "false_component_factor_ids": list(false_ids),
                            "target_component_position_jacobian_rank": int(
                                assignment_rank
                            ),
                        }
                    )
                    assignment_starts = sorted(
                        valid_coarse_rows,
                        key=lambda row: (
                            -assignment_objective(row, target_id_set),
                            float(np.linalg.norm(row["coordinate"])),
                            int(row["hypothesis_index"]),
                        ),
                    )[
                        : self.config.nonlinear_acquisition_latent_refinement_start_count
                    ]
                    for start_rank, start_row in enumerate(assignment_starts):
                        local_best_row = None
                        local_best_state = None
                        local_best_assignment_objective = -math.inf
                        local_evaluation_count = 0
                        local_invalid_evaluation_count = 0

                        def negative_assignment_objective(
                            coordinate: np.ndarray,
                        ) -> float:
                            nonlocal refinement_evaluation_count
                            nonlocal refinement_invalid_evaluation_count
                            nonlocal local_best_row
                            nonlocal local_best_state
                            nonlocal local_best_assignment_objective
                            nonlocal local_evaluation_count
                            nonlocal local_invalid_evaluation_count

                            refinement_evaluation_count += 1
                            local_evaluation_count += 1
                            evaluated_row, evaluated_state = evaluate_coordinate(
                                coordinate,
                                hypothesis_index=-1,
                                source=(
                                    "bounded_latent_assignment_refinement_"
                                    "evaluation"
                                ),
                            )
                            if not bool(evaluated_row.get("valid")):
                                refinement_invalid_evaluation_count += 1
                                local_invalid_evaluation_count += 1
                                return 1.0e100
                            objective = assignment_objective(
                                evaluated_row, target_id_set
                            )
                            if objective > local_best_assignment_objective:
                                local_best_assignment_objective = objective
                                local_best_row = evaluated_row
                                local_best_state = evaluated_state
                            return -objective

                        result = minimize(
                            negative_assignment_objective,
                            np.asarray(start_row["coordinate"], dtype=float),
                            method="Powell",
                            bounds=bounds,
                            options={
                                "maxfev": int(
                                    self.config.nonlinear_acquisition_refinement_max_evaluations_per_start
                                ),
                                "xtol": float(
                                    self.config.nonlinear_acquisition_refinement_coordinate_tolerance
                                ),
                                "ftol": float(
                                    self.config.nonlinear_acquisition_refinement_objective_tolerance
                                ),
                                "disp": False,
                            },
                        )
                        if (
                            local_best_row is not None
                            and local_best_state is not None
                        ):
                            refined_index = len(rows) + len(refined_rows)
                            retained_row = {
                                **local_best_row,
                                "hypothesis_index": refined_index,
                                "candidate_id": (
                                    f"acquisition-{refined_index:04d}"
                                ),
                                "source": (
                                    "bounded_latent_assignment_refinement"
                                ),
                                "latent_assignment_index": assignment_index,
                                "target_component_factor_ids": list(target_ids),
                                "false_component_factor_ids": list(false_ids),
                                "target_component_position_jacobian_rank": int(
                                    assignment_rank
                                ),
                                "latent_assignment_proposal_objective": float(
                                    local_best_assignment_objective
                                ),
                                "final_selection_uses_full_marginalized_mixture": True,
                                "coarse_start_rank": start_rank,
                                "coarse_start_hypothesis_index": int(
                                    start_row["hypothesis_index"]
                                ),
                            }
                            refined_rows.append(retained_row)
                            candidate_states[refined_index] = local_best_state
                        latent_assignment_refinement_runs.append(
                            {
                                "assignment_index": assignment_index,
                                "start_rank": start_rank,
                                "coarse_start_hypothesis_index": int(
                                    start_row["hypothesis_index"]
                                ),
                                "coarse_start_assignment_objective": float(
                                    assignment_objective(
                                        start_row, target_id_set
                                    )
                                ),
                                "optimizer": "scipy_Powell_bounded",
                                "optimizer_success": bool(result.success),
                                "optimizer_status": int(result.status),
                                "optimizer_message": str(result.message),
                                "optimizer_function_evaluations": int(
                                    result.nfev
                                ),
                                "optimizer_iterations": int(
                                    getattr(result, "nit", 0)
                                ),
                                "evaluated_coordinate_count": (
                                    local_evaluation_count
                                ),
                                "invalid_coordinate_count": (
                                    local_invalid_evaluation_count
                                ),
                                "retained_coordinate": (
                                    None
                                    if local_best_row is None
                                    else list(local_best_row["coordinate"])
                                ),
                                "retained_assignment_objective": (
                                    None
                                    if local_best_row is None
                                    else float(
                                        local_best_assignment_objective
                                    )
                                ),
                                "retained_full_mixture_log_posterior_objective": (
                                    None
                                    if local_best_row is None
                                    else float(
                                        local_best_row[
                                            "log_posterior_objective"
                                        ]
                                    )
                                ),
                            }
                        )

        valid_rows = valid_coarse_rows + refined_rows
        best = max(
            valid_rows,
            key=lambda row: (
                float(row["log_posterior_objective"]),
                -float(np.linalg.norm(row["coordinate"])),
                -int(row["hypothesis_index"]),
            ),
        )
        improvement = float(best["log_posterior_objective"]) - float(
            central["log_posterior_objective"]
        )
        select_noncentral = bool(
            int(best["hypothesis_index"]) != int(central["hypothesis_index"])
            and improvement
            >= self.config.nonlinear_acquisition_min_log_posterior_improvement
        )
        selected_row = best if select_noncentral else central
        selected_state = candidate_states[int(selected_row["hypothesis_index"])]
        replacements: list[DynamicFactor] = []
        if select_noncentral:
            for record in active_records:
                replacement, _ = self.relinearize_mixture_factor(
                    belief_world,
                    record,
                    initial_state_normalized=initial,
                    current_state_normalized=selected_state,
                    current_covariance_normalized=covariance,
                    score_step_scale=1.0,
                )
                if replacement is None:
                    raise RuntimeError(
                        "selected nonlinear acquisition basin produced an "
                        "empty installed factor"
                    )
                replacements.append(replacement)
        return selected_state.copy(), tuple(replacements), {
            **base_audit,
            "status": "selected" if select_noncentral else "prior_mean_retained",
            "position_prior_rank": position_rank,
            "observable_position_rank": observable_position_rank,
            "hypothesis_count": len(rows) + len(refined_rows),
            "coarse_hypothesis_count": len(rows),
            "retained_refined_hypothesis_count": len(refined_rows),
            "valid_hypothesis_count": len(valid_rows),
            "sigma_radius": sigma,
            "sigma_source": (
                "registered_nonlinear_acquisition_sigma"
                if sigma_override is None
                else "causal_innovation_reacquisition_override"
            ),
            "position_uncertainty_marginalized_twice": False,
            "velocity_and_nuisance_uncertainty_marginalized": True,
            "central_hypothesis_index": int(central["hypothesis_index"]),
            "best_coarse_hypothesis_index": int(
                best_coarse["hypothesis_index"]
            ),
            "best_coarse_log_posterior_objective": float(
                best_coarse["log_posterior_objective"]
            ),
            "best_coarse_coordinate": list(best_coarse["coordinate"]),
            "best_coarse_position_m": list(best_coarse["position_m"]),
            "best_hypothesis_index": int(best["hypothesis_index"]),
            "selected_hypothesis_index": int(selected_row["hypothesis_index"]),
            "selected_candidate_id": str(selected_row["candidate_id"]),
            "selected_hypothesis_source": str(selected_row["source"]),
            "selected_noncentral_hypothesis": select_noncentral,
            "best_over_central_log_posterior_improvement": improvement,
            "minimum_required_log_posterior_improvement": float(
                self.config.nonlinear_acquisition_min_log_posterior_improvement
            ),
            "selected_position_m": selected_row["position_m"],
            "replacement_factor_count": len(replacements),
            "hypotheses": rows,
            "refined_hypotheses": refined_rows,
            "continuous_refinement_start_count": len(refinement_runs),
            "continuous_refinement_evaluation_count": (
                refinement_evaluation_count
            ),
            "continuous_refinement_invalid_evaluation_count": (
                refinement_invalid_evaluation_count
            ),
            "continuous_refinement_runs": refinement_runs,
            "continuous_refinement_bounded_in_prior_whitened_coordinates": True,
            "latent_assignment_definition_count": len(
                latent_assignment_definitions
            ),
            "latent_assignment_definitions": latent_assignment_definitions,
            "latent_assignment_refinement_run_count": len(
                latent_assignment_refinement_runs
            ),
            "latent_assignment_refinement_runs": (
                latent_assignment_refinement_runs
            ),
            "latent_assignment_search_is_exact_global_enumeration": False,
            "latent_assignment_search_mutated_live_graph": False,
            "latent_assignment_search_mutated_live_world": False,
        }

    @staticmethod
    def _truth_jammer(world: PhysicalWorld) -> Jammer | None:
        return world.jammers[0] if world.jammers else None

    def _r1(
        self,
        truth_world: PhysicalWorld,
        truth_action: RadarAction,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        *,
        epoch: int,
        seed: int,
        fallback_reason: str | None,
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        truth_jammer = self._truth_jammer(truth_world)
        belief_factor = build_action_factor(belief_world, belief_action, None)
        controlled_local_gaussian = str(
            belief_world.external_channel_provenance.get(
                "experiment_condition_profile", "native"
            )
        ) == "controlled_gaussian_localization"
        if controlled_local_gaussian:
            observation = draw_controlled_local_linear_gaussian_sufficient_statistic(
                belief_world,
                truth_world,
                belief_action,
                seed=seed,
            )
        else:
            # A validation gate is opened around the causal target estimate,
            # while the received delay/Doppler still contains the physical
            # receiver nuisance realization.  Build that simulator-only centre
            # by combining the belief target kinematics with the truth-world
            # receiver state.  No numerical gate centre or truth nuisance is
            # exposed to the controller; inference receives only the sampled
            # observation and its declared likelihood family.
            gate_target = replace(
                truth_world.target,
                position_m=tuple(belief_world.target.position_m),
                velocity_m_s=tuple(belief_world.target.velocity_m_s),
            )
            gate_reference_world = replace(truth_world, target=gate_target)
            gate_reference_factor = build_action_factor(
                gate_reference_world,
                truth_action,
                None,
            )
            if (
                tuple(gate_reference_factor.measurement_names)
                != tuple(belief_factor.measurement_names)
            ):
                raise RuntimeError(
                    "causal association gate and belief measurement schemas disagree"
                )
            observation = draw_r1_sufficient_statistic(
                truth_world,
                truth_action,
                jammer=truth_jammer,
                seed=seed,
                sample_association_mixture=self.config.realized_association_mixture,
                association_gate_reference_prediction=(
                    gate_reference_factor.prediction
                ),
                association_gate_reference_covariance=belief_factor.covariance,
            )
        # The simulator may use hidden jammer truth to generate a capture.  The
        # inference model may not.  Unmodelled disturbance is handled by the
        # robust graph factor, not by passing oracle jammer parameters into the
        # controller likelihood.
        drawn = np.asarray(observation.value, dtype=float)
        value = None if not observation.detected else drawn
        residual = None
        false_component_reference_residual = None
        if value is not None and np.all(np.isfinite(value)):
            residual = _wrap_residual(
                belief_factor.measurement_names,
                value - belief_factor.prediction,
            )
            primary_rows = int(
                observation.metadata.get(
                    "primary_measurement_rows", len(belief_factor.measurement_names)
                )
            )
            displacement_sigma = np.asarray(
                belief_factor.metadata.get(
                    "false_target_displacement_sigma",
                    [0.0] * len(belief_factor.measurement_names),
                ),
                dtype=float,
            )[:primary_rows]
            primary_covariance = np.asarray(
                belief_factor.covariance[:primary_rows, :primary_rows],
                dtype=float,
            )
            false_offset = np.linalg.cholesky(primary_covariance) @ displacement_sigma
            false_component_reference_residual = (
                np.asarray(residual[:primary_rows], dtype=float) - false_offset
            )
        # Simultaneous free calibration is delivered whether or not the target
        # return was detected.  When the radar rows are void, the calibration
        # rows are still valid evidence and are emitted as their own factors so
        # that the realized expectation matches what the planner scored:
        #   E[dJ] = P_D * J_radar|D  +  J_cal.
        calibration_only: list[tuple[RadarAction, object, np.ndarray]] = []
        if not observation.detected:
            calibrations = automatic_free_calibration_actions(belief_world, belief_action)
            cursor = int(drawn.size) - int(
                observation.metadata.get("detection_independent_calibration_rows", 0)
            )
            for calibration in calibrations:
                calibration_factor = build_action_factor(belief_world, calibration, None)
                width = len(calibration_factor.measurement_names)
                rows = drawn[cursor : cursor + width]
                cursor += width
                if rows.size != width or not np.all(np.isfinite(rows)):
                    continue
                calibration_only.append(
                    (
                        calibration,
                        calibration_factor,
                        _wrap_residual(
                            calibration_factor.measurement_names,
                            rows - calibration_factor.prediction,
                        ),
                    )
                )
        return (
            bool(observation.detected and residual is not None),
            value,
            belief_factor.prediction,
            belief_factor.covariance,
            RuntimeFidelity.R1_SUFFICIENT.value,
            {
                "fallback_reason": fallback_reason,
                "truth_observation_likelihood": observation.metadata.get("likelihood"),
                "controlled_measurement_model": observation.metadata.get(
                    "controlled_measurement_model"
                ),
                "conditional_on_detection": observation.metadata.get(
                    "conditional_on_detection", False
                ),
                "truth_state_error_exposed_to_controller": observation.metadata.get(
                    "truth_state_error_exposed_to_controller", False
                ),
                "oracle_disturbance_parameters_used_by_inference": False,
                "belief_disturbance_model": "nominal_likelihood_plus_robust_factor",
                "future_environment_state_read": False,
                "association_probabilities": observation.association_probabilities.tolist(),
                "association_model": observation.metadata.get("association_model"),
                "residual": residual,
                "false_component_reference_residual": (
                    false_component_reference_residual
                ),
                "belief_factor": belief_factor,
                "calibration_only_factors": tuple(calibration_only),
                "detection_independent_calibration_rows": int(
                    observation.metadata.get("detection_independent_calibration_rows", 0)
                ),
                "primary_measurement_rows": int(
                    observation.metadata.get(
                        "primary_measurement_rows", len(belief_factor.measurement_names)
                    )
                ),
                "association_component_sampled": observation.metadata.get(
                    "association_component_sampled"
                ),
                "association_mixture_realized": bool(
                    observation.metadata.get("association_mixture_realized", False)
                ),
                "association_gate_reference_supplied": bool(
                    observation.metadata.get(
                        "association_gate_reference_supplied", False
                    )
                ),
                "association_gate_reference_applied": bool(
                    observation.metadata.get(
                        "association_gate_reference_applied", False
                    )
                ),
                "association_gate_reference_is_causal_preobservation_belief": bool(
                    observation.metadata.get(
                        "association_gate_reference_is_causal_preobservation_belief",
                        False,
                    )
                ),
                "association_gate_reference_numeric_value_exposed_to_controller": False,
                "simulator_truth_jammer_present_for_offline_audit": truth_jammer is not None,
            },
        )

    @staticmethod
    def _r2_primary_prediction_covariance(
        belief_world: PhysicalWorld,
        belief_factor: object,
        belief_covariance_normalized: np.ndarray | None,
        primary_rows: int,
    ) -> tuple[np.ndarray, bool]:
        layout = StateLayout.from_world(belief_world)
        jacobian = np.asarray(
            getattr(belief_factor, "jacobian_physical"), dtype=float
        )[:primary_rows]
        jacobian_normalized = jacobian * layout.scales()[None, :]
        if belief_covariance_normalized is None:
            covariance = np.asarray(
                getattr(belief_factor, "covariance"), dtype=float
            )[:primary_rows, :primary_rows]
            propagated = False
        else:
            posterior = np.asarray(belief_covariance_normalized, dtype=float)
            if posterior.shape != (layout.dimension, layout.dimension):
                raise ValueError(
                    "R2 belief covariance dimension disagrees with measurement state"
                )
            covariance = jacobian_normalized @ posterior @ jacobian_normalized.T
            propagated = True
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = max(
            float(np.max(np.abs(eigenvalues))) * 1e-10,
            np.finfo(float).tiny,
        )
        if float(eigenvalues.min()) < -tolerance:
            raise np.linalg.LinAlgError(
                "R2 predicted measurement covariance is materially indefinite"
            )
        return covariance, propagated

    def _finish_r2_extraction(
        self,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        *,
        value: np.ndarray | None,
        covariance: np.ndarray,
        prediction_covariance: np.ndarray,
        detected: bool,
        capture: object,
        statistic: dict[str, object],
        metadata: dict[str, object],
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or not np.all(np.isfinite(covariance))
            or float(np.linalg.eigvalsh(covariance).min()) <= 0.0
        ):
            raise ValueError("R2 extractor covariance must be finite positive definite")
        belief_factor = build_action_factor(
            belief_world,
            belief_action,
            None,
            measurement_covariance_override=covariance,
        )
        if len(belief_factor.measurement_names) != covariance.shape[0]:
            raise ValueError(
                "strict R2 currently requires exclusive calibration resources; "
                "implicit free-calibration rows need their own IQ capture"
            )
        innovation = 0.5 * (
            prediction_covariance
            + covariance
            + (prediction_covariance + covariance).T
        )
        innovation_cholesky = np.linalg.cholesky(innovation)
        residual: np.ndarray | None = None
        # No residual exists when the extractor did not produce a detection.
        # Preserve that semantic distinction as JSON null instead of NaN so
        # traces remain strict-JSON and downstream causal audits cannot crash.
        whitened_norm: float | None = None
        gate_reason: str | None = None
        ambiguity_periods = tuple(
            metadata.get(
                "measurement_ambiguity_periods",
                (None,) * len(belief_factor.measurement_names),
            )
        )
        if detected and value is not None:
            candidate = _wrap_residual(
                belief_factor.measurement_names,
                np.asarray(value, dtype=float) - belief_factor.prediction,
                ambiguity_periods,
            )
            whitened_norm = float(
                np.linalg.norm(np.linalg.solve(innovation_cholesky, candidate))
            )
            if whitened_norm > self.config.r2_residual_gate_sigma:
                detected = False
                value = None
                gate_reason = "R2_extraction_residual_exceeded_consistency_gate"
            else:
                residual = candidate
        capture_power_report = getattr(capture, "power_report")()
        actor_features = receiver_observable_feature_summary(capture, statistic)
        extra = {
            "fallback_reason": gate_reason,
            "capture_fidelity": getattr(capture, "fidelity").level.value,
            "capture_power_report": capture_power_report,
            "matched_filter_statistic": statistic,
            "actor_receiver_features": actor_features,
            "path_count": len(getattr(capture, "path_components")),
            "oracle_disturbance_parameters_used_by_inference": False,
            "belief_disturbance_model": "nominal_likelihood_plus_robust_factor",
            "future_environment_state_read": False,
            "r2_whitened_residual_norm": whitened_norm,
            "r2_residual_gate_covariance": innovation,
            "r2_residual_gate_includes_prior_state_uncertainty": True,
            "r2_extraction_variance_added": True,
            "calibration_only_factors": (),
            "r2_predicted_measurement_covariance": prediction_covariance,
            "r2_likelihood_is_capture_limited_not_analytic_crlb": True,
            "residual": residual,
            "belief_factor": belief_factor,
            **metadata,
        }
        final_detection_accepted = bool(detected and residual is not None)
        # The extractor-specific classification is formed before the posterior
        # consistency gate.  Preserve that diagnostic, then make the public
        # "accepted" fields describe the measurement that can actually enter
        # the factor graph.  This keeps association counts and accepted-
        # detection counts on the same statistical population.
        extra = _finalize_offline_association_metadata(
            extra,
            detection_accepted=final_detection_accepted,
        )
        return (
            final_detection_accepted,
            value,
            belief_factor.prediction,
            covariance,
            RuntimeFidelity.R2_COMPLEX_BASEBAND.value,
            extra,
        )

    def _r2_difference(
        self,
        truth_world: PhysicalWorld,
        truth_action: RadarAction,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        belief_covariance_normalized: np.ndarray | None,
        *,
        seed: int,
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        if truth_action.kind not in {"tdoa", "fdoa"}:
            raise ValueError("difference extractor requires TDOA or FDOA")
        if len(truth_action.rx_node_ids) != 2:
            raise ValueError("strict R2 difference extraction requires two receivers")
        truth_jammer = self._truth_jammer(truth_world)
        belief_factor = build_action_factor(belief_world, belief_action, None)
        prediction_covariance, propagated = self._r2_primary_prediction_covariance(
            belief_world,
            belief_factor,
            belief_covariance_normalized,
            1,
        )
        waveform_spec = truth_world.waveform(truth_action.waveform_id)
        predicted_channels = {
            receiver_id: receiver_observable_delay_doppler(
                belief_world, belief_action, receiver_id
            )
            for receiver_id in belief_action.rx_node_ids
        }
        capture_samples = _commanded_capture_samples(self.config, belief_action)
        starts = {
            receiver_id: _belief_centered_capture_start_s(
                delay, waveform_spec, capture_samples
            )
            for receiver_id, (delay, _) in predicted_channels.items()
        }
        capture = simulate_complex_baseband(
            truth_world,
            truth_action,
            jammer=truth_jammer,
            samples=capture_samples,
            seed=seed,
            impairments=True,
            capture_start_s=starts[truth_action.rx_node_ids[0]],
            capture_start_s_by_channel=starts,
        )
        waveform = synthesize_waveform(
            waveform_spec,
            capture_samples,
            truth_world.seed + truth_action.action_id + 31,
        )
        support = _coherent_capture_support(capture, waveform_spec)
        capture_duration_s = float(support["fast_window_s"])
        capture_cpi_s = float(support["actual_cpi_s"])
        gain_fraction = float(support["gain_fraction"])
        planning_predicted_capture_sinr = max(
            float(belief_factor.link_budget.sinr_linear) * gain_fraction,
            1e-12,
        )
        differential_std = math.sqrt(
            max(float(prediction_covariance[0, 0]), 0.0)
        )
        predicted_delay_crlb = delay_variance_s2(
            waveform_spec, planning_predicted_capture_sinr
        )
        capture_time_rms = float(support["time_rms_s"])
        predicted_doppler_crlb = 1.0 / (
            8.0
            * math.pi**2
            * capture_time_rms**2
            * planning_predicted_capture_sinr
        )
        if truth_action.kind == "tdoa":
            delay_prior_std = differential_std
            doppler_prior_std = math.sqrt(predicted_doppler_crlb)
        else:
            delay_prior_std = math.sqrt(predicted_delay_crlb)
            doppler_prior_std = differential_std
        delay_half_width = max(
            self.config.r2_delay_search_sigmas
            * max(delay_prior_std, math.sqrt(predicted_delay_crlb)),
            self.config.r2_delay_search_minimum_cells
            / max(waveform_spec.bandwidth_hz, 1.0),
            8.0 / waveform_spec.sample_rate_hz,
        )
        doppler_half_width = max(
            self.config.r2_doppler_search_sigmas
            * max(doppler_prior_std, math.sqrt(predicted_doppler_crlb)),
            self.config.r2_doppler_search_minimum_cells
            * slow_time_window_contract(
                waveform_spec.coherent_pulse_count,
                waveform_spec.pulse_repetition_hz,
                self.config.r2_slow_time_window,
            )[1]
            / capture_cpi_s,
        )
        detector_parameters = _r2_detector_parameters(
            self.config,
            belief_action,
            waveform_spec,
            capture_cpi_s,
        )
        statistics: list[dict[str, object]] = []
        for receiver_id in truth_action.rx_node_ids:
            channel_capture = receiver_channel_capture(capture, receiver_id)
            expected_delay, expected_doppler = predicted_channels[receiver_id]
            statistics.append(
                matched_filter_sufficient_statistic(
                    channel_capture,
                    waveform,
                    expected_delay_s=expected_delay,
                    expected_doppler_hz=expected_doppler,
                    delay_search_half_width_s=delay_half_width,
                    doppler_search_half_width_hz=doppler_half_width,
                    coherent_processing_interval_s=capture_cpi_s,
                    slow_time_window=self.config.r2_slow_time_window,
                    **detector_parameters,
                )
            )
        detected = bool(all(bool(item["detected"]) for item in statistics))
        receiver_sinrs = [
            max(float(item["receiver_estimated_output_sinr_linear"]), 1e-12)
            for item in statistics
        ]
        receiver_delay_crlbs = [
            delay_variance_s2(waveform_spec, sinr) for sinr in receiver_sinrs
        ]
        receiver_doppler_crlbs = [
            1.0
            / (
                8.0
                * math.pi**2
                * float(statistic["slow_time_window_weighted_rms_s"]) ** 2
                * sinr
            )
            for statistic, sinr in zip(statistics, receiver_sinrs, strict=True)
        ]
        per_receiver_truth_attribution: list[dict[str, object]] = []
        for receiver_id, statistic in zip(
            truth_action.rx_node_ids,
            statistics,
            strict=True,
        ):
            channel_capture = receiver_channel_capture(capture, receiver_id)
            target_power: float | None = None
            non_target_power: float | None = None
            target_fraction: float | None = None
            if (
                bool(statistic["detected"])
                and channel_capture.coherent_target_origin_iq is not None
                and channel_capture.coherent_pre_front_end_iq is not None
            ):
                target_response = latent_component_ambiguity_response_at_peak(
                    channel_capture,
                    statistic,
                    channel_capture.coherent_target_origin_iq,
                )
                total_response = latent_component_ambiguity_response_at_peak(
                    channel_capture,
                    statistic,
                    channel_capture.coherent_pre_front_end_iq,
                )
                non_target_response = total_response - target_response
                target_power = float(abs(target_response) ** 2)
                non_target_power = float(abs(non_target_response) ** 2)
                target_fraction = float(
                    target_power
                    / max(
                        target_power + non_target_power,
                        np.finfo(float).tiny,
                    )
                )
            per_receiver_truth_attribution.append(
                {
                    "receiver_node_id": receiver_id,
                    "raw_detector_accepted": bool(statistic["detected"]),
                    "target_origin_response_power": target_power,
                    "non_target_response_power": non_target_power,
                    "target_origin_response_fraction": target_fraction,
                }
            )
        pulse_repetition_values = tuple(
            float(item["pulse_repetition_hz"])
            for item in statistics
            if item.get("pulse_repetition_hz") is not None
        )
        if len(pulse_repetition_values) != len(statistics) or not np.allclose(
            pulse_repetition_values,
            pulse_repetition_values[0],
            rtol=0.0,
            atol=max(abs(pulse_repetition_values[0]) * 1.0e-12, 1.0e-12),
        ):
            raise ValueError(
                "strict R2 difference extraction requires a common coherent PRF"
            )
        if truth_action.kind == "tdoa":
            difference = float(statistics[0]["absolute_delay_s"]) - float(
                statistics[1]["absolute_delay_s"]
            )
            variance = sum(receiver_delay_crlbs) + sum(
                float(item["extraction_variance_delay_s2"])
                for item in statistics
            )
            feature_statistic = dict(statistics[0])
            feature_statistic["delay_s"] = difference
            measurement_name = "tdoa_s"
            ambiguity_period: float | None = None
            truth_channels = [
                receiver_observable_delay_doppler(
                    truth_world,
                    truth_action,
                    receiver_id,
                )
                for receiver_id in truth_action.rx_node_ids
            ]
            truth_difference = float(truth_channels[0][0] - truth_channels[1][0])
        else:
            difference = float(statistics[0]["doppler_hz"]) - float(
                statistics[1]["doppler_hz"]
            )
            variance = sum(receiver_doppler_crlbs) + sum(
                float(item["extraction_variance_doppler_hz2"])
                for item in statistics
            )
            feature_statistic = dict(statistics[0])
            feature_statistic["doppler_hz"] = difference
            measurement_name = "fdoa_hz"
            ambiguity_period = float(pulse_repetition_values[0])
            truth_channels = [
                receiver_observable_delay_doppler(
                    truth_world,
                    truth_action,
                    receiver_id,
                )
                for receiver_id in truth_action.rx_node_ids
            ]
            truth_difference = float(truth_channels[0][1] - truth_channels[1][1])
        feature_statistic["detected"] = detected
        value = np.asarray([difference], dtype=float) if detected else None
        difference_truth_audit = _offline_difference_target_origin_audit(
            measurement_name=measurement_name,
            observed_difference=difference if detected else None,
            truth_difference=truth_difference,
            variance=variance,
            raw_detected=detected,
            ambiguity_period=ambiguity_period,
            per_receiver_target_origin_response_fractions=[
                item["target_origin_response_fraction"]
                for item in per_receiver_truth_attribution
            ],
        )
        return self._finish_r2_extraction(
            belief_world,
            belief_action,
            value=value,
            covariance=np.asarray([[max(variance, 1e-30)]], dtype=float),
            prediction_covariance=prediction_covariance,
            detected=detected,
            capture=capture,
            statistic=feature_statistic,
            metadata={
                "r2_difference_kind": truth_action.kind,
                "measurement_ambiguity_periods": (
                    [float(pulse_repetition_values[0])]
                    if truth_action.kind == "fdoa"
                    else [None]
                ),
                "doppler_integer_ambiguity_model": (
                    "R_modulo_PRF"
                    if truth_action.kind == "fdoa"
                    else "not_applicable"
                ),
                "r2_per_receiver_statistics": statistics,
                "simulator_truth_per_receiver_attribution": (
                    per_receiver_truth_attribution
                ),
                "r2_per_receiver_belief_predictions": {
                    key: list(value) for key, value in predicted_channels.items()
                },
                "r2_independent_receiver_range_gates": True,
                "r2_common_illuminator_leg_cancelled_by_difference": True,
                "r2_capture_processing_gain_fraction": gain_fraction,
                "r2_planning_predicted_capture_sinr_linear": (
                    planning_predicted_capture_sinr
                ),
                "r2_receiver_estimated_output_sinr_linear_by_channel": (
                    receiver_sinrs
                ),
                "r2_capture_limited_sinr_linear": min(receiver_sinrs),
                "r2_measurement_covariance_uses_receiver_observable_sinr": True,
                "r2_planning_sinr_used_only_for_search_support": True,
                "r2_detector_parameters": detector_parameters,
                "r2_fast_time_window_s": capture_duration_s,
                "r2_commanded_fast_time_samples": capture_samples,
                "r2_coherent_pulse_count": int(support["pulse_count"]),
                "r2_coherent_slow_time_aperture_s": capture_cpi_s,
                "r2_captured_on_air_dwell_s": float(
                    support["captured_dwell_s"]
                ),
                "r2_doppler_information_time_axis": "pulse_slow_time",
                "r2_fast_time_used_as_doppler_aperture": False,
                "r2_search_uses_propagated_belief_covariance": propagated,
                "simulator_truth_jammer_present_for_offline_audit": (
                    truth_jammer is not None
                ),
                "simulator_truth_target_present_for_offline_audit": bool(
                    capture.target_present
                ),
                **difference_truth_audit,
            },
        )

    def _r2_aoa(
        self,
        truth_world: PhysicalWorld,
        truth_action: RadarAction,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        belief_covariance_normalized: np.ndarray | None,
        *,
        seed: int,
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        truth_jammer = self._truth_jammer(truth_world)
        belief_factor = build_action_factor(belief_world, belief_action, None)
        prediction_covariance, propagated = self._r2_primary_prediction_covariance(
            belief_world,
            belief_factor,
            belief_covariance_normalized,
            2,
        )
        waveform_spec = truth_world.waveform(truth_action.waveform_id)
        receiver_id = belief_action.rx_node_ids[0]
        predicted_delay, predicted_doppler = receiver_observable_delay_doppler(
            belief_world, belief_action, receiver_id
        )
        capture_samples = _commanded_capture_samples(self.config, belief_action)
        capture_start = _belief_centered_capture_start_s(
            predicted_delay, waveform_spec, capture_samples
        )
        capture = simulate_complex_baseband(
            truth_world,
            truth_action,
            jammer=truth_jammer,
            samples=capture_samples,
            capture_start_s=capture_start,
            seed=seed,
            impairments=True,
        )
        waveform = synthesize_waveform(
            waveform_spec,
            capture_samples,
            truth_world.seed + truth_action.action_id + 31,
        )
        support = _coherent_capture_support(capture, waveform_spec)
        capture_cpi_s = float(support["actual_cpi_s"])
        gain_fraction = float(support["gain_fraction"])
        planning_predicted_capture_sinr = max(
            float(belief_factor.link_budget.sinr_linear) * gain_fraction,
            1e-12,
        )
        delay_standard_deviation_by_receiver = belief_action.metadata.get(
            "planning_delay_standard_deviation_s_by_receiver", {}
        )
        if not isinstance(delay_standard_deviation_by_receiver, Mapping):
            raise ValueError(
                "planning delay uncertainty must be a receiver mapping"
            )
        delay_prior_std_s = float(
            delay_standard_deviation_by_receiver.get(receiver_id, 0.0)
        )
        if not math.isfinite(delay_prior_std_s) or delay_prior_std_s < 0.0:
            raise ValueError(
                "planning delay standard deviation must be finite and nonnegative"
            )
        predicted_capture_delay_variance = delay_variance_s2(
            waveform_spec, planning_predicted_capture_sinr
        )
        delay_search_half_width_s = max(
            self.config.r2_delay_search_sigmas
            * max(
                delay_prior_std_s,
                math.sqrt(predicted_capture_delay_variance),
            ),
            self.config.r2_delay_search_minimum_cells
            / max(waveform_spec.bandwidth_hz, 1.0),
            8.0 / waveform_spec.sample_rate_hz,
        )
        doppler_search_half_width_hz = (
            0.5
            * waveform_spec.pulse_repetition_hz
            * (1.0 - 1.0e-12)
        )
        receiver = belief_world.node(receiver_id)
        antenna = antenna_from_node(
            receiver,
            waveform_spec.carrier_hz,
            mutual_coupling_strength=float(
                belief_action.metadata.get("rx_mutual_coupling", 0.0)
            ),
            station_phase_rad=float(
                belief_action.metadata.get("station_phase_rad", 0.0)
            ),
        )
        predicted_analytic_covariance = angle_covariance_rad2(
            receiver,
            waveform_spec,
            planning_predicted_capture_sinr,
            float(belief_factor.prediction[0]),
            float(belief_factor.prediction[1]),
        )
        azimuth_half_width = max(
            4.0
            * max(
                math.sqrt(max(float(prediction_covariance[0, 0]), 0.0)),
                math.sqrt(float(predicted_analytic_covariance[0, 0])),
            ),
            math.radians(8.0),
        )
        elevation_half_width = max(
            4.0
            * max(
                math.sqrt(max(float(prediction_covariance[1, 1]), 0.0)),
                math.sqrt(float(predicted_analytic_covariance[1, 1])),
            ),
            math.radians(6.0),
        )
        detector_parameters = _r2_detector_parameters(
            self.config,
            belief_action,
            waveform_spec,
            capture_cpi_s,
        )
        estimator_transmitter_id = (
            belief_action.tx_node_id or belief_action.reference_node_id
        )
        estimator_transmitter_position = (
            belief_world.node(estimator_transmitter_id).position_m
            if estimator_transmitter_id is not None
            else belief_world.spectrum_emitters[0].position_m
        )
        predicted_target_range_m = float(
            np.linalg.norm(
                np.asarray(belief_world.target.position_m, dtype=float)
                - np.asarray(receiver.position_m, dtype=float)
            )
        )
        aoa_hpbw_rad = rectangular_array_hpbw_rad(
            receiver, waveform_spec.carrier_hz
        )
        statistic = pulse_compressed_array_angle_sufficient_statistic(
            truth_world,
            truth_action,
            capture,
            waveform,
            antenna,
            jammer=truth_jammer,
            expected_delay_s=predicted_delay,
            expected_doppler_hz=predicted_doppler,
            expected_azimuth_rad=float(belief_factor.prediction[0]),
            expected_elevation_rad=float(belief_factor.prediction[1]),
            azimuth_half_width_rad=min(azimuth_half_width, math.pi),
            elevation_half_width_rad=min(elevation_half_width, 0.49 * math.pi),
            delay_search_half_width_s=delay_search_half_width_s,
            doppler_search_half_width_hz=doppler_search_half_width_hz,
            estimator_receiver_position_m=receiver.position_m,
            estimator_target_range_m=predicted_target_range_m,
            estimator_target_position_m=belief_world.target.position_m,
            estimator_transmitter_position_m=(
                estimator_transmitter_position
            ),
            estimator_ground_height_m=float(belief_world.ground_height_m),
            estimator_surface=belief_world.surface,
            slow_time_window=self.config.r2_slow_time_window,
            seed=seed,
            **detector_parameters,
        )
        temporal_output_sinr = max(
            float(statistic["receiver_estimated_output_sinr_linear"]),
            0.0,
        )
        spatial_output_sinr = max(
            float(statistic["spatial_output_sinr_linear"]),
            0.0,
        )
        receiver_estimated_capture_sinr = max(
            min(temporal_output_sinr, spatial_output_sinr),
            1e-12,
        )
        receiver_covariance = angle_covariance_rad2(
            receiver,
            waveform_spec,
            receiver_estimated_capture_sinr,
            float(belief_factor.prediction[0]),
            float(belief_factor.prediction[1]),
        )
        extraction_covariance = np.diag(
            [
                float(statistic["extraction_variance_azimuth_rad2"]),
                float(statistic["extraction_variance_elevation_rad2"]),
            ]
        )
        detected = bool(statistic["detected"])
        profile_covariance = np.asarray(
            statistic["profile_likelihood_covariance_rad2"], dtype=float
        )
        discrepancy_covariance = np.asarray(
            statistic["model_discrepancy_covariance_rad2"], dtype=float
        )
        if detected:
            covariance = (
                _loewner_upper_envelope(
                    receiver_covariance,
                    profile_covariance,
                )
                + extraction_covariance
                + discrepancy_covariance
            )
            covariance_semantics = (
                "Loewner_envelope_of_ideal_CRLB_and_observed_profile_curvature_"
                "plus_grid_and_direct_vs_ground_image_model_discrepancy"
            )
        else:
            covariance = np.diag(
                [
                    min(azimuth_half_width, math.pi) ** 2 / 3.0,
                    min(elevation_half_width, 0.49 * math.pi) ** 2 / 3.0,
                ]
            )
            covariance_semantics = (
                "uniform_variance_over_unresolved_search_support_not_admitted_"
                "to_the_factor_graph"
            )
        covariance = 0.5 * (covariance + covariance.T)
        raw_value = np.asarray(
            [statistic["azimuth_rad"], statistic["elevation_rad"]],
            dtype=float,
        )
        value = (
            raw_value.copy()
            if detected
            else None
        )
        truth_factor = build_action_factor(truth_world, truth_action, None)
        truth_prediction = np.asarray(truth_factor.prediction, dtype=float)
        truth_residual = _wrap_residual(
            truth_factor.measurement_names,
            raw_value - truth_prediction,
        )
        truth_target_range_m = float(
            np.linalg.norm(
                np.asarray(truth_world.target.position_m, dtype=float)
                - np.asarray(
                    truth_world.node(truth_action.rx_node_ids[0]).position_m,
                    dtype=float,
                )
            )
        )
        truth_cross_range_error_m = (
            truth_target_range_m * float(np.linalg.norm(truth_residual))
            if detected
            else None
        )
        truth_mahalanobis2 = (
            float(
                truth_residual
                @ np.linalg.pinv(covariance, rcond=1.0e-12, hermitian=True)
                @ truth_residual
            )
            if detected
            else None
        )
        target_response_power = float(
            statistic["simulator_truth_target_component_response_power"]
        )
        nontarget_response_power = float(
            statistic["simulator_truth_nontarget_component_response_power"]
        )
        target_response_fraction = target_response_power / max(
            target_response_power + nontarget_response_power,
            np.finfo(float).tiny,
        )
        truth_gate_threshold = 9.21034037197618
        offline_classification = _offline_target_origin_classification(
            truth_mahalanobis2,
            truth_gate_threshold,
            target_response_fraction,
        )
        return self._finish_r2_extraction(
            belief_world,
            belief_action,
            value=value,
            covariance=covariance,
            prediction_covariance=prediction_covariance,
            detected=detected,
            capture=capture,
            statistic=statistic,
            metadata={
                "r2_aoa_element_level_iq": False,
                "r2_aoa_pulse_compressed_array_sufficient_statistic": True,
                "r2_aoa_full_raw_element_pulse_cube_materialized": False,
                "r2_commanded_fast_time_samples": capture_samples,
                "r2_aoa_array_elements": int(antenna.element_count),
                "r2_aoa_predicted_target_range_m": predicted_target_range_m,
                "r2_aoa_azimuth_hpbw_rad": float(aoa_hpbw_rad[0]),
                "r2_aoa_elevation_hpbw_rad": float(aoa_hpbw_rad[1]),
                "r2_aoa_truth_angle_used_by_estimator": False,
                "r2_aoa_truth_fields_used_only_for_offline_audit": True,
                "r2_aoa_coherent_slow_time_used_for_detection": True,
                "r2_aoa_coherent_processing_interval_s": capture_cpi_s,
                "r2_aoa_captured_on_air_dwell_s": float(
                    support["captured_dwell_s"]
                ),
                "r2_capture_processing_gain_fraction": gain_fraction,
                "r2_aoa_delay_prior_standard_deviation_s": (
                    delay_prior_std_s
                ),
                "r2_aoa_delay_search_half_width_s": (
                    delay_search_half_width_s
                ),
                "r2_aoa_doppler_search_half_width_hz": (
                    doppler_search_half_width_hz
                ),
                "r2_aoa_search_uses_posterior_delay_uncertainty": True,
                "r2_planning_predicted_capture_sinr_linear": (
                    planning_predicted_capture_sinr
                ),
                "r2_receiver_estimated_output_sinr_linear": (
                    receiver_estimated_capture_sinr
                ),
                "r2_temporal_output_sinr_linear": temporal_output_sinr,
                "r2_spatial_output_sinr_linear": spatial_output_sinr,
                "r2_aoa_covariance_uses_minimum_temporal_spatial_sinr": True,
                "r2_aoa_covariance_semantics": covariance_semantics,
                "r2_aoa_covariance_conditioned_on_detection": detected,
                "r2_aoa_ideal_crlb_covariance_rad2": receiver_covariance,
                "r2_aoa_profile_likelihood_covariance_rad2": (
                    profile_covariance
                ),
                "r2_aoa_model_discrepancy_covariance_rad2": (
                    discrepancy_covariance
                ),
                "r2_aoa_extraction_grid_covariance_rad2": (
                    extraction_covariance
                ),
                "r2_aoa_estimator_profiles_common_target_complex_amplitude": True,
                "r2_aoa_ground_reflection_ratio_is_Fresnel_waveform_constrained": (
                    True
                ),
                "r2_aoa_multipath_profile_uses_belief_range_not_truth": True,
                "r2_capture_limited_sinr_linear": (
                    receiver_estimated_capture_sinr
                ),
                "r2_measurement_covariance_uses_receiver_observable_sinr": True,
                "r2_planning_sinr_used_only_for_search_support": True,
                "r2_detector_parameters": detector_parameters,
                "r2_search_uses_propagated_belief_covariance": propagated,
                "simulator_truth_jammer_present_for_offline_audit": (
                    truth_jammer is not None
                ),
                "simulator_truth_measurement_vector": truth_prediction,
                "simulator_truth_raw_measurement_residual": truth_residual,
                "simulator_truth_aoa_error_rad": truth_residual,
                "simulator_truth_aoa_cross_range_error_m": (
                    truth_cross_range_error_m
                ),
                "simulator_truth_aoa_receiver_target_range_m": (
                    truth_target_range_m
                ),
                "simulator_truth_aoa_mahalanobis2": truth_mahalanobis2,
                "simulator_truth_aoa_threshold_chi2_2d_99pct": (
                    truth_gate_threshold
                ),
                "simulator_truth_target_origin_response_power": (
                    target_response_power
                ),
                "simulator_truth_non_target_response_power": (
                    nontarget_response_power
                ),
                "simulator_truth_target_origin_response_fraction": (
                    target_response_fraction
                ),
                "simulator_truth_target_origin_dominant_at_selected_cell": (
                    offline_classification["target_origin_dominant"]
                ),
                "simulator_truth_raw_target_peak_within_99pct_gate": (
                    offline_classification["near_truth"]
                ),
                "simulator_truth_near_truth_peak_but_nontarget_dominated": (
                    offline_classification[
                        "near_truth_but_nontarget_dominated"
                    ]
                ),
                "simulator_truth_target_association_accepted": bool(
                    detected and offline_classification["target_associated"]
                ),
                "simulator_truth_false_alarm_like_accepted": bool(
                    detected and not offline_classification["target_associated"]
                ),
                "simulator_truth_target_peak_rejected_by_prior_gate": bool(
                    not statistic.get(
                        "simulator_truth_target_observable_inside_search",
                        False,
                    )
                ),
                "simulator_truth_target_peak_rejected_by_spatial_cfar": bool(
                    target_response_fraction > 0.5
                    and bool(statistic.get("temporal_cfar_detected", False))
                    and not bool(statistic.get("spatial_cfar_passed", False))
                ),
                "simulator_truth_target_present_for_offline_audit": bool(
                    capture.target_present
                ),
                "simulator_truth_measurement_modality": "aoa",
            },
        )

    def _r2_calibration(
        self,
        truth_world: PhysicalWorld,
        truth_action: RadarAction,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        belief_covariance_normalized: np.ndarray | None,
        *,
        seed: int,
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        belief_factor = build_action_factor(belief_world, belief_action, None)
        prediction_covariance, propagated = self._r2_primary_prediction_covariance(
            belief_world,
            belief_factor,
            belief_covariance_normalized,
            1,
        )
        predicted = float(belief_factor.prediction[0])
        waveform_spec = truth_world.waveform(truth_action.waveform_id)
        capture_start = (
            predicted
            - min(48.0, self.config.r2_samples / 6.0)
            / waveform_spec.sample_rate_hz
            if truth_action.kind == "calibration_time"
            else None
        )
        capture = simulate_calibration_baseband(
            truth_world,
            truth_action,
            samples=self.config.r2_samples,
            capture_start_s=capture_start,
            seed=seed,
            impairments=True,
        )
        if truth_action.kind == "calibration_time":
            waveform = synthesize_waveform(
                waveform_spec,
                self.config.r2_samples,
                truth_world.seed + truth_action.action_id + 131,
            )
            receiver = belief_world.node(belief_action.rx_node_ids[0])
            reference = (
                belief_world.node(belief_action.reference_node_id)
                if belief_action.reference_node_id is not None
                else None
            )
            expected_frequency = float(receiver.frequency_bias_hz) - (
                0.0 if reference is None else float(reference.frequency_bias_hz)
            )
            prior_std = math.sqrt(
                max(float(prediction_covariance[0, 0]), 0.0)
            )
            declared_std = math.sqrt(float(belief_factor.covariance[0, 0]))
            statistic = matched_filter_sufficient_statistic(
                capture,
                waveform,
                expected_delay_s=predicted,
                expected_doppler_hz=expected_frequency,
                # A scheduled time-transfer burst estimates delay conditional
                # on the causal differential-CFO prediction.  Its sub-
                # microsecond fast-time record does not provide an independent
                # slow-time Doppler aperture.  Searching a full Nyquist Doppler
                # interval therefore creates a flat, non-identifiable nuisance
                # dimension whose random boundary maximum incorrectly rejects
                # a strong delay peak.  Frequency calibration remains a
                # separate registered action and nuisance factor.
                doppler_grid_hz=np.asarray([expected_frequency], dtype=float),
                delay_search_half_width_s=max(
                    self.config.r2_delay_search_sigmas
                    * max(prior_std, declared_std),
                    self.config.r2_delay_search_minimum_cells
                    / max(waveform_spec.bandwidth_hz, 1.0),
                    8.0 / waveform_spec.sample_rate_hz,
                ),
                coherent_processing_interval_s=float(
                    capture.metadata["calibration_duration_s"]
                ),
            )
            detected = bool(statistic["detected"])
            value = (
                np.asarray([statistic["absolute_delay_s"]], dtype=float)
                if detected
                else None
            )
            variance = float(belief_factor.covariance[0, 0]) + float(
                statistic["extraction_variance_delay_s2"]
            )
        else:
            statistic = calibration_frequency_sufficient_statistic(capture)
            detected = bool(statistic["detected"])
            value = (
                np.asarray([statistic["frequency_hz"]], dtype=float)
                if detected
                else None
            )
            variance = max(
                float(belief_factor.covariance[0, 0]),
                float(statistic["frequency_variance_hz2"]),
            )
        return self._finish_r2_extraction(
            belief_world,
            belief_action,
            value=value,
            covariance=np.asarray([[max(variance, 1e-30)]], dtype=float),
            prediction_covariance=prediction_covariance,
            detected=detected,
            capture=capture,
            statistic=statistic,
            metadata={
                "r2_calibration_kind": truth_action.kind,
                "r2_calibration_finite_window_iq": True,
                "r2_calibration_truth_used_by_estimator": False,
                "r2_time_calibration_doppler_estimand": (
                    "delay_conditioned_on_causal_differential_CFO_prediction"
                    if truth_action.kind == "calibration_time"
                    else "not_applicable"
                ),
                "r2_time_calibration_independent_doppler_claimed": False,
                "r2_search_uses_propagated_belief_covariance": propagated,
                "simulator_truth_jammer_present_for_offline_audit": False,
                "simulator_truth_target_present_for_offline_audit": False,
                "simulator_truth_measurement_modality": truth_action.kind,
            },
        )

    def _r2(
        self,
        truth_world: PhysicalWorld,
        truth_action: RadarAction,
        belief_world: PhysicalWorld,
        belief_action: RadarAction,
        belief_covariance_normalized: np.ndarray | None = None,
        *,
        epoch: int,
        seed: int,
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray, str, dict[str, object]]:
        truth_jammer = self._truth_jammer(truth_world)
        belief_factor = build_action_factor(belief_world, belief_action, None)
        if truth_action.kind not in _R2_SUPPORTED_KINDS:
            if not self.config.allow_r2_to_r1_fallback:
                raise ValueError(
                    f"R2 extraction is not registered for action kind {truth_action.kind!r}"
                )
            return self._r1(
                truth_world,
                truth_action,
                belief_world,
                belief_action,
                epoch=epoch,
                seed=seed,
                fallback_reason="R2_extractor_not_registered_for_action_or_implicit_calibration_rows",
            )
        if truth_action.kind in {"tdoa", "fdoa"}:
            return self._r2_difference(
                truth_world,
                truth_action,
                belief_world,
                belief_action,
                belief_covariance_normalized,
                seed=seed,
            )
        if truth_action.kind == "aoa":
            return self._r2_aoa(
                truth_world,
                truth_action,
                belief_world,
                belief_action,
                belief_covariance_normalized,
                seed=seed,
            )
        if truth_action.kind.startswith("calibration"):
            return self._r2_calibration(
                truth_world,
                truth_action,
                belief_world,
                belief_action,
                belief_covariance_normalized,
                seed=seed,
            )
        if (
            len(belief_factor.measurement_names) != 2
            or not (
                ("delay" in belief_factor.measurement_names[0])
                and ("doppler" in belief_factor.measurement_names[1])
            )
        ):
            raise ValueError(
                "strict R2 delay-Doppler extraction encountered implicit rows "
                "or an incompatible measurement schema"
            )
        # The range gate and the ambiguity search are receiver configuration,
        # so both are derived from the *belief* prediction.  Using the truth
        # referenced default gate would make the reported delay equal to truth
        # plus a grid artefact -- an oracle whose accuracy is independent of SNR.
        physical_waveform = truth_world.waveform(truth_action.waveform_id)
        predicted_delay_s = float(belief_factor.prediction[0])
        predicted_doppler_hz = float(belief_factor.prediction[1])
        layout = StateLayout.from_world(belief_world)
        primary_jacobian_normalized = (
            belief_factor.jacobian_physical[:2]
            * layout.scales()[None, :]
        )
        if belief_covariance_normalized is None:
            prediction_covariance = np.asarray(
                belief_factor.covariance[:2, :2],
                dtype=float,
            )
            propagated_belief_covariance = False
        else:
            posterior_covariance = np.asarray(
                belief_covariance_normalized,
                dtype=float,
            )
            if posterior_covariance.shape != (layout.dimension, layout.dimension):
                raise ValueError(
                    "R2 belief covariance dimension disagrees with the measurement state"
                )
            prediction_covariance = (
                primary_jacobian_normalized
                @ posterior_covariance
                @ primary_jacobian_normalized.T
            )
            propagated_belief_covariance = True
        prediction_covariance = 0.5 * (
            prediction_covariance + prediction_covariance.T
        )
        prediction_eigenvalues = np.linalg.eigvalsh(prediction_covariance)
        tolerance = max(
            float(np.max(np.abs(prediction_eigenvalues))) * 1e-10,
            np.finfo(float).tiny,
        )
        if float(prediction_eigenvalues.min()) < -tolerance:
            raise np.linalg.LinAlgError(
                "R2 predicted measurement covariance is materially indefinite"
            )
        delay_prior_std_s = math.sqrt(
            max(float(prediction_covariance[0, 0]), 0.0)
        )
        doppler_prior_std_hz = math.sqrt(
            max(float(prediction_covariance[1, 1]), 0.0)
        )
        capture_samples = _commanded_capture_samples(self.config, belief_action)
        capture_start_s = _belief_centered_capture_start_s(
            predicted_delay_s, physical_waveform, capture_samples
        )
        capture = simulate_complex_baseband(
            truth_world,
            truth_action,
            jammer=truth_jammer,
            samples=capture_samples,
            seed=seed,
            impairments=True,
            capture_start_s=capture_start_s,
        )
        waveform = synthesize_waveform(
            physical_waveform,
            capture_samples,
            truth_world.seed + truth_action.action_id + 31,
        )
        # ---- honest R2 evidence boundary ---------------------------------
        # The range gate retains ``r2_samples`` of fast time from *each* pulse;
        # the pulse cube spans the declared slow-time aperture.  Consequently
        # delay information depends on captured on-air dwell, while Doppler
        # information depends on the slow-time second moment.  Treating the
        # microsecond fast-time window as the CPI produced MHz-scale artefacts
        # for targets whose physical Doppler is O(10^2--10^3 Hz).
        support = _coherent_capture_support(capture, physical_waveform)
        capture_duration_s = float(support["fast_window_s"])
        capture_cpi_s = float(support["actual_cpi_s"])
        declared_cpi_s = max(float(physical_waveform.coherent_processing_interval_s), 1e-30)
        declared_dwell_s = max(float(physical_waveform.dwell_s), 1e-30)
        analytic_processing_gain = float(support["analytic_gain"])
        capture_processing_gain = float(support["capture_gain"])
        processing_gain_fraction = float(support["gain_fraction"])
        planning_predicted_capture_sinr = max(
            float(belief_factor.link_budget.sinr_linear)
            * processing_gain_fraction,
            1e-12,
        )
        predicted_capture_delay_variance = delay_variance_s2(
            physical_waveform, planning_predicted_capture_sinr
        )
        capture_time_rms = float(support["time_rms_s"])
        predicted_capture_doppler_variance = 1.0 / (
            8.0
            * math.pi**2
            * capture_time_rms**2
            * planning_predicted_capture_sinr
        )
        # Search half-widths are belief quantities: several prediction sigmas,
        # floored by the resolution cells the capture can actually resolve, so
        # the mainlobe is always inside the search region.
        delay_half_width_s = max(
            self.config.r2_delay_search_sigmas
            * max(delay_prior_std_s, math.sqrt(predicted_capture_delay_variance)),
            self.config.r2_delay_search_minimum_cells / max(physical_waveform.bandwidth_hz, 1.0),
            8.0 / physical_waveform.sample_rate_hz,
        )
        doppler_half_width_hz = max(
            self.config.r2_doppler_search_sigmas
            * max(
                doppler_prior_std_hz,
                math.sqrt(predicted_capture_doppler_variance),
            ),
            self.config.r2_doppler_search_minimum_cells
            * slow_time_window_contract(
                physical_waveform.coherent_pulse_count,
                physical_waveform.pulse_repetition_hz,
                self.config.r2_slow_time_window,
            )[1]
            / capture_cpi_s,
        )
        detector_parameters = _r2_detector_parameters(
            self.config,
            belief_action,
            physical_waveform,
            capture_cpi_s,
        )
        statistic = matched_filter_sufficient_statistic(
            capture,
            waveform,
            expected_delay_s=predicted_delay_s,
            expected_doppler_hz=predicted_doppler_hz,
            delay_search_half_width_s=delay_half_width_s,
            doppler_search_half_width_hz=doppler_half_width_hz,
            coherent_processing_interval_s=capture_cpi_s,
            slow_time_window=self.config.r2_slow_time_window,
            **detector_parameters,
        )
        detected = bool(statistic["detected"])
        value = None
        residual = None
        # The planning link budget only configures the causal search support.
        # The realized likelihood covariance is conditioned on the receiver's
        # own ambiguity peak/background estimate after CFAR and clutter/direct-
        # path exclusion; simulator component powers are audit-only and cannot
        # make a weak or absent return artificially precise.
        receiver_estimated_capture_sinr = max(
            float(statistic["receiver_estimated_output_sinr_linear"]),
            1e-12,
        )
        capture_delay_variance = delay_variance_s2(
            physical_waveform, receiver_estimated_capture_sinr
        )
        receiver_effective_time_rms_s = float(
            statistic.get("slow_time_window_weighted_rms_s", capture_time_rms)
        )
        capture_doppler_variance = 1.0 / (
            8.0
            * math.pi**2
            * receiver_effective_time_rms_s**2
            * receiver_estimated_capture_sinr
        )
        # Total R2 error variance = capture-limited CRLB + residual grid variance
        # of the discrete-maximum search.  The two contributions are independent.
        covariance = np.diag(
            [
                capture_delay_variance + float(statistic["extraction_variance_delay_s2"]),
                capture_doppler_variance
                + float(statistic["extraction_variance_doppler_hz2"]),
            ]
        )
        # Fuse the R2 observation with its *own* likelihood.  Using the analytic
        # R1 covariance for an R2 extraction would silently label one tier's
        # measurement with another tier's uncertainty.
        r2_belief_factor = build_action_factor(
            belief_world,
            belief_action,
            None,
            measurement_covariance_override=covariance,
        )
        belief_factor = r2_belief_factor
        r2_innovation_covariance = 0.5 * (
            prediction_covariance
            + covariance
            + (prediction_covariance + covariance).T
        )
        r2_innovation_cholesky = np.linalg.cholesky(r2_innovation_covariance)
        gate_reason: str | None = None
        # Undefined is not a numerical residual.  Use None so the runtime
        # trace records a missing diagnostic rather than a nonfinite number.
        whitened_norm: float | None = None
        if detected:
            value = np.asarray(
                [float(statistic["absolute_delay_s"]), float(statistic["doppler_hz"])],
                dtype=float,
            )
            pulse_repetition_hz = statistic.get("pulse_repetition_hz")
            if pulse_repetition_hz is None:
                raise RuntimeError(
                    "strict coherent R2 delay-Doppler extraction omitted its PRF"
                )
            ambiguity_periods = (None, float(pulse_repetition_hz))
            candidate = _wrap_residual(
                belief_factor.measurement_names,
                value - belief_factor.prediction,
                ambiguity_periods,
            )
            whitened_norm = float(
                np.linalg.norm(
                    np.linalg.solve(r2_innovation_cholesky, candidate)
                )
            )
            if whitened_norm > self.config.r2_residual_gate_sigma:
                # Fail closed. The gate uses predictive innovation covariance
                # H P H^T + R: measurement covariance alone would reject a valid
                # return merely because the non-oracle prior mean is uncertain.
                detected = False
                gate_reason = "R2_extraction_residual_exceeded_consistency_gate"
                value = None
            else:
                residual = candidate
        # Offline-only detector attribution.  The controller and every Agent
        # receive the recursively sanitised observation projection, which
        # removes all ``simulator_truth_*`` keys.  The full trace nevertheless
        # needs a falsifiable distinction between a target-associated ambiguity
        # peak, a clutter/noise peak, and a true peak subsequently rejected by
        # the causal prior-consistency gate.
        raw_detector_accepted = bool(statistic["detected"])
        truth_prediction = np.asarray(
            receiver_observable_delay_doppler(
                truth_world,
                truth_action,
                truth_action.rx_node_ids[0],
            ),
            dtype=float,
        )
        truth_residual: np.ndarray | None = None
        truth_mahalanobis2: float | None = None
        if raw_detector_accepted:
            raw_value = np.asarray(
                [
                    float(statistic["absolute_delay_s"]),
                    float(statistic["doppler_hz"]),
                ],
                dtype=float,
            )
            truth_residual = _wrap_residual(
                belief_factor.measurement_names,
                raw_value - truth_prediction,
                (None, float(statistic["pulse_repetition_hz"])),
            )
            truth_mahalanobis2 = float(
                truth_residual
                @ np.linalg.pinv(covariance, rcond=1.0e-12)
                @ truth_residual
            )
        truth_association_threshold = 9.21034037197618
        target_origin_response_power: float | None = None
        non_target_response_power: float | None = None
        target_origin_response_fraction: float | None = None
        target_to_non_target_response_ratio: float | None = None
        if (
            raw_detector_accepted
            and capture.coherent_target_origin_iq is not None
            and capture.coherent_pre_front_end_iq is not None
        ):
            target_response = latent_component_ambiguity_response_at_peak(
                capture,
                statistic,
                capture.coherent_target_origin_iq,
            )
            total_pre_front_end_response = latent_component_ambiguity_response_at_peak(
                capture,
                statistic,
                capture.coherent_pre_front_end_iq,
            )
            non_target_response = total_pre_front_end_response - target_response
            target_origin_response_power = float(abs(target_response) ** 2)
            non_target_response_power = float(abs(non_target_response) ** 2)
            response_power_sum = (
                target_origin_response_power + non_target_response_power
            )
            target_origin_response_fraction = float(
                target_origin_response_power
                / max(response_power_sum, np.finfo(float).tiny)
            )
            target_to_non_target_response_ratio = float(
                target_origin_response_power
                / max(non_target_response_power, np.finfo(float).tiny)
            )
        offline_classification = _offline_target_origin_classification(
            truth_mahalanobis2,
            truth_association_threshold,
            target_origin_response_fraction,
        )
        raw_target_associated = offline_classification["target_associated"]
        false_peak_density = _r2_delay_doppler_false_peak_density(statistic)
        return (
            detected,
            value,
            belief_factor.prediction,
            covariance,
            RuntimeFidelity.R2_COMPLEX_BASEBAND.value,
            {
                "fallback_reason": gate_reason,
                "capture_fidelity": capture.fidelity.level.value,
                "capture_power_report": capture.power_report(),
                "matched_filter_statistic": statistic,
                "actor_receiver_features": receiver_observable_feature_summary(
                    capture, statistic
                ),
                "path_count": len(capture.path_components),
                "oracle_disturbance_parameters_used_by_inference": False,
                "oracle_range_gate_used": (
                    capture.metadata.get("range_gate_source")
                    != "caller_supplied_belief_referenced_gate"
                ),
                "belief_disturbance_model": "nominal_likelihood_plus_robust_factor",
                "future_environment_state_read": False,
                "r2_whitened_residual_norm": whitened_norm,
                "r2_residual_gate_covariance": r2_innovation_covariance,
                "r2_residual_gate_includes_prior_state_uncertainty": True,
                "r2_extraction_variance_added": True,
                "calibration_only_factors": (),
                "r2_capture_duration_s": capture_duration_s,
                "r2_commanded_fast_time_samples": capture_samples,
                "r2_fast_time_window_s": capture_duration_s,
                "r2_coherent_pulse_count": int(support["pulse_count"]),
                "r2_coherent_slow_time_aperture_s": capture_cpi_s,
                "r2_captured_on_air_dwell_s": float(
                    support["captured_dwell_s"]
                ),
                "r2_declared_coherent_processing_interval_s": declared_cpi_s,
                "r2_capture_fraction_of_declared_cpi": min(
                    1.0, capture_cpi_s / declared_cpi_s
                ),
                "r2_declared_on_air_dwell_s": declared_dwell_s,
                "r2_analytic_processing_gain_linear": analytic_processing_gain,
                "r2_capture_processing_gain_linear": capture_processing_gain,
                "r2_capture_processing_gain_fraction": processing_gain_fraction,
                "r2_planning_capture_sinr_scaling_law": (
                    "min(1, B*T_captured_on_air/(B*T_declared_dwell))"
                ),
                "r2_doppler_information_time_axis": "pulse_slow_time",
                "r2_fast_time_used_as_doppler_aperture": False,
                "r2_planning_predicted_capture_sinr_linear": (
                    planning_predicted_capture_sinr
                ),
                "r2_receiver_estimated_output_sinr_linear": (
                    receiver_estimated_capture_sinr
                ),
                "r2_capture_limited_sinr_linear": (
                    receiver_estimated_capture_sinr
                ),
                "r2_measurement_covariance_uses_receiver_observable_sinr": True,
                "r2_planning_sinr_used_only_for_search_support": True,
                "r2_detector_parameters": detector_parameters,
                "r2_search_uses_propagated_belief_covariance": (
                    propagated_belief_covariance
                ),
                "r2_predicted_measurement_covariance": prediction_covariance,
                "r2_capture_delay_crlb_s2": float(capture_delay_variance),
                "r2_capture_doppler_crlb_hz2": float(capture_doppler_variance),
                "r2_doppler_crlb_time_rms_s": receiver_effective_time_rms_s,
                "r2_likelihood_is_capture_limited_not_analytic_crlb": True,
                "measurement_ambiguity_periods": [
                    None,
                    float(statistic["pulse_repetition_hz"]),
                ],
                "doppler_integer_ambiguity_model": "R_modulo_PRF",
                "doppler_residual_principal_interval_hz": [
                    -0.5 * float(statistic["pulse_repetition_hz"]),
                    0.5 * float(statistic["pulse_repetition_hz"]),
                ],
                "doppler_alias_branch_truth_unwrapped": False,
                "residual": residual,
                "belief_factor": belief_factor,
                "simulator_truth_measurement_vector": truth_prediction,
                "simulator_truth_raw_measurement_residual": truth_residual,
                "simulator_truth_target_association_mahalanobis2": (
                    truth_mahalanobis2
                ),
                "simulator_truth_target_association_threshold_chi2_2d_99pct": (
                    truth_association_threshold
                ),
                "simulator_truth_raw_target_peak_within_99pct_gate": (
                    offline_classification["near_truth"]
                ),
                "simulator_truth_target_origin_response_power": (
                    target_origin_response_power
                ),
                "simulator_truth_non_target_response_power": (
                    non_target_response_power
                ),
                "simulator_truth_target_origin_response_fraction": (
                    target_origin_response_fraction
                ),
                "simulator_truth_target_to_non_target_response_ratio": (
                    target_to_non_target_response_ratio
                ),
                "simulator_truth_target_origin_dominant_at_selected_cell": (
                    offline_classification["target_origin_dominant"]
                ),
                "simulator_truth_near_truth_peak_but_nontarget_dominated": (
                    offline_classification[
                        "near_truth_but_nontarget_dominated"
                    ]
                ),
                "simulator_truth_target_association_accepted": bool(
                    detected and raw_target_associated
                ),
                "simulator_truth_false_alarm_like_accepted": bool(
                    detected and not raw_target_associated
                ),
                "simulator_truth_target_peak_rejected_by_prior_gate": bool(
                    raw_target_associated and not detected
                ),
                "simulator_truth_jammer_present_for_offline_audit": truth_jammer is not None,
                "simulator_truth_target_present_for_offline_audit": bool(
                    capture.target_present
                ),
                "simulator_truth_measurement_modality": str(truth_action.kind),
                **false_peak_density,
            },
        )

    def observe(
        self,
        truth_world: PhysicalWorld,
        belief_world: PhysicalWorld,
        action_ids: Sequence[int],
        belief: GraphBeliefEngine,
        *,
        epoch: int,
        belief_covariance_override_normalized: np.ndarray | None = None,
    ) -> tuple[tuple[ObservationRecord, ...], tuple[DynamicFactor, ...]]:
        truth_by_id = {action.action_id: action for action in truth_world.actions}
        belief_by_id = {action.action_id: action for action in belief_world.actions}
        x0 = belief.current_state_vector()
        records: list[ObservationRecord] = []
        factors: list[DynamicFactor] = []
        from .receiver_admission import issue_capture_transport_event
        dispatches = []
        self._capture_dispatches = ()
        for action_id in action_ids:
            factor_start = len(factors)
            if action_id not in truth_by_id or action_id not in belief_by_id:
                raise ValueError(f"selected action {action_id} is missing from a world view")
            if not self.availability.action_available(action_id):
                if self.config.fail_on_unavailable_selected_action:
                    raise RuntimeError(
                        f"selected action {action_id} bypassed distributed availability"
                    )
                continue
            truth_action = truth_by_id[action_id]
            belief_action = belief_by_id[action_id]
            commanded_metadata = {
                key: value
                for key, value in belief_action.metadata.items()
                if key.startswith("commanded_beam_")
                or key.startswith("commanded_capture_")
            }
            if commanded_metadata:
                truth_action = replace(
                    truth_action,
                    metadata={
                        **dict(truth_action.metadata),
                        **commanded_metadata,
                    },
                )
            observation_master_seed = self.config.observation_master_seed
            seed = _action_keyed_observation_seed(
                observation_master_seed, epoch, action_id
            )
            delivery_probability = self.availability.delivery_probability(
                action_id,
                epoch=epoch,
            )
            transport_delivery_sampled = (
                self.config.fidelity is not RuntimeFidelity.R0_EXPECTED
            )
            transport_delivered = (
                True
                if not transport_delivery_sampled
                else bool(
                    np.random.default_rng(
                        observation_master_seed * 1_000_033
                        + epoch * 65_551
                        + action_id * 1013
                        + 17
                    ).random()
                    < delivery_probability
                )
            )
            fallback_reason: str | None = None
            if self.config.fidelity is RuntimeFidelity.R0_EXPECTED:
                belief_factor = build_action_factor(belief_world, belief_action, None)
                detected = True
                value = belief_factor.prediction.copy()
                prediction = belief_factor.prediction.copy()
                covariance = belief_factor.covariance.copy()
                residual = np.zeros_like(prediction)
                used_fidelity = RuntimeFidelity.R0_EXPECTED.value
                extra: dict[str, object] = {
                    "belief_factor": belief_factor,
                    "residual": residual,
                    "expected_information_only": True,
                    "future_environment_state_read": False,
                    "oracle_disturbance_parameters_used_by_inference": False,
                    "calibration_only_factors": (),
                }
                mode = FactorInformationMode.EXPECTED_OBSERVED
                kernel = RobustKernel.NONE
            elif self.config.fidelity is RuntimeFidelity.R2_COMPLEX_BASEBAND:
                extraction_covariance = (
                    np.asarray(
                        belief_covariance_override_normalized,
                        dtype=float,
                    )
                    if belief_covariance_override_normalized is not None
                    else belief.current_state_covariance()
                )
                detected, value, prediction, covariance, used_fidelity, extra = self._r2(
                    truth_world,
                    truth_action,
                    belief_world,
                    belief_action,
                    extraction_covariance,
                    epoch=epoch,
                    seed=seed,
                )
                extra[
                    "r2_causal_search_covariance_override_applied"
                ] = bool(belief_covariance_override_normalized is not None)
                extra[
                    "r2_search_covariance_is_authoritative_posterior"
                ] = bool(belief_covariance_override_normalized is None)
                extra[
                    "r2_search_covariance_truth_or_future_event_read"
                ] = False
                residual = extra.pop("residual")  # type: ignore[assignment]
                fallback_reason = extra.get("fallback_reason")  # type: ignore[assignment]
                mode = FactorInformationMode.CONDITIONAL_DETECTED
                kernel = RobustKernel.CAUCHY
            else:
                detected, value, prediction, covariance, used_fidelity, extra = self._r1(
                    truth_world,
                    truth_action,
                    belief_world,
                    belief_action,
                    epoch=epoch,
                    seed=seed,
                    fallback_reason=None,
                )
                residual = extra.pop("residual")  # type: ignore[assignment]
                fallback_reason = extra.get("fallback_reason")  # type: ignore[assignment]
                mode = FactorInformationMode.CONDITIONAL_DETECTED
                kernel = (
                    RobustKernel.NONE
                    if extra.get("controlled_measurement_model")
                    == "exact_local_linear_gaussian_sufficient_statistic"
                    else RobustKernel.CAUCHY
                )
            belief_factor = extra.pop("belief_factor")
            if not hasattr(belief_factor, "measurement_names"):
                raise RuntimeError("observation adapter did not return an ActionFactor")
            layout = StateLayout.from_world(belief_world)
            h_normalized = (
                belief_factor.jacobian_physical * layout.scales()[None, :]
            )
            current_state_covariance = belief.current_state_covariance()
            predicted_state_covariance = (
                h_normalized
                @ current_state_covariance
                @ h_normalized.T
            )
            innovation_covariance = (
                np.asarray(covariance, dtype=float) + predicted_state_covariance
            )
            innovation_covariance = 0.5 * (
                innovation_covariance + innovation_covariance.T
            )
            if float(np.linalg.eigvalsh(innovation_covariance).min()) <= 0.0:
                raise np.linalg.LinAlgError(
                    "observation innovation covariance is not positive definite"
                )
            extra["predicted_measurement_state_covariance"] = (
                predicted_state_covariance
            )
            extra["innovation_covariance"] = innovation_covariance
            extra["innovation_covariance_includes_prior_state_uncertainty"] = True
            arrival_epoch = self.availability.arrival_epoch(action_id, epoch)
            provenance = f"capture:{truth_world.world_id}:{epoch}:{action_id}:{used_fidelity}"
            factor_id = f"operational_observation:{epoch}:{action_id}:{used_fidelity}"
            # Detection-independent free-calibration evidence acquired during a
            # dwell whose target return was missed.
            calibration_rows = extra.pop("calibration_only_factors", ())
            if transport_delivered:
                for calibration, calibration_factor, calibration_residual in calibration_rows:
                    calibration_dynamic = make_operational_action_factor(
                        belief_world,
                        calibration,
                        calibration_factor,
                        epoch=epoch,
                        arrival_epoch=arrival_epoch,
                        participants=self.availability.factor_participants(belief_action),
                        mode=FactorInformationMode.CONDITIONAL_DETECTED,
                        linearization_state_normalized=x0,
                        measurement_residual=np.asarray(calibration_residual, dtype=float),
                        provenance_id=(
                            f"free_calibration:{truth_world.world_id}:{epoch}:"
                            f"{action_id}:{calibration.action_id}"
                        ),
                        factor_id=(
                            f"operational_free_calibration:{epoch}:{action_id}:"
                            f"{calibration.action_id}"
                        ),
                        fidelity_level=used_fidelity,
                        robust_kernel=RobustKernel.NONE,
                        eigenvalue_floor=self.config.information_eigenvalue_floor,
                    )
                    if calibration_dynamic is not None:
                        factors.append(calibration_dynamic)
            dynamic_factor = None
            if detected and transport_delivered and residual is not None:
                use_mixture = (
                    self.config.likelihood_update_mode
                    is not LikelihoodUpdateMode.LEGACY_FISHER_PSEUDOFACTOR
                    and not belief_action.kind.startswith("calibration")
                    and self.config.fidelity is not RuntimeFidelity.R0_EXPECTED
                )
                if use_mixture:
                    primary_rows = int(
                        extra.get(
                            "primary_measurement_rows",
                            len(belief_factor.measurement_names),
                        )
                    )
                    ambiguity_periods = tuple(
                        extra.get(
                            "measurement_ambiguity_periods",
                            (None,) * len(belief_factor.measurement_names),
                        )
                    )
                    false_component_residual = extra.get(
                        "false_component_reference_residual"
                    )
                    if false_component_residual is None:
                        displacement_sigma = np.asarray(
                            belief_factor.metadata.get(
                                "false_target_displacement_sigma",
                                [0.0] * len(belief_factor.measurement_names),
                            ),
                            dtype=float,
                        )[:primary_rows]
                        primary_covariance = np.asarray(
                            belief_factor.covariance[:primary_rows, :primary_rows],
                            dtype=float,
                        )
                        false_component_residual = _wrap_residual(
                            belief_factor.measurement_names[:primary_rows],
                            (
                                np.asarray(residual, dtype=float)[:primary_rows]
                                - np.linalg.cholesky(primary_covariance)
                                @ displacement_sigma
                            ),
                            ambiguity_periods[:primary_rows],
                        )
                        extra["false_component_reference_residual"] = (
                            false_component_residual
                        )
                    captured_correct_probability = float(
                        np.clip(
                            belief_factor.link_budget.correct_association_probability,
                            0.0,
                            1.0,
                        )
                    )
                    association_component_priors_at_capture = (
                        captured_correct_probability,
                        1.0 - captured_correct_probability,
                    )
                    extra["association_component_priors_at_capture"] = list(
                        association_component_priors_at_capture
                    )
                    extra[
                        "association_component_priors_conditioning"
                    ] = "causal_pre_observation_controller_belief"
                    extra[
                        "association_component_priors_truth_or_future_read"
                    ] = False
                    dynamic_factor, mixture_result = make_operational_mixture_factor(
                        belief_world,
                        belief_action,
                        belief_factor,
                        epoch=epoch,
                        arrival_epoch=arrival_epoch,
                        participants=self.availability.factor_participants(
                            belief_action
                        ),
                        linearization_state_normalized=x0,
                        measurement_residual=np.asarray(residual, dtype=float),
                        primary_rows=primary_rows,
                        provenance_id=provenance,
                        factor_id=factor_id,
                        fidelity_level=used_fidelity,
                        update_mode=self.config.likelihood_update_mode,
                        sandwich_temperature=self.config.sandwich_temperature,
                        robust_kernel=kernel,
                        eigenvalue_floor=self.config.information_eigenvalue_floor,
                        communication_snapshot_epoch=epoch,
                        predictive_state_covariance_normalized=(
                            current_state_covariance
                        ),
                        false_component_residual=false_component_residual,
                        false_component_log_density=extra.get(
                            "false_component_log_density"
                        ),
                        false_component_state_independent=bool(
                            extra.get("false_component_state_independent", False)
                        ),
                        false_component_density_model=extra.get(
                            "false_component_density_model"
                        ),
                        association_model_override=extra.get(
                            "association_model_override"
                        ),
                        association_component_priors=(
                            association_component_priors_at_capture
                        ),
                    )
                    extra["mixture_likelihood_contract"] = mixture_contract_report(
                        mixture_result
                    )
                    extra["mixture_likelihood_contract"].update(
                        {
                            "likelihood_update_mode": (
                                self.config.likelihood_update_mode.value
                            ),
                            "uses_PSD_Louis_predictive_mixture_curvature": bool(
                                self.config.likelihood_update_mode
                                is LikelihoodUpdateMode.MIXTURE_SCORE_LOUIS_PSD
                            ),
                        }
                    )
                    extra["association_responsibilities"] = list(
                        mixture_result.responsibilities
                    )
                else:
                    dynamic_factor = make_operational_action_factor(
                        belief_world,
                        belief_action,
                        belief_factor,
                        epoch=epoch,
                        arrival_epoch=arrival_epoch,
                        participants=self.availability.factor_participants(belief_action),
                        mode=mode,
                        linearization_state_normalized=x0,
                        measurement_residual=np.asarray(residual, dtype=float),
                        provenance_id=provenance,
                        factor_id=factor_id,
                        fidelity_level=used_fidelity,
                        robust_kernel=kernel,
                        eigenvalue_floor=self.config.information_eigenvalue_floor,
                        information_scale=(
                            delivery_probability
                            if self.config.fidelity is RuntimeFidelity.R0_EXPECTED
                            else 1.0
                        ),
                    )
                if dynamic_factor is not None:
                    factors.append(dynamic_factor)
            record = ObservationRecord(
                epoch=epoch,
                action_id=action_id,
                requested_fidelity=self.config.fidelity.value,
                used_fidelity=used_fidelity,
                detected=bool(detected),
                measurement_names=tuple(belief_factor.measurement_names),
                value=None if value is None else np.asarray(value, dtype=float),
                prediction=np.asarray(prediction, dtype=float),
                residual=None if residual is None else np.asarray(residual, dtype=float),
                covariance=np.asarray(covariance, dtype=float),
                observation_epoch=epoch,
                arrival_epoch=arrival_epoch,
                provenance_id=provenance,
                factor_id=None if dynamic_factor is None else dynamic_factor.factor_id,
                fallback_reason=fallback_reason,
                metadata={
                    "observation_random_stream_key": {
                        "master_seed": int(observation_master_seed),
                        "world_and_controller_seed": int(self.config.seed),
                        "epoch": int(epoch),
                        "action_id": int(action_id),
                        "component": "committed_action_observation",
                    },
                    "observation_random_stream_schedule_order_invariant": True,
                    "transport_delivery_probability": float(delivery_probability),
                    "transport_delivery_sampled": bool(transport_delivery_sampled),
                    "transport_delivered": bool(transport_delivered),
                    "transport_loss_conditioned_information_factor": (
                        self.config.fidelity is not RuntimeFidelity.R0_EXPECTED
                    ),
                    "dynamic_factor_contract": (
                        None
                        if dynamic_factor is None
                        else {
                            "factor_id": str(dynamic_factor.factor_id),
                            "robust_kernel": dynamic_factor.robust_kernel.value,
                            "information_mode": dynamic_factor.metadata.get(
                                "information_mode"
                            ),
                            "information_scale": dynamic_factor.metadata.get(
                                "information_scale"
                            ),
                            "factor_semantics": dynamic_factor.metadata.get(
                                "factor_semantics"
                            ),
                            "score_mapping": dynamic_factor.metadata.get(
                                "score_mapping"
                            ),
                            "score_Bartlett_identity_exact": bool(
                                dynamic_factor.metadata.get(
                                    "score_Bartlett_identity_exact", False
                                )
                            ),
                            "expected_information_counterfactual": bool(
                                dynamic_factor.metadata.get(
                                    "expected_information_counterfactual", True
                                )
                            ),
                        }
                    ),
                    **{
                        key: value
                        for key, value in extra.items()
                        if key not in {"fallback_reason"}
                    },
                },
            )
            records.append(record)
            dispatches.append((issue_capture_transport_event(
                capture_id=provenance, action_id=action_id,
                observation_epoch=epoch,
                recipient_id=str(self.availability.fusion_node_id),
                arrival_epoch=arrival_epoch, delivered=transport_delivered,
                factor_ids=tuple(f.factor_id for f in factors[factor_start:]),
                factor_provenance=tuple(tuple(f.provenance_ids) for f in factors[factor_start:]),
            ), record, tuple(factors[factor_start:])))
        self._capture_dispatches = tuple(dispatches)
        return tuple(records), tuple(factors)
