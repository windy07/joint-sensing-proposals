from __future__ import annotations

"""Typed contracts for the single operational radar execution architecture.

The conceptual cycle is

    world -> observation -> joint inference -> distributed availability
          -> safe decision -> world transition.

The executable epoch transaction is necessarily ordered more precisely:
arrived factors are fused, the observable topology bounds the action domain,
counterfactual factors are evaluated against the same authoritative graph,
the safety shield chooses the executable action, and the resulting observation
is either fused immediately or queued for its declared arrival epoch.  No
secondary covariance or controller belief is allowed to become an independent
source of truth.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np


class RuntimeFidelity(str, Enum):
    R0_EXPECTED = "R0_expected_information"
    R1_SUFFICIENT = "R1_sufficient_statistic"
    R2_COMPLEX_BASEBAND = "R2_complex_baseband"


class FactorInformationMode(str, Enum):
    EXPECTED_OBSERVED = "expected_observed_information"
    CONDITIONAL_DETECTED = "conditional_on_detection_information"
    COMPLETE_DATA = "complete_data_information"


class FusionMode(str, Enum):
    FUSION_COMPONENT_ONLY = "fusion_component_only"
    OUT_OF_BAND_CI = "out_of_band_covariance_intersection"


class LikelihoodUpdateMode(str, Enum):
    """Realized-observation factor used by the authoritative graph."""

    LEGACY_FISHER_PSEUDOFACTOR = "legacy_fisher_pseudofactor"
    MIXTURE_SCORE_FISHER = "mixture_score_fisher"
    MIXTURE_SCORE_SANDWICH = "mixture_score_sandwich"
    MIXTURE_SCORE_LOUIS_PSD = "mixture_score_louis_psd"


class ProposalBackend(str, Enum):
    """Candidate proposal backend; exact graph reranking remains mandatory."""

    ANALYTIC = "analytic"
    V71_FACTOR_GNN = "v71_factor_gnn"


class SequentialPolicyMode(str, Enum):
    """Long-horizon policy applied to the exact-reranked candidate table."""

    ROBUST_CVAR = "robust_cvar"
    BELIEF_MYOPIC = "belief_myopic"
    BELIEF_POMCP = "belief_pomcp"
    INDEPENDENT_RL = "independent_rl"


class MultiTargetAssociationMode(str, Enum):
    """Association law for the controller-owned finite target-set posterior."""

    DISABLED = "disabled"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    JPDA = "jpda"


class PlanningDetectionProbabilityMode(str, Enum):
    """Probability law used only for controller-side outcome planning.

    The realized receiver remains governed by the committed-action observation
    kernel.  This switch therefore changes the controller's forecast, never the
    simulator event draw or the authoritative likelihood update.
    """

    NOMINAL_TEMPORAL = "nominal_temporal"
    FROZEN_JOINT_AOA = "frozen_joint_aoa"


@dataclass(frozen=True)
class RuntimeConfig:
    horizon_steps: int = 3
    dt_s: float = 1.0
    fixed_lag_steps: int = 4
    seed: int = 7001
    # Optional observational random-stream authority.  Keeping this distinct
    # from ``seed`` permits conditional replication with the physical world,
    # target trajectory, controller initialization, and proposal randomness
    # frozen.  ``None`` preserves the historical single-seed contract exactly.
    observation_seed: int | None = None
    fidelity: RuntimeFidelity = RuntimeFidelity.R1_SUFFICIENT
    allow_r2_to_r1_fallback: bool = True
    r2_samples: int = 256
    #: Fast-time support is a receiver command, not an incidental compute
    #: constant.  During acquisition it expands from ``r2_samples`` until the
    #: requested posterior delay coverage is met or this hard hardware limit is
    #: reached; confirmed tracks naturally return to the minimum window.
    r2_adaptive_fast_time_enabled: bool = True
    r2_max_samples: int = 8192
    r2_capture_delay_sigmas: float = 6.0
    #: Full commanded 3-dB widths cover this many posterior angular sigmas on
    #: each side.  Broadening pays an explicit peak-gain solid-angle penalty.
    r2_beam_coverage_sigmas: float = 2.5
    r2_max_commanded_beamwidth_rad: float = math.pi / 2.0
    #: Belief-referenced cross-ambiguity search half-widths, expressed in
    #: prediction sigmas and floored by physical resolution cells
    #: (1/B in delay, 1/CPI in Doppler).  The search is never centred on truth.
    r2_delay_search_sigmas: float = 6.0
    r2_doppler_search_sigmas: float = 6.0
    r2_delay_search_minimum_cells: float = 8.0
    r2_doppler_search_minimum_cells: float = 8.0
    #: Slow-time taper used before Doppler steering.  The periodic Hann option
    #: suppresses the rectangular-window sidelobes that otherwise accumulate at
    #: the edge of a stationary-clutter notch.  ``rectangular`` remains available
    #: for identity controls and backward-compatible replays.
    r2_slow_time_window: str = "rectangular"
    #: Receiver-side pulse-Doppler rejection of stationary clutter and residual
    #: direct-path energy.  The notch is the larger of a declared clutter-spread
    #: multiple and a fixed number of coherent Doppler resolution cells.  The
    #: blind zone is explicit evidence, not silently counted as target data.
    r2_clutter_notch_spread_multiples: float = 3.0
    r2_clutter_notch_resolution_cells: float = 2.0
    r2_direct_path_guard_delay_cells: float = 2.0
    #: Whitened-residual gate for an R2 extraction.  An extraction outside its
    #: own declared likelihood fails closed instead of entering the graph.
    r2_residual_gate_sigma: float = 8.0
    planning_detection_probability_mode: PlanningDetectionProbabilityMode = (
        PlanningDetectionProbabilityMode.NOMINAL_TEMPORAL
    )
    joint_aoa_probability_model_path: str | None = None
    joint_aoa_probability_model_sha256: str | None = None
    schedule_update_probability_calibration_enabled: bool = False
    schedule_update_probability_calibration_checkpoint_path: str | None = None
    schedule_update_probability_calibration_checkpoint_sha256: str | None = None
    fusion_node_id: str | None = None
    fusion_mode: FusionMode = FusionMode.FUSION_COMPONENT_ONLY
    maximum_candidates: int = 8
    shortlist_size: int = 5
    maximum_robust_scenarios: int = 3
    #: Deterministic quadrature order used to marginalize the registered target-
    #: class log-RCS prior and its Swerling-I scan-to-scan texture.  This is a
    #: controller prior calculation; it never reads the realized simulator RCS.
    scattering_quadrature_order: int = 1
    scattering_prior_marginalization: bool = False
    lower_tail_probability: float = 0.25
    graph_diagnostics: bool = False
    process_noise_floor: float = 1e-12
    information_eigenvalue_floor: float = 1e-12
    same_epoch_fusion_window_s: float = 0.10
    decision_deadline_s: float = 0.75
    fail_on_unavailable_selected_action: bool = True
    include_noop_candidate: bool = True
    exact_rerank_required: bool = True
    modeled_compute_base_s: float = 0.002
    modeled_compute_per_candidate_s: float = 0.001
    modeled_compute_per_scenario_s: float = 0.0005
    likelihood_update_mode: LikelihoodUpdateMode = (
        LikelihoodUpdateMode.LEGACY_FISHER_PSEUDOFACTOR
    )
    realized_association_mixture: bool = False
    sandwich_temperature: float = 1.0
    mixture_relinearization_iterations: int = 4
    mixture_relinearization_tolerance: float = 1.0e-4
    mixture_trust_region_sigma: float = 1.0
    mixture_line_search_min_scale: float = 0.0625
    mixture_log_likelihood_tolerance: float = 1.0e-8
    #: Optional first-acquisition multi-start guard for strongly nonlinear
    #: range/angle geometry.  Hypotheses are deterministic points from the
    #: causal graph prior covariance; they never use simulator truth and only
    #: the winning local factorization is installed in the sole graph.
    nonlinear_acquisition_enabled: bool = False
    nonlinear_acquisition_max_epoch: int = 0
    nonlinear_acquisition_sigma: float = 2.0
    nonlinear_acquisition_max_hypotheses: int = 27
    nonlinear_acquisition_min_log_posterior_improvement: float = 2.0
    nonlinear_acquisition_minimum_altitude_m: float = 0.0
    #: Optional deterministic continuous refinement of the strongest coarse
    #: acquisition basins.  The search remains inside the registered prior-
    #: whitened box and has a hard evaluation budget for every start.
    nonlinear_acquisition_refinement_enabled: bool = False
    nonlinear_acquisition_refinement_start_count: int = 6
    nonlinear_acquisition_refinement_max_evaluations_per_start: int = 128
    nonlinear_acquisition_refinement_coordinate_tolerance: float = 1.0e-3
    nonlinear_acquisition_refinement_objective_tolerance: float = 1.0e-6
    #: Optional bounded proposal search over latent target/false association
    #: assignments.  Assignment-conditioned modes only seed candidates; the
    #: full marginalized mixture posterior remains the selection authority.
    nonlinear_acquisition_latent_assignment_refinement_enabled: bool = False
    nonlinear_acquisition_max_latent_assignments: int = 4
    nonlinear_acquisition_latent_refinement_start_count: int = 1
    proposal_backend: ProposalBackend = ProposalBackend.ANALYTIC
    proposal_checkpoint: str | None = None
    proposal_device: str = "cpu"
    sequential_policy_mode: SequentialPolicyMode = SequentialPolicyMode.ROBUST_CVAR
    sequential_policy_activation_epoch: int = 0
    #: Optional causal model-health layer.  A time-uniform innovation alarm
    #: forces one exact, E-optimal verification decision; it never reads truth.
    innovation_health_enabled: bool = False
    innovation_health_episode_alpha: float = 0.01
    innovation_health_min_detected_actions_to_clear: int = 2
    innovation_health_min_consistent_verification_epochs_to_clear: int = 1
    #: When an anytime-valid innovation alarm remains active, broaden the
    #: receiver command support with a covariance envelope and rerun the
    #: registered bounded nonlinear acquisition search on the new arrived
    #: observations.  The envelope is search support only: it is not a second
    #: posterior and never replaces the sole DynamicFactorGraph covariance.
    innovation_health_reacquisition_search_enabled: bool = False
    #: Standard-deviation multiplier applied to the registered initial graph
    #: covariance before forming the Loewner upper envelope with the current
    #: posterior covariance.
    innovation_health_reacquisition_covariance_floor_scale: float = 1.0
    #: Permit one causal change-point covariance increment in the differential
    #: clock subspace for each newly raised differential-clock alarm.  This is
    #: a model transition in the sole DynamicFactorGraph, not receiver search
    #: support and not a truth-conditioned reset.  It is default-off so frozen
    #: predecessor experiments remain byte-for-byte semantically unchanged.
    innovation_health_clock_change_process_support_enabled: bool = False
    #: Pre-registered RMS magnitude of a zero-network-mean clock change point.
    #: The same registered scale may support receiver search and, when the
    #: explicit process-support switch is enabled, a one-time causal transition
    #: covariance.  It never encodes an event epoch or realized clock-step draw.
    innovation_health_reacquisition_clock_step_rms_s: float = 100.0e-9
    #: Prior-whitened half-width of the bounded nonlinear reacquisition lattice.
    innovation_health_reacquisition_sigma: float = 3.0
    #: R31 separates internal track memory, sensing mode, and the externally
    #: reported finite track set.  When enabled, an innovation alarm on an
    #: otherwise healthy confirmed track preserves the graph track identity
    #: and requests verification sensing, but abstains from reporting that
    #: location until the model-health alarm clears.  This avoids treating an
    #: unverified, potentially biased posterior mean as a valid reported track.
    innovation_alarm_preserves_confirmed_track: bool = False
    #: Permit a confirmed track to remain in the reported finite set during a
    #: *clock-only* model-health alarm, but only under a causal certificate:
    #: the sole DynamicFactorGraph posterior remains inside the registered
    #: confirmation radius and a recently arrived JPDA-associated
    #: active-monostatic observation supplies a clock-offset-invariant target
    #: anchor.  The controller continues verification sensing while reporting;
    #: non-clock alarms, stale anchors, excessive covariance, or true coast
    #: loss still force abstention.  Default-off preserves predecessor runs.
    innovation_alarm_clock_robust_track_reporting_enabled: bool = False
    #: Retain a probability/geometry-admissible verification set for the exact
    #: downstream risk selector.  The historical single-best local E-optimal
    #: rule remains the default for frozen predecessor experiments.
    innovation_verification_diverse_candidate_set_enabled: bool = False
    #: If a feasible full-rank candidate carries an active-monostatic delay
    #: measurement, require that causal absolute-range anchor during an active
    #: verification alarm.  No action identifier is special-cased.
    innovation_verification_require_absolute_range_anchor: bool = False
    #: When the causal receiver signature isolates a cross-node timing
    #: inconsistency, require an available verification schedule to contain a
    #: clock-invariant monostatic timing anchor, a clock-sensitive cross-node
    #: observation, and an explicit time-calibration factor with joint target / 
    #: differential-clock structural rank.  The requirement is hypothesis-
    #: conditional and never reads simulator truth or scheduled future events.
    innovation_verification_require_differential_clock_identification: bool = False
    #: Require rank-three target-position support after removing every
    #: measurement row that is sensitive to the unknown target velocity.  In
    #: particular, Doppler rows cannot manufacture a position certificate
    #: during acquisition or reacquisition.  Default-off preserves frozen
    #: predecessor semantics.
    innovation_verification_require_position_separating_rank: bool = False
    #: Optional acquisition/reacquisition certificate for stable inversion after
    #: a bounded sensing-action erasure.  The certificate is built from
    #: controller-visible, velocity-independent, unit-normalized position
    #: Jacobian rows.  It is a local directional-frame condition, not a claim of
    #: global nonlinear identifiability or realized tracking success.  The
    #: default remains disabled so frozen predecessor experiments retain their
    #: exact selection semantics.
    erasure_robust_position_geometry_enabled: bool = False
    erasure_robust_position_geometry_max_erasures: int = 1
    #: Maximum condition number of the reduced directional Jacobian after every
    #: registered erasure.  A value of 10 bounds the corresponding normal-matrix
    #: condition number by 100.
    erasure_robust_position_geometry_max_jacobian_condition_number: float = 10.0
    #: Maximum scattering-prior probability mass excluded by the rigorous
    #: verification lower bound.  The retained event uses a union-bound
    #: quantile over target-class uncertainty and every carrier-group Swerling
    #: texture represented by the candidate; it is distinct from the robust-
    #: objective CVaR level.
    innovation_verification_scattering_outage_probability: float = 0.10
    #: Use a scan-indexed, carrier-grouped slow-fluctuation texture for
    #: Swerling I/III truth generation.  Texture is common within one declared
    #: carrier and independent across carriers as an explicit R2 proxy; this
    #: is not a full frequency/aspect correlation model.  It remains hidden
    #: from the controller.  Default-off keeps historical traces byte-
    #: compatible.
    scan_correlated_swerling_truth_enabled: bool = False
    innovation_verification_probability_indifference_band: float = 0.05
    innovation_verification_geometry_floor_fraction: float = 0.25
    #: Optional pre-alarm observability contract.  At registered epochs a
    #: deterministic sequence of cross-node timing probes is designed from
    #: controller-visible action physics only.  Dedicated target-independent
    #: time-transfer pilots are preferred; target-dependent bistatic/TDOA
    #: returns are a declared fallback.  The exact selector is then restricted,
    #: when a mission-admissible schedule exists, to candidates containing the
    #: scheduled probe and an active-monostatic range anchor.
    #: This is default-off and never reads an event flag, event epoch, realized
    #: clock state, receiver residual, or simulator truth.
    clock_sentinel_enabled: bool = False
    clock_sentinel_start_epoch: int = 0
    clock_sentinel_interval_epochs: int = 1
    clock_sentinel_maximum_unique_edges: int = 5
    clock_sentinel_require_monostatic_anchor: bool = True
    sequential_game_horizon: int = 3
    sequential_game_discount: float = 0.95
    sequential_game_simulations: int = 512
    sequential_game_evaluation_rollouts: int = 48
    independent_rl_training_episodes: int = 240
    multi_target_count: int = 1
    multi_target_association_mode: MultiTargetAssociationMode = (
        MultiTargetAssociationMode.DISABLED
    )
    multi_target_information_weight: float = 0.20
    #: Deprecated fallback used only when the physical link budget does not
    #: provide an expected false-return count.  Its units are the reciprocal
    #: volume of the action's declared measurement coordinates.
    jpda_clutter_density: float = 1.0e-3
    jpda_gate_sigma: float = 5.0
    jpda_coalescence_entropy_threshold: float = 0.35
    jpda_birth_unassigned_probability_min: float = 0.40
    jpda_max_confirmed_births_per_epoch: int = 1
    jpda_birth_confirmation_min_distinct_epochs: int = 2
    #: Optional conservative track-phase contract.  When enabled, raw receiver
    #: detections do not confirm or refresh a track unless the current JPDA
    #: joint MAP hypothesis assigns at least one non-calibration measurement to
    #: a track.  The default remains disabled so historical experiments retain
    #: their original semantics.
    track_phase_requires_associated_update_evidence: bool = False
    #: Admit a stochastic Bernoulli birth component from the declared episode
    #: transition model.  The controller receives the hazard and spatial prior,
    #: never the realized or scheduled birth events.
    multi_target_birth_prior_enabled: bool = False
    #: Optional V72 final-selector term.  The evaluator samples hypothetical
    #: measurements from the frozen controller posterior and updates isolated
    #: JPDA forks; it never receives DynamicWorldEngine truth.
    nonlocal_task_risk_enabled: bool = False
    nonlocal_task_risk_weight: float = 0.0
    nonlocal_task_risk_replicates: int = 8
    nonlocal_task_risk_cutoff_m: float = 1000.0
    nonlocal_task_risk_alpha: float = 2.0
    nonlocal_task_risk_confidence_z: float = 1.96
    #: Required when no stochastic birth prior is active.  This is an explicit
    #: assertion that the target-set support is closed over the decision scope.
    nonlocal_task_risk_closed_world_support: bool = False
    #: V73 uses the existing graph-native candidate generator and exact
    #: counterfactual rows.  These fields alter only the final selector between
    #: exact rerank and the already-authoritative campaign safety shield.
    v73_adaptive_enabled: bool = False
    v73_certificate_risk_only: bool = False
    v73_mission_tolerance: float = 0.10
    v73_max_certificate_relative_se: float = 1.0
    v73_confidence_z: float = 1.96
    v73_hard_mission_gate: bool = True
    #: Optional R27 task guard.  The local exact-information winner in the
    #: current frozen candidate set becomes the reference.  Risk/resource
    #: preferences may act only inside its mission-loss and track-update
    #: noninferiority region.  The explicit no-op remains available solely as a
    #: fail-closed fallback when no physical certificate survives.
    v73_reference_noninferiority_gate: bool = False
    #: Optional R28 fallback hierarchy.  If nonlocal-risk uncertainty rejects
    #: every otherwise task-admissible candidate, commit the physical task
    #: reference without claiming a risk improvement.  No-op remains reserved
    #: for an empty physical domain or a downstream safety intervention.
    v73_reference_fallback_on_risk_uncertainty: bool = False
    v73_reference_mission_loss_margin: float = 0.0
    v73_reference_track_update_margin: float = 0.0
    v73_gate_checkpoint: str | None = None
    v73_gate_checkpoint_sha256: str | None = None
    v73_require_frozen_gate_checkpoint: bool = False

    def validate(self) -> None:
        if self.horizon_steps <= 0 or self.fixed_lag_steps <= 0:
            raise ValueError("runtime horizon and fixed lag must be positive")
        # A lag shorter than the bounded episode is supported by exact Schur
        # separator messages for the current frozen linearization and final IRLS
        # weights.  The lag must nevertheless retain at least two epochs so that
        # an expired state can be eliminated into its immediate dynamic boundary.
        if self.fixed_lag_steps < 2:
            raise ValueError(
                "fixed_lag_steps must be at least two for separator marginalization"
            )
        if self.dt_s <= 0.0 or not math.isfinite(self.dt_s):
            raise ValueError("runtime dt_s must be finite and positive")
        if self.seed < 0 or (
            self.observation_seed is not None and self.observation_seed < 0
        ):
            raise ValueError("runtime and observation seeds must be nonnegative")
        if self.r2_samples < 128:
            raise ValueError("R2 sample count must be at least 128")
        if self.r2_max_samples < self.r2_samples:
            raise ValueError("R2 maximum sample count cannot be below its minimum")
        if not isinstance(self.r2_adaptive_fast_time_enabled, bool):
            raise ValueError("R2 adaptive-fast-time flag must be boolean")
        if any(
            value <= 0.0 or not math.isfinite(value)
            for value in (
                self.r2_delay_search_sigmas,
                self.r2_capture_delay_sigmas,
                self.r2_beam_coverage_sigmas,
                self.r2_max_commanded_beamwidth_rad,
                self.r2_doppler_search_sigmas,
                self.r2_delay_search_minimum_cells,
                self.r2_doppler_search_minimum_cells,
                self.r2_clutter_notch_spread_multiples,
                self.r2_clutter_notch_resolution_cells,
                self.r2_direct_path_guard_delay_cells,
                self.r2_residual_gate_sigma,
            )
        ):
            raise ValueError("R2 search and gate parameters must be finite and positive")
        if self.r2_max_commanded_beamwidth_rad > math.pi:
            raise ValueError("R2 commanded beamwidth cannot exceed pi radians")
        if self.r2_slow_time_window not in {"rectangular", "hann_periodic"}:
            raise ValueError(
                "R2 slow-time window must be rectangular or hann_periodic"
            )
        joint_model_supplied = self.joint_aoa_probability_model_path is not None
        joint_hash_supplied = self.joint_aoa_probability_model_sha256 is not None
        if joint_model_supplied != joint_hash_supplied:
            raise ValueError(
                "joint AOA probability model path and SHA-256 must be supplied together"
            )
        if (
            self.planning_detection_probability_mode
            is PlanningDetectionProbabilityMode.FROZEN_JOINT_AOA
        ):
            if not joint_model_supplied:
                raise ValueError(
                    "frozen joint AOA planning requires a hash-bound model"
                )
            if self.fidelity is not RuntimeFidelity.R2_COMPLEX_BASEBAND:
                raise ValueError(
                    "the frozen joint AOA probability is validated only for R2 observations"
                )
        elif joint_model_supplied:
            raise ValueError(
                "the nominal planning arm must not load a joint AOA model"
            )
        if joint_hash_supplied:
            digest = str(self.joint_aoa_probability_model_sha256).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("joint AOA probability model SHA-256 is malformed")
        calibration_path_supplied = (
            self.schedule_update_probability_calibration_checkpoint_path
            is not None
        )
        calibration_hash_supplied = (
            self.schedule_update_probability_calibration_checkpoint_sha256
            is not None
        )
        if calibration_path_supplied != calibration_hash_supplied:
            raise ValueError(
                "schedule-update calibration checkpoint path and SHA-256 must "
                "be supplied together"
            )
        if not isinstance(
            self.schedule_update_probability_calibration_enabled, bool
        ):
            raise ValueError("schedule-update calibration flag must be boolean")
        if self.schedule_update_probability_calibration_enabled and not (
            calibration_path_supplied and calibration_hash_supplied
        ):
            raise ValueError(
                "enabled schedule-update calibration requires a hash-bound checkpoint"
            )
        if calibration_hash_supplied:
            digest = str(
                self.schedule_update_probability_calibration_checkpoint_sha256
            ).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "schedule-update calibration checkpoint SHA-256 is malformed"
                )
        if self.maximum_candidates <= 0 or self.shortlist_size <= 0:
            raise ValueError("candidate and shortlist limits must be positive")
        if self.maximum_robust_scenarios <= 0:
            raise ValueError("at least one robust scenario is required")
        if self.scattering_quadrature_order not in {1, 2, 3}:
            raise ValueError("scattering quadrature order must be one, two, or three")
        if not isinstance(self.scattering_prior_marginalization, bool):
            raise ValueError("scattering-prior marginalization flag must be boolean")
        if not 0.0 < self.lower_tail_probability <= 1.0:
            raise ValueError("lower-tail probability must lie in (0, 1]")
        if self.process_noise_floor <= 0.0 or self.information_eigenvalue_floor <= 0.0:
            raise ValueError("numerical floors must be positive")
        if self.same_epoch_fusion_window_s < 0.0:
            raise ValueError("same-epoch fusion window must be nonnegative")
        if self.decision_deadline_s <= 0.0:
            raise ValueError("decision deadline must be positive")
        if not self.exact_rerank_required:
            raise ValueError("the unified runtime forbids proposal-only execution")
        if any(
            value < 0.0 or not math.isfinite(value)
            for value in (
                self.modeled_compute_base_s,
                self.modeled_compute_per_candidate_s,
                self.modeled_compute_per_scenario_s,
            )
        ):
            raise ValueError("modeled compute costs must be finite and nonnegative")
        if not isinstance(self.realized_association_mixture, bool):
            raise ValueError("realized association-mixture flag must be boolean")
        if self.sandwich_temperature < 1.0 or not math.isfinite(
            self.sandwich_temperature
        ):
            raise ValueError("sandwich temperature must be finite and at least one")
        if self.mixture_relinearization_iterations <= 0:
            raise ValueError("mixture relinearization iterations must be positive")
        if (
            self.mixture_relinearization_tolerance <= 0.0
            or not math.isfinite(self.mixture_relinearization_tolerance)
        ):
            raise ValueError("mixture relinearization tolerance is invalid")
        if (
            self.mixture_trust_region_sigma <= 0.0
            or not math.isfinite(self.mixture_trust_region_sigma)
        ):
            raise ValueError("mixture trust-region radius is invalid")
        if (
            not 0.0 < self.mixture_line_search_min_scale <= 1.0
            or not math.isfinite(self.mixture_line_search_min_scale)
        ):
            raise ValueError("mixture line-search minimum scale is invalid")
        if (
            self.mixture_log_likelihood_tolerance < 0.0
            or not math.isfinite(self.mixture_log_likelihood_tolerance)
        ):
            raise ValueError("mixture log-likelihood tolerance is invalid")
        if not isinstance(self.nonlinear_acquisition_enabled, bool):
            raise ValueError("nonlinear acquisition enable flag must be boolean")
        if self.nonlinear_acquisition_max_epoch < 0:
            raise ValueError("nonlinear acquisition maximum epoch must be nonnegative")
        if (
            self.nonlinear_acquisition_sigma <= 0.0
            or not math.isfinite(self.nonlinear_acquisition_sigma)
        ):
            raise ValueError("nonlinear acquisition sigma must be finite and positive")
        if not 1 <= self.nonlinear_acquisition_max_hypotheses <= 27:
            raise ValueError(
                "the registered three-axis nonlinear acquisition lattice "
                "supports between one and 27 hypotheses"
            )
        if (
            self.nonlinear_acquisition_min_log_posterior_improvement < 0.0
            or not math.isfinite(
                self.nonlinear_acquisition_min_log_posterior_improvement
            )
        ):
            raise ValueError(
                "nonlinear acquisition posterior-improvement threshold is invalid"
            )
        if not math.isfinite(self.nonlinear_acquisition_minimum_altitude_m):
            raise ValueError("nonlinear acquisition altitude floor must be finite")
        if not isinstance(self.nonlinear_acquisition_refinement_enabled, bool):
            raise ValueError(
                "nonlinear acquisition refinement enable flag must be boolean"
            )
        if not 1 <= self.nonlinear_acquisition_refinement_start_count <= 27:
            raise ValueError(
                "nonlinear acquisition refinement requires between one and "
                "27 coarse starts"
            )
        if self.nonlinear_acquisition_refinement_max_evaluations_per_start < 16:
            raise ValueError(
                "nonlinear acquisition refinement requires at least 16 "
                "objective evaluations per start"
            )
        if (
            self.nonlinear_acquisition_refinement_coordinate_tolerance <= 0.0
            or not math.isfinite(
                self.nonlinear_acquisition_refinement_coordinate_tolerance
            )
        ):
            raise ValueError(
                "nonlinear acquisition refinement coordinate tolerance is invalid"
            )
        if (
            self.nonlinear_acquisition_refinement_objective_tolerance < 0.0
            or not math.isfinite(
                self.nonlinear_acquisition_refinement_objective_tolerance
            )
        ):
            raise ValueError(
                "nonlinear acquisition refinement objective tolerance is invalid"
            )
        if (
            self.nonlinear_acquisition_refinement_enabled
            and not self.nonlinear_acquisition_enabled
        ):
            raise ValueError(
                "continuous acquisition refinement requires coarse acquisition"
            )
        if not isinstance(
            self.nonlinear_acquisition_latent_assignment_refinement_enabled,
            bool,
        ):
            raise ValueError(
                "latent-assignment acquisition refinement enable flag must be "
                "boolean"
            )
        if not 1 <= self.nonlinear_acquisition_max_latent_assignments <= 32:
            raise ValueError(
                "latent-assignment acquisition refinement supports between "
                "one and 32 bounded assignments"
            )
        if not 1 <= self.nonlinear_acquisition_latent_refinement_start_count <= 8:
            raise ValueError(
                "latent-assignment acquisition refinement requires between "
                "one and eight starts per assignment"
            )
        if (
            self.nonlinear_acquisition_latent_assignment_refinement_enabled
            and not self.nonlinear_acquisition_refinement_enabled
        ):
            raise ValueError(
                "latent-assignment acquisition refinement requires bounded "
                "continuous refinement"
            )
        if (
            self.nonlinear_acquisition_enabled
            and self.likelihood_update_mode
            is LikelihoodUpdateMode.LEGACY_FISHER_PSEUDOFACTOR
        ):
            raise ValueError(
                "nonlinear acquisition requires an observed-data mixture likelihood"
            )
        if self.likelihood_update_mode is not LikelihoodUpdateMode.LEGACY_FISHER_PSEUDOFACTOR:
            if not self.realized_association_mixture:
                raise ValueError(
                    "mixture-score updates require realized association-mixture sampling"
                )
        if self.proposal_backend is ProposalBackend.V71_FACTOR_GNN:
            if not self.proposal_checkpoint:
                raise ValueError("the V71 factor-GNN backend requires a checkpoint")
        if self.proposal_device not in {"cpu", "cuda"}:
            raise ValueError("proposal device must be cpu or cuda")
        if self.sequential_game_horizon <= 0:
            raise ValueError("sequential game horizon must be positive")
        if self.sequential_policy_activation_epoch < 0:
            raise ValueError("sequential policy activation epoch must be nonnegative")
        if self.sequential_policy_activation_epoch >= self.horizon_steps:
            raise ValueError(
                "sequential policy activation epoch must precede the horizon"
            )
        if not 0.0 <= self.sequential_game_discount <= 1.0:
            raise ValueError("sequential game discount must lie in [0, 1]")
        if min(
            self.sequential_game_simulations,
            self.sequential_game_evaluation_rollouts,
            self.independent_rl_training_episodes,
        ) <= 1:
            raise ValueError("sequential search and RL budgets must exceed one")
        if self.multi_target_count <= 0:
            raise ValueError("multi-target count must be positive")
        if (
            self.multi_target_count > 1
            and self.multi_target_association_mode
            is MultiTargetAssociationMode.DISABLED
        ):
            raise ValueError(
                "multi-target runs require an explicit association mode"
            )
        if self.multi_target_information_weight < 0.0 or not math.isfinite(
            self.multi_target_information_weight
        ):
            raise ValueError("multi-target information weight is invalid")
        if not isinstance(self.innovation_health_enabled, bool):
            raise ValueError("innovation-health enable flag must be boolean")
        if not (
            math.isfinite(self.innovation_health_episode_alpha)
            and 0.0 < self.innovation_health_episode_alpha < 1.0
        ):
            raise ValueError("innovation-health episode alpha must lie in (0,1)")
        if self.innovation_health_min_detected_actions_to_clear < 1:
            raise ValueError(
                "innovation-health clear count must be a positive integer"
            )
        if self.innovation_health_min_consistent_verification_epochs_to_clear < 1:
            raise ValueError(
                "innovation-health consistent verification epoch count must be positive"
            )
        if not isinstance(self.innovation_alarm_preserves_confirmed_track, bool):
            raise ValueError(
                "innovation alarm track-preservation flag must be boolean"
            )
        if not isinstance(
            self.innovation_alarm_clock_robust_track_reporting_enabled, bool
        ):
            raise ValueError(
                "clock-robust alarm track-reporting flag must be boolean"
            )
        if (
            self.innovation_alarm_clock_robust_track_reporting_enabled
            and not self.innovation_alarm_preserves_confirmed_track
        ):
            raise ValueError(
                "clock-robust alarm track reporting requires preserved track memory"
            )
        if not isinstance(
            self.innovation_verification_diverse_candidate_set_enabled, bool
        ):
            raise ValueError(
                "innovation verification candidate-set flag must be boolean"
            )
        if not isinstance(
            self.innovation_verification_require_absolute_range_anchor, bool
        ):
            raise ValueError(
                "innovation verification absolute-range flag must be boolean"
            )
        if not isinstance(
            self.innovation_verification_require_differential_clock_identification,
            bool,
        ):
            raise ValueError(
                "innovation verification differential-clock flag must be boolean"
            )
        if not isinstance(
            self.innovation_verification_require_position_separating_rank,
            bool,
        ):
            raise ValueError(
                "innovation verification position-separating flag must be boolean"
            )
        if not isinstance(self.erasure_robust_position_geometry_enabled, bool):
            raise ValueError(
                "erasure-robust position-geometry enable flag must be boolean"
            )
        if not isinstance(
            self.erasure_robust_position_geometry_max_erasures, int
        ) or not 1 <= self.erasure_robust_position_geometry_max_erasures <= 2:
            raise ValueError(
                "erasure-robust position geometry supports one or two erasures"
            )
        if (
            not math.isfinite(
                self.erasure_robust_position_geometry_max_jacobian_condition_number
            )
            or self.erasure_robust_position_geometry_max_jacobian_condition_number
            < 1.0
        ):
            raise ValueError(
                "erasure-robust position-geometry condition limit must be "
                "finite and at least one"
            )
        if not (
            math.isfinite(
                self.innovation_verification_scattering_outage_probability
            )
            and 0.0
            < self.innovation_verification_scattering_outage_probability
            < 1.0
        ):
            raise ValueError(
                "innovation verification scattering outage probability must lie in (0,1)"
            )
        if not isinstance(self.scan_correlated_swerling_truth_enabled, bool):
            raise ValueError(
                "scan-correlated Swerling truth flag must be boolean"
            )
        if not (
            math.isfinite(
                self.innovation_verification_probability_indifference_band
            )
            and 0.0
            <= self.innovation_verification_probability_indifference_band
            <= 1.0
        ):
            raise ValueError(
                "innovation verification probability indifference band must lie in [0,1]"
            )
        if not (
            math.isfinite(self.innovation_verification_geometry_floor_fraction)
            and 0.0
            < self.innovation_verification_geometry_floor_fraction
            <= 1.0
        ):
            raise ValueError(
                "innovation verification geometry floor fraction must lie in (0,1]"
            )
        if (
            self.innovation_alarm_preserves_confirmed_track
            or self.innovation_alarm_clock_robust_track_reporting_enabled
            or self.innovation_verification_diverse_candidate_set_enabled
            or self.innovation_verification_require_absolute_range_anchor
            or self.innovation_verification_require_differential_clock_identification
            or self.innovation_verification_require_position_separating_rank
        ) and not self.innovation_health_enabled:
            raise ValueError(
                "R31 innovation verification semantics require innovation health"
            )
        if (
            self.innovation_verification_require_absolute_range_anchor
            and not self.innovation_verification_diverse_candidate_set_enabled
        ):
            raise ValueError(
                "absolute-range verification requires the diverse candidate-set rule"
            )
        if (
            self.innovation_verification_require_differential_clock_identification
            and not self.innovation_verification_diverse_candidate_set_enabled
        ):
            raise ValueError(
                "differential-clock verification requires the diverse candidate-set rule"
            )
        if (
            self.innovation_verification_require_position_separating_rank
            and not self.innovation_verification_diverse_candidate_set_enabled
        ):
            raise ValueError(
                "position-separating verification requires the diverse candidate-set rule"
            )
        if not isinstance(self.clock_sentinel_enabled, bool):
            raise ValueError("clock-sentinel enable flag must be boolean")
        if not isinstance(self.clock_sentinel_require_monostatic_anchor, bool):
            raise ValueError("clock-sentinel anchor flag must be boolean")
        if self.clock_sentinel_start_epoch < 0:
            raise ValueError("clock-sentinel start epoch must be nonnegative")
        if self.clock_sentinel_interval_epochs < 1:
            raise ValueError("clock-sentinel interval must be positive")
        if self.clock_sentinel_maximum_unique_edges < 1:
            raise ValueError("clock-sentinel edge budget must be positive")
        if self.clock_sentinel_enabled and not self.innovation_health_enabled:
            raise ValueError(
                "clock sentinel requires the causal innovation-health monitor"
            )
        if not isinstance(
            self.innovation_health_reacquisition_search_enabled, bool
        ):
            raise ValueError(
                "innovation-health reacquisition search flag must be boolean"
            )
        if not isinstance(
            self.innovation_health_clock_change_process_support_enabled, bool
        ):
            raise ValueError(
                "innovation-health clock-change process-support flag must be boolean"
            )
        if (
            self.innovation_health_clock_change_process_support_enabled
            and not self.innovation_health_enabled
        ):
            raise ValueError(
                "clock-change process support requires the causal innovation-health "
                "monitor"
            )
        if self.innovation_health_reacquisition_search_enabled:
            if not self.innovation_health_enabled:
                raise ValueError(
                    "innovation-health reacquisition search requires the causal "
                    "innovation-health monitor"
                )
            if not self.nonlinear_acquisition_enabled:
                raise ValueError(
                    "innovation-health reacquisition search requires the registered "
                    "nonlinear acquisition implementation"
                )
        if (
            self.innovation_health_reacquisition_covariance_floor_scale < 1.0
            or not math.isfinite(
                self.innovation_health_reacquisition_covariance_floor_scale
            )
        ):
            raise ValueError(
                "innovation-health reacquisition covariance floor scale must be "
                "finite and at least one"
            )
        if (
            self.innovation_health_reacquisition_clock_step_rms_s <= 0.0
            or not math.isfinite(
                self.innovation_health_reacquisition_clock_step_rms_s
            )
        ):
            raise ValueError(
                "innovation-health reacquisition clock-step RMS must be "
                "finite and positive"
            )
        if (
            self.innovation_health_reacquisition_sigma <= 0.0
            or not math.isfinite(self.innovation_health_reacquisition_sigma)
        ):
            raise ValueError(
                "innovation-health reacquisition sigma must be finite and positive"
            )
        if self.jpda_clutter_density <= 0.0 or not math.isfinite(
            self.jpda_clutter_density
        ):
            raise ValueError("JPDA clutter density must be finite and positive")
        if self.jpda_gate_sigma <= 0.0 or not math.isfinite(
            self.jpda_gate_sigma
        ):
            raise ValueError("JPDA gate radius must be finite and positive")
        if not 0.0 <= self.jpda_coalescence_entropy_threshold <= 1.0:
            raise ValueError(
                "JPDA coalescence entropy threshold must lie in [0, 1]"
            )
        if not 0.0 < self.jpda_birth_unassigned_probability_min < 1.0:
            raise ValueError(
                "JPDA birth unassigned-probability threshold must lie in (0, 1)"
            )
        if self.jpda_max_confirmed_births_per_epoch <= 0:
            raise ValueError(
                "JPDA maximum confirmed births per epoch must be positive"
            )
        if self.jpda_birth_confirmation_min_distinct_epochs < 2:
            raise ValueError(
                "JPDA birth confirmation requires at least two distinct epochs"
            )
        if not isinstance(
            self.track_phase_requires_associated_update_evidence, bool
        ):
            raise ValueError(
                "track-phase associated-update evidence flag must be boolean"
            )
        if (
            self.track_phase_requires_associated_update_evidence
            and self.multi_target_association_mode
            is MultiTargetAssociationMode.DISABLED
        ):
            raise ValueError(
                "track-phase associated-update evidence requires an explicit "
                "multi-target association mode"
            )
        if not isinstance(self.multi_target_birth_prior_enabled, bool):
            raise ValueError("multi-target birth-prior flag must be boolean")
        if (
            self.multi_target_birth_prior_enabled
            and self.multi_target_association_mode
            is MultiTargetAssociationMode.DISABLED
        ):
            raise ValueError(
                "a multi-target birth prior requires an explicit association mode"
            )
        if not isinstance(self.nonlocal_task_risk_enabled, bool):
            raise ValueError("nonlocal task-risk enable flag must be boolean")
        if not isinstance(self.nonlocal_task_risk_closed_world_support, bool):
            raise ValueError(
                "nonlocal task-risk closed-world support flag must be boolean"
            )
        if self.nonlocal_task_risk_weight < 0.0 or not math.isfinite(
            self.nonlocal_task_risk_weight
        ):
            raise ValueError("nonlocal task-risk weight must be finite and nonnegative")
        if self.nonlocal_task_risk_replicates < 2:
            raise ValueError("nonlocal task risk requires at least two replicates")
        if self.nonlocal_task_risk_cutoff_m <= 0.0 or not math.isfinite(
            self.nonlocal_task_risk_cutoff_m
        ):
            raise ValueError("nonlocal task-risk cutoff must be finite and positive")
        if not math.isclose(
            self.nonlocal_task_risk_alpha,
            2.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("the operational GOSPA task risk requires alpha=2")
        if self.nonlocal_task_risk_confidence_z < 0.0 or not math.isfinite(
            self.nonlocal_task_risk_confidence_z
        ):
            raise ValueError(
                "nonlocal task-risk confidence multiplier must be finite and nonnegative"
            )
        if self.nonlocal_task_risk_enabled:
            if self.multi_target_association_mode is MultiTargetAssociationMode.DISABLED:
                raise ValueError(
                    "nonlocal task risk requires an explicit multi-target association mode"
                )
            if self.multi_target_count > 4:
                raise ValueError(
                    "the bounded cubature task-risk evaluator supports at most four tracks"
                )
            if self.nonlocal_task_risk_weight <= 0.0 and not self.v73_certificate_risk_only:
                raise ValueError(
                    "enabled nonlocal task risk requires a positive selector weight"
                )
            if not (
                self.multi_target_birth_prior_enabled
                or self.nonlocal_task_risk_closed_world_support
            ):
                raise ValueError(
                    "nonlocal task-risk selection requires either a declared "
                    "stochastic birth prior or an explicit closed-world support "
                    "contract"
                )
        for name in ("v73_adaptive_enabled","v73_certificate_risk_only","v73_hard_mission_gate",
                     "v73_reference_noninferiority_gate",
                     "v73_reference_fallback_on_risk_uncertainty",
                     "v73_require_frozen_gate_checkpoint"):
            if not isinstance(getattr(self,name),bool): raise ValueError(f"{name} must be boolean")
        if (self.v73_gate_checkpoint is None)!=(self.v73_gate_checkpoint_sha256 is None):
            raise ValueError("V73 gate checkpoint path and SHA-256 must be supplied together")
        if self.v73_gate_checkpoint_sha256 is not None:
            digest=str(self.v73_gate_checkpoint_sha256).lower()
            if len(digest)!=64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("V73 gate checkpoint SHA-256 is malformed")
        if self.v73_mission_tolerance < 0.0 or not math.isfinite(self.v73_mission_tolerance):
            raise ValueError("V73 mission tolerance must be finite and nonnegative")
        if self.v73_max_certificate_relative_se <= 0.0 or not math.isfinite(self.v73_max_certificate_relative_se):
            raise ValueError("V73 certificate relative-SE limit must be finite and positive")
        if self.v73_confidence_z < 0.0 or not math.isfinite(self.v73_confidence_z):
            raise ValueError("V73 confidence multiplier must be finite and nonnegative")
        if (
            self.v73_reference_mission_loss_margin < 0.0
            or not math.isfinite(self.v73_reference_mission_loss_margin)
        ):
            raise ValueError(
                "V73 reference mission-loss margin must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.v73_reference_track_update_margin)
            or not 0.0 <= self.v73_reference_track_update_margin <= 1.0
        ):
            raise ValueError(
                "V73 reference track-update margin must lie in [0,1]"
            )
        if self.v73_certificate_risk_only and not self.nonlocal_task_risk_enabled:
            raise ValueError("V73 certificate-risk mode requires the nonlocal task-risk evaluator")
        if self.v73_adaptive_enabled:
            if not self.v73_certificate_risk_only:
                raise ValueError("V73 adaptive selection requires risk certificates without score double counting")
            if not self.include_noop_candidate or not self.exact_rerank_required:
                raise ValueError("V73 requires one explicit no-op and exact reranking")
            if abs(self.nonlocal_task_risk_weight) > 1.0e-15:
                raise ValueError("V73 set-risk certificates must not be counted again inside the information score")
            if self.v73_require_frozen_gate_checkpoint and self.v73_gate_checkpoint is None:
                raise ValueError("authoritative V73 runs require a frozen gate checkpoint")
            if self.v73_reference_noninferiority_gate and not self.v73_hard_mission_gate:
                raise ValueError(
                    "V73 reference noninferiority requires the hard mission gate"
                )
            if (
                self.v73_reference_fallback_on_risk_uncertainty
                and not self.v73_reference_noninferiority_gate
            ):
                raise ValueError(
                    "task-reference fallback requires reference noninferiority"
                )
        if self.fusion_mode is not FusionMode.FUSION_COMPONENT_ONLY:
            raise ValueError(
                "out-of-band covariance fusion is not an authoritative online "
                "belief path; import it as an explicitly provenance-tagged graph "
                "factor before enabling it"
            )

    @property
    def observation_master_seed(self) -> int:
        """Return the realized-observation stream root without changing legacy runs."""

        return int(self.seed if self.observation_seed is None else self.observation_seed)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise ValueError(f"unknown runtime configuration fields: {unknown}")
        values = dict(payload)
        if "fidelity" in values:
            values["fidelity"] = RuntimeFidelity(values["fidelity"])
        if "fusion_mode" in values:
            values["fusion_mode"] = FusionMode(values["fusion_mode"])
        if "likelihood_update_mode" in values:
            values["likelihood_update_mode"] = LikelihoodUpdateMode(
                values["likelihood_update_mode"]
            )
        if "proposal_backend" in values:
            values["proposal_backend"] = ProposalBackend(values["proposal_backend"])
        if "sequential_policy_mode" in values:
            values["sequential_policy_mode"] = SequentialPolicyMode(
                values["sequential_policy_mode"]
            )
        if "multi_target_association_mode" in values:
            values["multi_target_association_mode"] = MultiTargetAssociationMode(
                values["multi_target_association_mode"]
            )
        if "planning_detection_probability_mode" in values:
            values["planning_detection_probability_mode"] = (
                PlanningDetectionProbabilityMode(
                    values["planning_detection_probability_mode"]
                )
            )
        result = cls(**values)
        result.validate()
        return result


@dataclass(frozen=True)
class OperationalNetworkState:
    """Observable topology and node transport telemetry."""

    epoch: int
    communication_edges: tuple[tuple[str, str], ...]
    node_packet_delay_steps: Mapping[str, int] = field(default_factory=dict)
    node_queue_megabytes: Mapping[str, float] = field(default_factory=dict)
    node_compute_load_fraction: Mapping[str, float] = field(default_factory=dict)
    node_packet_loss_probability: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.epoch < 0:
            raise ValueError("operational network epoch must be nonnegative")
        for edge in self.communication_edges:
            if len(edge) != 2 or not edge[0] or not edge[1] or edge[0] == edge[1]:
                raise ValueError("operational communication edges must join distinct nodes")
        if any(int(value) < 0 for value in self.node_packet_delay_steps.values()):
            raise ValueError("node packet-delay telemetry must be nonnegative")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.node_queue_megabytes.values()
        ):
            raise ValueError("node queue telemetry must be finite and nonnegative")
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.5
            for value in self.node_compute_load_fraction.values()
        ):
            raise ValueError("node compute-load telemetry is outside [0, 1.5]")
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.node_packet_loss_probability.values()
        ):
            raise ValueError("node packet-loss telemetry is outside [0, 1]")


@dataclass(frozen=True)
class AvailabilityRecord:
    epoch: int
    fusion_node_id: str
    component_by_node: Mapping[str, int]
    fusion_component_nodes: tuple[str, ...]
    available_action_ids: tuple[int, ...]
    blocked_action_reasons: Mapping[int, tuple[str, ...]]
    action_transport_latency_s: Mapping[int, float]
    action_arrival_epoch: Mapping[int, int]
    communication_contract: Mapping[str, Any]
    status: str


@dataclass(frozen=True)
class ObservationRecord:
    epoch: int
    action_id: int
    requested_fidelity: str
    used_fidelity: str
    detected: bool
    measurement_names: tuple[str, ...]
    value: np.ndarray | None
    prediction: np.ndarray
    residual: np.ndarray | None
    covariance: np.ndarray
    observation_epoch: int
    arrival_epoch: int
    provenance_id: str
    factor_id: str | None
    fallback_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    candidate_index: int
    action_ids: tuple[int, ...]
    proposal_score: float
    exact_score: float
    position_information_gain_nats: float
    velocity_information_gain_nats: float
    lower_tail_position_gain_nats: float
    lower_tail_velocity_gain_nats: float
    nominal_d_radius_improvement_fraction: float
    worst_operational_bures_degradation_m: float
    normalized_resource_cost: float
    switching_cost: float
    scenario_position_gains_nats: tuple[float, ...]
    scenario_velocity_gains_nats: tuple[float, ...]
    scenario_ids: tuple[str, ...]
    post_position_covariance_m2: np.ndarray
    post_velocity_covariance_m2_s2: np.ndarray
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord:
    epoch: int
    candidate_library: tuple[tuple[int, ...], ...]
    safe_mask: tuple[bool, ...]
    selected_candidate: int
    selected_action_ids: tuple[int, ...]
    proposed_candidate: int
    shield_intervened: bool
    shield_reasons: tuple[str, ...]
    exact_rerank_rows: tuple[CandidateScore, ...]
    decision_latency_s: float
    deadline_miss: bool
    fail_closed: bool
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefRecord:
    epoch: int
    status: str
    target_mean_normalized: np.ndarray
    nuisance_mean_normalized: np.ndarray
    position_covariance_m2: np.ndarray
    velocity_covariance_m2_s2: np.ndarray
    accepted_factor_ids: tuple[str, ...]
    rejected_factors: tuple[tuple[str, str], ...]
    robust_weights: Mapping[str, float]
    information_trace: float
    covariance_trace: float
    exact_batch_incremental_residual: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeStepRecord:
    epoch: int
    world_id: str
    availability: AvailabilityRecord
    prior_belief: BeliefRecord
    decision: DecisionRecord
    observations: tuple[ObservationRecord, ...]
    posterior_belief: BeliefRecord
    next_epoch: int
    resources_before: Mapping[str, float]
    resources_after: Mapping[str, float]
    status: str


def to_builtin(value: Any) -> Any:
    """Convert nested runtime records to deterministic JSON-compatible data."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return to_builtin(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    return value
