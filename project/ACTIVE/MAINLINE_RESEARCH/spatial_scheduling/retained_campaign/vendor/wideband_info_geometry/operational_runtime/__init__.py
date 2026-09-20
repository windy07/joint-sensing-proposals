"""Single operational execution architecture for the radar research stack."""

from .availability import DistributedAvailabilityManager
from .belief import GraphBeliefEngine
from .contracts import (
    AvailabilityRecord,
    BeliefRecord,
    CandidateScore,
    DecisionRecord,
    FactorInformationMode,
    FusionMode,
    LikelihoodUpdateMode,
    MultiTargetAssociationMode,
    ObservationRecord,
    OperationalNetworkState,
    PlanningDetectionProbabilityMode,
    RuntimeConfig,
    RuntimeFidelity,
    RuntimeStepRecord,
    ProposalBackend,
    SequentialPolicyMode,
    to_builtin,
)
from .decision import GraphSafeDecisionEngine
from .kernel import OperationalFusionRuntime
from .observation import ObservationEngine
from .world import DynamicWorldEngine, WorldViews

__all__ = [
    "AvailabilityRecord",
    "BeliefRecord",
    "CandidateScore",
    "DecisionRecord",
    "DistributedAvailabilityManager",
    "DynamicWorldEngine",
    "FactorInformationMode",
    "FusionMode",
    "LikelihoodUpdateMode",
    "MultiTargetAssociationMode",
    "GraphBeliefEngine",
    "GraphSafeDecisionEngine",
    "ObservationEngine",
    "ObservationRecord",
    "OperationalNetworkState",
    "OperationalFusionRuntime",
    "PlanningDetectionProbabilityMode",
    "RuntimeConfig",
    "RuntimeFidelity",
    "RuntimeStepRecord",
    "ProposalBackend",
    "SequentialPolicyMode",
    "WorldViews",
    "to_builtin",
]

# V73 truth-isolated adaptive belief-risk policy.
from .v73_adaptive_belief_risk import (
    AdaptiveMissionConstrainedController,
    AdaptiveDecision,
    CandidateCertificate,
    ControllerVisibleState,
    ControllerFeatureError,
    MissionDual,
    MonotoneAdaptiveLambdaGate,
    reject_forbidden_features,
)
from .v73_runtime_cycle import V73OperationalCycle, CycleTrace, AvailabilityAudit, derive_crn_seed
from .v73_baseline_contracts import (
    MatchedAction, BaselineBudget, BaselineContext, BaselineDecision,
    default_ten_paper_policies, run_matched_baseline_campaign,
)
