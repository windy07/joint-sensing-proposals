"""ICLR 2027 finite active-information graph proposal stack.

The source-native integration point is
``wideband_info_geometry.experiments_v75_part1.config.load_part1_runtime``
with :func:`iclr2027_graph_agent.runtime_policy.proposal_policy_factory`.
The learned model proposes only; the existing counterfactual evaluator and
commit path retain decision authority.
"""
from .schema import (
    ActionCandidate,
    DynamicInformationGraph,
    GraphEdge,
    GraphHyperedge,
    GraphNode,
    NodeKind,
)
from .builder import InformationGraphBuilder, LeakageError, CausalityError
from .coverage import (
    CertificateSearchResult,
    bounded_certificate_search,
    robust_frame_ratio,
)
from .risk import AssociationBelief, CandidateRiskVector, TemporalRiskState

__all__ = [
    "ActionCandidate",
    "AssociationBelief",
    "CandidateRiskVector",
    "CertificateSearchResult",
    "CausalityError",
    "DynamicInformationGraph",
    "GraphEdge",
    "GraphHyperedge",
    "GraphNode",
    "InformationGraphBuilder",
    "LeakageError",
    "NodeKind",
    "TemporalRiskState",
    "bounded_certificate_search",
    "robust_frame_ratio",
]

# ICLR 2027 experiment-contract exports
from .experiment_contract import (ExperimentContract, MatchedWorldContract, MethodArm, ReferenceSemantics, ExecutionTier)
from .metrics import wilson_lower_bound, tail_summary
