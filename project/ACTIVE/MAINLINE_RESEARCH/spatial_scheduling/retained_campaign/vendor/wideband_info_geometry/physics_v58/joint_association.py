"""Owned finite association components of one DynamicFactorGraph posterior.

This is an explicitly selected frozen affine Bernoulli return-set model. It
does not claim closure for the legacy R2 extractor, JPDA, state-dependent pD,
nonlinear return covariance, correlated undeclared noise or multiple targets.
Each component is a conditional Gaussian graph, never a second online owner.
No pruning or moment merging is performed; insufficient capacity is UNKNOWN.
"""
from __future__ import annotations

import copy
import math
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gaussian_history import immutable, solve_spd, sym


class JointModelUnavailable(ValueError):
    """A declared joint-model operation cannot safely publish a posterior."""


class JointOperationBudget:
    """Explicit per-run computation ledger; failed transactions do not refund it.

    These are operation counts, not a wall-time, RAM or hardware cost model.
    A branch fork means one actual DynamicFactorGraph.fork invocation. Leaf
    likelihood rows and branch alternatives are separate from Gaussian solves.
    """
    NAMES = frozenset(('branch_forks', 'gaussian_solves', 'factor_constructions',
        'likelihood_branches', 'predictive_integrals', 'leaf_density_terms',
        'history_admissions', 'cut_stages', 'cut_commits', 'joint_batches',
        'joint_predictions', 'joint_snapshots', 'posterior_forks'))

    def __init__(self, limits):
        if set(limits) != self.NAMES or any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError('Declare a nonnegative integer cap for every joint operation')
        self.limits = dict(limits)
        self.used = dict.fromkeys(self.NAMES, 0)
        self.events = []

    def charge(self, name, amount=1, **detail):
        if name not in self.NAMES or type(amount) is not int or amount < 0:
            raise ValueError('Invalid joint operation charge')
        allowed = self.used[name] + amount <= self.limits[name]
        self.events.append(dict(event=name, amount=amount, allowed=allowed, **detail))
        if not allowed:
            raise JointModelUnavailable('UNKNOWN_COMPUTATION_CAP:' + name)
        self.used[name] += amount


@dataclass(frozen=True)
class GaussianComponent:
    mode_id: str
    exists: bool
    log_weight: float
    graph: Any = field(repr=False)
    solution: Any = field(repr=False)
    graph_digest: str
    solution_digest: str


@dataclass(frozen=True)
class JointPosteriorState:
    epoch: int
    recipient_id: str
    components: tuple[GaussianComponent, ...]
    acquisitions: tuple[tuple[str, str, str], ...]
    branch_cap: int
    acquisition_cap: int
    revision: int
    model: str = 'finite_frozen_affine_Bernoulli_complete_return_set'


@dataclass(frozen=True, eq=False)
class JointGraphStage:
    state: JointPosteriorState = field(repr=False)
    source_digest: str
    batch: Any = field(repr=False)


_ISSUED = weakref.WeakKeyDictionary()


def state_digest(state):
    from .dynamic_factor_graph import _frozen_digest
    return _frozen_digest((state.epoch, state.recipient_id, state.acquisitions, state.branch_cap,
        state.acquisition_cap, state.revision, state.model,
        tuple((c.mode_id, c.exists, c.log_weight, c.graph._model_digest(),
               c.graph_digest, _frozen_digest(c.solution), c.solution_digest)
              for c in state.components)))


def _require(owner):
    state = getattr(owner, '_joint_state', None)
    if state is None:
        raise JointModelUnavailable('Joint complete-return-set mode is not enabled')
    from .dynamic_factor_graph import _frozen_digest
    for c in state.components:
        if (c.graph._model_digest() != c.graph_digest
                or _frozen_digest(c.solution) != c.solution_digest):
            raise JointModelUnavailable('UNKNOWN_MUTATED_JOINT_COMPONENT')
    return state


def _solve(graph, epoch, budget):
    budget.charge('gaussian_solves', epoch=epoch)
    result = graph.solve(epoch, irls_iterations=1, diagnostics=False)
    if result.status != 'passed' or result.rejected_factors:
        raise JointModelUnavailable('UNKNOWN_GAUSSIAN_COMPONENT_HEALTH_OR_REJECTED_EVIDENCE')
    if (set(result.variable_slices) != {k.token() for k in graph.variables}
            or set(result.accepted_factor_ids) != {f.factor_id for f in graph.factors}):
        raise JointModelUnavailable('UNKNOWN_PARTIAL_GAUSSIAN_SCOPE_OR_PENDING_FACTORS')
    # Singular support is outside this declared proper Gaussian implementation.
    solve_spd(result.joint_covariance, np.zeros(result.joint_covariance.shape[0]))
    return result


def _component(mode, exists, log_weight, graph, epoch, budget, solution=None):
    from .dynamic_factor_graph import _frozen_digest
    solution = _solve(graph, epoch, budget) if solution is None else solution
    return GaussianComponent(mode, bool(exists), float(log_weight), graph, solution,
                             graph._model_digest(), _frozen_digest(solution))


def _copy_graph(graph, budget):
    budget.charge('branch_forks')
    return graph.fork()


def enable(owner, *, existence_probability, branch_cap, acquisition_cap, recipient_id, budget):
    """Promote the current proper Gaussian into one Bernoulli joint owner.

    Initially the declared continuous state has the same distribution under
    E=0 and E=1. It may include background/nuisance variables. A nonexistent
    target's mathematical coordinates are not claimed to describe a physical
    target; an absence likelihood must declare the background G explicitly.
    """
    from .dynamic_factor_graph import RobustKernel
    if getattr(owner, '_joint_state', None) is not None:
        raise JointModelUnavailable('Joint mode cannot be initialized twice')
    r = float(existence_probability)
    if not math.isfinite(r) or not 0 <= r <= 1:
        raise ValueError('Existence probability must be in [0,1]')
    if (type(branch_cap) is not int or branch_cap <= 0
            or type(acquisition_cap) is not int or acquisition_cap <= 0
            or type(budget) is not JointOperationBudget):
        raise ValueError('Explicit finite branch/acquisition/computation capacities are required')
    if not isinstance(recipient_id, str) or not recipient_id.strip():
        raise ValueError('A fixed recipient identity is required')
    if not owner.variables:
        raise ValueError('A proper declared Gaussian seed graph is required')
    if any(f.robust_kernel is not RobustKernel.NONE or f.association_probabilities != (1.0,)
           or f.correlation_group is not None or f.missing for f in owner.factors):
        raise JointModelUnavailable('UNKNOWN_NON_GAUSSIAN_OR_UNDECLARED_CORRELATED_SEED')
    epoch = max(k.epoch for k in owner.variables)
    possibilities = tuple((exists, p) for exists, p in ((False, 1-r), (True, r)) if p > 0)
    if len(possibilities) > branch_cap:
        raise JointModelUnavailable('UNKNOWN_BRANCH_CAP')
    original = owner._model_digest()
    components = []
    for exists, p in possibilities:
        graph = _copy_graph(owner, budget)
        components.append(_component('E1' if exists else 'E0', exists, math.log(p), graph, epoch, budget))
    state = JointPosteriorState(epoch, recipient_id, tuple(components), (), branch_cap, acquisition_cap, 0)
    if owner._model_digest() != original:
        raise JointModelUnavailable('UNKNOWN_SEED_MUTATION')
    owner._joint_budget = budget
    owner._joint_state = state


def scope_moments(component, keys, dimensions, units):
    """Full current/history joint marginal, including all cross-covariances."""
    graph, solution = component.graph, component.solution
    live = {v.key.token(): v for v in graph.variables.values()}
    history = graph._retained_history
    historic = {} if history is None else {v.key.token(): v for v in history.history_variables}
    descriptions = {**live, **historic}
    if len(set(keys)) != len(keys) or len(keys) != len(dimensions) or len(keys) != len(units):
        raise JointModelUnavailable('UNKNOWN_CAPTURE_COORDINATE_SCHEMA')
    for key, dimension, unit in zip(keys, dimensions, units):
        v = descriptions.get(key)
        if v is None or v.dimension != dimension or tuple(v.units) != tuple(unit):
            raise JointModelUnavailable('UNKNOWN_CAPTURE_COORDINATE_OR_UNITS:' + key)
    nq = solution.joint_covariance.shape[0]
    qmean = np.zeros(nq)
    for key, (start, stop) in solution.variable_slices.items():
        qmean[start:stop] = solution.estimates[key]
    n = sum(dimensions)
    a, transform, noise = np.zeros(n), np.zeros((n, nq)), np.zeros((n, n))
    hislices, offset = {}, 0
    if history is not None:
        for v in history.history_variables:
            hislices[v.key.token()] = slice(offset, offset+v.dimension)
            offset += v.dimension
        t_full = np.zeros((history.dimension, nq))
        offset = 0
        for v in history.boundary_variables:
            start, stop = solution.variable_slices[v.key.token()]
            t_full[:, start:stop] = history.boundary_map[:, offset:offset+v.dimension]
            offset += v.dimension
    destinations, offset = {}, 0
    for key, dimension in zip(keys, dimensions):
        dest = slice(offset, offset+dimension)
        destinations[key] = dest
        offset += dimension
        if key in hislices:
            sl = hislices[key]
            a[dest], transform[dest] = history.offset[sl], t_full[sl]
        else:
            start, stop = solution.variable_slices[key]
            transform[dest, start:stop] = np.eye(dimension)
    for left in keys:
        for right in keys:
            if left in hislices and right in hislices:
                noise[destinations[left], destinations[right]] = history.covariance[hislices[left], hislices[right]]
    return a + transform @ qmean, sym(noise + transform @ solution.joint_covariance @ transform.T)


def _predictive_log_density(mean, covariance, branch, budget):
    h, y, r = np.asarray(branch.H), np.asarray(branch.y), np.asarray(branch.R)
    if not y.size:
        return 0.0
    budget.charge('predictive_integrals', rows=int(y.size))
    s = sym(r + h @ covariance @ h.T)
    residual = y - h @ mean
    # Dimensionless diagonal equilibration preserves the declared density units.
    d = np.sqrt(np.diag(s))
    c = s / d[:, None] / d[None, :]
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        raise JointModelUnavailable('UNKNOWN_PREDICTIVE_COVARIANCE')
    logdet += 2 * float(np.log(d).sum())
    value = -.5 * (y.size*math.log(2*math.pi) + logdet + float(residual @ solve_spd(s, residual)))
    if not math.isfinite(value):
        raise JointModelUnavailable('UNKNOWN_PREDICTIVE_NORMALIZER')
    return value


def _apply_likelihood(component, law, payload, branch, epoch, budget):
    from .dynamic_factor_graph import DynamicFactor, FactorKind, RobustKernel
    graph = _copy_graph(component.graph, budget)
    h, y, r = np.asarray(branch.H), np.asarray(branch.y), np.asarray(branch.R)
    if y.size:
        historic = {} if graph._retained_history is None else {
            v.key.token(): v for v in graph._retained_history.history_variables}
        descriptions = {v.key.token(): v for v in graph.variables.values()} | historic
        keys = tuple(descriptions[k].key for k in law.capture_keys)
        jacobians, offset = [], 0
        for dim in law.capture_dimensions:
            jacobians.append(h[:, offset:offset+dim]); offset += dim
        budget.charge('factor_constructions')
        factor = DynamicFactor('joint_packet:' + payload.provenance_id,
            FactorKind.MEASUREMENT, payload.observation_epoch, epoch, keys,
            tuple(jacobians), y, r, (), (payload.provenance_id,),
            robust_kernel=RobustKernel.NONE,
            metadata={'provenance_roots': (payload.provenance_id,),
                'conditional_noise_independent_given_scope': True,
                'whole_set_law_digest': law.digest,
                'whole_set_payload_digest': payload.digest,
                'association_alternative': branch.event,
                'association_index': branch.association_index,
                'observation_units': law.measurement_units * (int(y.size)//len(law.measurement_units)),
                'mode_alternatives_are_mutually_exclusive': True})
        if set(law.capture_keys).intersection(historic):
            budget.charge('history_admissions')
            budget.charge('factor_constructions', induced_historical_likelihood=True)
            graph.add_historical_factor(factor, current_epoch=epoch)
        else:
            graph.add_factor(factor)
    # Even an empty/constant channel consumes its packet at the joint owner.
    # A new solve is charged because the graph was actually forked; no copied
    # solution is promoted to graph-issued marginalization authority.
    solution = _solve(graph, epoch, budget)
    return graph, solution


def prepare_batch(owner, batch):
    from wideband_info_geometry.operational_runtime.joint_receipt import validate_joint_batch
    state = _require(owner)
    validate_joint_batch(batch, epoch=state.epoch, recipient_id=state.recipient_id)
    if batch.status != 'passed' or batch.epoch != state.epoch:
        raise JointModelUnavailable('UNKNOWN_BATCH_OR_CURRENT_EPOCH')
    budget = owner._joint_budget
    budget.charge('joint_batches')
    source = owner._model_digest()
    components, acquisitions = state.components, list(state.acquisitions)
    seen = {p[0] for p in acquisitions}
    if len(acquisitions)+len(batch.receipts) > state.acquisition_cap:
        raise JointModelUnavailable('UNKNOWN_ACQUISITION_CAP')
    for receipt in batch.receipts:
        law, payload = batch.law_for(receipt), receipt.payload
        law.validate_payload(payload)
        if payload.observation_epoch > state.epoch:
            raise JointModelUnavailable('UNKNOWN_FUTURE_CAPTURE')
        if payload.provenance_id in seen:
            raise JointModelUnavailable('UNKNOWN_REUSED_WHOLE_PACKET_EVIDENCE')
        # Detect ancestor roots already in the Gaussian seed/history as well.
        for c in components:
            if (payload.provenance_id in c.graph._represented_root_owners
                    or any(payload.provenance_id in c.graph._fusion_provenance_lineages(f)
                           for f in c.graph.factors)):
                raise JointModelUnavailable('UNKNOWN_REUSED_SEED_EVIDENCE')
        plans = []
        m = len(payload.measurements)
        rho_m = law.rho[m] if m < len(law.rho) else 0.0
        rho_previous = law.rho[m-1] if 0 < m <= len(law.rho) else 0.0
        counts = tuple((int(rho_m > 0 and (not c.exists or law.p_D < 1))
                        + m*int(c.exists and law.p_D > 0 and rho_previous > 0))
                       for c in components)
        if sum(counts) > state.branch_cap:
            raise JointModelUnavailable('UNKNOWN_BRANCH_CAP')
        for c, count in zip(components, counts):
            budget.charge('likelihood_branches', count, packet=payload.provenance_id)
            budget.charge('leaf_density_terms', m*max(0,m-1)*int(
                c.exists and law.p_D > 0 and rho_previous > 0))
            alternatives = law.factor_branches(payload, c.exists)
            if len(alternatives) != count:
                raise JointModelUnavailable('UNKNOWN_BRANCH_ENUMERATION_CONTRACT')
            # Miss Gaussian rows are integrated jointly below; detected clutter
            # constants each call one leaf Gaussian density. Count their work.
            plans.extend((c, a) for a in alternatives)
        if not plans:
            raise JointModelUnavailable('UNKNOWN_ZERO_MODEL_SUPPORT')
        if len(plans) > state.branch_cap:
            raise JointModelUnavailable('UNKNOWN_BRANCH_CAP')
        updated = []
        for c, branch in plans:
            mean, covariance = scope_moments(c, law.capture_keys, law.capture_dimensions, law.capture_units)
            logw = c.log_weight + branch.log_constant + _predictive_log_density(mean, covariance, branch, budget)
            graph, solution = _apply_likelihood(c, law, payload, branch, state.epoch, budget)
            updated.append(_component(c.mode_id + '/' + payload.provenance_id + ':' + branch.event
                + ':' + str(branch.association_index), c.exists, logw, graph, state.epoch, budget, solution))
        maximum = max(c.log_weight for c in updated)
        normalizer = maximum + math.log(math.fsum(math.exp(c.log_weight-maximum) for c in updated))
        components = tuple(GaussianComponent(c.mode_id,c.exists,c.log_weight-normalizer,c.graph,c.solution,
                                             c.graph_digest,c.solution_digest) for c in updated)
        acquisitions.append((payload.provenance_id, payload.digest, law.digest)); seen.add(payload.provenance_id)
    proposed = (JointPosteriorState(state.epoch, state.recipient_id, components, tuple(acquisitions),
                                  state.branch_cap, state.acquisition_cap, state.revision+1)
                if batch.receipts else state)
    if owner._model_digest() != source:
        raise JointModelUnavailable('UNKNOWN_CONCURRENT_GRAPH_CHANGE')
    stage = JointGraphStage(proposed, source, batch)
    _ISSUED[owner] = (stage, state_digest(proposed))
    return stage


def validate_stage(owner, stage, batch):
    from wideband_info_geometry.operational_runtime.joint_receipt import validate_joint_batch
    state = _require(owner)
    validate_joint_batch(batch, epoch=state.epoch, recipient_id=state.recipient_id)
    issued = _ISSUED.get(owner)
    if (issued is None or issued[0] is not stage or stage.batch is not batch
            or owner._model_digest() != stage.source_digest
            or state_digest(stage.state) != issued[1]):
        raise JointModelUnavailable('UNKNOWN_STALE_OR_UNISSUED_JOINT_STAGE')
    # Allocation and validation happen before either inbox or graph publishes.
    return {'_joint_state': stage.state}


def predict(owner, *, new_variable, previous_keys, transition_matrix, offset, covariance, provenance_id):
    """One normalized affine Markov transition; no hidden new prior or evidence."""
    from .dynamic_factor_graph import DynamicFactor, FactorKind, RobustKernel
    state = _require(owner); budget = owner._joint_budget
    budget.charge('joint_predictions')
    if (new_variable.prior_enabled or new_variable.key.epoch != state.epoch+1
            or not previous_keys or len(set(previous_keys)) != len(previous_keys)):
        raise JointModelUnavailable('UNKNOWN_NON_MARKOV_PREDICTION_SCHEMA')
    q = sym(covariance); f = np.asarray(transition_matrix, dtype=float); b = np.asarray(offset, dtype=float)
    d = new_variable.dimension
    if q.shape != (d,d) or b.shape != (d,) or not np.isfinite(f).all():
        raise ValueError('Prediction coordinate shape/values disagree')
    solve_spd(q, np.zeros(d))
    original = owner._model_digest(); out=[]
    for c in state.components:
        graph = _copy_graph(c.graph, budget)
        if (new_variable.key in graph.variables or any(k not in graph.variables for k in previous_keys)
                or any(k.epoch != state.epoch for k in previous_keys)):
            raise JointModelUnavailable('UNKNOWN_PREDICTION_SCOPE')
        dimensions = [graph.variables[k].dimension for k in previous_keys]
        if f.shape != (d,sum(dimensions)):
            raise ValueError('Prediction matrix shape disagrees with complete scope')
        if provenance_id in dict((a[0],a[1]) for a in state.acquisitions):
            raise JointModelUnavailable('UNKNOWN_REUSED_PROCESS_OR_PACKET_ROOT')
        graph.add_variable(copy.deepcopy(new_variable))
        jacobians=[]; start=0
        for dim in dimensions:
            jacobians.append(-f[:,start:start+dim]); start+=dim
        budget.charge('factor_constructions')
        factor=DynamicFactor('joint_process:'+provenance_id,FactorKind.TRANSITION,
            state.epoch+1,state.epoch+1,(*previous_keys,new_variable.key),(*jacobians,np.eye(d)),b,q,(),
            (provenance_id,),robust_kernel=RobustKernel.NONE,
            metadata={'provenance_roots':(provenance_id,), 'normalized_transition':True})
        if graph.evidence_admission_reason(factor) is not None:
            raise JointModelUnavailable('UNKNOWN_REUSED_PROCESS_ROOT')
        graph.add_factor(factor)
        out.append(_component(c.mode_id,c.exists,c.log_weight,graph,state.epoch+1,budget))
    proposed=JointPosteriorState(state.epoch+1,state.recipient_id,tuple(out),state.acquisitions,
                                state.branch_cap,state.acquisition_cap,state.revision+1)
    if owner._model_digest()!=original:raise JointModelUnavailable('UNKNOWN_CONCURRENT_GRAPH_CHANGE')
    owner._joint_state=proposed


def cut(owner, *, eliminate_epoch, boundary_keys):
    from wideband_info_geometry.operational_runtime.belief import make_scoped_information_factor
    state=_require(owner); budget=owner._joint_budget; original=owner._model_digest(); out=[]
    for c in state.components:
        graph=_copy_graph(c.graph,budget)
        solution=_solve(graph,state.epoch,budget)
        budget.charge('cut_stages')
        stage=graph.marginalize_epoch_to_boundary(eliminate_epoch=eliminate_epoch,
            boundary_keys=tuple(boundary_keys),solved_epoch=state.epoch,solution=solution,arrival_epoch=state.epoch)
        budget.charge('factor_constructions')
        factor=make_scoped_information_factor(factor_id='joint_separator:'+str(state.revision)+':'+str(eliminate_epoch),
            variables=tuple(graph.variables[k] for k in stage.boundary_keys),information=stage.information,
            information_vector=stage.information_vector,epoch=min(k.epoch for k in stage.boundary_keys),
            arrival_epoch=stage.arrival_epoch,provenance_ids=(stage.separator_provenance_id,),
            fidelity_level='frozen_joint_Gaussian_branch',metadata=stage.separator_metadata)
        budget.charge('cut_commits')
        graph.commit_marginalization(stage,factor)
        out.append(_component(c.mode_id,c.exists,c.log_weight,graph,state.epoch,budget))
    proposed=JointPosteriorState(state.epoch,state.recipient_id,tuple(out),state.acquisitions,
                                state.branch_cap,state.acquisition_cap,state.revision+1)
    if owner._model_digest()!=original:raise JointModelUnavailable('UNKNOWN_CONCURRENT_GRAPH_CHANGE')
    owner._joint_state=proposed


def snapshot(owner, *, capture_keys=None, capture_dimensions=None, capture_units=None):
    from .dynamic_factor_graph import _frozen_digest
    state=_require(owner); owner._joint_budget.charge('joint_snapshots')
    if capture_keys is None:
        variables=tuple(sorted(state.components[0].graph.variables.values(),key=lambda v:v.key.token()))
        capture_keys=tuple(v.key.token() for v in variables)
        capture_dimensions=tuple(v.dimension for v in variables)
        capture_units=tuple(v.units for v in variables)
    modes=[]
    for c in state.components:
        mean,cov=scope_moments(c,capture_keys,capture_dimensions,capture_units)
        modes.append({'mode_id':c.mode_id,'exists':c.exists,'weight':math.exp(c.log_weight),
            'log_weight':c.log_weight,'mean':immutable(mean),'covariance':immutable(cov),
            'graph_digest':c.graph_digest,'history_digest':None if c.graph._retained_history is None else
                _frozen_digest(c.graph._retained_history)})
    return {'schema_version':'belief-snapshot.v1','timestamp':state.epoch,
        'posterior_authority':'DynamicFactorGraph','graph_digest':owner._model_digest(),
        'admitted_factor_digest':state_digest(state),'health':'healthy','truth_access':False,
        'model_scope':state.model,'recipient_id':state.recipient_id,
        'capture_keys':tuple(capture_keys),'capture_dimensions':tuple(capture_dimensions),
        'capture_units':tuple(tuple(u) for u in capture_units),'modes':tuple(modes),
        'existence_probability':math.fsum(m['weight'] for m in modes if m['exists']),
        'acquisitions':state.acquisitions,'branch_cap':state.branch_cap,'revision':state.revision,
        'pruning_or_moment_merge':False,'full_R2_physical_Bayes_claimed':False}
