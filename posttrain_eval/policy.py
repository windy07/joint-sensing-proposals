"""Independent numeric-generation / feedback intervention for Stage6 evaluation.

The body retains the registered candidate evaluator, resource projection, coverage
rule and physical model. Only the two inference interventions differ. This module
does not change any training source or checkpoint metadata.
"""
from pathlib import Path
import hashlib
from spatial_lab import scheduling as base
from spatial_lab.scheduling import *

EXPECTED_SCHEDULING_SHA = 'ef71188fad63eb57142cff304bf344fa6842ddf95b9044ddfffcdd2e44e9bca6'


def check_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != EXPECTED_SCHEDULING_SHA:
        raise ContractError('POSTTRAIN_SCHEDULING_SOURCE_CHANGED')


def _factorial_policy(view,config,method,seed,learned,numeric_parameters,feedback_enabled):
    validate_view(view);start=time.perf_counter();config=dict(config)
    K=config['return_capacity'];B=config['root_score_budget']
    if not 1<=K<=8 or not 0<=B<=64:raise ContractError('BUDGET_DOMAIN')
    meter=Meter(B,config['construction_cap']);allow=config.get('allow_calibration',True) and method!='no_calibration'
    gen=Generator(view,config,meter,allow);reserve=gen.initial();initial_pool=set(gen.pool)
    guard,guard_cell=coverage_guard(view,config,gen) if config.get('enforce_search_revisit',False) else ((),None)
    calibration=[]
    ages=view.get('last_calibration_epochs',[-1]*4)
    due=[a for a in gen.atoms if a.kind=='calibrate' and view['epoch']-ages[a.node]>=config.get('calibration_proposal_interval',12)]
    if due:
        candidate=(min(due,key=lambda a:(ages[a.node],a.node)),)
        gen.add(candidate,'calibration_due')
        if candidate in gen.pool:calibration=[candidate]
    numeric=bool(numeric_parameters)
    network=method in ('learned','numeric_network','fixed_network')
    extra=[]
    if network:
        if learned is None:raise ContractError('NETWORK_REQUIRED')
        t0=time.perf_counter();learned.prepare(view,gen.atoms,config);meter.network_forwards+=1
        meter.network_seconds=time.perf_counter()-t0
        if numeric:
            for proposal in learned.proposals(config['max_simultaneous'],meter=meter):
                try:a=gen.add(proposal,'network_generated')
                except BudgetExhausted:break
                if a is not None:extra.append(a)
    elif numeric:
        t0=time.perf_counter();ordered=sorted(gen.atoms,key=lambda t:-gen.value((t,)))
        # Deterministic bounded low-discrepancy exploration of ALL eight controls.
        # It is a finite numerical search, not a continuous-global optimum claim.
        from scipy.stats import qmc
        controls=qmc.Halton(16,scramble=True,seed=int(seed)+view['epoch']).random(config.get('numeric_proposal_cap',16))
        for i,u in enumerate(controls):
            if not ordered:break
            t=ordered[i%len(ordered)]
            try:meter.construct()
            except BudgetExhausted:break
            meter.numeric_materializations+=1;proposal=(materialize(t,view,config,u[:8]),)
            if i%2 and len(ordered)>1:
                other=next((a for a in ordered if a.node!=t.node),None)
                if other is not None:
                    try:meter.construct()
                    except BudgetExhausted:break
                    meter.numeric_materializations+=1;proposal+=(materialize(other,view,config,u[8:]),)
            meter.numeric_optimization_calls+=1
            try:a=gen.add(proposal,'parameterized_nonlearning_search')
            except BudgetExhausted:break
            if a is not None:extra.append(a)
        meter.optimization_seconds=time.perf_counter()-t0
    H=1 if method=='myopic' else config['forecast_horizon'];ref=ProxyReference(view,config,meter,H,allow)
    def rank(pool):
        if network:return learned.rank_schedules(pool)
        if method in ('random','random_priority','numeric_random_priority'):
            p=list(pool);rng(seed,'priority',view['epoch']).shuffle(p);return p
        return sorted(pool,key=lambda a:(-gen.value(a),a))
    if method=='functional_contract':
        from spatial_lab.functional_fixture import proposal
        candidate=gen.add(proposal(view,config),'preregistered_functional_contract')
        if candidate is None:raise ContractError('FUNCTIONAL_PROPOSAL_NOT_EXECUTABLE')
        queue=[candidate,()]
    elif method=='random_execution':
        choices=[a for a in gen.pool if a];chosen=choices[int(rng(seed,'execute',view['epoch']).integers(len(choices)))] if choices else ()
        queue=[chosen];ref.horizon=1
    elif method=='periodic_scan':
        choices=[a for a in gen.pool if any(t.kind=='search' and t.item==view['epoch']%len(cell_list(config)) for t in a)];queue=choices[:1]+[()]
    else:
        # A protected paid reference does not consume a mandatory return slot.
        # At B>=2 the numeric generator has an immediate paid opportunity.
        queue=([guard] if guard else [])+([reserve] if reserve and reserve!=guard else [])+calibration+rank(extra)+[()]+rank([a for a in gen.pool if a not in extra and a!=reserve and a])
    meter.proposal_seconds=time.perf_counter()-start
    rows=[];failures=[];seen=set();feedback_done=False;feedback_object=None
    while meter.score_calls<B:
        if feedback_enabled and not feedback_done and meter.score_calls>=max(1,B//2) and meter.construction<meter.construction_cap:
            from spatial_lab.feedback import create_feedback
            best=tuple(Task(**t) for t in min(rows,key=lambda r:r['proxy_m2'])['schedule']) if rows else ()
            _feedback_start=time.perf_counter()
            feedback_object=create_feedback(view,config,gen.projections,rows,failures,meter)
            refined=[]
            if network:
                t0=time.perf_counter();learned.refine(feedback_object);meter.network_forwards+=1
                meter.network_seconds=(meter.network_seconds or 0.)+time.perf_counter()-t0
                for proposal in (learned.proposals(config['max_simultaneous'],meter=meter) if numeric else []):
                    try:a=gen.add(proposal,'network_feedback_generated')
                    except BudgetExhausted:break
                    if a is not None:refined.append(a);extra.append(a)
            # Construction charges include every projection, also rejections.
            if numeric:
                queue=gen.challenge(best,feedback_object)+refined+queue
            else:
                # Frozen projected actions: only the unscored order changes.
                queue=rank([a for a in queue if a not in seen])
                gen.feedback.append({'feedback':feedback_object,'mode':'fixed_pool_reranking',
                                     'continuous_parameters_changed':False})
            feedback_done=True
            meter.feedback_seconds=time.perf_counter()-_feedback_start
            meter.proposal_seconds+=meter.feedback_seconds
        if not queue:break
        a=queue.pop(0)
        if a in seen:continue
        seen.add(a)
        try:rows.append(ref.evaluate(a))
        except Exception as exc:failures.append({'schedule':[t.to_dict() for t in a],'error':repr(exc)})
    successful=sorted(rows,key=lambda r:(r['proxy_m2'],digest(r['schedule'])))
    if guard_cell is not None:
        guarded=[r for r in successful if r.get('root_diagnostics',{}).get('event_forecast',{}).get('discovery_probability',[0.]*4)[guard_cell]>=.5]
        if guarded:successful=guarded+[r for r in successful if r not in guarded]
    output=[r['schedule'] for r in successful[:K]];selected=tuple(Task(**t) for t in output[0]) if output else ()
    body={'schema':'ain.spatial.numeric.commit.v4','context_digest':view['digest'],'method':method,'K':K,'B':B,
          'forecast_horizon':H,'selected':[t.to_dict() for t in selected],'output':output,'archive':rows,'failures':failures,
          'reserved':[t.to_dict() for t in reserve],'policy_config':config,'input_actions':[t.to_dict() for t in gen.atoms],
          'input_action_semantic_ids':[t.semantic_id() for t in gen.atoms],
          'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,'graph_input_binding':graph_input_binding(view,gen.atoms,config),
          'network_raw_output':getattr(learned,'raw_numeric_proposals',None),'proposals':gen.projections,'feedback':gen.feedback,'feedback_object':feedback_object,'meter':meter.report(),'seconds':time.perf_counter()-start,
          'status':'VERIFIED_PROXY' if output else 'ABSTAINED','coverage_guard_cell':guard_cell,'coverage_guard_satisfied':bool(guard_cell is not None and output and successful[0] in guarded),
          'coverage_guard_status':('DISABLED' if not config.get('enforce_search_revisit',False) else 'NOT_DUE' if max(view.get('search_age_cycles',[0]))<config.get('search_revisit_cycles',6) else 'NO_FEASIBLE_OVERDUE_SEARCH') if guard_cell is None else 'NOMINAL_PAID_COVERAGE' if output and successful[0] in guarded else 'UNSATISFIED',
          'future_value_use':'numeric-conditioned future cost orders finite paid candidates; physical verification remains final',
          'capacity_audit':{'mandatory_protected_return_slots':0,'competitive_return_slots':K,
              'network_present':network,'network_role':'STRUCTURE_AND_NUMERIC_GENERATION' if network and numeric else 'FIXED_ACTION_RANKING' if network else None,
              'network_exclusive_constructed_candidates':len(set(extra)-initial_pool) if network else 0,
              'emergency_idle_used':not bool(output),'numeric_paid_candidates':sum(tuple(Task(**t) for t in r['schedule']) in extra for r in rows)},
          'future_model':'emission_conditioned_joint_delivery_reference_detection_masks; zero-innovation posterior; future association not enumerated',
          'selection_scope':'FINITE_PAID_CANDIDATES_NOT_GLOBAL_OPTIMAL','resource_pacing':bool(config.get('pace_resources',False)),'physical_safety_authority':False}
    body['digest']=digest(body);return selected,body

def run_policy(view, config, method, seed, learned=None, *, numeric_parameters, feedback_enabled):
    check_base()
    if method not in ('fixed_network', 'numeric_network') or learned is None:
        raise ContractError('FACTORIAL_REQUIRES_LEARNED_POLICY')
    with decision_linear_work() as work:
        selected, commit = _factorial_policy(view, config, method, seed, learned,
                                             numeric_parameters, feedback_enabled)
        legal(selected, view, config)
    commit['meter']['numpy_linear_algebra_calls'] = work
    commit['meter']['joint_decoder_calls'] = learned.decode_calls
    commit['meter']['joint_decoder_seconds'] = learned.decoder_seconds
    commit['intervention'] = {
        'numeric_parameters': bool(numeric_parameters), 'feedback_enabled': bool(feedback_enabled),
        'fixed_feedback_semantics': 'rerank remaining fixed projected schedules; no new numeric actions',
        'feedback_applied': commit['feedback_object'] is not None,
        'temporal_memory_updates_per_cycle': 1,
    }
    if not numeric_parameters:
        if commit['meter']['numeric_materializations'] or commit['meter']['numeric_optimization_calls']:
            raise ContractError('FIXED_PARAMETERS_CHANGED')
    if commit['meter']['score_calls'] > config['root_score_budget']:
        raise ContractError('POSTTRAIN_SCORE_BUDGET')
    if commit['meter']['construction'] > config['construction_cap']:
        raise ContractError('POSTTRAIN_CONSTRUCTION_BUDGET')
    commit['digest'] = digest({k: v for k, v in commit.items() if k != 'digest'})
    return selected, commit
