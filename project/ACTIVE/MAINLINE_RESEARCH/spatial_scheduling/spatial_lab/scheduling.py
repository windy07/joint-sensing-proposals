"""Budgeted candidate construction, independent proxy verification and feedback.

Every candidate is a concrete simultaneous node-task schedule. A forecast reads
only the frozen posterior, resource ledger and public sensor model. It never
reads measurement realizations, truth paths or audit results.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations
from fractions import Fraction
import math,time
import numpy as np
from .core import *
from .physics import *
from .actions import materialize,project,visits,resource,CONTROL_NAMES
from .shared_events import captures,unique_captures,packet_gate,any_detection_probability,emission_branches,moment_match,EVENT_SPEC
from dataclasses import replace
from scipy.linalg import block_diag
from .binding import FEATURE_BINDING,LABEL_BINDING,graph_input_binding

def validate_view(view):
    d=dict(view);old=d.pop('digest',None)
    if old!=digest(d):raise ContractError('CONTEXT_BINDING')
    for key in ('truth','true_targets','future_observations','audit','world_template'):
        if key in d:raise ContractError('CAUSAL_LEAK')
    if d['units']!={'position':'m','velocity':'m/s','bearing':'rad','time':'s','RF_energy':'J'}:raise ContractError('UNITS')
    P=np.asarray(d['covariance']);m=np.asarray(d['mean'])
    if P.shape!=(len(m),len(m)) or not np.isfinite(P).all() or not np.isfinite(m).all():raise ContractError('BELIEF_SCHEMA')
    return d

def bind(d):
    d=dict(d);d.pop('digest',None);d['digest']=digest(d);return d

def atoms(view,config,allow_calibration=True):
    ans=[]
    for sensor in SENSORS:
        roles=('monostatic',) if sensor.mode=='active' else ('bistatic','direct')
        for role in roles:
            proto=[Task(sensor.id,'search',c,role=role) for c in range(len(cell_list(config)))]
            proto += [Task(sensor.id,'track',t['id'],role=role) for t in view['tracks']]
            if allow_calibration and config.get('allow_calibration',True):
                # One known reference calibration role per node.
                if role==roles[0]:proto += [Task(sensor.id,'calibrate',role=role)]
            expanded=[]
            for t in proto:
                for mode in config.get('allowed_measurement_modes',['full']):
                    if mode=='aoa' and (t.kind=='calibrate' or t.role=='monostatic'):continue
                    expanded.append(replace(t,mode=mode))
            for t in expanded:
                t=materialize(t,view,config)
                try:legal((t,),view,config);ans.append(t)
                except ContractError:
                    minimal=materialize(t,view,config,[0.,0.,.5,0.,.5,.5,.5,0.])
                    try:legal((minimal,),view,config);ans.append(minimal)
                    except ContractError:pass
    return sorted(ans)

def discovery_prob(schedule,view,config,delivery_probability=None,*,capture_rows=None):
    """Nine-point area quadrature, with shared emission and per-capture events.

    Admission provides observed delivery/reference conditions, not new captures.
    A pure direct-source opportunity cannot exceed the emission probability.
    """
    rows=captures(schedule,config) if capture_rows is None else unique_captures(capture_rows)
    ps=np.zeros(len(cell_list(config)));width=float(config.get('search_cell_halfwidth_m',350.))
    def localized(t,sensor,pos):
        if t.mode=='aoa':return False
        if t.role=='monostatic':return True
        H=jacobian(np.r_[pos,0.,0.],sensor,NBIAS+4,NBIAS,t)[:,NBIAS:NBIAS+2]
        R=noise_cov(pos,sensor,t,view['epoch'],config)
        return eigvalsh(H.T@solve(R,H)).min()>=1/config.get('birth_position_variance_limit_m2',1e6)
    for c,centre in enumerate(cell_list(config)):
        values=[any_detection_probability(rows,centre+[dx,dy],view,config,
                  localization_test=localized,delivery_probability=delivery_probability)
                for dx in (-width,0.,width) for dy in (-width,0.,width)]
        ps[c]=sum(values)/len(values)
    return ps

def admitted_captures(packets,results,config):
    """Actual accepted packet roots; do not add turn/settle time a second time."""
    if len(packets)!=len(results):raise ContractError('CREDIT_PACKET_RESULT_BINDING')
    admitted={'CALIBRATED','NO_EXISTING_TRACK','GATED_JOINT_MOMENT_UPDATE','GNN_CAP_APPROXIMATION'}
    rows=[]
    for p,r in zip(packets,results):
        if r['status'] not in admitted:continue
        t=Task(**p['task'])
        rows.append({'task':t,'capture_id':p['root'],'mid_s':p['acquire_s']-p['acquire_epoch']*config['dt_s'],
                     'delivered_condition':True,'reference_condition':True,'reference_capture_root':p.get('reference_capture_root')})
    return unique_captures(rows)


def admitted_schedule(packets,results):
    accepted=[];roots=set()
    admitted={'CALIBRATED','NO_EXISTING_TRACK','GATED_JOINT_MOMENT_UPDATE','GNN_CAP_APPROXIMATION'}
    for p,r in zip(packets,results):
        if r['status'] not in admitted or p['root'] in roots:continue
        roots.add(p['root']);accepted.append(Task(**p['task']))
    return tuple(sorted(accepted))


def coverage_guard(view,config,gen):
    """Reserve a paid search of the oldest feasible sector; no target truth.

    Ages describe admitted broad searches, not confirmed target discovery.
    The best attainable nominal localization probability is used as an ordering
    criterion. An empty guard does not certify area coverage.
    """
    ages=np.asarray(view.get('search_age_cycles',[view['epoch']+1]*len(cell_list(config))))
    threshold=int(config.get('search_revisit_cycles',6))
    overdue=[c for c in range(len(cell_list(config))) if ages[c]>=threshold]
    if not overdue:return (),None
    searches=[a for a in gen.atoms if a.kind=='search']
    options=[(a,) for a in searches]
    options += [p for p in combinations(searches,2)
                if p[0].node!=p[1].node and p[0].item==p[1].item
                and all(SENSORS[t.node].mode=='passive' for t in p)]
    feasible=[]
    for a in options:
        try:legal(tuple(a),view,config)
        except ContractError:continue
        feasible.append((tuple(a),discovery_prob(a,view,config)))
    for cell in sorted(overdue,key=lambda c:(-ages[c],c)):
        choices=[(a,p[cell]) for a,p in feasible if p[cell]>=.5]
        if choices:
            chosen=max(choices,key=lambda x:(x[1],-sum(cost(t)['energy_J'] for t in x[0]),x[0]))[0]
            # Construct and account for this proposal before the general pool.
            gen.add(chosen)
            return chosen,cell
    return (),None

def criterion(view,config):
    P=np.asarray(view['covariance']);loc=sum(float(np.trace(P[t['start']:t['start']+2,t['start']:t['start']+2])) for t in view['tracks'])
    missed=config['missed_target_scale_m']**2*float(np.sum(view['unseen_intensity']))
    coast=float(view.get('last_expected_missed_tracks',0.))*config.get('coast_penalty_m2',1000.)
    tentative=config.get('tentative_track_penalty_m2',100.)*sum(not t['confirmed'] for t in view['tracks'])
    return {'position_uncertainty_m2':loc,'unlocalized_target_cost_m2':missed,
            'expected_coast_cost_m2':coast,'tentative_track_cost_m2':tentative,
            'selection_proxy_m2':loc+missed+coast+tentative}

def predicted_receipts(schedule,view,config):
    """Causal processor/transport queue prediction, no received values or truth."""
    pending=[];epoch_start=view['decision_s'];clock=max(epoch_start,float(view.get('communication_available_s',epoch_start)))
    processors=list(view.get('processing_available_s',[epoch_start]*4))
    for visit in captures(schedule,config):
            task=visit['task'];dv=device(config,task.node);work=resource(task,config)
            pending.append((epoch_start+visit['end_s'],(task.node,visit['index']),visit,work['processing_ops']/dv['processing_ops_s'],work['communication_bytes']*8/dv['communication_bps']))
    ready=[]
    for end,ident,visit,proc,tx in sorted(pending,key=lambda row:(row[0],row[1])):
        n=visit['task'].node;finished=max(end,processors[n])+proc;processors[n]=finished
        ready.append((finished,ident,visit,tx))
    accepted=[];receipts=[]
    for available,_,visit,tx in sorted(ready,key=lambda row:(row[0],row[1])):
        arrival=max(available,clock)+tx;clock=arrival
        accepted_now=arrival<=epoch_start+config['dt_s']
        receipts.append({'task_id':visit['task'].execution_id(),'predicted_arrival_s':arrival,'accepted_this_cycle':accepted_now})
        if accepted_now:accepted.append(visit)
    return accepted,clock,processors,receipts

def advance_time(view,config):
    """Pure next-epoch state prediction. No action, missed event, or RF charge."""
    d=deepcopy(view);m=np.asarray(d['mean']);P=np.asarray(d['covariance'])
    F=np.eye(len(m));Q=np.zeros_like(P);Q[:NBIAS,:NBIAS]=np.diag(BIAS_WALK_SD**2)
    f,q=transition(config['dt_s'],config['acceleration_psd_m2_s3'])
    for t in d['tracks']:
        j=t['start'];F[j:j+4,j:j+4]=f;Q[j:j+4,j:j+4]=q
    m=F@m;P=F@P@F.T+Q;d['epoch']+=1;d['decision_s']+=config['dt_s']
    d['unseen_intensity']=(np.asarray(d['unseen_intensity'])+config['birth_intensity_per_cell_per_cycle']).tolist()
    d['known_node_states']=[availability(d['epoch'],s.id,config) for s in sensor_list(config)]
    d['search_age_cycles']=[x+1 for x in d.get('search_age_cycles',[0]*len(cell_list(config)))]
    d['last_expected_missed_tracks']=0.;d['predicted_receipts']=[];d['emission_event_posterior']={}
    # Absolute queue availability times are retained. At the next action the
    # queue uses max(now, available); no processing/transmission is charged twice.
    d['mean']=m.tolist();d['covariance']=((P+P.T)/2).tolist();return bind(d)

def forecast_once(view,schedule,config,meter,advance=False):
    """One actual candidate effect, conditional on finite shared emission masks.

    Within each emitter branch, independent capture/detection masks are moment
    matched after a capture. This remains a mean-innovation proxy, not exact
    future JPDA or an expected truth-error calculation. The common emitter is
    NEVER collapsed between its visits. Shared packet/reference gates mix the
    full joint posterior, not target-by-target marginal covariance gains.
    """
    if advance:
        if schedule:raise ContractError('TIME_ONLY_REQUIRES_EMPTY_ACTION')
        return advance_time(view,config)
    legal(schedule,view,config);d=deepcopy(view);m=np.asarray(d['mean']);prior=np.asarray(d['covariance']);sensors=sensor_list(config)
    arrivals,link_free,processors,receipt_audit=predicted_receipts(schedule,view,config)
    direct_ids=[tr['id'] for tr in d['tracks'] if any(v['task'].role=='direct' and covers(m[tr['start']:tr['start']+2]+m[tr['start']+2:tr['start']+4]*v['mid_s'],v['task'],d,sensors[v['task'].node]) for v in arrivals)]
    branch_weights=[];branch_P=[];branch_miss=[]
    for ew,emits in emission_branches(direct_ids,config.get('emission_probability',.7)):
        P=prior.copy();missed=np.ones(len(d['tracks']))
        for visit in arrivals:
            task=visit['task'];sensor=sensors[task.node];off=visit['mid_s'];gate=packet_gate(visit,d['epoch'],config)
            if gate<=0:continue
            if task.kind=='calibrate':
                try:_,H,R,pd=calibration_model(task,m[:NBIAS],len(m),d['epoch'],config)
                except ContractError:continue
                # Calibration's acquired surveyed reference is deterministic
                # when available; its single detection event is pd, not pd^2.
                pd=(1.-config.get('packet_loss_probability',0.))*pd
                _,post,_,_=gaussian_update(m,P,H,R,np.zeros(len(R)),meter)
                _,P=moment_match([1-pd,pd],[m,m],[P,post]);meter.forecast_updates+=1;continue
            Hs=[];Rs=[];pds=[];common=None
            for i,tr in enumerate(d['tracks']):
                j=tr['start'];state=m[j:j+4];pos=state[:2]+state[2:]*off
                if not covers(pos,task,d,sensor):continue
                pd=conditional_detection_probability(pos,sensor,task,d['epoch'],config,emission=emits.get(tr['id'],0) if task.role=='direct' else None)
                if pd<=0:continue
                Hs.append(jacobian(state,sensor,len(m),j,task,off));R,C=noise_parts(pos,sensor,task,d['epoch'],config)
                Rs.append(R+C);common=C;pds.append(pd);missed[i]*=1-gate*pd
            if not Hs:continue
            if len(Hs)>6:raise ContractError('DETECTION_BRANCH_LIMIT_SIX_TARGETS')
            weights=[1-gate];posts=[P]
            for mask in range(1<<len(Hs)):
                chosen=[i for i in range(len(Hs)) if mask>>i&1]
                weight=gate*math.prod(pds[i] if i in chosen else 1-pds[i] for i in range(len(Hs)))
                if weight==0:continue
                post=P
                if chosen:
                    H=np.vstack([Hs[i] for i in chosen]);R=block_diag(*[Rs[i] for i in chosen]);k=len(Rs[0])
                    for i in range(len(chosen)):
                        for j in range(len(chosen)):
                            if i!=j:R[i*k:(i+1)*k,j*k:(j+1)*k]=common
                    _,post,_,_=gaussian_update(m,P,H,R,np.zeros(len(R)),meter);meter.forecast_updates+=1
                weights.append(weight);posts.append(post)
            _,P=moment_match(weights,[m]*len(weights),posts)
        branch_weights.append(ew);branch_P.append(P);branch_miss.append(missed)
    _,P=moment_match(branch_weights,[m]*len(branch_weights),branch_P)
    missed=sum(w*x for w,x in zip(branch_weights,branch_miss))
    d['last_expected_missed_tracks']=float(sum(missed));d['missed_probability_by_track']={str(tr['id']):float(p) for tr,p in zip(d['tracks'],missed)}
    d['expected_missed_track_steps']=float(d.get('expected_missed_track_steps',0.)+sum(missed))
    d['candidate_action_steps']=int(d.get('candidate_action_steps',0))+1
    pdisc=discovery_prob((),view,config,capture_rows=arrivals);d['unseen_intensity']=(np.asarray(d['unseen_intensity'])*(1-pdisc)).tolist()
    E=sum((Fraction(resource(t,config)['energy_exact_J']) for t in schedule),Fraction());T=sum((Fraction(resource(t,config)['tx_exact_s']) for t in schedule),Fraction())
    d['energy_remaining_exact_J']=str(Fraction(d['energy_remaining_exact_J'])-E);d['tx_remaining_exact_s']=str(Fraction(d['tx_remaining_exact_s'])-T)
    d['communication_available_s']=link_free;d['processing_available_s']=processors;d['predicted_receipts']=receipt_audit
    d['event_forecast']={'emission_branches':len(branch_weights),'capture_count':len(arrivals),'discovery_probability':pdisc.tolist(),'semantics':EVENT_SPEC['schema']}
    pointing=list(d.get('node_pointing_rad',[0.]*4))
    for t in schedule:pointing[t.node]=t.scan[-1][0]
    d['node_pointing_rad']=pointing
    d['mean']=m.tolist();d['covariance']=P.tolist();return bind(d)


def primitive_priority(t,view,config,meter):
    """No objective call: geometric gradients, declared quality, search mass."""
    meter.priority_comparisons+=1;s=sensor_list(config)[t.node];m=np.asarray(view['mean']);P=np.asarray(view['covariance'])
    if t.kind=='search':
        p=discovery_prob((t,),view,config);score=config['missed_target_scale_m']**2*(p@np.asarray(view['unseen_intensity']))
        # Passive single bearing is useful for existing tracks, not free birth range.
        for tr in view['tracks']:
            j=tr['start']
            if covers(m[j:j+2],t,view,s):score+=.15*np.trace(P[j:j+2,j:j+2])
    elif t.kind=='track':
        tr=next(x for x in view['tracks'] if x['id']==t.item);j=tr['start'];pos=m[j:j+2]
        pd=detection_probability(pos,s,t,view['epoch'],config)
        score=pd*np.trace(P[j:j+2,j:j+2])*(.9 if s.mode=='active' else .35)
    else:
        # Future leverage only; not an independently verified risk bound.
        score=.35*P[s.range_bias,s.range_bias]*max(1,len(view['tracks']))
    # Scarce RF energy is a feasibility resource. This is an ordering heuristic,
    # not an energy term added to the verification objective.
    return float(score)/(1.+cost(t)['energy_J']/max(.05,float(Fraction(view['energy_remaining_exact_J']))))

class Generator:
    def __init__(self,view,config,meter,allow_calibration=True):
        self.view=view;self.config=config;self.meter=meter;self.pool=[];self.seen=set();self.feedback=[];self.projections=[]
        self.atoms=atoms(view,config,allow_calibration)
        self.rank={a.semantic_id():primitive_priority(a,view,config,meter) for a in self.atoms}
    def value(self,a):return sum(self.rank.get(t.semantic_id(),0.) for t in a)
    def add(self,a,origin='constructive'):
        self.meter.construct();fixed,receipt=project(tuple(a),self.view,self.config)
        self.projections.append({**receipt,'origin':origin})
        if fixed is None:self.meter.feasibility_rejections+=1;return None
        if fixed in self.seen:return None
        self.seen.add(fixed);self.pool.append(fixed);return fixed
    def initial(self):
        self.add(());ordered=sorted(self.atoms,key=lambda a:(-self.value((a,)),a))
        # Preserve complementary geometry and calibration-plus-track BEFORE fill.
        complementary=[]
        for a,b in combinations(ordered,2):
            self.meter.structural_pair_inspections+=1
            if a.node==b.node:continue
            if (a.kind=='calibrate' and b.kind=='track') or (b.kind=='calibrate' and a.kind=='track') or (a.kind==b.kind and a.item==b.item):
                complementary.append((a,b))
        for a in sorted(complementary,key=lambda a:-self.value(a))[:min(6,self.config['candidate_pool_cap']//3)]:self.add(a)
        for t in ordered:
            if len(self.pool)>=self.config['candidate_pool_cap']:break
            self.add((t,))
        nonempty=[a for a in self.pool if a]
        return max(nonempty,key=self.value) if nonempty else ()
    def challenge(self,incumbent,feedback):
        from .feedback import validate_feedback,targeted_proposals
        validate_feedback(feedback,self.view)
        if self.meter.regenerations:raise ContractError('ONE_FEEDBACK_REGENERATION_PER_CONTEXT')
        if not all(feedback['remaining'][k]>0 for k in ('scores','construction','iterations')):return []
        self.meter.regenerations+=1;added=[];branches=[]
        def charged_materialize(*args,**kwargs):
            self.meter.construct();self.meter.numeric_materializations+=1
            return materialize(*args,**kwargs)
        proposals=iter(targeted_proposals(feedback,incumbent,self.view,self.config,self.atoms,charged_materialize))
        while True:
            try:proposal,branch,record=next(proposals)
            except (StopIteration,BudgetExhausted):break
            self.meter.numeric_optimization_calls+=1
            try:a=self.add(proposal,'reasoned_feedback:'+branch)
            except BudgetExhausted:break
            branches.append({'branch':branch,'source_action_id':record['action_id'],'source_reason':record['reason'],
                             'proposed':[t.to_dict() for t in proposal],'accepted_projection':a is not None})
            if a is not None:added.append(a)
        self.feedback.append({'feedback':feedback,'incumbent':[a.to_dict() for a in incumbent],'branches':branches})
        return added

class ProxyReference:
    def __init__(self,view,config,meter,horizon,allow_calibration=True):
        validate_view(view);self.view=deepcopy(view);self.config=config;self.m=meter;self.horizon=horizon;self.cache={}
        self.allow_calibration=allow_calibration
    def evaluate(self,schedule):
        self.m.charge();start=time.perf_counter()
        try:
            if self.m.risk_kernel_calls is not None:self.m.risk_kernel_calls+=1
            if schedule in self.cache:
                self.m.cache_hits+=1;row=deepcopy(self.cache[schedule])
            else:
                d=forecast_once(self.view,schedule,self.config,self.m)
                total=criterion(d,self.config)['selection_proxy_m2'];trajectory=[total];component_steps=[criterion(d,self.config)];root_diagnostics={k:d.get(k) for k in ('missed_probability_by_track','last_expected_missed_tracks','predicted_receipts','event_forecast')}
                for depth in range(1,self.horizon):
                    d=advance_time(d,self.config)
                    tail=atoms(d,self.config,allow_calibration=self.allow_calibration)
                    tail=sorted(tail,key=lambda t:(-primitive_priority(t,d,self.config,self.m),t))[:self.config['tail_width']]
                    options=[()]+[(a,) for a in tail]
                    if self.config.get('future_joint_actions',False):
                        for x,y in combinations(tail,2):
                            if x.node!=y.node:
                                fixed,_=project((x,y),d,self.config)
                                if fixed is not None:options.append(fixed)
                    trials=[forecast_once(d,a,self.config,self.m) for a in options]
                    vals=[criterion(x,self.config)['selection_proxy_m2'] for x in trials]
                    k=min(range(len(vals)),key=lambda i:vals[i]);d=trials[k];total+=vals[k];trajectory.append(vals[k]);component_steps.append(criterion(d,self.config))
                row={'proxy_m2':total/self.horizon,'terminal_proxy_m2':trajectory[-1],
                    'forecast_trajectory_m2':trajectory,'proxy_sum_m2':total,'proxy_reduction':'arithmetic_mean_of_per_action_stage_costs',
                    'components':{k:sum(x[k] for x in component_steps)/self.horizon for k in component_steps[0]},'component_steps':component_steps,
                    'terminal_components':criterion(d,self.config),'root_diagnostics':root_diagnostics,
                    'expected_missed_track_steps':d.get('expected_missed_track_steps',0.),'candidate_action_steps':d.get('candidate_action_steps',0),
                    'assurance':'NUMERICAL_MEAN_BELIEF_ROLLOUT','formal_certificate':False}
                self.cache[schedule]=deepcopy(row)
            if not math.isfinite(row['proxy_m2']):raise ContractError('NONFINITE_SCORE')
            self.m.score_completed+=1
            return {**row,'schedule':[t.to_dict() for t in schedule],'context_digest':self.view['digest']}
        except Exception:
            self.m.score_failures+=1;self.m.risk_kernel_calls=None;raise
        finally:self.m.reference_seconds+=time.perf_counter()-start

def _run_policy(view,config,method,seed,learned=None):
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
    numeric=method in ('numeric_optimizer','numeric_network','numeric_random_priority')
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
        from .functional_fixture import proposal
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
        if numeric and not feedback_done and meter.score_calls>=max(1,B//2) and meter.construction<meter.construction_cap:
            from .feedback import create_feedback
            best=tuple(Task(**t) for t in min(rows,key=lambda r:r['proxy_m2'])['schedule']) if rows else ()
            _feedback_start=time.perf_counter()
            feedback_object=create_feedback(view,config,gen.projections,rows,failures,meter)
            refined=[]
            if network:
                t0=time.perf_counter();learned.refine(feedback_object);meter.network_forwards+=1
                meter.network_seconds=(meter.network_seconds or 0.)+time.perf_counter()-t0
                for proposal in learned.proposals(config['max_simultaneous'],meter=meter):
                    try:a=gen.add(proposal,'network_feedback_generated')
                    except BudgetExhausted:break
                    if a is not None:refined.append(a);extra.append(a)
            # Construction charges include every projection, also rejections.
            queue=gen.challenge(best,feedback_object)+refined+queue;feedback_done=True
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


def run_policy(view,config,method,seed,learned=None):
    with decision_linear_work() as work:
        selected,commit=_run_policy(view,config,method,seed,learned)
        legal(selected,view,config)
    commit['meter']['numpy_linear_algebra_calls']=work
    commit['meter']['joint_decoder_calls']=getattr(learned,'decode_calls',0) if learned is not None else 0
    commit['meter']['joint_decoder_seconds']=getattr(learned,'decoder_seconds',None) if learned is not None else None
    commit['meter']['timing_semantics']='construction counts projection and numeric materialization attempts; structural pair/decoder/network work counted separately. proposal_seconds includes network and optimization; total seconds is the comparison endpoint'
    commit['digest']=digest({k:v for k,v in commit.items() if k!='digest'})
    return selected,commit
