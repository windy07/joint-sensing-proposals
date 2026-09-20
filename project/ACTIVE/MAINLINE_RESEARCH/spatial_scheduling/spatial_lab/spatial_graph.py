"""Joint spatial information tokens on the retained graph/set backbones.

Every backbone receives the same numeric marginal and joint-block information.
The posterior and paid reference remain external. This revision still ranks
the same device-bounded numerical domain; proposals are executed by the mainline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from dataclasses import asdict, fields
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import numpy as np
from .binding import (ROOT, FEATURE_BINDING, FEATURE_SPEC, bootstrap_vendor,
                      graph_input_binding, verify_checkpoint, import_origins)
bootstrap_vendor()
import torch
from iclr2027_graph_agent.schema import (GraphNode, GraphEdge, GraphHyperedge,
    DynamicInformationGraph, NodeKind, EdgeKind)
from iclr2027_graph_agent.tensorize import tensorize_graph
from iclr2027_graph_agent.model import GraphAgentConfig
from iclr2027_graph_agent.baselines import make_graph_agent
from iclr2027_graph_agent.training import joint_schedule_logits
from .core import ContractError, Task, digest, read_json, cholesky
from .physics import NBIAS, CELLS,cell_list, SENSORS, sensor_list, cost
from .scheduling import validate_view, atoms
from .sequence_memory import SequenceMemory

from .feature_state import WIDTH,PUBLIC_START,FEEDBACK_START,public_values,feedback_values
FEATURES = tuple(f'f{i:03d}' for i in range(WIDTH))
BINDING = FEATURE_BINDING
ROLES = {'<unk>': 0, 'action': 1, 'sensor': 2, 'track': 3, 'bias': 4,
         'cell': 5, 'resource': 6, 'joint_covariance': 7,'geometry':8}
EDGES = {e.value: i for i, e in enumerate(EdgeKind)}
HYPERS = {'sensor_root': 0, 'joint_bias': 1}
ARMS = {'integrated_agent': 'proposed', 'original_agent': 'proposed',
        'deep_sets': 'deep_sets', 'set_transformer': 'set_transformer'}


def variable_blocks(view):
    """(semantic ID, state indices, physical scales, causal identity code)."""
    blocks = [(f'bias/{i}', [i], np.array([1000. if i in (0,1,6,7) else (50. if i>=8 else .05)]), -(i+1))
              for i in range(NBIAS)]
    used = list(range(NBIAS))
    ids = set()
    for t in view['tracks']:
        ident, start = t['id'], t['start']
        if type(ident) is not int or not 0 <= ident < 2**20 or ident in ids:
            raise ContractError('TRACK_SEMANTIC_ID')
        if type(start) is not int:
            raise ContractError('TRACK_STATE_LAYOUT')
        ids.add(ident)
        ix = list(range(start, start+4))
        used.extend(ix)
        blocks.append((f'track/{ident}', ix, np.array([1000.,1000.,50.,50.]), ident+1))
    if sorted(used) != list(range(len(view['mean']))):
        raise ContractError('JOINT_STATE_BLOCK_PARTITION')
    return sorted(blocks, key=lambda row: row[0])


def graph(view, actions, config, device='cpu',feedback=None):
    validate_view(view)
    actions = tuple(actions)
    if len(set(actions)) != len(actions):
        raise ContractError('DUPLICATE_GRAPH_ACTION')
    m, P = np.asarray(view['mean']), np.asarray(view['covariance'])
    if not np.allclose(P, P.T, rtol=1e-9, atol=1e-9):
        raise ContractError('ASYMMETRIC_JOINT_GRAPH_COVARIANCE')
    cholesky(P)
    nodes, edges, hypers = [], [], []
    now = view['decision_s']

    def add(ident, kind, role, values, endpoints=(0, 0),node=None,task=None):
        v = np.zeros(WIDTH, np.float32)
        a = np.asarray(values, float).ravel()
        if len(a) > 88 or not np.isfinite(a).all():
            raise ContractError('SPATIAL_GRAPH_FEATURES')
        v[:len(a)] = a
        v[88:90] = np.asarray(endpoints)/1024.
        if role in ('sensor','resource','action'):
            v[90:93]=[float(view['epoch']+k not in config.get('calibration_reference_off_epochs',[])) for k in (1,2,3)]
            v[93:96]=[config.get('birth_position_variance_limit_m2',1e6)/1e6,config['forecast_horizon']/3,config['association_gate_chi2']/16.27]
            v[PUBLIC_START:FEEDBACK_START]=public_values(view,config,node,task)
        v[FEEDBACK_START:]=feedback_values(feedback,view,ident if role=='action' else None)
        if not np.isfinite(v).all():
            raise ContractError('SPATIAL_GRAPH_FLOAT32_OVERFLOW')
        nodes.append(GraphNode(ident, kind, role, now,
            features=dict(zip(FEATURES, map(float, v))),
            units={f: '1' for f in FEATURES}))

    def edge(a, b, kind, dependence=0.):
        edges.append(GraphEdge(a, b, kind, timestamp_s=now,
            features={'age_s': 0., 'delay_s': 0., 'dependence': float(dependence)}))

    for i in range(NBIAS):
        scale = 1000. if i in (0,1,6,7) else (50. if i>=8 else .05)
        add(f'bias/{i}', NodeKind.NUISANCE_VARIABLE, 'bias',
            [m[i]/scale, P[i,i]/scale**2], (-(i+1),0))
    sensors = sensor_list(config)
    for s in sensors:
        add(f'sensor/{s.id}', NodeKind.SENSOR, 'sensor',
            [*list(np.array(s.position)/1000), float(s.mode=='active'),
             s.bandwidth_Hz/1e7, s.carrier_Hz/3e9,
             float(view['known_node_states'][s.id]),view.get('node_pointing_rad',[0.]*4)[s.id]/math.pi], (-(201+s.id),0),node=s.id)
    for t in view['tracks']:
        j = t['start']
        scale = np.array([1000.,1000.,50.,50.])
        cov = P[j:j+4,j:j+4]/np.outer(scale,scale)
        add(f'track/{t["id"]}', NodeKind.TRACK_VARIABLE, 'track',
            [*(m[j:j+4]/scale), *cov.ravel(), t['confirmed'],
             (view['epoch']-t['last_hit'])/10], (t['id']+1,0))
    for i, pos in enumerate(cell_list(config)):
        add(f'cell/{i}', NodeKind.TRACK_VARIABLE, 'cell',
            [*list(pos/1000), view['unseen_intensity'][i],
             view.get('search_age_cycles',[0]*len(cell_list(config)))[i]/10], (-(101+i),0))
    add('resource', NodeKind.RESOURCE, 'resource',
        [float(Fraction(view['energy_remaining_exact_J']))/10,
         float(Fraction(view['tx_remaining_exact_s']))/.1,
         config.get('forecast_horizon',config.get('lookahead_cycles',1))/4, view['epoch']/40,
         config['cycles']/40, float(config.get('pace_resources',False)),
         float(config.get('exposure_energy_density_limit_J_m2_per_cycle',0.))/1e-9,
         float(config.get('future_resource_reserve_fraction',.5))])

    # Public opaque geometry is not hidden truth. Variable-length obstacles
    # receive their own tokens rather than silently truncating a dense vector.
    for i,disc in enumerate(config.get('opaque_discs',[])):
        add(f'opaque/{i}',NodeKind.RESOURCE,'geometry',
            [*list(np.asarray(disc['centre_m'])/1000),disc['radius_m']/1000],(-(401+i),0))
    from .physics import BEACON,ILLUMINATOR,MONITOR
    for i,(name,pos) in enumerate([('beacon',BEACON),('illuminator',ILLUMINATOR),('monitor',MONITOR)]):
        add(f'surveyed/{name}',NodeKind.RESOURCE,'geometry',list(pos/1000),(-(301+i),0))

    # Complete joint block tokens, including zeros. Pairwise marginal changes
    # cannot disappear merely because typed edges carry only a scalar label.
    # Endpoint identities/descriptors are numeric too, so edge-ignoring set
    # models are not denied the covariance information given to the graph model.
    blocks = variable_blocks(view)
    for bi, (left, ix, sx, code_x) in enumerate(blocks):
        for right, iy, sy, code_y in blocks[bi:]:
            nx, ny = len(ix), len(iy)
            raw = P[np.ix_(ix, iy)]
            cov = np.zeros((4,4)); corr = np.zeros((4,4))
            cov[:nx,:ny] = raw/np.outer(sx,sy)
            denom = np.sqrt(np.outer(P[ix,ix],P[iy,iy]))
            corr[:nx,:ny] = raw/denom
            lx, ry, vx, vy = [np.zeros(4) for _ in range(4)]
            lx[:nx], ry[:ny] = m[ix]/sx, m[iy]/sy
            vx[:nx], vy[:ny] = P[ix,ix]/sx**2, P[iy,iy]/sy**2
            ident = f'joint/{left}|{right}'
            add(ident, NodeKind.FACTOR, 'joint_covariance',
                [nx,ny,*cov.ravel(),*corr.ravel(),*lx,*ry,*vx,*vy], (code_x,code_y))
            for endpoint in sorted({left,right}):
                edge(endpoint, ident, EdgeKind.INCIDENCE)
                edge(ident, endpoint, EdgeKind.CONDITIONS)
            hypers.append(GraphHyperedge(ident, 'joint_bias',
                tuple(dict.fromkeys([left,right,ident])), now))

    for a in actions:
        s = sensors[a.node]
        onehot = [float(a.kind==k) for k in ('search','track','calibrate')]
        values = [*onehot,*list(np.array(s.position)/1000),float(s.mode=='active'),
                  cost(a)['energy_J'],cost(a)['tx_s']/.01,
                  float(Fraction(view['energy_remaining_exact_J']))/10,view['epoch']/40]
        if a.kind == 'track':
            t = next(t for t in view['tracks'] if t['id']==a.item)
            j = t['start']
            values += [*list((m[j:j+2]-s.position)/1000),
                       float(np.trace(P[j:j+2,j:j+2]))/1e6]
            target, target_code = f'track/{a.item}', a.item+1
        elif a.kind == 'search':
            values += [*list((cell_list(config)[a.item]-s.position)/1000),view['unseen_intensity'][a.item]]
            target, target_code = f'cell/{a.item}', -(101+a.item)
        else:
            values += [0.,0.,P[s.range_bias,s.range_bias]/1e6]
            target, target_code = f'bias/{s.range_bias}', -(s.range_bias+1)
        from .actions import device as device_spec
        dv=device_spec(config,a.node)
        values += [float(a.role==r) for r in ('monostatic','bistatic','direct')]
        values += [a.power_W/100.,a.dwell_s/.2,a.bearing_rad/math.pi,a.start_s/config['dt_s'],a.bandwidth_Hz/1e7,a.reference_node/4.]
        values += [dv['power_min_W']/100.,dv['power_max_W']/100.,dv['dwell_min_s']/.2,dv['dwell_max_s']/.2,dv['point_offset_limit_rad'],dv['scan_span_max_rad'],dv['turn_rate_rad_s']/100.]
        ident = a.semantic_id()
        add(ident, NodeKind.ACTION, 'action', values, (-(201+a.node),target_code),node=a.node,task=a)
        edge(f'sensor/{a.node}',ident,EdgeKind.OWNS_ACTION)
        edge(ident,'resource',EdgeKind.CONSUMES_RESOURCE)
        edge(ident,target,EdgeKind.OBSERVES)
        edge(target,ident,EdgeKind.CONDITIONS)
        if a.kind == 'calibrate':
            edge(ident,f'bias/{s.angle_bias}',EdgeKind.OBSERVES)
            edge(f'bias/{s.angle_bias}',ident,EdgeKind.CONDITIONS)
    for s in sensors:
        ids = [a.semantic_id() for a in actions if a.node==s.id]
        if ids:
            hypers.append(GraphHyperedge(f'sensor-group/{s.id}','sensor_root',
                                          (f'sensor/{s.id}',*ids),now))
    raw = DynamicInformationGraph('spatial/'+view['digest'],'spatial',view['epoch'],now,
        tuple(nodes),tuple(edges),tuple(hypers),system_posterior_ref=view['digest'],
        metadata={'feature_binding':BINDING,
                  'graph_input_binding':graph_input_binding(view,actions,config)})
    G = tensorize_graph(raw,feature_names=FEATURES,role_vocabulary=ROLES,
        edge_vocabulary=EDGES,hyperedge_vocabulary=HYPERS,device=device)
    # tensorize_graph sorts node IDs; callers supply actions in causal-list
    # order. Bind outputs explicitly, never infer correspondence from sorting.
    index = {name:i for i,name in enumerate(G.node_ids)}
    G.action_node_index = torch.tensor([index[a.semantic_id()] for a in actions],
                                      dtype=torch.long,device=device)
    return G


def graph_document(G):
    return {f.name:(getattr(G,f.name).detach().cpu().tolist()
                    if torch.is_tensor(getattr(G,f.name)) else list(getattr(G,f.name)))
            for f in fields(G)}


def graph_digest(G):
    return digest(graph_document(G))


def recover_joint_covariance(G, view):
    """Testable information-preservation reference, not a second estimator."""
    blocks = variable_blocks(view)
    ix_node = {node:i for i,node in enumerate(G.node_ids)}
    P = np.zeros_like(np.asarray(view['covariance']),dtype=float)
    x = G.node_features.detach().cpu().numpy()
    for bi,(left,ix,sx,_) in enumerate(blocks):
        for right,iy,sy,_ in blocks[bi:]:
            row = x[ix_node[f'joint/{left}|{right}']]
            block = row[2:18].reshape(4,4)[:len(ix),:len(iy)]*np.outer(sx,sy)
            P[np.ix_(ix,iy)] = block
            P[np.ix_(iy,ix)] = block.T
    return P


def make(arm='integrated_agent',device='cpu',seed=541,config=None):
    from .stage6_model import make as create
    return create(arm,device,seed,config)


def scores(model,G,actions,pool,memory=None,*,view=None,config=None):
    from .stage6_model import numeric_features
    if view is None or config is None:raise ContractError('NUMERIC_SCORE_REQUIRES_CAUSAL_CONTEXT')
    out=model(G,memory);ids={a.semantic_id():i for i,a in enumerate(actions)}
    k=max(1,max(map(len,pool),default=0));n=len(pool);device=out.controls.device
    ix=torch.zeros(1,n,k,dtype=torch.long,device=device);features=torch.zeros(1,n,k,16,device=device);mask=torch.zeros(1,n,k,device=device)
    for j,schedule in enumerate(pool):
        count=len(schedule)
        if count:
            ix[0,j,:count]=torch.tensor([ids[a.semantic_id()] for a in schedule],device=device)
            features[0,j,:count]=torch.tensor(numeric_features(schedule,view,config),device=device);mask[0,j,:count]=1
    return -model.group_values(out,ix,features,mask)[0,:,0],out


class Priority:
    def __init__(self,path,device='cpu',allow_untrained=False,evaluation_split='confirmation'):
        ck = torch.load(path,map_location=device,weights_only=True)
        if evaluation_split not in ('development','confirmation','interface_check'):raise ContractError('EVALUATION_SPLIT')
        self.evaluation_split=evaluation_split
        verify_checkpoint(ck)
        if ck.get('training_purpose')=='development_diagnostic' and evaluation_split!='interface_check':
            raise ContractError('DIAGNOSTIC_CHECKPOINT_NOT_FOR_SCIENTIFIC_EVALUATION')
        if not ck.get('trained',False) and not allow_untrained:raise ContractError('UNTRAINED_CHECKPOINT_REQUIRES_EXPLICIT_OPT_IN')
        self.training_law_seeds = set(ck['training_law_seeds'])
        self.development_law_seeds = set(ck.get('development_law_seeds',[]))
        self.allowed_configurations = ck['allowed_policy_configurations']
        self.model,self.cfg = make(ck['arm'],device,config=ck['cfg'])
        self.model.load_state_dict(ck['state'],strict=True)
        self.model.eval()
        self.memory = SequenceMemory(self.cfg.memory_dim)
        self.device = device
        self.checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.checkpoint_arm = ck['arm']
        self.last_context = None
        self.out = None
        self.actual_imports = import_origins()
        self.untrained_initialization = not ck.get('trained',False)
        self.training_configurations = self.allowed_configurations

    @classmethod
    def untrained(cls,arm='integrated_agent',seed=32003,device='cpu',config=None):
        """Explicit interface check only; never recorded as trained weights."""
        self=cls.__new__(cls);self.model,self.cfg=make(arm,device,seed,config);self.model.eval()
        self.training_law_seeds=set();self.development_law_seeds=set();self.allowed_configurations={}
        self.device=device;self.memory=SequenceMemory(self.cfg.memory_dim);self.last_context=None;self.out=None
        self.checkpoint_sha256=None;self.checkpoint_arm=arm;self.untrained_initialization=True
        self.initialization_seed=seed;self.evaluation_split='interface_check';self.actual_imports=import_origins();return self

    def proposals(self,width=2,meter=None):
        from .actions import materialize
        from .core import BudgetExhausted
        if self.out is None:raise ContractError('MODEL_NOT_PREPARED')
        controls=self.out.controls.detach().cpu().numpy()
        order=sorted(range(len(self.actions)),key=lambda i:-float(self.out.candidate_score[i]))[:8]
        actions={}
        for i in order:
            if meter is not None:
                try:meter.construct()
                except BudgetExhausted:break
                meter.numeric_materializations+=1
            actions[i]=materialize(self.actions[i],self.view,self.policy_config,controls[i])
        order=[i for i in order if i in actions]
        proposals=[(actions[i],) for i in order[:4]]
        pairs=[]
        if width>=2:
            for i,j in combinations(order,2):
                if meter is not None:meter.structural_pair_inspections+=1
                if actions[i].node!=actions[j].node:pairs.append((actions[i],actions[j]))
            proposals=self.rank_schedules(pairs)[:2]+proposals
        self.raw_numeric_proposals={'semantic_ids':[a.semantic_id() for a in self.actions],
            'normalized_controls':controls.tolist(),'materialized_indices':order,
            'initialization_only':self.untrained_initialization,
            'feedback_digest':self.feedback_object['digest'] if getattr(self,'feedback_object',None) else None}
        return proposals

    def begin_world(self,seed,config):
        if seed in self.training_law_seeds or (self.evaluation_split=='confirmation' and seed in self.development_law_seeds):
            raise ContractError('TRAIN_EVALUATION_WORLD_OVERLAP')
        self.memory.begin(f'spatial-world/{int(seed)}')
        self.last_context = None
        self.out = None
        self.raw_numeric_proposals=None
        self.decode_calls=0

    def prepare(self,view,actions,config):
        if not self.untrained_initialization and digest(config) not in self.allowed_configurations:
            raise ContractError('CHECKPOINT_CONFIGURATION_DOMAIN')
        actions = tuple(actions)
        if set(actions) != set(atoms(view,config)) or len(set(actions)) != len(actions):
            raise ContractError('INCOMPLETE_RUNTIME_ACTION_GRAPH')
        key = graph_input_binding(view,tuple(sorted(actions)),config)
        if key == self.last_context:
            return  # feedback may reread one frozen decision; never advance twice
        self.decode_calls=0;self.decoder_seconds=0.
        self.policy_config = config
        self.actions = actions
        self.view = view
        self.G = graph(view,actions,config,'cpu')
        mem = self.memory.before(view,self.G,self.device)
        self.context_memory=mem.detach().clone();self.refined=False;self.feedback_object=None
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        with torch.no_grad():
            self.out = self.model(self.G,mem)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        self.memory.after(view,self.G,self.out.next_memory)
        self.last_context = key

    def refine(self,feedback):
        """Same-cycle proposal iteration; temporal memory is not advanced."""
        from .feedback import validate_feedback
        if self.out is None or self.refined:raise ContractError('FEEDBACK_FORWARD_BOUND')
        validate_feedback(feedback,self.view)
        if not all(feedback['remaining'][k]>0 for k in ('scores','construction','iterations')):
            raise ContractError('FEEDBACK_REQUIRES_REMAINING_BUDGET')
        self.G=graph(self.view,self.actions,self.policy_config,'cpu',feedback=feedback)
        with torch.no_grad():self.out=self.model(self.G,self.context_memory)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        self.refined=True;self.feedback_object=feedback
        # No SequenceMemory.before/after here. The next epoch inherits exactly
        # the first causal encoding, not the number of within-epoch retries.

    def rank_schedules(self,schedules):
        import time
        from .stage6_model import numeric_features
        if not schedules:return []
        start=time.perf_counter();self.decode_calls=getattr(self,'decode_calls',0)+1
        ids={a.semantic_id():i for i,a in enumerate(self.actions)}
        count=len(schedules);width=max(1,max(map(len,schedules)))
        index=torch.zeros(1,count,width,dtype=torch.long,device=self.device)
        features=torch.zeros(1,count,width,16,device=self.device)
        mask=torch.zeros(1,count,width,device=self.device)
        for j,schedule in enumerate(schedules):
            n=len(schedule)
            if n:
                index[0,j,:n]=torch.tensor([ids[a.semantic_id()] for a in schedule],device=self.device)
                features[0,j,:n]=torch.tensor(numeric_features(schedule,self.view,self.policy_config),device=self.device)
                mask[0,j,:n]=1
        with torch.no_grad():
            values=self.model.group_values(self.out,index,features,mask)[0]
            # Both heads use log1p(cost_m2 / 10000). Future return is an
            # executed-group offline label, used here to allocate paid scores.
            costs=torch.expm1(values.clamp(max=12))*10000.
            priority=costs[:,0]+self.policy_config.get('learned_future_priority_weight',.25)*costs[:,1]
        priorities=priority.detach().cpu().tolist()
        self.decoder_seconds=getattr(self,'decoder_seconds',0.)+time.perf_counter()-start
        return [schedules[i] for i in sorted(range(count),key=lambda i:(priorities[i],i))]

    def priority(self,schedule):
        # Compatibility for callers requesting a single scalar.
        from .stage6_model import numeric_features
        ids={a.semantic_id():i for i,a in enumerate(self.actions)}
        n=len(schedule);width=max(1,n)
        index=torch.zeros(1,1,width,dtype=torch.long,device=self.device)
        features=torch.zeros(1,1,width,16,device=self.device);mask=torch.zeros(1,1,width,device=self.device)
        if n:
            index[0,0,:n]=torch.tensor([ids[a.semantic_id()] for a in schedule],device=self.device)
            features[0,0,:n]=torch.tensor(numeric_features(schedule,self.view,self.policy_config),device=self.device)
            mask[0,0,:n]=1
        with torch.no_grad():
            cost=torch.expm1(self.model.group_values(self.out,index,features,mask)[0,0].clamp(max=12))*10000
        return float(cost[0]+self.policy_config.get('learned_future_priority_weight',.25)*cost[1])


def train(records,output,updates=16,arm='integrated_agent',seed=541,device='cpu'):
    from .sequence_training import train as sequence_train
    return sequence_train(records,output,updates,arm,seed,device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--updates',type=int,default=400)
    ap.add_argument('--arm',choices=list(ARMS),default='integrated_agent')
    ap.add_argument('--device',default='cpu')
    args = ap.parse_args()
    selected = {}
    for path in sorted((Path(args.results)/'cases').glob('*.json')):
        d = read_json(path)
        if d.get('method')=='lookahead' and d.get('status')=='COMPLETE':
            if d['seed'] in selected:
                raise ContractError('DUPLICATE_TRAIN_WORLD')
            selected[d['seed']] = path
    if not selected:
        raise ContractError('NO_LOOKAHEAD_TEACHER_TRAJECTORIES')
    print(json.dumps(train(list(selected.values()),args.output,args.updates,args.arm,
                           device=args.device)))

if __name__ == '__main__':
    main()
