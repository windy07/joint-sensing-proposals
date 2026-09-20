"""Finite-domain candidate omission audit on a saved causal decision.

No world is advanced. The restricted numeric grid plus recorded proposals is an
explicit finite reference set, not a continuous/global optimum certificate.
"""
from __future__ import annotations
import argparse,itertools,time
from pathlib import Path
from .core import *
from .actions import materialize,project
from .scheduling import atoms,ProxyReference
from .case_identity import case_identity,validate_cached

def audit(result,epoch=0,max_atoms=2,max_evaluations=48):
    if not 1<=max_atoms<=4 or not 1<=max_evaluations<=128:raise ContractError('AUDIT_BOUND')
    validate_cached(result,case_identity(result['seed'],result['method'],result['configuration'],result['case_identity']['checkpoint']))
    frame=result['trace'][epoch];view=frame['causal_view'];commit=frame['commit'];cfg=commit['policy_config']
    generated=set();labels={}
    for row in commit['proposals']:
        if row['status']=='ACCEPTED':
            a=tuple(Task(**t) for t in row['executed']);generated.add(a);labels.setdefault(a,set()).add(row['origin'])
    paid={tuple(Task(**t) for t in row['schedule']):row for row in commit['archive']}
    generated|=set(paid);grid=set();domain=atoms(view,cfg)[:max_atoms];rejections=[]
    for atom in domain:
        for power,dwell in itertools.product((.2,.8),repeat=2):
            a=materialize(atom,view,cfg,[power,dwell,.5,.4,.5,.5,.5,0.]);fixed,receipt=project((a,),view,cfg)
            if fixed is not None:grid.add(fixed)
            else:rejections.append(receipt)
    representatives=[materialize(a,view,cfg,[.5,.5,.5,.4,.5,.5,.5,0.]) for a in domain]
    for a,b in itertools.combinations(representatives,2):
        fixed,receipt=project((a,b),view,cfg)
        if fixed is not None:grid.add(fixed)
        else:rejections.append(receipt)
    union=sorted(generated,key=lambda a:digest([t.to_dict() for t in a]))+sorted(grid-generated,key=lambda a:digest([t.to_dict() for t in a]))
    if len(generated)>max_evaluations:raise ContractError('AUDIT_BUDGET_CANNOT_COVER_GENERATED_SET')
    meter=Meter(max_evaluations);reference=ProxyReference(view,cfg,meter,commit['forecast_horizon']);rows=[];errors=[]
    for a in union[:max_evaluations]:
        try:
            r=reference.evaluate(a);r['in_generated']=a in generated;r['in_paid']=a in paid;r['origins']=sorted(labels.get(a,set()));rows.append(r)
        except Exception as exc:errors.append({'schedule':[t.to_dict() for t in a],'error':repr(exc)})
    all_min=min((r['proxy_m2'] for r in rows),default=None);gen_min=min((r['proxy_m2'] for r in rows if r['in_generated']),default=None)
    paid_min=min((r['proxy_m2'] for r in rows if r['in_paid']),default=None)
    selected=tuple(Task(**t) for t in commit['selected']);selection=next((r['proxy_m2'] for r in rows if tuple(Task(**t) for t in r['schedule'])==selected),None)
    return {'schema':'ain.stage3.finite_candidate_audit.v1','case_digest':result['case_identity']['digest'],'epoch':epoch,'context_digest':view['digest'],
      'scope':'recorded feasible proposals UNION restricted two-level numeric grid; no continuous-domain optimality claim',
      'grid_roles':[a.semantic_id() for a in domain],'grid_controls':{'power':[.2,.8],'dwell':[.2,.8],'others':'canonical fixed values'},
      'candidate_count':len(union),'evaluated_count':len(rows),'not_evaluated_due_to_bound':max(0,len(union)-max_evaluations),
      'reference_min_m2':all_min,'generated_min_m2':gen_min,'paid_min_m2':paid_min,'selected_m2':selection,
      'generation_omission_m2':gen_min-all_min if all_min is not None and gen_min is not None else None,
      'paid_subset_omission_m2':paid_min-gen_min if gen_min is not None and paid_min is not None else None,
      'selection_error_m2':selection-paid_min if selection is not None and paid_min is not None else None,
      'continuous_reference_approximation_error':None,'continuous_reference_error_reason':'finite grid has no certified continuous-domain error bound',
      'all_recomputed_rows':rows,'evaluation_failures':errors,'feasibility_rejections':rejections,'meter':meter.report(),
      'optimizer_updates':0,'world_advances':0}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--epoch',type=int,default=0);p.add_argument('--max-atoms',type=int,default=2);p.add_argument('--max-evaluations',type=int,default=48)
    a=p.parse_args(argv);r=audit(read_json(a.case),a.epoch,a.max_atoms,a.max_evaluations);write_json(a.output,r)
    print({k:r[k] for k in ('evaluated_count','generation_omission_m2','paid_subset_omission_m2','selection_error_m2')});return 0
