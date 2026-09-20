"""Bounded causal feedback from projection receipts and already paid scores.

One proposal iteration per frozen context. No extra reference evaluations are
hidden in this interface. Numeric values retain their units in the audit object.
"""
from __future__ import annotations
import math
from fractions import Fraction
from .core import Task,ContractError,digest
from .actions import resource,device,materialize,control_target,owners
from .physics import availability,illumination

FEEDBACK_SCHEMA='ain.spatial.paid_projection_feedback.v4'
REASONS=('RF_RESOURCE','TIME_RESOURCE','PROCESSING_RESOURCE','REFERENCE','GEOMETRY','RISK','SCORING')

def reason_of(text,task,view,config):
    text=str(text).upper()
    if 'ENERGY' in text:return 'RF_RESOURCE'
    if any(x in text for x in ('EMISSION_TIME','CYCLE_TIME','SPECTRUM','CONFLICT','ONE_TASK')):return 'TIME_RESOURCE'
    if 'RESOURCE_DOMAIN' in text:return 'PROCESSING_RESOURCE'
    if any(x in text for x in ('REFERENCE','ILLUMINATION','NODE_UNAVAILABLE')):return 'REFERENCE'
    if any(x in text for x in ('GEOMETRY','POINT','BEAM','SCAN')):return 'GEOMETRY'
    return 'SCORING'

def create_feedback(view,config,projections,paid,failures,meter):
    remaining={'scores':max(0,meter.B-meter.score_calls),
               'construction':max(0,meter.construction_cap-meter.construction),
               'iterations':1 if meter.regenerations==0 else 0}
    records=[]
    # Most recent actual projection failures get a bounded opportunity; the
    # source receipt digest and exact candidate identify what was rejected.
    for receipt in reversed(projections):
        if receipt.get('status')!='REJECTED':continue
        tasks=[Task(**t) for t in receipt.get('raw',[])]
        for t in tasks:
            r=resource(t,config);reason=reason_of(receipt.get('reason'),t,view,config)
            limits={'RF_RESOURCE':float(Fraction(view['energy_remaining_exact_J'])),
                    'TIME_RESOURCE':float(Fraction(view['tx_remaining_exact_s'])),
                    'PROCESSING_RESOURCE':float(config.get('processing_budget_ops_per_cycle',0))}
            requested={'RF_RESOURCE':sum(resource(x,config)['energy_J'] for x in tasks),
                       'TIME_RESOURCE':sum(resource(x,config)['tx_s'] for x in tasks),
                       'PROCESSING_RESOURCE':sum(resource(x,config)['processing_ops'] for x in tasks)}
            records.append({'action_id':t.execution_id(),'semantic_id':t.semantic_id(),'action':t.to_dict(),
                'reason':reason,'origin':'actual_projection','receipt_digest':digest(receipt),
                'detail':str(receipt.get('reason')),'value':requested.get(reason,1.),
                'limit':limits.get(reason,0.),'units':{'RF_RESOURCE':'J','TIME_RESOURCE':'s','PROCESSING_RESOURCE':'ops'}.get(reason,'1')})
            if len(records)>=4:break
        if len(records)>=4:break
    best=min(paid,key=lambda x:x['proxy_m2']) if paid else None
    if best is not None:
        for d in best['schedule']:
            t=Task(**d);diag=best.get('root_diagnostics',{});miss=diag.get('last_expected_missed_tracks',0.)
            # This is a *paid proxy insufficiency*, not a claim that the true
            # target was missed. For search the proxy still carries area mass.
            loc=best['components'].get('unlocalized_target_cost_m2',0.)
            reason='GEOMETRY' if t.kind in ('search','track') and (miss>0 or loc>0) else 'RISK'
            records.append({'action_id':t.execution_id(),'semantic_id':t.semantic_id(),'action':d,
                'reason':reason,'origin':'paid_score','receipt_digest':digest(best),
                'detail':'conditional detection/localization insufficiency' if reason=='GEOMETRY' else 'paid posterior risk',
                'value':float(miss if t.kind=='track' else loc if t.kind=='search' else best['proxy_m2']),
                'limit':0.,'units':'expected_tracks' if t.kind=='track' else 'm2'})
            if len(records)>=7:break
    for fail in failures[-1:]:
        records.append({'action_id':digest(fail.get('schedule',[])),'semantic_id':None,'action':None,
                        'reason':'SCORING','origin':'paid_score_failure','receipt_digest':digest(fail),
                        'detail':str(fail.get('error')),'value':1.,'limit':0.,'units':'1'})
    obj={'schema':FEEDBACK_SCHEMA,'context_digest':view['digest'],'iteration':1,
         'remaining':remaining,'records':records[:8],
         'paid_components':dict(best['components']) if best else {},
         'paid_proxy_m2':float(best['proxy_m2']) if best else None,
         'score_calls_already_paid':meter.score_calls,'score_failures_already_paid':meter.score_failures}
    obj['digest']=digest(obj);validate_feedback(obj,view);return obj

def validate_feedback(feedback,view):
    if feedback.get('schema')!=FEEDBACK_SCHEMA or feedback.get('context_digest')!=view['digest']:raise ContractError('FEEDBACK_CONTEXT_BINDING')
    if feedback.get('iteration')!=1 or len(feedback.get('records',[]))>8:raise ContractError('FEEDBACK_BOUND')
    if feedback.get('digest')!=digest({k:v for k,v in feedback.items() if k!='digest'}):raise ContractError('FEEDBACK_CONTENT_BINDING')
    if any(type(v) is not int or v<0 for v in feedback['remaining'].values()):raise ContractError('FEEDBACK_BUDGET')
    for r in feedback['records']:
        if r['reason'] not in REASONS or not all(math.isfinite(r[k]) for k in ('value','limit')):raise ContractError('FEEDBACK_VALUE')
        if r['action'] is not None:
            t=Task(**r['action'])
            if t.execution_id()!=r['action_id'] or t.semantic_id()!=r['semantic_id']:raise ContractError('FEEDBACK_ACTION_IDENTITY')
    return True

def targeted_proposals(feedback,seed,view,config,all_atoms,materializer=materialize):
    """Reason- and value-dependent analytic repair, without an optimizer.

    Yield at most six proposed groups. Final projection/reference remain the
    authority. A repaired but unscored group is never marked verified.
    """
    validate_feedback(feedback,view)
    if not all(feedback['remaining'][k]>0 for k in ('scores','construction','iterations')):return
    produced=0
    for r in feedback['records']:
        if r['action'] is None:continue
        t=Task(**r['action']);reason=r['reason'];group=tuple(seed)
        # The rejected source, not just the best group, matters to the repair.
        if not any(a.semantic_id()==t.semantic_id() for a in group):group=(t,)
        try:u,_=control_target(t,view,config)
        except ContractError:
            # A malformed imported numerical value cannot bypass materialize.
            prototype=next((a for a in all_atoms if a.semantic_id()==t.semantic_id()),None)
            if prototype is None:continue
            t=prototype;u,_=control_target(t,view,config)
        adjusted=None;branch=''
        if reason in ('RF_RESOURCE','TIME_RESOURCE','PROCESSING_RESOURCE'):
            ratio=min(.9,max(0.,r['limit']/max(r['value'],1e-30)))
            d=device(config,t.node)
            if reason=='RF_RESOURCE' and t.role=='monostatic':
                target_power=max(d['power_min_W'],t.power_W*ratio)
                u[0]=(target_power-d['power_min_W'])/(d['power_max_W']-d['power_min_W'])
                if target_power<=d['power_min_W']:u[1]=0.
                branch='reduce_peak_power_then_pulses'
            else:u[1]=max(0.,u[1]*ratio);u[7]=0.;branch='reduce_pulses_and_start_earlier'
            adjusted=materializer(t,view,config,u)
        elif reason=='REFERENCE':
            # Change role/reference *structure*, not merely a power multiplier.
            options=[a for a in all_atoms if a.kind==t.kind and a.item==t.item and a.role!=t.role]
            options=sorted(options,key=lambda a:(a.role!='monostatic',a.node))
            if not options:
                options=[a for a in all_atoms if a.kind==t.kind and a.item==t.item and a.reference_node!=t.reference_node]
            if options:adjusted=options[0];branch='replace_unavailable_reference_role'
        elif reason=='GEOMETRY':
            # The paid deficit determines the amount of pointing/span change.
            severity=min(1.,max(0.,r['value'])/(1.+abs(r['value'])))
            u[3]=min(1.,u[3]+.2+.3*severity) if t.kind=='search' else u[3]
            u[2]=min(1.,max(0.,u[2]+(.2 if u[2]<=.5 else -.2)*severity))
            adjusted=materializer(t,view,config,u);branch='change_scan_or_look_direction'
            other=next((a for a in all_atoms if a.kind==t.kind and a.item==t.item and a.node!=t.node
                        and not set(owners(a))&set(owners(t))),None)
            if other is not None and config['max_simultaneous']>=2:
                yield (adjusted,other),'add_complementary_geometry',r;produced+=1
                if produced>=6:return
        elif reason=='RISK':
            u[1]=min(1.,u[1]+.2);adjusted=materializer(t,view,config,u);branch='increase_information_dwell'
        if adjusted is not None:
            repaired=tuple(adjusted if a.semantic_id()==t.semantic_id() else a for a in group)
            yield repaired,branch,r;produced+=1
        if produced>=6:return
