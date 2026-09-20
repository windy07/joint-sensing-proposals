"""Finite shared-event algebra used by proposal forecasts and admitted credit.

No random draws, truth states, or future received values are used here.
"""
from __future__ import annotations
from itertools import product
import math
import numpy as np
from .core import ContractError, digest
from .actions import visits
from .physics import sensor_list, covers, detection_components, reference_response

EVENT_SPEC={
 'schema':'ain.spatial.event_scopes.v4',
 'direct_emission':'one Bernoulli per world/epoch/target; shared by every direct beam and receiver; independent across epochs and targets',
 'bistatic_reference':'one Bernoulli and one common extraction error per physical capture; shared by all targets in that capture',
 'direct_reference':'target-specific two-receiver reception within each capture, conditional on the shared target emission',
 'delivery':'one Bernoulli per packet root; copies have exactly the same event and no additional credit',
 'thermal_detection':'independent target/capture reception conditional on the above events',
 'forced_faults':'realized private test interventions never read by prediction; observed failures receive no credit',
 'posterior_approximation':'conditional Gaussian/finite-association moment closure; emission branches retained until epoch transition',
}

def moment_match(weights,means,covariances):
    weights=np.asarray(weights,float)
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ContractError('EVENT_BRANCH_MASS')
    weights=weights/weights.sum();mean=sum(w*np.asarray(m) for w,m in zip(weights,means))
    P=sum(w*(np.asarray(C)+np.outer(np.asarray(m)-mean,np.asarray(m)-mean)) for w,m,C in zip(weights,means,covariances))
    return mean,(P+P.T)/2

def emission_branches(ids,q):
    if not 0<=q<=1:raise ContractError('EMISSION_PROBABILITY_DOMAIN')
    ids=tuple(ids)
    if len(ids)>6:raise ContractError('EMISSION_BRANCH_LIMIT_SIX_TARGETS')
    for bits in product(*[((0,) if q==0 else (1,) if q==1 else (0,1)) for _ in ids]):
        w=math.prod(q if b else 1-q for b in bits)
        if w>0:yield w,dict(zip(ids,bits))

def captures(schedule,config):
    """Unique physical visits. Repeating the same task is not another capture."""
    out=[];seen=set()
    for action in schedule:
        for v in visits(action,config):
            key=v['task'].execution_id()
            if key in seen:continue
            seen.add(key);out.append({**v,'capture_id':key,'parent_action_id':action.execution_id()})
    return out

def unique_captures(rows):
    out=[];seen={}
    for row in rows:
        key=row.get('capture_id',row['task'].execution_id())
        body=digest(row['task'].to_dict())
        if key in seen:
            if seen[key]!=body:raise ContractError('CAPTURE_CONTENT_CONFLICT')
            continue
        seen[key]=body;out.append(row)
    return out

def packet_gate(capture,epoch,config,pos=None,delivery_probability=None):
    task=capture['task']
    delivered=capture.get('delivered_condition',None)
    p=(1.-config.get('packet_loss_probability',0.) if delivery_probability is None else delivery_probability) if delivered is None else float(delivered)
    acquired=capture.get('reference_condition',None)
    if acquired is not None:return p*float(acquired)
    return p*(reference_response(task,epoch,config,pos)['pd'] if task.role=='bistatic' else 1.)

def any_detection_probability(rows,pos,view,config,*,localization_test=None,delivery_probability=None):
    """P(at least one usable observation) by conditioning on target emission."""
    miss0=miss1=1.;sensors=sensor_list(config)
    for row in unique_captures(rows):
        task=row['task'];sensor=sensors[task.node]
        if task.kind=='calibrate' or not covers(pos,task,view,sensor):continue
        f=detection_components(pos,sensor,task,view['epoch'],config)
        p=packet_gate(row,view['epoch'],config,pos,delivery_probability)*f['receiver']
        if localization_test is not None and p>0 and not localization_test(task,sensor,pos):p=0.
        if not 0<=p<=1:raise ContractError('CONDITIONAL_DETECTION_DOMAIN')
        miss1*=1-p
        if task.role!='direct':miss0*=1-p
    q=config.get('emission_probability',.7)
    return (1-q)*(1-miss0)+q*(1-miss1)
