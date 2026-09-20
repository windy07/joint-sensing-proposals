"""Numeric sequence supervision and explicit SERVER-ONLY optimization.

Importing, loading records, or computing objective NEVER performs optimizer
updates. Training requires an explicit call/CLI execute flag. Stage1 training
and weights remain in the immutable cumulative baseline.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib,math,json,random,time
import numpy as np
import torch
import torch.nn.functional as F
from .core import ContractError,Task,digest,read_json,write_json,atomic_save
from .binding import (source_manifest,source_digest,FEATURE_BINDING,LABEL_BINDING,LABEL_SPEC,
    dependency_contract,state_digest,import_origins,policy_configuration)
from .case_identity import case_identity,validate_cached
from .spatial_graph import graph,graph_document,graph_digest,make,ARMS
from .sequence_memory import SequenceMemory
from .actions import teacher_control_labels
from .scheduling import atoms
from iclr2027_graph_agent.training import joint_schedule_logits

DATA_SCHEMA='ain.spatial.numeric.sequence.v4'

def training_context(view,commit,config,realized_return_m2=None):
    actions=tuple(Task(**a) for a in commit['input_actions'])
    if actions!=tuple(atoms(view,config)):
        raise ContractError('DATA_INCOMPLETE_CAUSAL_ACTION_DOMAIN')
    ids={a.semantic_id():i for i,a in enumerate(actions)}
    paid=commit['archive']
    if not paid:raise ContractError('NO_PAID_LABELS')
    if any(r['context_digest']!=view['digest'] for r in paid):raise ContractError('LABEL_CONTEXT_MISMATCH')
    labels=[]
    for row in paid:
        tasks=[Task(**d) for d in row['schedule']]
        labels.append({'action_indices':[ids[a.semantic_id()] for a in tasks],
                       'schedule':[t.to_dict() for t in tasks],'proxy_m2':row['proxy_m2'],
                       'components':row['components']})
    best=next((x for x in labels if digest(x['schedule'])==digest(commit['selected'])),None)
    if best is None:raise ContractError('EXECUTED_GROUP_NOT_IN_PAID_LABELS')
    teacher=tuple(Task(**action) for action in best['schedule'])
    controls=teacher_control_labels(teacher,view,config)
    numeric=[{'index':ids[t.semantic_id()],**label} for t,label in zip(teacher,controls)]
    feedback=commit.get('feedback_object')
    G=graph(view,actions,config)
    return {'schema':DATA_SCHEMA,'view':view,'policy_configuration':config,
      'actions':[a.to_dict() for a in actions],'full_input_graph':graph_document(G),'graph_digest':graph_digest(G),
      'paid_labels':labels,'numeric_labels':numeric,'proposals':commit.get('proposals',[]),
      'executed':commit['selected'],'cost_accounting':commit['meter'],
      'realized_future_target_m2':realized_return_m2,'future_target_mask':realized_return_m2 is not None,
      'future_target_definition':'normalized discounted mean post-decision GOSPA squared, offline evaluation labels only',
      'memory_reset':view['epoch']==0,'feedback_object':feedback,
      'feedback_graph_digest':graph_digest(graph(view,actions,config,feedback=feedback)) if feedback else None,
      'proxy_reduction':'arithmetic_mean_of_per_action_stage_costs'}

def load_sequences(paths, gamma=.95):
    sequences=[];manifest=[];seen=set()
    for path in paths:
        p=Path(path);result=read_json(p)
        if result.get('fixture_only'):raise ContractError('CACHE_FIXTURE_IS_NOT_TRAINING_DATA')
        config=policy_configuration(result['configuration'],result['method'])
        expected=case_identity(result['seed'],result['method'],result['configuration'],result['case_identity']['checkpoint'])
        from .training_provenance import validate_training_case
        provenance=validate_training_case(result,expected)
        if result['seed'] in seen:raise ContractError('DUPLICATE_WORLD_ACROSS_DATA')
        seen.add(result['seed']);trace=result['trace'];returns=[];ret=0.;normalizer=0.
        for frame in reversed(trace):
            ret=frame['metric']['gospa_m']**2+gamma*ret;normalizer=1.+gamma*normalizer;returns.append(ret/normalizer)
        returns.reverse();sequence=[]
        for epoch,(frame,ret) in enumerate(zip(trace,returns)):
            if frame['epoch']!=epoch:raise ContractError('NONCONSECUTIVE_TRAINING_DATA')
            try:
                row=training_context(frame['causal_view'],frame['commit'],config,ret)
            except Exception as exc:
                exc.add_note(f'Training export file={p.name}, world={result["seed"]}, epoch={epoch}')
                raise
            row['arrived_evidence']=[{'root':p['root'],'acquire_s':p['acquire_s'],'arrival_s':p['arrival_s']} for p in frame['packets']]
            row['acceptance']=frame['update_results'];row['physical_results']=frame['metric'];sequence.append(row)
        sequences.append({'world_seed':result['seed'],'contexts':sequence})
        manifest.append({'world_seed':result['seed'],'case_identity':result['case_identity']['digest'],
                         'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'contexts':len(sequence),
                         **({'training_provenance':provenance} if provenance else {})})
    if not sequences:raise ContractError('EMPTY_TRAINING_DATA')
    return sequences,manifest


def train(*args,**kwargs):
    raise ContractError('USE_STAGE6_BATCHED_TRAINING_ENTRYPOINT')
