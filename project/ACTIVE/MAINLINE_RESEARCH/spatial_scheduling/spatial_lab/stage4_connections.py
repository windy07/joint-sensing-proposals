"""Two consecutive fixed contexts per backbone, exactly one backward, no step.

Task gradients and numeric-output input gradients are separated in a single
backward graph: the latter forward uses DETACHED parameters via functional_call.
It cannot contribute to a model parameter gradient. This is a connectivity
check, not an optimizer update or a learned-policy effectiveness experiment.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import math,time
import numpy as np
import torch
from .core import Task,Meter,ContractError,read_json,write_json,digest
from .local_checks import config,fixture
from .actions import materialize,project
from .scheduling import bind,atoms,run_policy
from .feedback import create_feedback
from .spatial_graph import Priority,make,graph
from .sequence_training import training_context,objective,_checkpoint,architecture
from .binding import source_manifest,state_digest,policy_configuration,FEATURE_BINDING,LABEL_BINDING
from .feature_state import PUBLIC_INDEX,FEEDBACK_START
from .work_budget import charge_backward,log_event

def memory_digest(memory):
    d=memory.snapshot()
    d['entries']=[{**r,'state':r['state'].tolist()} for r in d['entries']]
    return digest(d)

def actual_feedback(view,c):
    active=materialize(Task(0,'search',0),view,c,[1.,1.,.5,0,.5,.5,.5,0.])
    external=materialize(Task(2,'search',0,role='bistatic'),view,c)
    _,rf=project((active,),view,c);_,ref=project((external,),view,c)
    if rf['status']!='REJECTED' or ref['status']!='REJECTED':raise ContractError('CONNECTION_FEEDBACK_FIXTURE_EXPECTED_ACTUAL_FAILURES')
    m=Meter(4,96)
    return create_feedback(view,c,[rf],[],[],m),create_feedback(view,c,[ref],[],[],m)

def check_connections(out):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=policy_configuration({**config(),'forecast_horizon':1,'root_score_budget':3,'return_capacity':2,
                            'illumination_off_epochs':[0,1]},'numeric_network')
    tracker,_=fixture(c);views=[]
    for epoch in range(2):
        if epoch:tracker.predict(epoch)
        views.append(bind({**tracker.view(Fraction('.03'),Fraction('.1'),[.4]*4),'search_age_cycles':[epoch+1]*4,
                           'node_pointing_rad':[0.]*4,'communication_available_s':0.,'processing_available_s':[0.]*4}))
    report={'schema':'ain.stage4.zero_update_connections.v1','source_manifest':source_manifest(),
            'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,'world_advances':0,'optimizer_updates':0,
            'distinct_consecutive_contexts_per_backbone':2,'backward_limit_per_backbone':1,
            'extra_forwards':'same-context queue/feedback perturbations and exact reload; no additional temporal contexts',
            'numeric_input_gradient':'single isolated numeric output; functional parameters detached; model task gradients exclude input probe',
            'arms':{}}
    for arm in ('integrated_agent','deep_sets','set_transformer'):
        p=Priority.untrained(arm,seed=64011);p.begin_world(64010,c);m=p.model;m.eval();before=state_digest(m.state_dict())
        forward_count=[0]
        handle=m.register_forward_hook(lambda *args:forward_count.__setitem__(0,forward_count[0]+1))
        refinement=[];original_refine=p.refine
        def traced_refine(fb):
            b=memory_digest(p.memory);epoch=p.memory.last_epoch;original_refine(fb);a=memory_digest(p.memory)
            if b!=a or epoch!=p.memory.last_epoch:raise ContractError('SAME_CYCLE_FEEDBACK_ADVANCED_TIME_MEMORY')
            refinement.append({'epoch':epoch,'before':b,'after':a,'unchanged':True,'feedback_digest':fb['digest']})
        p.refine=traced_refine
        rows=[];incoming=[];commits=[]
        for i,view in enumerate(views):
            _,commit=run_policy(view,c,'numeric_network',64010,p)
            if commit['meter']['network_forwards']!=2 or commit['meter']['regenerations']!=1:raise ContractError('REAL_FEEDBACK_DRIVER_NOT_CONNECTED')
            if commit['meter']['construction']>c['construction_cap'] or commit['meter']['score_calls']>3:raise ContractError('FEEDBACK_BUDGET_OVERSPEND')
            if len(p.memory.audit)!=i+1 or p.memory.last_epoch!=i:raise ContractError('TEMPORAL_MEMORY_CADENCE')
            incoming.append(p.context_memory.detach().clone());commits.append(commit)
            rows.append(training_context(view,commit,c,10000.-i*2000.))
        task_loss=0.;parts=[];expected_outputs=[]
        for row,mem in zip(rows,incoming):
            acts=tuple(Task(**a) for a in row['actions']);G=graph(row['view'],acts,c)
            o=m(G,mem);loss,terms=objective(m,o,G,row);expected_outputs.append(o.controls.detach().clone())
            GF=graph(row['view'],acts,c,feedback=row['feedback_object']);of=m(GF,mem);lf,tf=objective(m,of,GF,row)
            task_loss+=(loss+lf)/2;parts.append({'initial':terms,'feedback':tf})
        view=views[1];acts=tuple(Task(**a) for a in rows[1]['actions']);rf,ref=actual_feedback(view,c)
        late=bind({**view,'communication_available_s':2.,'processing_available_s':[2.]*4})
        late_rf,_=actual_feedback(late,c)
        # Same temporal context, intervention only on causal queue input.
        detached={name:x.detach() for name,x in m.named_parameters()}
        detached.update({name:x.detach() for name,x in m.named_buffers()})
        G0=graph(view,acts,c,feedback=rf);GL=graph(late,acts,c,feedback=late_rf);GR=graph(view,acts,c,feedback=ref)
        GL.node_features.requires_grad_(True)
        with torch.no_grad():
            o0=torch.func.functional_call(m,detached,(G0,incoming[1]))
            orf=torch.func.functional_call(m,detached,(GR,incoming[1]))
        probe=torch.func.functional_call(m,detached,(GL,incoming[1]))
        queue_diff=float((probe.controls.detach()-o0.controls).abs().max())
        feedback_diff=float((orf.controls-o0.controls).abs().max())
        if queue_diff<=0 or feedback_diff<=0:raise ContractError('NUMERIC_INPUT_PERTURBATION_NOT_REACHABLE')
        # A single numerical output has no structural/objective path. Therefore
        # GL.grad is a genuine local derivative of that control, not of loss.
        numeric_probe=probe.controls[0,0]
        total=task_loss+numeric_probe
        charge_backward(arm)  # charged before the only backward attempt
        total.backward()
        gradients={name:float(x.grad.norm()) if x.grad is not None else None for name,x in m.named_parameters()}
        numeric=sum(v or 0 for n,v in gradients.items() if n.startswith('numeric_head.'))
        memory={n:v for n,v in gradients.items() if n.endswith('memory.weight_hh') or n.endswith('memory_cell.weight_hh')}
        joint=sum(v or 0 for n,v in gradients.items() if 'joint_schedule_decoder' in n)
        qcols=[PUBLIC_INDEX[k] for k in ('processing_wait','reference_processing_wait','shared_link_wait')]
        queue_grad=float(GL.node_features.grad[:,qcols].abs().sum());feedback_grad=float(GL.node_features.grad[:,FEEDBACK_START:].abs().sum())
        # Preserve raw diagnostic scalars BEFORE any later assertion or reload.
        write_json(out/f'BACKWARD_RAW_{arm}.json',{'source_manifest':source_manifest(),'arm':arm,
            'weights_before':before,'weights_after':state_digest(m.state_dict()),'all_task_parameter_gradients':gradients,
            'numeric_queue_input_gradient_l1':queue_grad,'numeric_feedback_input_gradient_l1':feedback_grad,
            'queue_output_difference':queue_diff,'feedback_output_difference':feedback_diff,
            'optimizer_updates':0,'backward_attempts':1})
        if not numeric>0 or not any(v is not None and v>0 for v in memory.values()) or not joint>0:raise ContractError('TASK_GRADIENT_DISCONNECTED')
        if queue_grad<=0 or feedback_grad<=0:raise ContractError('NUMERIC_CONTROL_FEATURE_GRADIENT_DISCONNECTED')
        if state_digest(m.state_dict())!=before:raise ContractError('PARAMETERS_CHANGED_DURING_ZERO_UPDATE_CHECK')
        loaded,mc=make(arm,seed=64012);loaded.load_state_dict(m.state_dict());loaded.eval();diff=0.
        for row,mem,expected in zip(rows,incoming,expected_outputs):
            G=graph(row['view'],tuple(Task(**a) for a in row['actions']),c)
            with torch.enable_grad():actual=loaded(G,mem)
            diff=max(diff,float((actual.controls-expected).abs().max()))
        if diff!=0.:raise ContractError('RELOAD_NOT_EXACT')
        ck=_checkpoint(m,mc,arm,[],{digest(c):c},[],[],64011,0)
        checkpoint=out/f'UNTRAINED_CONNECTION_{arm}.pt';torch.save(ck,checkpoint)
        handle.remove()
        report['arms'][arm]={'weights_before':before,'weights_after':state_digest(m.state_dict()),'weights_unchanged':True,
            'optimizer_updates':0,'backward_attempts':1,'task_numeric_gradient_norm':numeric,'task_memory_gradients':memory,
            'task_joint_decoder_gradient_norm':joint,'all_task_parameter_gradients':gradients,
            'pure_numeric_queue_input_gradient_l1':queue_grad,'pure_numeric_feedback_input_gradient_l1':feedback_grad,
            'same_context_queue_controls_max_difference':queue_diff,'same_context_feedback_controls_max_difference':feedback_diff,
            'same_cycle_memory':refinement,'temporal_memory_audit':p.memory.audit,
            'model_forwards':forward_count[0],'additional_reload_forwards':2,'exact_reload_max_difference':diff,
            'architecture':architecture(m,mc),'fixed_objectives':parts,'commits':commits,
            'checkpoint':checkpoint.name,'checkpoint_trained_flag':False,'checkpoint_dataset_is_empty':True,
            'fixed_context_digests':[v['digest'] for v in views]}
        write_json(out/'CONNECTIONS.json',report)
        write_json(out/f'FIXED_TWO_CONTEXTS_{arm}.json',rows)
        log_event('backbone_connection_complete',arm=arm,source_digest=report['source_manifest']['digest'],
                  forwards=forward_count[0]+2,backwards=1,optimizer_updates=0,parameters_unchanged=True)
    report['status']='PASSED';write_json(out/'CONNECTIONS.json',report);return report
