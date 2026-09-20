"""Public decision-state feature registry, shared unchanged by all backbones.

The first 96 slots retain the complete joint-block representation. Additions
below carry dimensioned public hardware/model state; missing quantities have a
separate presence bit, never a fabricated successful future event.
"""
from __future__ import annotations
import math
import numpy as np

# (name, original unit, normalization scale). Linear transforms except named
# log10 fields (log10(1+x), finite). Zero for not applicable, with masks below.
PUBLIC_FIELDS=[
 ('processing_wait','s',1.),('reference_processing_wait','s',1.),('shared_link_wait','s',1.),
 ('processing_queue_present','1',1.),('reference_queue_present','1',1.),('link_queue_present','1',1.),
 ('processing_rate','ops/s',2e9),('communication_rate','bit/s',1e6),('operations_per_sample','ops',8.),
 ('packet_bytes','byte',512.),('receive_power','W',5.),('reference_processing_rate','ops/s',2e9),
 ('reference_communication_rate','bit/s',1e6),('reference_present','1',1.),('reference_available','1',1.),
 ('receive_budget','s',1.),('processing_budget','ops',1e8),('communication_budget','byte',4096.),
 ('total_rf_budget','J',10.),('total_tx_budget','s',.1),('simultaneous_limit','tasks',4.),
 ('emission_probability','1',1.),('delivery_probability','1',1.),('pfa_neg_log10','1',10.),
 ('emitter_eirp_log10','W',1.),('illuminator_eirp_log10','W',1.),('beacon_eirp_log10','W',1.),
 ('aperture_diameter','m',1.),('aperture_efficiency','1',1.),('noise_factor','1',4.),('loss','1',4.),
 ('temperature','K',290.),('reference_gain','1',10.),('gain','1',1000.),
 ('pulse_width','s',1e-6),('pri','s',1e-4),('power_quantum','W',.1),('point_quantum','rad',math.radians(.05)),
 ('settle_time','s',.001),('passive_delay_window','s',2e-4),('reference_pd','1',1.),
 ('reference_snr_log10','1',1.),('reference_delay_variance_present','1',1.),('reference_delay_sd','s',1e-8),
 ('reference_frequency_variance_present','1',1.),('reference_frequency_sd','Hz',100.),
 ('model_snr_log10','1',1.),('thermal_detection','1',1.),('emission_event_posterior','1',1.),
 ('single_action_first_arrival_wait','s',1.),('single_action_last_arrival_wait','s',1.),
 ('single_action_timely_fraction','1',1.),('single_action_receipts_present','1',1.),
 ('processing_operations','ops',1e7),('communication_bytes','byte',4096.),('listen_time','s',.2),
 ('occupied_end','s',1.),('requested_start','s',1.),('request_present','1',1.),
 ('pulse_count','pulses',1800.),('scan_count','beams',3.),
 ('beam0_angle','rad',math.pi),('beam1_angle','rad',math.pi),('beam2_angle','rad',math.pi),
 ('beam0_pulses','pulses',1800.),('beam1_pulses','pulses',1800.),('beam2_pulses','pulses',1800.),
 ('own_next1_available','1',1.),('own_next2_available','1',1.),('own_next3_available','1',1.),
 ('ref_next1_available','1',1.),('ref_next2_available','1',1.),('ref_next3_available','1',1.),
 ('illumination_now','1',1.),('illumination_next1','1',1.),('illumination_next2','1',1.),('illumination_next3','1',1.),
 ('calibration_reference_now','1',1.),('allow_calibration','1',1.),('dt','s',1.),
 ('acceleration_psd','m2/s3',1.),('birth_intensity','1/cycle',.01),('search_halfwidth','m',350.),
 ('missed_target_scale','m',300.),('coast_penalty','m2',1000.),('tentative_penalty','m2',100.),
 ('reference_x','m',1000.),('reference_y','m',1000.),('role_public_state_present','1',1.),
 ('geometry_localization_information_log10','1',10.),('geometry_estimate_present','1',1.),
 ('belief_position_trace','m2',1e6),('clock_horizon_remaining','s',40.),('clutter_mean','returns',1.),
 ('delay_deadline_slack','s',1.),('calibration_reflector_rcs','m2',1.),
]
RESERVED_PUBLIC_FIELDS=[
 {'index':90,'name':'calibration_reference_next1','unit':'1','scale':1.},
 {'index':91,'name':'calibration_reference_next2','unit':'1','scale':1.},
 {'index':92,'name':'calibration_reference_next3','unit':'1','scale':1.},
 {'index':93,'name':'birth_position_variance_limit','unit':'m2','scale':1e6},
 {'index':94,'name':'forecast_horizon','unit':'cycles','scale':3.},
 {'index':95,'name':'association_gate_chi2','unit':'1','scale':16.27},
]
PUBLIC_START=96
FEEDBACK_START=PUBLIC_START+len(PUBLIC_FIELDS)
FEEDBACK_FIELDS=['present',*[f'reason_{r}' for r in ('RF_RESOURCE','TIME_RESOURCE','PROCESSING_RESOURCE','REFERENCE','GEOMETRY','RISK','SCORING')],
 'affected_semantic_action','value_log10','limit_log10','relative_deficit','remaining_scores','remaining_construction',
 'paid_calls','paid_failures','paid_proxy_log10','position_cost_log10','unlocalized_cost_log10','coast_cost_log10',
 'tentative_cost_log10','projection_origin','paid_origin','iteration']
WIDTH=FEEDBACK_START+len(FEEDBACK_FIELDS)
PUBLIC_INDEX={name:PUBLIC_START+i for i,(name,_,_) in enumerate(PUBLIC_FIELDS)}
FEEDBACK_INDEX={name:FEEDBACK_START+i for i,name in enumerate(FEEDBACK_FIELDS)}
REGISTRY={'reserved_public_fields':RESERVED_PUBLIC_FIELDS,'public_start':PUBLIC_START,'public_fields':[{'index':PUBLIC_START+i,'name':n,'original_unit':u,'scale':s} for i,(n,u,s) in enumerate(PUBLIC_FIELDS)],
 'feedback_start':FEEDBACK_START,'feedback_fields':[{'index':FEEDBACK_START+i,'name':n} for i,n in enumerate(FEEDBACK_FIELDS)],
 'missing':'zero plus explicit *_present; queues absent mean idle at decision time, accompanied by presence=0',
 'public_probabilities':'configured probabilities and arrived-event posterior only, not random draws',
 'arrivals':'deterministic model prediction from public capacity and already accumulated queues, not future packets',
 'horizon':'public availability for current and next three epochs; preflight limits forecast horizon to 3',
 'forbidden':'target truth, true births, actual future delivery, forced private fault epochs, unarrived measurements',
 'same_information':'identical tensor and feature registry for graph, deep sets and set attention',
 'geometry_presence':'search/track only, positive receiver response and available modeled reference; absent is value=0 with presence=0; calibration constrains receiver nuisance, not a direct target-position observation',
 'log10':'log10(1+nonnegative value); geometry information uses log10(1+lambda_min(Fisher)*1e6 m2)/10; no clipping of late queues to an on-time state'}

def log10p(value):
    return math.log10(1.+max(0.,float(value))) if math.isfinite(float(value)) else 0.

def public_values(view,config,node=None,task=None):
    from .actions import device,resource
    from .physics import (availability,illumination,reference_response,sensor_list,CELLS,cell_list,BEACON,ILLUMINATOR,
                          detection_components,snr,jacobian,noise_cov,NBIAS,resolution_cell_count)
    from .scheduling import predicted_receipts
    from .core import ContractError
    epoch=view['epoch'];now=view['decision_s'];v={};n=0 if node is None else node;dv=device(config,n)
    processors=view.get('processing_available_s',[now]*4);comm=view.get('communication_available_s',now)
    if len(processors)!=4 or not np.isfinite([*processors,comm]).all():raise ContractError('PUBLIC_QUEUE_STATE')
    ref=task.reference_node if task is not None and task.role=='direct' else -1
    rd=device(config,ref) if ref>=0 else None
    v.update(processing_wait=max(0.,processors[n]-now) if node is not None else 0.,
      reference_processing_wait=max(0.,processors[ref]-now) if ref>=0 else 0.,shared_link_wait=max(0.,comm-now),
      processing_queue_present=float(node is not None and 'processing_available_s' in view),
      reference_queue_present=float(ref>=0 and 'processing_available_s' in view),link_queue_present=float('communication_available_s' in view),
      processing_rate=dv['processing_ops_s'],communication_rate=dv['communication_bps'],operations_per_sample=dv['processing_ops_per_sample'],
      packet_bytes=dv['packet_bytes'],receive_power=dv['rx_power_W'],reference_processing_rate=rd['processing_ops_s'] if rd else 0.,
      reference_communication_rate=rd['communication_bps'] if rd else 0.,reference_present=float(ref>=0),
      reference_available=float(availability(epoch,ref,config)) if ref>=0 else 0.,
      receive_budget=config.get('receive_budget_s_per_cycle',0.),processing_budget=config.get('processing_budget_ops_per_cycle',0.),
      communication_budget=config.get('communication_budget_bytes_per_cycle',0.),total_rf_budget=config['rf_energy_budget_J'],
      total_tx_budget=config['transmit_time_budget_s'],simultaneous_limit=config['max_simultaneous'],
      emission_probability=config.get('emission_probability',.7),delivery_probability=1-config.get('packet_loss_probability',0.),
      pfa_neg_log10=-math.log10(config['pfa_cell']),emitter_eirp_log10=log10p(config.get('emitter_eirp_W',.01)),
      illuminator_eirp_log10=log10p(config.get('illuminator_eirp_W',1e4)),beacon_eirp_log10=log10p(config.get('calibration_beacon_eirp_W',.01)),
      aperture_diameter=dv['aperture_diameter_m'],aperture_efficiency=dv['aperture_efficiency'],noise_factor=dv['noise_factor'],
      loss=dv['loss'],temperature=dv['temperature_K'],reference_gain=dv['direct_reference_gain'],gain=dv['gain'],
      pulse_width=dv['pulse_width_s'],pri=dv['pri_s'],power_quantum=dv['power_quantum_W'],point_quantum=dv['point_quantum_rad'],
      settle_time=dv['settle_s'],passive_delay_window=dv['passive_delay_window_s'],
      illumination_now=float(illumination(epoch,config)),calibration_reference_now=float(epoch not in config.get('calibration_reference_off_epochs',[])),
      allow_calibration=float(config.get('allow_calibration',True)),dt=config['dt_s'],acceleration_psd=config['acceleration_psd_m2_s3'],
      birth_intensity=config.get('birth_intensity_per_cell_per_cycle',.01),search_halfwidth=config.get('search_cell_halfwidth_m',350.),
      missed_target_scale=config['missed_target_scale_m'],coast_penalty=config.get('coast_penalty_m2',1000.),
      tentative_penalty=config.get('tentative_track_penalty_m2',100.),clock_horizon_remaining=(config['cycles']-epoch)*config['dt_s'],
      calibration_reflector_rcs=config.get('calibration_reflector_rcs_m2',1.))
    for k in (1,2,3):
        v[f'own_next{k}_available']=float(availability(epoch+k,n,config)) if node is not None else 0.
        v[f'ref_next{k}_available']=float(availability(epoch+k,ref,config)) if ref>=0 else 0.
        v[f'illumination_next{k}']=float(illumination(epoch+k,config))
    if task is not None:
        sensors=sensor_list(config);sensor=sensors[n];work=resource(task,config)
        track=next((t for t in view['tracks'] if t['id']==task.item),None) if task.kind=='track' else None
        pos=np.asarray(view['mean'][track['start']:track['start']+2]) if track else BEACON if task.kind=='calibrate' else cell_list(config)[task.item]
        response=reference_response(task,epoch,config,pos);f=detection_components(pos,sensor,task,epoch,config)
        _,_,_,receipts=predicted_receipts((task,),view,config)
        arrivals=[r['predicted_arrival_s']-now for r in receipts]
        v.update(reference_pd=response['pd'],reference_snr_log10=log10p(response['snr']),
          reference_delay_variance_present=float(response['delay_variance_s2'] is not None),
          reference_delay_sd=math.sqrt(response['delay_variance_s2'] or 0.),
          reference_frequency_variance_present=float(response['frequency_variance_Hz2'] is not None),
          reference_frequency_sd=math.sqrt(response['frequency_variance_Hz2'] or 0.),
          model_snr_log10=log10p(snr(pos,sensor,task,1.,epoch,config)),thermal_detection=f['receiver'],
          emission_event_posterior=view.get('emission_event_posterior',{}).get(str(task.item),config.get('emission_probability',.7)) if task.role=='direct' else 1.,
          single_action_first_arrival_wait=min(arrivals) if arrivals else 0.,single_action_last_arrival_wait=max(arrivals) if arrivals else 0.,
          single_action_timely_fraction=sum(r['accepted_this_cycle'] for r in receipts)/len(receipts) if receipts else 0.,single_action_receipts_present=float(bool(receipts)),
          processing_operations=work['processing_ops'],communication_bytes=work['communication_bytes'],listen_time=work['listen_s'],
          occupied_end=work['occupied_end_s'],requested_start=task.requested_start_s or 0.,request_present=float(task.requested_start_s is not None),
          pulse_count=round(task.dwell_s/task.pri_s),scan_count=len(task.scan),role_public_state_present=1.,
          clutter_mean=config['pfa_cell']*resolution_cell_count(task,config),delay_deadline_slack=config['dt_s']-(max(arrivals) if arrivals else config['dt_s']))
        for i,(angle,duration) in enumerate(task.scan[:3]):v[f'beam{i}_angle']=angle;v[f'beam{i}_pulses']=round(duration/task.pri_s)
        rp=sensors[ref].position if ref>=0 else BEACON if task.kind=='calibrate' else ILLUMINATOR if task.role=='bistatic' else sensor.position
        v['reference_x'],v['reference_y']=rp
        if track:
            j=track['start'];P=np.asarray(view['covariance']);v['belief_position_trace']=float(np.trace(P[j:j+2,j:j+2]))
        # A covariance is conditional on an obtainable measurement. A visible
        # echo does not supply a missing direct reference. Calibration measures
        # receiver nuisance at a surveyed beacon, not an unknown target position.
        if task.kind in ('search','track') and f['receiver']>0 and response['available']:
            H=jacobian(np.r_[pos,0.,0.],sensor,NBIAS+4,NBIAS,task)[:,NBIAS:NBIAS+2]
            R=noise_cov(pos,sensor,task,epoch,config);info=H.T@np.linalg.solve(R,H)
            v['geometry_localization_information_log10']=log10p(max(0.,float(np.linalg.eigvalsh(info).min()))*1e6);v['geometry_estimate_present']=1.
    return np.array([v.get(name,0.)/scale for name,_,scale in PUBLIC_FIELDS],dtype=float)

def feedback_values(feedback,view,semantic_id=None):
    from .feedback import validate_feedback,REASONS
    out=np.zeros(len(FEEDBACK_FIELDS));idx={n:i for i,n in enumerate(FEEDBACK_FIELDS)}
    if feedback is None:return out
    validate_feedback(feedback,view);out[idx['present']]=1.;records=feedback['records']
    local=[r for r in records if r['semantic_id']==semantic_id] if semantic_id else records
    out[idx['affected_semantic_action']]=float(bool(local) and semantic_id is not None)
    for reason in REASONS:out[idx['reason_'+reason]]=float(any(r['reason']==reason for r in local))
    if local:
        r=max(local,key=lambda r:abs(r['value']-r['limit']))
        out[idx['value_log10']]=log10p(r['value']);out[idx['limit_log10']]=log10p(r['limit'])
        out[idx['relative_deficit']]=(r['value']-r['limit'])/(1+abs(r['value'])+abs(r['limit']))
        out[idx['projection_origin']]=float(any(r['origin']=='actual_projection' for r in local))
        out[idx['paid_origin']]=float(any(r['origin']=='paid_score' for r in local))
    out[idx['remaining_scores']]=feedback['remaining']['scores']/64
    out[idx['remaining_construction']]=feedback['remaining']['construction']/256
    out[idx['paid_calls']]=feedback['score_calls_already_paid']/64;out[idx['paid_failures']]=feedback['score_failures_already_paid']/64
    out[idx['paid_proxy_log10']]=log10p(feedback['paid_proxy_m2'] or 0.)
    for short,key in [('position','position_uncertainty_m2'),('unlocalized','unlocalized_target_cost_m2'),('coast','expected_coast_cost_m2'),('tentative','tentative_track_cost_m2')]:
        out[idx[short+'_cost_log10']]=log10p(feedback['paid_components'].get(key,0.))
    out[idx['iteration']]=feedback['iteration'];return out
