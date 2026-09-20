"""Truth is permitted here only: disaggregated physical performance metrics."""
from __future__ import annotations
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from .physics import *

def gospa_parts(truth,estimate,cutoff=250.):
    x=np.asarray(truth,float).reshape(-1,2);y=np.asarray(estimate,float).reshape(-1,2)
    n,m=len(x),len(y);pairs=[]
    if n and m:
        costs=np.minimum(((x[:,None,:]-y[None,:,:])**2).sum(2),cutoff**2)
        ii,jj=linear_sum_assignment(costs)
        pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if np.linalg.norm(x[i]-y[j])<cutoff]
    loc=sum(float(np.sum((x[i]-y[j])**2)) for i,j in pairs)
    missed=n-len(pairs);false=m-len(pairs)
    return {'localization_sum_m2':loc,'missed_count':missed,'false_count':false,
        'gospa_m':math.sqrt(loc+cutoff**2/2*(missed+false)),'pairs':pairs}

class Evaluator:
    def __init__(self,config):
        self.config=config;self.first_seen={};self.previous={};self.switches=0;self.gaps={};self.max_gaps={};self.rows=[];self.first_detected={};self.first_received={};self.first_initiated={};self.last_matched={};self.reacquisitions=[];self.lost_gaps={};self.max_lost_gaps={}
    def frame(self,world,tracker,schedule,costs,policy,packets,acceptance=None):
        accepted_roots={p['root'] for p,r in zip(packets,acceptance or []) if r['status'] in ('NO_EXISTING_TRACK','GATED_JOINT_MOMENT_UPDATE','GNN_CAP_APPROXIMATION')}
        for event in world.evaluation_events:
            if event['kind']!='detected_before_transport':continue
            tid=event['target_id'];self.first_detected.setdefault(tid,event['epoch'])
            if event['root'] in accepted_roots:self.first_received.setdefault(tid,world.epoch)
        truths=sorted(world.targets.items());tracks=[t for t in tracker.tracks if t['confirmed']]
        x=[t['state'][:2] for _,t in truths]
        all_estimates=[tracker.mean[t['start']:t['start']+2] for t in tracker.tracks]
        initial_pairs=gospa_parts(x,all_estimates,self.config['metric_cutoff_m'])['pairs']
        for i,j in initial_pairs:self.first_initiated.setdefault(truths[i][0],world.epoch)
        y=[tracker.mean[t['start']:t['start']+2] for t in tracks]
        row=gospa_parts(x,y,self.config['metric_cutoff_m']);errors=[];ves=[];nees=[];covered=[];matched=set();links=[]
        for i,j in row['pairs']:
            tid,true=truths[i];tr=tracks[j];start=tr['start'];delta=tracker.mean[start:start+4]-true['state']
            P=tracker.P[start:start+4,start:start+4];errors.append(float(delta[:2]@delta[:2]));ves.append(float(delta[2:]@delta[2:]))
            nees.append(float(delta@np.linalg.solve(P,delta)));covered.append(float(delta[:2]@np.linalg.solve(P[:2,:2],delta[:2]))<=5.991464547107979)
            if tid not in self.first_seen:self.first_seen[tid]=world.epoch
            if tid in self.previous and self.previous[tid]!=tr['id']:self.switches+=1
            if tid in self.last_matched and self.last_matched[tid]<world.epoch-1:self.reacquisitions.append({'truth_id':tid,'epoch':world.epoch,'track_id':tr['id'],'gap_cycles':world.epoch-self.last_matched[tid]-1})
            self.last_matched[tid]=world.epoch
            self.previous[tid]=tr['id'];matched.add(tid);links.append({'truth_id':tid,'track_id':tr['id'],'position_error_m':math.sqrt(errors[-1])})
        for tid,_ in truths:
            self.gaps[tid]=0 if tid in matched else self.gaps.get(tid,0)+1
            self.max_gaps[tid]=max(self.gaps[tid],self.max_gaps.get(tid,0))
            if tid in self.first_seen:
                self.lost_gaps[tid]=0 if tid in matched else self.lost_gaps.get(tid,0)+1
                self.max_lost_gaps[tid]=max(self.lost_gaps[tid],self.max_lost_gaps.get(tid,0))
        row.pop('pairs');row.update({'epoch':world.epoch,'time_s':world.epoch*self.config['dt_s'],
            'truth_count':len(truths),'confirmed_track_count':len(tracks),'tentative_track_count':len(tracker.tracks)-len(tracks),
            'matched_count':len(errors),'position_sq_errors_m2':errors,'velocity_sq_errors_m2_s2':ves,
            'position_rmse_m':math.sqrt(np.mean(errors)) if errors else None,
            'velocity_rmse_mps':math.sqrt(np.mean(ves)) if ves else None,
            'position_errors_m':[math.sqrt(e) for e in errors], 'confidence_matching':'GOSPA assignment conditional; not identity-fixed filter NEES','state_nees':nees,'position_95_covered':covered,
            'association_id_switches_cumulative':self.switches,
            'bias_rmse_m':float(np.sqrt(np.mean((tracker.mean[[0,1,6,7]]-world.bias[[0,1,6,7]])**2))),
            'angle_bias_rmse_deg':float(np.sqrt(np.mean((tracker.mean[2:6]-world.bias[2:6])**2))*180/math.pi),
            'rf_energy_J':costs['energy_J'],'transmit_on_s':costs['tx_s'],'receive_on_s':costs['listen_s'],
            'receiver_electrical_energy_J':costs['rx_energy_J'],
            'processing_ops':costs['processing_ops'],'communication_bytes':costs['communication_bytes'],
            'frequency_bias_rmse_Hz':float(np.sqrt(np.mean((tracker.mean[8:12]-world.bias[8:12])**2))),
            'network_seconds':policy['meter'].get('network_seconds'),'optimization_seconds':policy['meter'].get('optimization_seconds'),
            'report_bytes':sum(p.get('communication_bytes',0) for p in packets),
            'exposure_energy_density_proxy_J_m2':exposure_density(schedule,self.config),
            'exposure_limit_J_m2_per_cycle':self.config.get('exposure_energy_density_limit_J_m2_per_cycle'),
            'coverage_guard_status':policy.get('coverage_guard_status'),'coverage_guard_satisfied':policy.get('coverage_guard_satisfied'),
            'calibration_action_count':sum(t.kind=='calibrate' for t in schedule),
            'links':links,'policy_status':policy['status'],'K':policy['K'],'B':policy['B'],
            'paid_root_evaluations':policy['meter']['score_calls'],'forecast_updates':policy['meter']['forecast_updates'],
            'forecast_factorizations':policy['meter']['cholesky'],'construction_attempts':policy['meter']['construction'],
            'policy_seconds':policy['seconds']})
        self.rows.append(row);return row
    def summary(self,world,tracker):
        rows=self.rows;pe=[v for r in rows for v in r['position_sq_errors_m2']];ve=[v for r in rows for v in r['velocity_sq_errors_m2_s2']]
        counts=sum(r['truth_count'] for r in rows);matches=sum(r['matched_count'] for r in rows)
        delays={str(i):None if i not in self.first_seen else (self.first_seen[i]-t['birth'])*self.config['dt_s'] for i,t in world.all_targets.items()}
        def delays_from(events):return {str(i):None if i not in events else (events[i]-t['birth'])*self.config['dt_s'] for i,t in world.all_targets.items()}
        cov=[x for r in rows for x in r['position_95_covered']];ns=[x for r in rows for x in r['state_nees']]
        return {'scenario':self.config['scenario'],'seed':world.seed,'cycles':len(rows),
            'true_target_time_count':counts,'matched_target_time_count':matches,
            'missed_target_time_count':sum(r['missed_count'] for r in rows),'false_confirmed_track_time_count':sum(r['false_count'] for r in rows),
            'conditional_position_samples':len(pe),'conditional_velocity_samples':len(ve),'state_nees_samples':len(ns),'position_coverage_samples':len(cov),
            'independent_world_count':1,'metrics_unit':'one world; target-times are repeated dependent observations',
            'coverage_guard_requests':sum(r.get('coverage_guard_requested',False) for r in rows),
            'coverage_guard_satisfied_cycles':sum(r.get('coverage_guard_satisfied',False) for r in rows),
            'maximum_search_age_cycles':max(r.get('maximum_search_age_cycles',0) for r in rows),
            'position_rmse_m':math.sqrt(np.mean(pe)) if pe else None,
            'position_error_p95_m':float(np.percentile(np.sqrt(pe),95)) if pe else None,
            'velocity_rmse_mps':math.sqrt(np.mean(ve)) if ve else None,'velocity_error_p95_mps':float(np.percentile(np.sqrt(ve),95)) if ve else None,
            'matched_target_time_fraction':matches/counts if counts else None,
            'missed_target_time_fraction':sum(r['missed_count'] for r in rows)/counts if counts else None,
            'mean_false_tracks':float(np.mean([r['false_count'] for r in rows])),
            'mean_gospa_m':float(np.mean([r['gospa_m'] for r in rows])),
            'mean_localization_sum_m2':float(np.mean([r['localization_sum_m2'] for r in rows])),
            'confirmed_discovery_delays_s':delays,'not_confirmed_count':sum(x is None for x in delays.values()),
            'first_echo_delays_s':delays_from(self.first_detected),'first_received_delays_s':delays_from(self.first_received),
            'initiation_delays_s':delays_from(self.first_initiated),'not_discovered_count':sum(i not in self.first_received for i in world.all_targets),
            'discovery_definition':'first admitted nonduplicate reception, attribution from evaluation-only event log',
            'reacquisition_events':self.reacquisitions,'null_metric_reasons':{'conditional_errors':'null when no matched confirmed target times','delay':'null when event absent before horizon','lost_interval':'null when no confirmed track established'},
            'max_missing_interval_s':max(self.max_gaps.values(),default=0)*self.config['dt_s'],
            'max_lost_interval_s':max(self.max_lost_gaps.values())*self.config['dt_s'] if self.max_lost_gaps else None,
            'id_switches':self.switches,'confidence_matching':'GOSPA assignment conditional; not identity-fixed filter NEES','mean_state_nees':float(np.mean(ns)) if ns else None,
            'position_95_coverage':float(np.mean(cov)) if cov else None,
            'final_bias_rmse_m':rows[-1]['bias_rmse_m'] if rows else None,
            'final_frequency_bias_rmse_Hz':rows[-1]['frequency_bias_rmse_Hz'] if rows else None,
            'final_angle_bias_rmse_deg':rows[-1]['angle_bias_rmse_deg'] if rows else None,
            'rf_energy_J':sum(r['rf_energy_J'] for r in rows),'transmit_on_s':sum(r['transmit_on_s'] for r in rows),
            'receiver_electrical_energy_J':sum(r['receiver_electrical_energy_J'] for r in rows),
            'report_bytes':sum(r['report_bytes'] for r in rows),
            'exposure_energy_density_proxy_J_m2':sum(r['exposure_energy_density_proxy_J_m2'] for r in rows),
            'calibration_action_count':sum(r['calibration_action_count'] for r in rows),
            'coverage_guard_satisfied_cycles':sum(bool(r['coverage_guard_satisfied']) for r in rows),
            'coverage_guard_unavailable_cycles':sum(r['coverage_guard_status']=='NO_FEASIBLE_OVERDUE_SEARCH' for r in rows),
            'paid_root_evaluations':sum(r['paid_root_evaluations'] for r in rows),
            'forecast_updates':sum(r['forecast_updates'] for r in rows),
            'forecast_factorizations':sum(r['forecast_factorizations'] for r in rows),
            'construction_attempts':sum(r['construction_attempts'] for r in rows),
            'policy_seconds':sum(r['policy_seconds'] for r in rows),
            'tracker_diagnostics':dict(tracker.stats),'measurement_diagnostics':dict(world.audit),
            'assurance':'MEASUREMENT_LEVEL_DEVELOPMENT_NOT_PHYSICAL_CALIBRATION',
            'geometry_checkpoint_compatible_with_legacy':False,
            'receive_on_s':sum(r['receive_on_s'] for r in rows),
            'effective_measurement_packets':len(tracker.seen),'processing_ops':sum(r['processing_ops'] for r in rows),'communication_bytes':sum(r['communication_bytes'] for r in rows),
            'gospa_parameters':{'p':2,'alpha':2,'cutoff_m':self.config['metric_cutoff_m']},
            'receiver_IQ_full_multitarget_validation':False}
