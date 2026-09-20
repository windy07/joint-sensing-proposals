"""One entry point for tiny local and bounded Windows spatial diagnostics."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,csv,sys,time,platform,traceback,hashlib
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
import numpy as np
from .core import *
from .physics import *
from .world import World
from .filtering import MultiTracker
from .scheduling import run_policy,discovery_prob,admitted_schedule,admitted_captures,bind
from .metrics import Evaluator
from .binding import source_digest,source_manifest,dependency_contract,import_origins,FEATURE_BINDING,LABEL_BINDING,policy_configuration

ROOT=Path(__file__).resolve().parents[1]

def run_case(seed,config,method,learned=None,progress_path=None,resume_state=None):
    """Execute the one authoritative cycle order; persist completed-cycle state.

    World advancement is separately authorized in World.step. A failure receipt
    can include a partially observed cycle, but continuation uses only the last
    fully completed cycle snapshot, never stale measurements from that receipt.
    """
    control_config=policy_configuration(config,method)
    if learned is not None:learned.begin_world(seed,control_config)
    run_source=source_digest();start=time.perf_counter()
    world=World(seed,config);tracker=MultiTracker(control_config);evaluator=Evaluator(config)
    energy=Fraction(str(config['rf_energy_budget_J']));tx=Fraction(str(config['transmit_time_budget_s']))
    unseen=np.full(len(cell_list(config)),config['initial_unseen_per_cell']);trace=[]
    search_age=np.zeros(len(cell_list(config)),int);pointing=[0.]*4;start_epoch=0
    if learned is not None and seed in learned.training_law_seeds:raise ContractError('TRAIN_EVALUATION_WORLD_OVERLAP')
    if resume_state is not None:
        from .state_io import restore_cycle
        restored=restore_cycle(resume_state,seed,config,method,world,tracker,evaluator,learned)
        trace=restored['trace'];energy=Fraction(restored['energy_exact_J']);tx=Fraction(restored['tx_exact_s'])
        unseen=np.asarray(restored['unseen']);search_age=np.asarray(restored['search_age'],int);pointing=restored['pointing']
        start_epoch=restored['completed_cycles']
    last_progress=[0.]
    def progress(phase,**extra):
        if progress_path is not None and (phase.startswith('FAILED') or time.monotonic()-last_progress[0]>=2.):
            write_json(progress_path,{'status':'FAILED' if phase.startswith('FAILED') else 'RUNNING',
                'phase':phase,'seed':seed,'method':method,'completed_cycles':len(trace),
                'target_cycles':config['cycles'],'world_advances':world.audit['world_advances'],
                'updated_ns':time.time_ns(),**{k:v for k,v in extra.items() if k in ('attempted_epoch','error','traceback')}})
            last_progress[0]=time.monotonic()

    try:
        for epoch in range(start_epoch,config['cycles']):
            progress('BEFORE_ADVANCE',attempted_epoch=epoch)
            world.step();tracker.predict(epoch)
            pre_packets=world.arrived(epoch*config['dt_s']);pre_results=[]
            for packet in pre_packets:
                result=tracker.receive(packet,now_s=epoch*config['dt_s']);pre_results.append(result)
                tracker.initiate([packet],[result])
            if epoch:unseen+=config['birth_intensity_per_cell_per_cycle']
            search_age+=1
            view=bind({**tracker.view(energy,tx,unseen),'search_age_cycles':search_age.tolist(),
                'node_pointing_rad':list(pointing),'communication_available_s':world.link_free_s,
                'processing_available_s':list(world.processing_free_s)})
            fingerprint=tracker.fingerprint()
            selected,commit=run_policy(view,control_config,method,seed,learned)
            if tracker.fingerprint()!=fingerprint:raise ContractError('PLANNING_STATE_MUTATION')
            legal(selected,view,control_config)
            costs={k:sum(resource(t,config)[k] for t in selected)
                   for k in ('energy_J','tx_s','listen_s','rx_energy_J','processing_ops','communication_bytes')}
            energy-=sum((Fraction(resource(t,config)['energy_exact_J']) for t in selected),Fraction())
            tx-=sum((Fraction(resource(t,config)['tx_exact_s']) for t in selected),Fraction())
            if min(energy,tx)<0:raise ContractError('RESOURCE_OVERSPEND')
            progress('BEFORE_OBSERVE',attempted_epoch=epoch,current_view=view,current_commit=commit)
            generated_packets=world.observe(selected,view)
            progress('CAPTURED_BEFORE_ACCEPTANCE',attempted_epoch=epoch,current_view=view,
                     current_commit=commit,generated_packets=generated_packets)
            world.queue(generated_packets)
            for action in selected:pointing[action.node]=action.scan[-1][0]
            packets=world.arrived((epoch+1)*config['dt_s']);results=[]
            for packet in packets:
                result=tracker.receive(packet,now_s=(epoch+1)*config['dt_s']);results.append(result)
                tracker.initiate([packet],[result])
            admitted=admitted_captures(packets,results,config)
            unseen*=1-discovery_prob((),view,config,capture_rows=admitted)
            broad=[v for v in admitted if v['task'].kind=='search']
            search_age[discovery_prob((),view,config,capture_rows=broad)>=.5]=0
            # Only the evaluator sees target identities and true states.
            metric=evaluator.frame(world,tracker,selected,costs,commit,packets,results)
            metric.update({'maximum_search_age_cycles':int(max(search_age)),
                'coverage_guard_requested':commit['coverage_guard_cell'] is not None,
                'coverage_guard_satisfied':commit['coverage_guard_satisfied']})
            trace.append({'epoch':epoch,'causal_view':view,'commit':commit,'packets':packets,
                'predecision_packets':pre_packets,'predecision_acceptance':pre_results,
                'generated_packets':generated_packets,'pending_packet_roots':[p['root'] for p in world.pending],
                'update_results':results,'truth_for_evaluation_only':{str(i):t['state'].tolist() for i,t in world.targets.items()},
                'posterior_mean':tracker.mean.tolist(),'posterior_covariance':tracker.P.tolist(),
                'tracks':deepcopy(tracker.tracks),'metric':metric,
                'remaining_RF_energy_J':float(energy),'remaining_transmit_time_s':float(tx),
                'post_search_age_cycles':search_age.tolist(),'post_unseen_intensity':unseen.tolist()})
            if progress_path is not None and ((epoch+1)%config.get('restart_snapshot_interval',8)==0 or epoch+1==config['cycles']):
                from .state_io import save_cycle
                save_cycle(str(progress_path)+'.state.json.gz',seed,config,method,world,tracker,evaluator,
                           learned,trace,energy,tx,unseen,search_age,pointing)
                progress('COMPLETED_CYCLE',completed_epoch=epoch)
    except BaseException as exc:
        progress('FAILED_CURRENT_CYCLE',attempted_epoch=world.epoch,error=repr(exc),traceback=traceback.format_exc())
        raise
    summary=evaluator.summary(world,tracker);summary['method']=method;summary['wall_seconds']=time.perf_counter()-start
    return {'schema':'ain.spatial.numeric.result.v4','status':'COMPLETE','configuration':config,
        'source_digest':run_source,'source_manifest':source_manifest(),'feature_binding':FEATURE_BINDING,
        'label_binding':LABEL_BINDING,'dependency_contract':dependency_contract(),'actual_imports':import_origins(),
        'checkpoint_sha256':learned.checkpoint_sha256 if learned is not None else None,
        'model_state_digest':__import__('spatial_lab.binding',fromlist=['state_digest']).state_digest(learned.model.state_dict()) if learned is not None else None,
        'initialization_seed':getattr(learned,'initialization_seed',None),
        'untrained_initialization':getattr(learned,'untrained_initialization',False),'checkpoint_arm':getattr(learned,'checkpoint_arm',None),
        'memory_lifecycle':learned.memory.audit if learned is not None else None,'seed':seed,'method':method,
        'summary':summary,'trace':trace,'training_result':False,'physical_validation':False,'legacy_checkpoint_used':False,
        'world_advances':world.audit['world_advances'],'resumed_from_completed_cycles':start_epoch,
        'evaluation_events_only':world.evaluation_events,
        'pending_after_horizon':[{'root':p['root'],'arrival_s':p['arrival_s'],'delivered':p['delivered']} for p in world.pending]}

def main(argv=None):
    from .case_identity import case_identity, checkpoint_identity, seal_result, validate_cached
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--case-index',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--checkpoint')
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--allow-untrained-checkpoint',action='store_true')
    args=ap.parse_args(argv);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    receipt={'commands':sys.argv,'python':sys.version,'platform':platform.platform(),
             'status':'RUNNING','cases':[],'optimizer_updates':0}
    receipt_path=out/'RUN_RECEIPT.json';write_json(receipt_path,receipt)
    start=time.perf_counter()
    try:
        plan=read_json(args.config)
        binding={'schema':'ain.spatial.run.v4','plan_digest':digest(plan),
                 'source_digest':source_digest(),'checkpoint':checkpoint_identity(args.checkpoint),
                 'dependencies':dependency_contract()}
        path=out/'BINDING.json'
        if path.exists() and read_json(path)!=binding:
            raise ContractError('INCOMPATIBLE_RUN_BINDING')
        cases=[]
        for spec in plan['cases']:
            for seed in spec['seeds']:
                for method in plan['methods']:
                    cases.append((seed,{**plan['model'],**spec.get('overrides',{}),'scenario':spec['scenario']},method))
        if not cases:raise ContractError('EMPTY_EXPERIMENT')
        if args.case_index is not None:cases=[cases[args.case_index]]
        receipt.update(binding);receipt['planned_case_ids']=[f'{c[1]["scenario"]}_{c[0]}_{c[2]}' for c in cases]
        if args.dry_run:
            # Dependency and identity checks only; no model instantiation, world or training.
            receipt['status']='DRY_RUN';receipt['case_identities']=[case_identity(s,m,c,binding['checkpoint']) for s,c,m in cases]
        else:
            write_json(path,binding)
            for seed,cfg,method in cases:
                tag=f'{cfg["scenario"]}_{seed}_{method}';dest=out/'cases'/f'{tag}.json'
                try:
                    expected=case_identity(seed,method,cfg,binding['checkpoint'])
                    learned=None
                    if method in ('learned','numeric_network','fixed_network'):
                        if not args.checkpoint:raise ContractError('CHECKPOINT_REQUIRED')
                        # Validate weights and actual imports BEFORE any cache lookup.
                        from .spatial_graph import Priority
                        learned=Priority(args.checkpoint,allow_untrained=args.allow_untrained_checkpoint)
                    if args.resume and dest.exists():
                        validate_cached(read_json(dest),expected)
                        receipt['cases'].append({'id':tag,'status':'REUSED','case_digest':expected['digest']})
                    else:
                        if dest.exists():raise ContractError('CASE_EXISTS_USE_NEW_OUTPUT_OR_RESUME')
                        result=run_case(seed,cfg,method,learned,progress_path=out/'partial'/f'{tag}.json')
                        write_json(dest,seal_result(result,expected))
                        receipt['cases'].append({'id':tag,'status':'COMPLETE','case_digest':expected['digest']})
                except Exception as exc:
                    failure={'id':tag,'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()}
                    # Preserve any original cache; validation errors have their own receipts.
                    write_json(out/'failures'/f'{tag}.json',failure);receipt['cases'].append(failure)
                write_json(receipt_path,receipt)
            receipt['status']='FAILED' if any(c['status']=='FAILED' for c in receipt['cases']) else 'COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        receipt['wall_seconds']=time.perf_counter()-start;write_json(receipt_path,receipt)
    print(json.dumps(clean(receipt),ensure_ascii=False))
    return 2 if receipt['status']=='FAILED' else 0
if __name__=='__main__':sys.exit(main())
