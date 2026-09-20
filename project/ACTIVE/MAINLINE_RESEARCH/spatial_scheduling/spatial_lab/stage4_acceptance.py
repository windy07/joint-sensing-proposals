"""Zero-training release acceptance and explicitly selected finite fixtures."""
from __future__ import annotations
import argparse,datetime,hashlib,io,json,os,platform,sys,time,traceback,unittest
from pathlib import Path
from copy import deepcopy
from .core import ContractError,Task,read_json,write_json,digest,clean
from .binding import (ROOT,source_manifest,source_digest,dependency_contract,FEATURE_SPEC,FEATURE_BINDING,
                      LABEL_SPEC,LABEL_BINDING,import_origins)
from .preflight import preflight,SUPPORTED_PUBLIC_VERSIONS
from .work_budget import (LEDGER,locked_ledger,prohibit_optimizers,log_event,functional_authority)

RELEASE=ROOT.parents[2]
BINDING=RELEASE/'STAGE4_BINDINGS/FINAL_BINDING.json'

def current_environment():
    import importlib.metadata
    from threadpoolctl import threadpool_info
    return {'python':sys.version,'platform':platform.platform(),'public_dependencies':dependency_contract(),
      'installed_builds':{n:importlib.metadata.version(n) for n in dependency_contract()},
      'supported_public_dependencies':SUPPORTED_PUBLIC_VERSIONS,'threadpools':threadpool_info(),
      'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}

def frozen_matches():
    if not BINDING.exists():raise ContractError('FINAL_SOURCE_NOT_FROZEN')
    b=read_json(BINDING)
    if b['source_manifest']!=source_manifest():raise ContractError('FINAL_SOURCE_CHANGED_AFTER_FREEZE')
    for f,row in b['configuration_files'].items():
        p=RELEASE/f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ContractError('FINAL_CONFIGURATION_CHANGED:'+f)
    return b

def freeze(out):
    if BINDING.exists():frozen_matches();return read_json(BINDING)
    from .actions import ACTION_SCHEMA
    from .server_pipeline import check_plan,task_list
    plan=read_json(ROOT/'spatial_configs/stage4_server_plan.json');check_plan(plan)
    c=read_json(ROOT/'spatial_configs/stage4_model.json');env=preflight(c)
    cfgfiles=[ROOT/'spatial_configs/stage4_model.json',ROOT/'spatial_configs/stage4_server_plan.json',
              RELEASE/'STAGE4_BINDINGS/FUNCTIONAL_PREREGISTRATION.json']
    body={'schema':'ain.stage4.release.binding.v1','source_manifest':source_manifest(),
       'feature_spec':FEATURE_SPEC,'feature_binding':FEATURE_BINDING,'label_spec':LABEL_SPEC,'label_binding':LABEL_BINDING,
       'action_schema':ACTION_SCHEMA,'server_plan_digest':digest(plan),'data_status':'NEW_SCHEMA_NO_BATCH_DATA_GENERATED',
       'model_status':'THREE_UNTRAINED_INITIALIZATIONS_ONLY_ZERO_OPTIMIZER_UPDATES',
       'configuration_files':{p.relative_to(RELEASE).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in cfgfiles},
       'environment':env,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'historical_source_digest':'3e342e0c82efdf3da0caeb7e941199ad5df6dd9e7cbf6221505d2da52b8ca5db',
       'local_budget':{'optimizer_updates':0,'maximum_world_advances_including_failures':12,'maximum_fixtures':2,'backward_per_backbone':1}}
    write_json(BINDING,body);write_json(out/'FREEZE_RECEIPT.json',body)
    log_event('final_source_frozen',source_digest=source_digest(),optimizer_updates=0)
    return body

class DetailedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.rows=[];self.started={}
    def startTest(self,test):self.started[test.id()]=time.perf_counter();super().startTest(test)
    def row(self,test,status,error=None):self.rows.append({'test':test.id(),'status':status,'seconds':time.perf_counter()-self.started[test.id()],'error':error})
    def addSuccess(self,test):super().addSuccess(test);self.row(test,'PASSED')
    def addFailure(self,test,err):super().addFailure(test,err);self.row(test,'FAILED',self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.row(test,'ERROR',self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.row(test,'SKIPPED',reason)

def verify(out,groups=None,development=False):
    from . import stage4_tests
    c=read_json(ROOT/'spatial_configs/stage4_model.json')
    try:env=preflight(c)
    except ContractError as exc:
        if not development:raise
        env={'status':'DEVELOPMENT_ONLY_FORMAL_PREFLIGHT_REFUSED','refusal':repr(exc),**current_environment()}
    if BINDING.exists():frozen_matches()
    before=read_json(LEDGER) if LEDGER.exists() else {'world_advance_attempts':0,'optimizer_updates':0,'backward_attempts':{}}
    stage4_tests.OUTPUT=out;stage4_tests.DETAILS.clear();stream=io.StringIO()
    with prohibit_optimizers():
        result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=DetailedResult).run(stage4_tests.suite(groups))
    (out/'unittest.stdout.txt').write_text(stream.getvalue(),encoding='utf-8')
    after=read_json(LEDGER) if LEDGER.exists() else before
    for key in ('world_advance_attempts','optimizer_updates','backward_attempts'):
        if before.get(key)!=after.get(key):raise ContractError('FINITE_VERIFY_EXCEEDED_ZERO_WORK:'+key)
    report={'schema':'ain.stage4.fixed_acceptance.v1','status':'PASSED' if result.wasSuccessful() else 'FAILED',
        'source_manifest':source_manifest(),'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,
        'environment':env,'tests_run':result.testsRun,'passed':sum(r['status']=='PASSED' for r in result.rows),
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'tests':result.rows,
        'world_advances':0,'optimizer_updates':0,'backwards':0,'frozen_final_source':BINDING.exists()}
    write_json(out/'REPORT.json',report);write_json(out/'FIXED_DETAILS.json',stage4_tests.DETAILS)
    log_event('fixed_acceptance',source_digest=source_digest(),path=str(out.relative_to(RELEASE)) if out.is_relative_to(RELEASE) else str(out),
              tests_run=result.testsRun,status=report['status'],world_advances=0,optimizer_updates=0,backwards=0)
    print(stream.getvalue());return report

def replay_packets(out):
    """Reprocess six old direct-source packets without adopting old case bindings."""
    frozen_matches()
    from .filtering import MultiTracker
    from .core import Fraction
    from .physics import CELLS
    from .scheduling import admitted_captures,discovery_prob,bind
    root=RELEASE/'STAGE3_BASE/STAGE3_OUTPUT/evidence'
    candidates=sorted(root.glob('*/cases/06_numeric_optimizer_analytic.json'))
    if not candidates:candidates=sorted(root.rglob('06_numeric_optimizer_analytic.json'))
    if not candidates:raise ContractError('SAVED_DIRECT_CASE_NOT_FOUND')
    # E1 is a source of immutable observations, never relabelled a final run.
    p=candidates[0];old=read_json(p);cfg={**old['configuration'],'action_schema':'ain.spatial.executable.pulse_scan.v4'}
    tracker=MultiTracker(cfg);rows=[];count=0
    for frame in old['trace']:
        tracker.predict(frame['epoch']);results=[]
        for packet in frame['packets']:
            before=tracker.fingerprint();result=tracker.receive(packet,(frame['epoch']+1)*cfg['dt_s']);tracker.initiate([packet],[result])
            count+=1;results.append({'root':packet['root'],'status':result,'posterior_changed':before!=tracker.fingerprint()})
        rows.append({'epoch':frame['epoch'],'accepted_results':results,'mean':tracker.mean.tolist(),
                     'covariance':tracker.P.tolist(),'tracks':deepcopy(tracker.tracks)})
    report={'schema':'ain.stage4.historical_observation_replay.v1','source_manifest':source_manifest(),
        'original_case_path':p.relative_to(RELEASE).as_posix(),'original_case_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'original_case_source_digest':old['source_digest'],'replay_semantics':'original packets unchanged; final posterior only; NOT a final-source world run',
        'packets':count,'frames':rows,'new_world_advances':0,'optimizer_updates':0,
        'confirmed_tracks':sum(t['confirmed'] for t in tracker.tracks),'tentative_tracks':sum(not t['confirmed'] for t in tracker.tracks)}
    write_json(out/'SAVED_PACKET_REPLAY.json',report);return report

def functional(out,names=None):
    binding=frozen_matches()
    from .run import run_case
    from .case_identity import case_identity,seal_result,validate_cached
    from .numeric_replay import build
    from .functional_fixture import proposal
    registry=read_json(RELEASE/'STAGE4_BINDINGS/FUNCTIONAL_PREREGISTRATION.json')
    selected=names or list(registry['fixtures']);rows=[]
    for name in selected:
        if name not in registry['fixtures']:raise ContractError('UNREGISTERED_FIXTURE_NAME')
        spec=registry['fixtures'][name];cfg=spec['configuration'];seed=spec['seed'];preflight(cfg)
        path=out/'cases'/f'{name}.json';progress=out/'partial'/f'{name}.json';calls={}
        def profile(frame,event,arg):
            if event!='call':return
            module=frame.f_globals.get('__name__','');function=frame.f_code.co_name
            if (module=='receiver_revisions.passive_geometry' and function=='measurement_and_jacobian') or (module=='spatial_lab.world' and function in ('step','observe')) or (module=='spatial_lab.filtering' and function in ('receive','initiate','_association')):
                scope='prediction_or_inference';parent=frame
                while parent:
                    if parent.f_globals.get('__name__')=='spatial_lab.world' and parent.f_code.co_name=='observe':scope='world_reception';break
                    parent=parent.f_back
                key=f'{module}.{function}/{scope}/{frame.f_locals.get("mode","")}'
                calls[key]=calls.get(key,0)+1
        try:
            with prohibit_optimizers(),functional_authority(name,seed,cfg):
                sys.setprofile(profile)
                try:result=run_case(seed,cfg,'functional_contract',progress_path=progress)
                finally:sys.setprofile(None)
            result['functional_acceptance_only']=True;result['preregistration_digest']=digest(spec);result['actual_call_profile']=calls
            result['development_vs_formal_environment']='SUPPORTED_FORMAL_DEPENDENCIES';ident=case_identity(seed,'functional_contract',cfg)
            result=seal_result(result,ident);validate_cached(result,ident);write_json(path,result)
            # The complete snapshot can be restored and summarized without a
            # single further World.step. This does not restart the fixture.
            with prohibit_optimizers():restored=run_case(seed,cfg,'functional_contract',resume_state=str(progress)+'.state.json')
            if digest(restored['trace'])!=digest(result['trace']):raise ContractError('FINAL_SNAPSHOT_EXACT_RELOAD_FAILED')
            replay_final_packets(result,out/'posterior_replays'/f'{name}.json')
            rows.append({'fixture':name,'status':'COMPLETE','path':str(path.relative_to(RELEASE)),
                         'summary':result['summary'],'source_digest':result['source_digest'],
                         'resume_state_exact_trace':True,'additional_reload_world_advances':0,'actual_call_profile':calls})
        except BaseException as exc:
            sys.setprofile(None);failure={'fixture':name,'status':'FAILED','source_digest':source_digest(),
                'error':repr(exc),'traceback':traceback.format_exc(),'actual_call_profile':calls,
                'partial_result':str(progress),'ledger':read_json(LEDGER)}
            write_json(out/'failures'/f'{name}.json',failure);rows.append(failure)
        write_json(out/'FUNCTIONAL_RECEIPT.json',{'fixtures':rows,'source_digest':source_digest(),'ledger':read_json(LEDGER)})
    paths=list((out/'cases').glob('*.json'))
    replay=build(paths,out/'FINAL_REPLAY.html',max_worlds=2) if paths else None
    result={'status':'PASSED' if len(rows)==len(selected) and all(r['status']=='COMPLETE' for r in rows) else 'FAILED',
        'source_digest':source_digest(),'fixtures':rows,'ledger':read_json(LEDGER),'replay':replay,
        'claim':'finite preregistered function checks; not trained model effects or hardware calibration'}
    write_json(out/'FUNCTIONAL_RECEIPT.json',result);return result

def replay_final_packets(result,path):
    from .filtering import MultiTracker
    from .binding import policy_configuration
    import numpy as np
    cfg=result['configuration'];tracker=MultiTracker(policy_configuration(cfg,result['method']));errors=[];rows=[]
    for frame in result['trace']:
        tracker.predict(frame['epoch']);accept=[]
        for packets,now in [(frame['predecision_packets'],frame['epoch']*cfg['dt_s']),(frame['packets'],(frame['epoch']+1)*cfg['dt_s'])]:
            for packet in packets:
                response=tracker.receive(packet,now);tracker.initiate([packet],[response]);accept.append(response)
        em=float(np.max(np.abs(tracker.mean-np.asarray(frame['posterior_mean']))));ep=float(np.max(np.abs(tracker.P-np.asarray(frame['posterior_covariance']))))
        if em!=0. or ep!=0. or tracker.tracks!=frame['tracks']:raise ContractError('FINAL_PACKET_REPLAY_NOT_EXACT')
        rows.append({'epoch':frame['epoch'],'max_mean_difference':em,'max_covariance_difference':ep,'receive_results':accept,'tracks':deepcopy(tracker.tracks)})
    report={'schema':'ain.stage4.final_packet_replay.v1','source_digest':source_digest(),'case_digest':result['case_identity']['digest'],
            'frames':rows,'exact':True,'new_world_advances':0,'optimizer_updates':0}
    write_json(path,report);return report

def replay_functional_source(input_case,out):
    """Final-code closed-loop replay of immutable earlier received observations.

    This is NOT a new world run. Rebuild every causal belief and paid proposal
    using the final implementation; require exactly the same executed controls
    before admitting the saved packets. Evaluation truths are read only AFTER
    proposal and posterior update. Original case/source hashes are retained.
    """
    frozen_matches()
    import math,time
    from types import SimpleNamespace
    from fractions import Fraction
    import numpy as np
    from .filtering import MultiTracker
    from .metrics import Evaluator
    from .physics import NBIAS,CELLS
    from .actions import resource
    from .scheduling import bind,run_policy,admitted_captures,discovery_prob
    from .binding import policy_configuration
    from .case_identity import case_identity,seal_result,validate_cached
    path=Path(input_case);old=read_json(path)
    # Validate the ORIGINAL binding as original, not as a current reusable case.
    validate_cached(old,old['case_identity'])
    if old['method']!='functional_contract':raise ContractError('REPLAY_INPUT_NOT_DECLARED_FUNCTIONAL_CASE')
    cfg=old['configuration'];c=policy_configuration(cfg,'functional_contract');seed=old['seed']
    tracker=MultiTracker(c);ev=Evaluator(cfg);energy=Fraction(str(cfg['rf_energy_budget_J']));tx=Fraction(str(cfg['transmit_time_budget_s']))
    unseen=np.full(len(CELLS),cfg['initial_unseen_per_cell']);age=np.zeros(len(CELLS),int);pointing=[0.]*4
    frames=[];checks=[];all_truths={};audit={**old['summary']['measurement_diagnostics'],'world_advances':0,
                                       'original_input_world_advances':old['world_advances'],'mode':'RECORDED_OBSERVATIONS_NO_WORLD_GENERATION'}
    for oldframe in old['trace']:
        epoch=oldframe['epoch'];tracker.predict(epoch);pre=[]
        for p in oldframe['predecision_packets']:
            r=tracker.receive(p,epoch*cfg['dt_s']);tracker.initiate([p],[r]);pre.append(r)
        if epoch:unseen+=cfg['birth_intensity_per_cell_per_cycle']
        age+=1
        view=bind({**tracker.view(energy,tx,unseen),'search_age_cycles':age.tolist(),'node_pointing_rad':pointing,
            'communication_available_s':oldframe['causal_view']['communication_available_s'],
            'processing_available_s':oldframe['causal_view']['processing_available_s']})
        if view['digest']!=oldframe['causal_view']['digest']:raise ContractError('REPLAY_CAUSAL_CONTEXT_DIFFERS')
        selected,commit=run_policy(view,c,'functional_contract',seed)
        if digest([t.to_dict() for t in selected])!=digest(oldframe['commit']['selected']):raise ContractError('REPLAY_EXECUTED_ACTION_DIFFERS')
        costs={k:sum(resource(t,cfg)[k] for t in selected) for k in ('energy_J','tx_s','listen_s','rx_energy_J','processing_ops','communication_bytes')}
        energy-=sum((Fraction(resource(t,cfg)['energy_exact_J']) for t in selected),Fraction())
        tx-=sum((Fraction(resource(t,cfg)['tx_exact_s']) for t in selected),Fraction())
        results=[]
        for p in oldframe['packets']:
            r=tracker.receive(p,(epoch+1)*cfg['dt_s']);tracker.initiate([p],[r]);results.append(r)
        for t in selected:pointing[t.node]=t.scan[-1][0]
        admitted=admitted_captures(oldframe['packets'],results,cfg);unseen*=1-discovery_prob((),view,cfg,capture_rows=admitted)
        broad=[r for r in admitted if r['task'].kind=='search'];age[discovery_prob((),view,cfg,capture_rows=broad)>=.5]=0
        if not np.array_equal(tracker.mean,np.asarray(oldframe['posterior_mean'])) or not np.array_equal(tracker.P,np.asarray(oldframe['posterior_covariance'])):
            raise ContractError('REPLAY_FULL_POSTERIOR_DIFFERS')
        if digest(tracker.tracks)!=digest(oldframe['tracks']):raise ContractError('REPLAY_LIFECYCLE_DIFFERS')
        # Everything below is evaluation-only; unavailable historical bias truth
        # is explicitly null, not filled with the estimate or a sampled truth.
        truths={int(i):{'state':np.asarray(x),'birth':cfg['target_birth_epochs'][int(i)]} for i,x in oldframe['truth_for_evaluation_only'].items()}
        all_truths.update(truths)
        eval_world=SimpleNamespace(seed=seed,epoch=epoch,targets=truths,all_targets=all_truths,
            evaluation_events=old['evaluation_events_only'],bias=np.full(NBIAS,np.nan),audit=audit)
        metric=ev.frame(eval_world,tracker,selected,costs,commit,oldframe['packets'],results)
        metric.update(maximum_search_age_cycles=int(max(age)),coverage_guard_requested=False,coverage_guard_satisfied=False,
                      bias_truth_unavailable_in_saved_packet_record=True)
        frames.append({'epoch':epoch,'causal_view':view,'commit':commit,'packets':oldframe['packets'],
            'predecision_packets':oldframe['predecision_packets'],'predecision_acceptance':pre,
            'generated_packets':oldframe['generated_packets'],'pending_packet_roots':oldframe['pending_packet_roots'],
            'update_results':results,'truth_for_evaluation_only':oldframe['truth_for_evaluation_only'],
            'posterior_mean':tracker.mean.tolist(),'posterior_covariance':tracker.P.tolist(),'tracks':deepcopy(tracker.tracks),
            'metric':metric,'remaining_RF_energy_J':float(energy),'remaining_transmit_time_s':float(tx),
            'post_search_age_cycles':age.tolist(),'post_unseen_intensity':unseen.tolist()})
        checks.append({'epoch':epoch,'causal_context_exact':True,'executed_action_exact':True,
                       'full_joint_posterior_exact':True,'lifecycle_exact':True,'admitted_packet_count':len(admitted)})
    summary=ev.summary(eval_world,tracker);summary['method']='recorded_observation_replay';summary['execution_kind']='FINAL_SOURCE_SAVED_OBSERVATION_REPLAY'
    summary['null_metric_reasons']['bias']='historical true bias vectors are not in the saved observation record'
    result={'schema':'ain.spatial.numeric.result.v4','status':'COMPLETE','source_manifest':source_manifest(),'source_digest':source_digest(),
       'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,'dependency_contract':dependency_contract(),'actual_imports':import_origins(),
       'seed':seed,'method':'recorded_observation_replay','configuration':cfg,'summary':summary,'trace':frames,
       'execution_kind':'FINAL_SOURCE_SAVED_OBSERVATION_REPLAY','functional_acceptance_only':True,
       'world_advances':0,'original_input_world_advances':old['world_advances'],'input_case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'input_case_identity':old['case_identity'],'original_generation_source_digest':old['source_digest'],
       'checkpoint_sha256':None,'model_state_digest':None,'checkpoint_arm':None,'untrained_initialization':False,
       'training_result':False,'physical_validation':False,'legacy_checkpoint_used':False,'evaluation_events_only':old['evaluation_events_only'],
       'replay_checks':checks,'data_binding':'old immutable packet roots and original source retained; this is a newly executed final-source replay, not relabelled world advances'}
    result=seal_result(clean(result),case_identity(seed,'recorded_observation_replay',cfg));validate_cached(result,result['case_identity'])
    tag=cfg['fixture_name']+'_final_source_replay'
    dest=Path(out)/'cases'/f'{tag}.json';write_json(dest,result)
    replay_final_packets(result,Path(out)/'posterior_replays'/f'{tag}.json')
    return {'status':'PASSED','source_digest':source_digest(),'case':str(dest),'summary':result['summary'],
       'world_advances':0,'optimizer_updates':0,'original_generation_source_digest':old['source_digest'],'checks':checks}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['verify','connections','fixtures','replay-packets','freeze','preflight'])
    ap.add_argument('--output');ap.add_argument('--development',action='store_true');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--only',help='comma separated finite test groups or preregistered fixture names')
    args=ap.parse_args(argv);stamp=str(time.time_ns())
    out=Path(args.output).resolve() if args.output else RELEASE/'STAGE4_EVIDENCE'/(args.command+'_'+stamp)
    out.mkdir(parents=True,exist_ok=True);result={'status':'RUNNING','command':args.command,'source_digest':source_digest()}
    try:
        if args.command=='verify':result=verify(out,args.only.split(',') if args.only else None,args.development)
        elif args.command=='freeze':result=freeze(out)
        elif args.command=='preflight':result=preflight(read_json(ROOT/'spatial_configs/stage4_model.json'))
        elif args.command=='replay-packets':
            with prohibit_optimizers():result=replay_packets(out)
        elif args.command=='connections':
            if not args.execute:raise ContractError('ONE_BACKWARD_PER_ARM_REQUIRES_EXPLICIT_EXECUTE')
            frozen_matches()
            from .stage4_connections import check_connections
            with prohibit_optimizers():result=check_connections(out)
        elif args.command=='fixtures':
            if not args.execute:raise ContractError('WORLD_FIXTURES_REQUIRE_EXPLICIT_EXECUTE')
            result=functional(out,args.only.split(',') if args.only else None)
        if result.get('status') is None:result['status']='PASSED'
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),environment=current_environment())
    finally:
        result['output_directory']=str(out);write_json(out/'ENTRY_RECEIPT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','command','tests_run','passed','failures','errors','error','output_directory','source_digest')},ensure_ascii=False))
    return 2 if result.get('status')=='FAILED' else 0
if __name__=='__main__':raise SystemExit(main())
