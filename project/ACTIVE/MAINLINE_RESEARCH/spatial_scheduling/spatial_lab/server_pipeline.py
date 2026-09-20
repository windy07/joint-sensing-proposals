"""Frozen, serial server pipeline. No work is started without --execute.

Data generation, training and confirmation have different directories. Reuse is
case-identity checked; a worker error writes a receipt instead of disappearing.
No new optimizer is instantiated by preflight, dry-run, evaluate, or pack.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback,subprocess,os,gzip,zipfile,math
from pathlib import Path
from .core import ContractError,read_json,write_json,digest
from .binding import source_digest,source_manifest,dependency_contract,policy_configuration,FEATURE_BINDING,LABEL_BINDING
from .preflight import preflight
from .case_identity import case_identity,checkpoint_identity,validate_cached,seal_result

HERE=Path(__file__).resolve().parents[1]
DEFAULT_PLAN=HERE/'spatial_configs/stage4_server_plan.json'
ARMS=('integrated_agent','deep_sets','set_transformer')

def plan_cases(plan,split):
    ans=[];seeds=plan['world_splits'][split]
    for i,seed in enumerate(seeds):
        scenario=plan['scenarios'][i%len(plan['scenarios'])]
        cfg={**plan['model'],**scenario['overrides'],'scenario':scenario['name']}
        ans.append((seed,cfg))
    return ans

def check_plan(plan):
    if plan.get('schema')!='ain.stage4.server.plan.v1':raise ContractError('SERVER_PLAN_SCHEMA')
    if plan.get('feature_binding')!=FEATURE_BINDING or plan.get('label_binding')!=LABEL_BINDING:raise ContractError('SERVER_PLAN_FEATURE_LABEL_BINDING')
    from .binding import ROOT
    frozen=ROOT.parents[2]/'STAGE4_BINDINGS/FINAL_BINDING.json'
    if frozen.exists():
        authority=read_json(frozen)
        if authority['source_manifest']!=source_manifest() or authority['server_plan_digest']!=digest(plan):raise ContractError('SERVER_PLAN_FINAL_SOURCE_BINDING')
    seen=set()
    for split in ('training','development','confirmation'):
        seeds=plan['world_splits'][split]
        if not seeds or len(seeds)!=len(set(seeds)) or seen&set(seeds) or any(type(s)is not int for s in seeds):raise ContractError('WORLD_PARTITION_OVERLAP_OR_EMPTY')
        seen.update(seeds)
    if set(plan['backbones'])!=set(ARMS):raise ContractError('BACKBONE_COMPARISON_MATRIX')
    if not plan['model_seeds'] or len(set(plan['model_seeds']))!=len(plan['model_seeds']):raise ContractError('MODEL_SEEDS')
    for _,c in plan_cases(plan,'development')[:len(plan['scenarios'])]:preflight(c)
    if plan['decision_comparison']['claim']!='common bounded action/forecast domain; total cost reported, not presumed equal':raise ContractError('COMPUTE_COMPARISON_RULE')
    return True

def task_list(plan,root):
    tasks=[]
    for seed,cfg in plan_cases(plan,'training'):
        method=plan['teacher_policies'][seed%len(plan['teacher_policies'])]
        tasks.append({'stage':'data','seed':seed,'scenario':cfg['scenario'],'method':method,'cycles':cfg['cycles']})
    for arm in plan['backbones']:
        for ms in plan['model_seeds']:tasks.append({'stage':'train','arm':arm,'model_seed':ms,'updates':plan['optimizer_updates']})
    for split in ('development','confirmation'):
        for seed,cfg in plan_cases(plan,split):
            for method in ('fixed_analytic','numeric_optimizer'):
                tasks.append({'stage':split,'seed':seed,'scenario':cfg['scenario'],'method':method,'cycles':cfg['cycles']})
            for arm in plan['backbones']:
                for ms in plan['model_seeds']:
                    for method in ('fixed_network','numeric_network'):
                        tasks.append({'stage':split,'seed':seed,'scenario':cfg['scenario'],'method':method,'arm':arm,'model_seed':ms,'cycles':cfg['cycles']})
    return tasks

def checkpoint_map(plan,root):
    return {f'{a}/{s}':root/'models'/a/str(s)/'latest.pt' for a in plan['backbones'] for s in plan['model_seeds']}

def frozen_confirmation(plan,root,create=False):
    if create:validate_training_completion(plan,root)
    checkpoints=checkpoint_map(plan,root)
    body={'schema':'ain.stage4.confirmation.freeze.v1','plan_digest':digest(plan),'source_digest':source_digest(),
          'dependencies':dependency_contract(),'checkpoints':{k:checkpoint_identity(p) for k,p in checkpoints.items()},
          'development_receipt_sha256':hashlib.sha256((root/'development'/'STAGE_RECEIPT.json').read_bytes()).hexdigest(),
          'selection':'fixed registered final update; no confirmation or development hyperparameter selection',
          'confirmation_worlds':plan['world_splits']['confirmation']}
    path=root/'FROZEN_CONFIRMATION.json'
    if path.exists():
        if read_json(path)!=body:raise ContractError('CONFIRMATION_FREEZE_CHANGED_NEW_VERSION_REQUIRED')
    elif create:
        if (root/'confirmation').exists():raise ContractError('CANNOT_FREEZE_AFTER_CONFIRMATION_ACCESS')
        write_json(path,body)
    else:raise ContractError('FREEZE_CONFIRMATION_BEFORE_EVALUATION')
    return body


def validate_training_completion(plan,root):
    """Freeze actual completed training, never a partial-but-trained checkpoint."""
    import torch
    from .binding import verify_checkpoint
    receipt=read_json(root/'data/STAGE_RECEIPT.json')
    if receipt.get('status')!='COMPLETE':raise ContractError('CONFIRMATION_DATA_STAGE_INCOMPLETE')
    summary=read_json(root/'TRAINING_SUMMARY.json')
    if summary.get('status')!='COMPLETE' or len(summary.get('runs',[]))!=len(plan['backbones'])*len(plan['model_seeds']):
        raise ContractError('CONFIRMATION_TRAINING_STAGE_INCOMPLETE')
    expected_jobs=[t for t in task_list(plan,root) if t['stage']=='development']
    development=read_json(root/'development/STAGE_RECEIPT.json')
    if (development.get('status') not in ('COMPLETE','FAILED') or development.get('jobs')!=expected_jobs or
        len(development.get('results',[]))!=len(expected_jobs)):
        raise ContractError('CONFIRMATION_DEVELOPMENT_STAGE_INCOMPLETE')
    # Failed evaluation jobs remain in the planned population and retain the
    # registered penalty. Completeness must not silently exclude these cases.
    if any(r.get('status') not in ('COMPLETE','REUSED','FAILED','TIMEOUT') for r in development['results']):
        raise ContractError('CONFIRMATION_DEVELOPMENT_JOB_INCOMPLETE')
    for arm in plan['backbones']:
        for seed in plan['model_seeds']:
            ck=torch.load(root/'models'/arm/str(seed)/'latest.pt',weights_only=True,map_location='cpu')
            verify_checkpoint(ck)
            if (ck.get('arm')!=arm or ck.get('model_seed')!=seed or not ck.get('trained') or
                ck.get('optimizer_updates')!=plan['optimizer_updates'] or
                ck.get('training_purpose')!='registered_server_training' or
                set(ck.get('training_law_seeds',[]))!=set(plan['world_splits']['training']) or
                set(ck.get('development_law_seeds',[]))!=set(plan['world_splits']['development'])):
                raise ContractError('CONFIRMATION_CHECKPOINT_NOT_REGISTERED_COMPLETION')
            if any(not bool(torch.isfinite(t).all()) for t in ck['state'].values()):
                raise ContractError('CONFIRMATION_NONFINITE_CHECKPOINT')
    return True

def _one_case(root,seed,cfg,method,ck=None,arm=None,model_seed=None,split='training',device='cpu',resume=False):
    from .run import run_case
    from .spatial_graph import Priority
    ident=None;tag=f'{seed}_{method}'+(f'_{arm}_{model_seed}' if arm else '')
    dest=root/'cases'/f'{tag}.json';failure=root/'failures'/f'{tag}_{time.time_ns()}.json'
    try:
        ident=case_identity(seed,method,cfg,checkpoint_identity(ck))
        model=Priority(ck,device=device,evaluation_split=split if split!='training' else 'interface_check') if ck else None
        if dest.exists():
            if not resume:raise ContractError('EXISTING_CASE_REQUIRES_RESUME')
            validate_cached(read_json(dest),ident);return {'tag':tag,'status':'REUSED','path':str(dest)}
        partial=root/'partial'/f'{tag}.json';state=Path(str(partial)+'.state.json')
        result=run_case(seed,cfg,method,model,progress_path=partial,resume_state=state if resume and state.exists() else None);result['evaluation_split']=split
        write_json(dest,seal_result(result,ident));return {'tag':tag,'status':'COMPLETE','path':str(dest)}
    except Exception as exc:
        row={'tag':tag,'status':'FAILED','seed':seed,'method':method,'arm':arm,'model_seed':model_seed,'case_identity':ident,
             'error':repr(exc),'traceback':traceback.format_exc()};write_json(failure,row);return row

def worker(plan,root,task_index,split,device,resume,attempt_id):
    jobs=[t for t in task_list(plan,root) if t['stage']==split]
    job=jobs[task_index];cfg=next(c for s,c in plan_cases(plan,'training' if split=='data' else split) if s==job['seed'])
    ck=root/'models'/job['arm']/str(job['model_seed'])/'latest.pt' if 'arm'in job else None
    row=_one_case(root/split,job['seed'],cfg,job['method'],ck,job.get('arm'),job.get('model_seed'),
                  'training' if split=='data' else split,device,resume)
    row.update(attempt_id=attempt_id,job_digest=digest(job),source_digest=source_digest())
    write_json(root/split/'worker_receipts/history'/f'{task_index:06d}_{time.time_ns()}.json',row)
    write_json(root/split/'worker_receipts'/f'{task_index:06d}.json',row)
    return row['status']!='FAILED'

def run_stage(plan,plan_path,root,split,device,resume):
    jobs=[t for t in task_list(plan,root) if t['stage']==split];rows=[]
    if split=='confirmation':frozen_confirmation(plan,root)
    for i,job in enumerate(jobs):
        attempt_id=str(time.time_ns())
        cmd=[sys.executable,'-B','-m','spatial_lab.server_pipeline','worker','--plan',str(plan_path),
             '--output',str(root),'--split',split,'--index',str(i),'--device',device,'--attempt-id',attempt_id,'--execute']
        if resume:cmd.append('--resume')
        logdir=root/split/'logs'/str(time.time_ns());logdir.mkdir(parents=True,exist_ok=True)
        try:
            run=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,timeout=plan['case_timeout_s'],
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            (logdir/f'{i:06d}.stdout.txt').write_text(run.stdout,encoding='utf-8');(logdir/f'{i:06d}.stderr.txt').write_text(run.stderr,encoding='utf-8')
            receipt=root/split/'worker_receipts'/f'{i:06d}.json'
            row=read_json(receipt) if receipt.exists() else {}
            if run.returncode!=0 or row.get('attempt_id')!=attempt_id or row.get('job_digest')!=digest(job) or row.get('source_digest')!=source_digest():
                row={'status':'FAILED','job':job,'returncode':run.returncode,'reason':'MISSING_STALE_OR_FAILED_WORKER_RECEIPT','attempt_id':attempt_id,'observed_worker_receipt':row}
                write_json(root/split/'failures'/f'worker_{i:06d}_{attempt_id}.json',row)
        except subprocess.TimeoutExpired as exc:
            row={'status':'TIMEOUT','job':job,'timeout_s':plan['case_timeout_s']}
            for name,value in (('stdout',exc.stdout),('stderr',exc.stderr)):
                if isinstance(value,bytes):value=value.decode(errors='replace')
                (logdir/f'{i:06d}.{name}.txt').write_text(value or '',encoding='utf-8')
            write_json(root/split/'failures'/f'timeout_{i:06d}_{attempt_id}.json',row)
        rows.append(row);write_json(root/split/'STAGE_RECEIPT.json',{'stage':split,'jobs':jobs,'results':rows,'status':'RUNNING'})
    failures=sum(r['status'] not in ('COMPLETE','REUSED') for r in rows)
    receipt={'stage':split,'jobs':jobs,'results':rows,'failure_count':failures,'status':'FAILED' if failures else 'COMPLETE'}
    write_json(root/split/'STAGE_RECEIPT.json',receipt)
    return receipt

def _export_one(arg):
    from .sequence_training import load_sequences
    from .core import atomic_save
    root,p=map(Path,arg)
    seq,entries=load_sequences([p])
    body={'schema':'ain.stage4.saved_sequence.v1','source_digest':source_digest(),
          'case_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dataset_manifest':entries,'sequence':seq[0]}
    body['content_digest']=digest(body)
    out=root/'data/sequences'/f'{seq[0]["world_seed"]}.json.gz'
    payload=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
    def save(stream):
        with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as f:f.write(payload)
    atomic_save(out,save)
    return {'world_seed':seq[0]['world_seed'],'case':p.relative_to(root).as_posix(),'case_sha256':body['case_sha256'],
            'full_graph_sequence':out.relative_to(root).as_posix(),'sequence_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}

def export_sequences(root,workers=1):
    from concurrent.futures import ProcessPoolExecutor
    paths=sorted((root/'data/cases').glob('*.json'));manifest=[]
    def progress(status,**extra):
        row={'schema':'ain.stage5.sequence_export.receipt.v1','status':status,
             'source_digest':source_digest(),'worlds_total':len(paths),'worlds_exported':len(manifest),
             'actual_world_advances':0,'actual_optimizer_updates':0,**extra}
        write_json(root/'data/EXPORT_RECEIPT.json',row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    progress('RUNNING')
    pool=None
    try:
        args=[(str(root),str(p)) for p in paths]
        if workers>1:
            pool=ProcessPoolExecutor(max_workers=min(workers,len(paths)));iterator=pool.map(_export_one,args)
        else:iterator=map(_export_one,args)
        for row in iterator:
            manifest.append(row);progress('RUNNING',last_world_seed=row['world_seed'])
    except Exception as exc:
        progress('FAILED',failed_case=paths[len(manifest)].name,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
    write_json(root/'data/DATASET_MANIFEST.json',{'schema':'ain.stage4.training_dataset.v1','source_digest':source_digest(),'worlds':manifest})
    progress('COMPLETE')
    return paths

def validate_planned_record(plan,root,split,result):
    """Validate against the requested plan, not against a record's own claims."""
    configs=dict(plan_cases(plan,split));seed=result.get('seed');method=result.get('method')
    if seed not in configs:raise ContractError('UNREGISTERED_RESULT_WORLD')
    ck=checkpoint_identity(None)
    if split=='training':
        if method!=plan['teacher_policies'][seed%len(plan['teacher_policies'])]:raise ContractError('UNREGISTERED_TEACHER_METHOD')
    elif method in ('fixed_network','numeric_network'):
        arm=result.get('checkpoint_arm')
        if arm not in plan['backbones']:raise ContractError('UNREGISTERED_RESULT_BACKBONE')
        matches=[checkpoint_identity(root/'models'/arm/str(ms)/'latest.pt') for ms in plan['model_seeds']]
        matches=[x for x in matches if x['sha256']==result.get('checkpoint_sha256')]
        if len(matches)!=1:raise ContractError('UNREGISTERED_RESULT_CHECKPOINT')
        ck=matches[0]
    elif method not in ('fixed_analytic','numeric_optimizer'):raise ContractError('UNREGISTERED_EVALUATION_METHOD')
    expected=case_identity(seed,method,configs[seed],ck)
    if split=='training':
        from .training_provenance import validate_training_case
        validate_training_case(result,expected)
    else:
        validate_cached(result,expected)
    return result

def train_stage(plan,root,device,resume):
    from .sequence_training import train
    receipt=read_json(root/'data/STAGE_RECEIPT.json')
    if receipt['status']!='COMPLETE':raise ContractError('TRAINING_DATA_HAS_FAILURES_REPAIR_OR_NEW_REGISTERED_DATASET')
    data_paths=sorted((root/'data/cases').glob('*.json'))
    if len(data_paths)!=len(plan['world_splits']['training']):raise ContractError('TRAINING_WORLD_COUNT_MISMATCH')
    for path in data_paths:validate_planned_record(plan,root,'training',read_json(path))
    paths=export_sequences(root)
    allowed=[]
    for split in ('training','development','confirmation'):
        for _,c in plan_cases(plan,split)[:len(plan['scenarios'])]:allowed.append(policy_configuration(c,'numeric_network'))
    rows=[]
    for arm in plan['backbones']:
        for seed in plan['model_seeds']:
            dest=root/'models'/arm/str(seed);latest=dest/'latest.pt'
            if latest.exists() and not resume:raise ContractError('TRAINING_OUTPUT_EXISTS_USE_RESUME')
            rows.append(train(paths,dest,plan['optimizer_updates'],arm,seed,device,execute=True,
                 development_seeds=plan['world_splits']['development'],allowed_configurations=allowed,resume=latest if latest.exists() else None))
    write_json(root/'TRAINING_SUMMARY.json',{'status':'COMPLETE','runs':rows})
    return rows

def analyze(plan,root,split):
    """World-paired intervals; model seeds averaged WITHIN each physical world.

    Failure penalty is registered at the maximal all-missed/all-false GOSPA for
    the declared finite target/track counts. Failure-free metrics are separate.
    """
    import numpy as np
    from scipy.stats import t
    records={};failures=[]
    for p in sorted((root/split/'cases').glob('*.json')):
        r=validate_planned_record(plan,root,split,read_json(p))
        ck_arm=r.get('checkpoint_arm');ck_sha=r.get('checkpoint_sha256');records[r['seed'],r['method'],ck_arm,ck_sha]=r
    for p in sorted((root/split/'failures').glob('*.json')):failures.append(read_json(p))
    maps=checkpoint_map(plan,root);checkpoint_shas={k:checkpoint_identity(v)['sha256'] for k,v in maps.items()}
    penalty=plan['model']['metric_cutoff_m']*math.sqrt(plan['model']['max_tracks']);comparisons=[];world_rows=[]
    for seed,cfg in plan_cases(plan,split):
        base=records.get((seed,'numeric_optimizer',None,None));baseline=base['summary']['mean_gospa_m'] if base else penalty
        for arm in plan['backbones']:
            members=[records.get((seed,'numeric_network',arm,checkpoint_shas[f'{arm}/{ms}'])) for ms in plan['model_seeds']]
            score=float(np.mean([r['summary']['mean_gospa_m'] if r else penalty for r in members]))
            world_rows.append({'seed':seed,'scenario':cfg['scenario'],'arm':arm,'baseline_gospa_m':baseline,'network_seed_mean_gospa_m':score,
                               'paired_improvement_m':baseline-score,'baseline_failed':base is None,'failed_model_seeds':sum(r is None for r in members)})
    for arm in plan['backbones']:
        a=np.array([r['paired_improvement_m'] for r in world_rows if r['arm']==arm]);n=len(a);mean=float(a.mean())
        half=float(t.ppf(1-.05/(2*len(ARMS)),n-1)*a.std(ddof=1)/math.sqrt(n)) if n>1 else None
        comparisons.append({'arm':arm,'independent_worlds':n,'mean_paired_improvement_m':mean,'familywise_95_lower_m':mean-half if half is not None else None,
          'familywise_95_upper_m':mean+half if half is not None else None,'model_seeds_are_not_independent_worlds':True,
          'registered_practical_margin_m':plan['analysis']['practical_margin_m'],
          'practical_margin_met':mean-half>plan['analysis']['practical_margin_m'] if half is not None else None})
    answer={'schema':'ain.stage4.world_paired_analysis.v1','source_digest':source_digest(),'split':split,'failure_penalty_gospa_m':penalty,
            'comparisons':comparisons,'world_pairs':world_rows,'failures':failures,
            'physical_results':[{'seed':r['seed'],'method':r['method'],'arm':r.get('checkpoint_arm'),'checkpoint_sha256':r.get('checkpoint_sha256'),
                                 'summary':r['summary']} for r in records.values()],
            'interval':'Student-t world-paired mean, Bonferroni three numeric backbones, model seeds averaged within world',
            'cost_rule':plan['decision_comparison'],'histories_not_reinterpreted':True}
    write_json(root/split/'ANALYSIS.json',answer);return answer

def pack_results(root,dest):
    dest=Path(dest).resolve();root=root.resolve()
    if dest.is_relative_to(root):raise ContractError('ARCHIVE_MUST_BE_OUTSIDE_RESULT_TREE')
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json']
    manifest={'schema':'ain.stage4.server.results.v1','files':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)],
              'interrupted_partial_files':[str(p.relative_to(root)) for p in root.rglob('*.partial')]}
    write_json(root/'RESULT_MANIFEST.json',manifest)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in sorted(set(files+[root/'RESULT_MANIFEST.json'])):z.write(p,p.relative_to(root))
    return {'path':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','dry-run','data','train','development','freeze','confirmation','analyze','pack','all','worker'])
    ap.add_argument('--plan',default=str(DEFAULT_PLAN));ap.add_argument('--output',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--device',default='cpu');ap.add_argument('--index',type=int,default=0);ap.add_argument('--attempt-id',default='direct_worker_call');ap.add_argument('--split',choices=['data','development','confirmation'],default='confirmation');ap.add_argument('--archive')
    args=ap.parse_args(argv);root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    authority_token=None
    receipt={'status':'RUNNING','stage':args.stage,'optimizer_updates_in_driver':0};receiptpath=root/(f'DRIVER_{args.stage}_{args.index}.json' if args.stage=='worker' else f'DRIVER_{args.stage}.json')
    try:
        plan=read_json(args.plan);check_plan(plan);receipt['environment']=preflight(plan['model'],args.device,require_device=args.execute and args.stage in ('train','all'))
        binding={'schema':'ain.stage4.server.run.v1','plan_digest':digest(plan),'source_digest':source_digest(),'dependencies':dependency_contract()}
        bp=root/'SERVER_BINDING.json'
        if bp.exists() and read_json(bp)!=binding:raise ContractError('SERVER_OUTPUT_IDENTITY_MISMATCH')
        if args.stage in ('preflight','dry-run') or not args.execute:
            tasks=task_list(plan,root);receipt.update(status='DRY_RUN',tasks=tasks,task_count=len(tasks),
              planned_world_cycles=sum(t.get('cycles',0) for t in tasks),planned_optimizer_updates=sum(t.get('updates',0) for t in tasks),
              actual_world_cycles=0,actual_optimizer_updates=0)
        else:
            if os.environ.get('RADAR_LOCAL_BUDGET_FILE'):raise ContractError('SERVER_EXECUTION_FORBIDDEN_IN_LOCAL_CHECK_ENVIRONMENT')
            from .work_budget import enable_server
            authority_token=enable_server(digest(plan))
            write_json(bp,binding);write_json(root/'REGISTERED_PLAN.json',plan)
            if args.stage=='worker':receipt['status']='COMPLETE' if worker(plan,root,args.index,args.split,args.device,args.resume,args.attempt_id) else 'FAILED'
            elif args.stage=='pack':receipt.update(status='COMPLETE',archive=pack_results(root,args.archive or str(root)+'.zip'))
            else:
                stages=['data','train','development','freeze','confirmation','analyze'] if args.stage=='all' else [args.stage]
                receipt['evaluation_case_failures']=0
                for stage in stages:
                    if stage in ('data','development','confirmation'):
                        res=run_stage(plan,Path(args.plan).resolve(),root,stage,args.device,args.resume)
                        if stage=='data' and res['status']=='FAILED':raise ContractError('DATA_WORKERS_FAILED')
                        receipt['evaluation_case_failures']+=res.get('failure_count',0)
                    elif stage=='train':train_stage(plan,root,args.device,args.resume)
                    elif stage=='freeze':frozen_confirmation(plan,root,create=True)
                    elif stage=='analyze':
                        analyze(plan,root,'confirmation')
                        from .numeric_replay import build
                        build(sorted((root/'confirmation/cases').glob('*.json')),root/'confirmation/replay.html',max_worlds=1)
                receipt['status']='COMPLETE_WITH_CASE_FAILURES' if receipt['evaluation_case_failures'] else 'COMPLETE'
    except Exception as exc:receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if authority_token is not None:
            from .work_budget import disable_server
            disable_server(authority_token)
        receipt['wall_seconds']=time.perf_counter()-started;write_json(root/'driver_history'/f'{args.stage}_{time.time_ns()}.json',receipt);write_json(receiptpath,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('environment','tasks')},ensure_ascii=False))
    return 2 if receipt['status']=='FAILED' else 0
if __name__=='__main__':raise SystemExit(main())
