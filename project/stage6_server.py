"""Persistent Windows workers, a transactional task queue and source-bound resume.

Each CPU worker runs one world at a time and retains its imports. Two GPU workers
train independent model/seed runs. The coordinator never advances a world or optimizer.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,ctypes,hashlib,json,math,os,platform,shutil,sqlite3,subprocess,sys,time,traceback,uuid,zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'ACTIVE/MAINLINE_RESEARCH/spatial_scheduling'))
_PRIORITY_CACHE={}

def group_info():
    if os.name!='nt':return {'counts':[os.cpu_count() or 1],'placement':'os_default'}
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetActiveProcessorGroupCount.restype=ctypes.c_ushort
    kernel.GetActiveProcessorCount.argtypes=[ctypes.c_ushort];kernel.GetActiveProcessorCount.restype=ctypes.c_uint
    n=kernel.GetActiveProcessorGroupCount()
    return {'counts':[int(kernel.GetActiveProcessorCount(i)) for i in range(n)],'windows_version':platform.version()}

def place_worker(slot):
    info=group_info()
    if os.name!='nt':return info
    class Affinity(ctypes.Structure):_fields_=[('Mask',ctypes.c_size_t),('Group',ctypes.c_ushort),('Reserved',ctypes.c_ushort*3)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype=ctypes.c_void_p
    kernel.SetThreadGroupAffinity.argtypes=[ctypes.c_void_p,ctypes.POINTER(Affinity),ctypes.POINTER(Affinity)]
    kernel.SetThreadGroupAffinity.restype=ctypes.c_int
    group=slot%len(info['counts']);affinity=Affinity((1<<info['counts'][group])-1,group,(ctypes.c_ushort*3)(0,0,0));old=Affinity()
    if not kernel.SetThreadGroupAffinity(kernel.GetCurrentThread(),ctypes.byref(affinity),ctypes.byref(old)):
        raise OSError(ctypes.get_last_error(),'SetThreadGroupAffinity')
    return {**info,'worker_main_thread_group':group,'mask':int(affinity.Mask)}

def hardware(output):
    gpu=[]
    run=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if run.returncode:raise RuntimeError('NVIDIA_SMI_FAILED:'+run.stderr)
    for row in run.stdout.strip().splitlines():
        index,ident,name,total,free,util=[v.strip() for v in row.split(',')]
        gpu.append({'index':int(index),'uuid':ident,'name':name,'total_mib':float(total),'free_mib':float(free),'utilization_percent':float(util)})
    free_ram=None
    if os.name=='nt':
        class Memory(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
        m=Memory();m.length=ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):free_ram=m.available/2**30
    return {'processor_groups':group_info(),'logical_processors':sum(group_info()['counts']),'gpus':gpu,'free_ram_gib':free_ram,
            'free_disk_gib':shutil.disk_usage(output).free/2**30,'captured_ns':time.time_ns()}

def gpu_probe(planpath):
    """Synthetic maximum-shape backward; zero optimizer updates and zero worlds."""
    from spatial_lab.core import read_json
    from spatial_lab.stage6_model import make
    from spatial_lab.feature_state import WIDTH
    import torch
    plan=read_json(planpath);t=plan['training'];torch.set_num_threads(1);torch.cuda.set_device(0)
    arithmetic=float((torch.ones((2,2),device='cuda')@torch.ones((2,2),device='cuda')).sum())
    initial_free,_=torch.cuda.mem_get_info()
    model,cfg=make('integrated_agent','cuda:0',17,t['model']);b=t['world_batch'];n=240;a=64
    x=torch.zeros(b,n,WIDTH,device='cuda');valid=torch.ones(b,n,dtype=torch.bool,device='cuda')
    batch={'x':x,'valid':valid,'roles':torch.zeros(b,n,dtype=torch.long,device='cuda'),'types':torch.zeros(b,n,dtype=torch.long,device='cuda'),
           'adj':torch.ones(b,n,n,dtype=torch.bool,device='cuda'),'action_index':torch.arange(a,device='cuda').expand(b,a),
           'action_valid':torch.ones(b,a,dtype=torch.bool,device='cuda')}
    amp=bool(t.get('bf16',True) and torch.cuda.is_bf16_supported());memory=None;loss=0.
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        for _ in range(t['bptt_steps']):
            out=model.forward_batch(batch,memory);memory=out.next_memory
            values=model.group_values(out,torch.zeros(b,12,2,dtype=torch.long,device='cuda'),torch.zeros(b,12,2,16,device='cuda'),torch.ones(b,12,2,device='cuda'))
            loss=loss+out.value.float().mean()+out.controls.float().mean()+out.proxy.float().mean()+values.float().mean()
    loss.backward();torch.cuda.synchronize()
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters());peak=torch.cuda.max_memory_allocated()
    free,total=torch.cuda.mem_get_info()
    if peak+2*parameter_bytes+2**30>initial_free:raise RuntimeError('NO_OPTIMIZER_MEMORY_MARGIN')
    print(json.dumps({'status':'PASSED','physical_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'name':torch.cuda.get_device_name(0),
       'calculation':arithmetic,'synthetic_nodes':n,'world_batch':b,'bptt_steps':t['bptt_steps'],'precision':'bf16' if amp else 'float32',
       'peak_allocated_bytes':peak,'estimated_peak_with_adam_moments_bytes':peak+2*parameter_bytes,'total_memory_bytes':total,
       'actual_optimizer_updates':0,'actual_world_advances':0}),flush=True)

def configurations(plan,split):
    for i,seed in enumerate(plan['world_splits'][split]):
        scenario=plan['scenarios'][i%len(plan['scenarios'])]
        yield seed,{**plan['model'],**scenario['overrides'],'scenario':scenario['name']}

def check_plan(plan):
    from spatial_lab.core import ContractError,digest
    from spatial_lab.binding import source_digest,FEATURE_BINDING,LABEL_BINDING
    from spatial_lab.preflight import preflight
    if plan.get('schema')!='ain.stage6.plan.v1' or plan['source_digest']!=source_digest() or plan['feature_binding']!=FEATURE_BINDING or plan['label_binding']!=LABEL_BINDING:
        raise ContractError('STAGE6_PLAN_BINDING')
    seen=set()
    for split in ('training','validation','development','confirmation'):
        seeds=plan['world_splits'][split]
        if not seeds or len(set(seeds))!=len(seeds) or seen&set(seeds) or any(type(s)is not int for s in seeds):raise ContractError('WORLD_PARTITION')
        seen.update(seeds)
    if set(plan['backbones'])!={'integrated_agent','deep_sets','set_transformer'}:raise ContractError('BACKBONES')
    for _,cfg in list(configurations(plan,'development'))[:len(plan['scenarios'])]:preflight(cfg)
    t=plan['training']
    if not 1<=t['world_batch']<=64 or t['bptt_steps']!=8 or not 1<=t['epochs']<=100:raise ContractError('TRAINING_DOMAIN')
    return digest(plan)

def task_list(plan,stage):
    jobs=[]
    if stage in ('data','validation_data'):
        split='training' if stage=='data' else 'validation'
        for i,(seed,cfg) in enumerate(configurations(plan,split)):
            method=plan['teacher_policies'][(i//len(plan['scenarios']))%len(plan['teacher_policies'])]
            jobs.append({'kind':'case','stage':stage,'split':split,'seed':seed,'configuration':cfg,'method':method})
    elif stage in ('prepare','validation_prepare'):
        previous='data' if stage=='prepare' else 'validation_data'
        for job in task_list(plan,previous):jobs.append({**job,'kind':'prepare','stage':stage,'raw_stage':previous})
    elif stage=='train':
        jobs=[{'kind':'train','stage':stage,'arm':arm,'model_seed':seed} for arm in plan['backbones'] for seed in plan['model_seeds']]
    elif stage in ('development','confirmation'):
        for seed,cfg in configurations(plan,stage):
            for method in ('fixed_analytic','numeric_optimizer'):
                jobs.append({'kind':'case','stage':stage,'split':stage,'seed':seed,'configuration':cfg,'method':method})
            for arm in plan['backbones']:
                for ms in plan['model_seeds']:
                    for method in ('fixed_network','numeric_network'):
                        jobs.append({'kind':'case','stage':stage,'split':stage,'seed':seed,'configuration':cfg,'method':method,'arm':arm,'model_seed':ms})
    else:raise ValueError('UNKNOWN_STAGE:'+stage)
    if stage in ('development','confirmation'):
        jobs.sort(key=lambda j:(j['method'],j.get('arm',''),j.get('model_seed',0),j['seed']))
    return jobs

def tag(job):return str(job['seed'])+'_'+job['method']+('_'+job['arm']+'_'+str(job['model_seed']) if 'arm' in job else '')

def perform(plan,out,job,device,resume):
    from spatial_lab.core import write_json,read_json,digest
    from spatial_lab.binding import source_digest,policy_configuration
    from spatial_lab.stage6_dataset import prepare_world,sha
    if job['kind']=='train':
        from spatial_lab.stage6_training import train
        allowed={digest(policy_configuration(c,'numeric_network')):policy_configuration(c,'numeric_network') for split in plan['world_splits'] for _,c in configurations(plan,split)}
        return train(out/'prepare',out/'validation_prepare',out/'models'/job['arm']/str(job['model_seed']),plan['training'],job['arm'],job['model_seed'],device,
                     execute=True,resume=resume,development_seeds=plan['world_splits']['development'],allowed_configurations=allowed)
    if job['kind']=='prepare':
        return prepare_world(out/job['raw_stage']/'cases'/(tag(job)+'.json.gz'),out/job['stage']/(str(job['seed'])+'.pt'))
    from spatial_lab.run import run_case
    from spatial_lab.case_identity import checkpoint_identity,case_identity,validate_cached,seal_result
    from spatial_lab.spatial_graph import Priority
    path=out/job['stage']/'cases'/(tag(job)+'.json.gz');ck=out/'models'/job['arm']/str(job['model_seed'])/'best.pt' if 'arm'in job else None
    ident=case_identity(job['seed'],job['method'],job['configuration'],checkpoint_identity(ck))
    if path.exists():
        if not resume:raise RuntimeError('EXISTING_CASE_REQUIRES_RESUME')
        result=validate_cached(read_json(path),ident)
    else:
        learned=None
        if ck:
            key=(str(ck),ident['checkpoint']['sha256'],job['split'])
            if key not in _PRIORITY_CACHE:
                _PRIORITY_CACHE.clear();_PRIORITY_CACHE[key]=Priority(ck,device='cpu',evaluation_split=job['split'])
            learned=_PRIORITY_CACHE[key]
        partial=out/job['stage']/'partial'/(tag(job)+'.json');state=Path(str(partial)+'.state.json.gz')
        result=run_case(job['seed'],job['configuration'],job['method'],learned,partial,state if resume and state.exists() else None)
        result['evaluation_split']=job['split'];result=seal_result(result,ident);write_json(path,result)
    summary={'status':'COMPLETE','job':job,'summary':result['summary'],'case_digest':result['content_digest'],'source_digest':source_digest(),'case_file_sha256':sha(path)}
    write_json(out/job['stage']/'summaries'/(tag(job)+'.json'),summary)
    return {'path':str(path.relative_to(out)),'sha256':summary['case_file_sha256'],'case_digest':result['content_digest']}

def connect(path):
    db=sqlite3.connect(path,timeout=120);db.execute('PRAGMA busy_timeout=120000');return db

def pool_progress(stage,jobs,counts,processes,elapsed_s):
    """Separate launched processes, live processes and experiment health."""
    alive=sum(p.poll() is None for p in processes)
    status=('DRAINING_AFTER_FAILURE' if counts.get('FAILED',0) else
            'STARTING' if not counts.get('RUNNING',0) and not counts.get('COMPLETE',0) else 'RUNNING')
    return {'stage':stage,'status':status,'jobs':jobs,'counts':counts,
            'workers':len(processes),'workers_started':len(processes),'workers_alive':alive,
            'elapsed_s':elapsed_s,'updated_ns':time.time_ns()}

def worker(args):
    from spatial_lab.core import read_json,write_json,digest
    from spatial_lab.work_budget import enable_server,disable_server
    from spatial_lab.binding import source_digest
    plan=read_json(args.plan);check_plan(plan);out=Path(args.output);place=place_worker(args.slot)
    device='cuda:0' if args.gpu else 'cpu'
    if args.gpu:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU_WORKER_UNAVAILABLE')
        torch.cuda.set_device(0)
    db=connect(args.queue);token=enable_server(digest(plan));worker_id=f'{os.getpid()}_{uuid.uuid4().hex}'
    write_json(out/'CONTROL/workers'/(worker_id+'.json'),{'pid':os.getpid(),'group_placement':place,'device':device,'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_digest':source_digest()})
    try:
        while True:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM tasks WHERE status='FAILED' LIMIT 1").fetchone():db.commit();break
            row=db.execute("SELECT id,payload FROM tasks WHERE status='PENDING' ORDER BY id LIMIT 1").fetchone()
            if row is None:db.commit();break
            index,payload=row;db.execute("UPDATE tasks SET status='RUNNING',worker=?,started=? WHERE id=?",(worker_id,time.time_ns(),index));db.commit()
            job=json.loads(payload);start=time.monotonic();cpu_start=time.process_time()
            try:
                result=perform(plan,out,job,device,args.resume);receipt={'status':'COMPLETE','result':result}
            except BaseException as exc:receipt={'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc()}
            receipt.update(job=job,index=index,worker=worker_id,source_digest=source_digest(),wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu_start,completed_ns=time.time_ns())
            write_json(out/job['stage']/'receipt_history'/f'{index:06d}_{receipt["completed_ns"]}.json',receipt)
            dest=out/job['stage']/'receipts'/f'{index:06d}.json';write_json(dest,receipt)
            db.execute('UPDATE tasks SET status=?,receipt=? WHERE id=?',(receipt['status'],str(dest.relative_to(out)),index));db.commit()
            if receipt['status']=='FAILED':break
    finally:disable_server(token);db.close()

@contextmanager
def coordinator_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)

def run_stage(plan,planpath,out,stage,resume,resources):
    from spatial_lab.core import write_json,read_json,digest
    from spatial_lab.binding import source_digest
    jobs=task_list(plan,stage);queue=out/'CONTROL'/f'{stage}.sqlite';db=connect(queue)
    db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS meta (identity TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY,payload TEXT,status TEXT,worker TEXT,started INTEGER,receipt TEXT)')
    identity=digest({'source_digest':source_digest(),'plan':digest(plan),'stage':stage,'jobs':jobs})
    old=db.execute('SELECT identity FROM meta').fetchone()
    if old:
        if old[0]!=identity or not resume:raise RuntimeError('QUEUE_IDENTITY_OR_RESUME_REQUIRED')
        # The coordinator owns all its workers and waits for them on interruption.
        # An orphan still using this queue is an explicit blocker, never overwritten.
        active=db.execute("SELECT DISTINCT worker FROM tasks WHERE status='RUNNING'").fetchall()
        for (owner,) in active:
            pid=int(owner.split('_')[0])
            if os.name=='nt':
                kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.restype=ctypes.c_void_p
                handle=kernel.OpenProcess(0x1000,False,pid)
                if handle:kernel.CloseHandle(ctypes.c_void_p(handle));raise RuntimeError('OLD_WORKER_STILL_ALIVE:'+str(pid))
        db.execute("UPDATE tasks SET status='PENDING',worker=NULL WHERE status!='COMPLETE'")
    else:
        db.execute('INSERT INTO meta VALUES (?)',(identity,))
        db.executemany('INSERT INTO tasks(id,payload,status) VALUES (?,?,?)',[(i,json.dumps(j),'PENDING') for i,j in enumerate(jobs)])
    db.commit();profile=plan['resources'];cpu=sum(resources['processor_groups']['counts'])
    if stage=='train':slots=profile['gpu_indices']
    else:
        requested=profile['preparation_workers'] if 'prepare' in stage else profile['evaluation_workers'] if stage in ('development','confirmation') else profile['data_workers']
        slots=list(range(min(requested,max(1,cpu-profile['reserved_cpu_threads']),len(jobs))))
    running=[];logs=[];start=time.monotonic();last_print=0
    try:
        pending=db.execute("SELECT count(*) FROM tasks WHERE status='PENDING'").fetchone()[0]
        for i,slot in enumerate(slots[:pending]):
            logdir=out/stage/'logs';logdir.mkdir(parents=True,exist_ok=True);log=(logdir/f'{time.time_ns()}_slot_{slot}.txt').open('w',encoding='utf-8');logs.append(log)
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'stage6.py'),'worker','--plan',str(planpath),'--output',str(out),'--queue',str(queue),'--slot',str(i),'--execute']
            if resume:command.append('--resume')
            env=dict(os.environ);env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            if stage=='train':
                env['CUDA_VISIBLE_DEVICES']=next(g['uuid'] for g in resources['gpus'] if g['index']==slot)
                env['CUDA_DEVICE_ORDER']='PCI_BUS_ID';command.append('--gpu')
            else:env['CUDA_VISIBLE_DEVICES']=''
            running.append(subprocess.Popen(command,env=env,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        while True:
            counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())
            row=pool_progress(stage,len(jobs),counts,running,time.monotonic()-start)
            if counts.get('FAILED',0):
                failure=db.execute("SELECT id,receipt FROM tasks WHERE status='FAILED' ORDER BY started,id LIMIT 1").fetchone()
                failed=read_json(out/failure[1])
                row['failure_example']={'job_index':failure[0],'error':failed.get('error'),'receipt':failure[1]}
            write_json(out/stage/'STAGE_RECEIPT.json',row)
            if time.monotonic()-last_print>=30:print(json.dumps(row),flush=True);last_print=time.monotonic()
            if all(p.poll() is not None for p in running):break
            time.sleep(3)
        codes=[p.returncode for p in running];counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())
        complete=counts.get('COMPLETE',0)==len(jobs) and all(c==0 for c in codes)
        row.update(status='COMPLETE' if complete else 'FAILED',counts=counts,worker_returncodes=codes,elapsed_s=time.monotonic()-start)
        write_json(out/stage/'STAGE_RECEIPT.json',row)
        if not complete:raise RuntimeError('STAGE_FAILED:'+stage+': read '+str(out/stage/'receipts')+' and logs')
        if 'prepare' in stage:
            from spatial_lab.stage6_dataset import seal_index
            seal_index(out/stage,[read_json(out/stage/'receipts'/f'{i:06d}.json')['result'] for i in range(len(jobs))])
        if stage in ('development','confirmation'):summarize(out,stage)
        return row
    finally:
        # Only the exact child processes created by this coordinator are stopped.
        for p in running:
            if p.poll() is None:p.terminate()
        for p in running:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()
        db.close()

def summarize(out,stage):
    from spatial_lab.core import read_json,write_json
    rows=[read_json(p) for p in sorted((out/stage/'summaries').glob('*.json'))];groups={}
    for row in rows:
        j=row['job'];key=j['method']+('/'+j['arm']+'/'+str(j['model_seed']) if 'arm' in j else '')
        groups.setdefault(key,[]).append(row)
    result={}
    for key,items in groups.items():
        fields=set().union(*(r['summary'] for r in items));metrics={}
        for f in fields:
            values=[r['summary'].get(f) for r in items];valid=[v for v in values if type(v) in (int,float) and math.isfinite(v)]
            if valid:metrics[f]={'mean':sum(valid)/len(valid),'samples':len(valid),'missing':len(values)-len(valid)}
        result[key]={'worlds':len(items),'metrics':metrics}
    baseline={r['job']['seed']:r['summary'] for r in groups.get('numeric_optimizer',[])};paired={}
    for key,items in groups.items():
        deltas={}
        for row in items:
            for name,val in row['summary'].items():
                ref=baseline.get(row['job']['seed'],{}).get(name)
                if type(val) in (float,int) and type(ref) in (float,int) and math.isfinite(val) and math.isfinite(ref):deltas.setdefault(name,[]).append(val-ref)
        paired[key]={k:{'mean_delta':sum(v)/len(v),'pairs':len(v),'standard_error':math.sqrt(sum((x-sum(v)/len(v))**2 for x in v)/(len(v)-1)/len(v)) if len(v)>1 else None} for k,v in deltas.items()}
    write_json(out/stage/'SUMMARY.json',{'groups':result,'paired_against_numeric_optimizer':paired,'direction':'method minus reference; per metric units and direction retained','timing':'concurrent host wall time, not isolated latency'})

def freeze(plan,out,create=False):
    from spatial_lab.core import read_json,write_json,digest
    from spatial_lab.binding import source_digest,verify_checkpoint
    from spatial_lab.stage6_dataset import sha
    import torch
    for stage in ('data','validation_data','prepare','validation_prepare','train','development'):
        if read_json(out/stage/'STAGE_RECEIPT.json')['status']!='COMPLETE':raise RuntimeError('INCOMPLETE_FREEZE_STAGE:'+stage)
    checkpoints={}
    for arm in plan['backbones']:
        for seed in plan['model_seeds']:
            directory=out/'models'/arm/str(seed);receipt=read_json(directory/'TRAIN_RECEIPT.json')
            if receipt['status']!='COMPLETE' or receipt['optimizer_updates']!=receipt['target_updates']:raise RuntimeError('TRAINING_NOT_COMPLETE')
            ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=True);verify_checkpoint(ck)
            if ck['training_purpose']!='registered_server_training':raise RuntimeError('DIAGNOSTIC_CHECKPOINT')
            if (ck['arm']!=arm or ck['model_seed']!=seed or ck['training_configuration']!=plan['training'] or
                set(ck['training_law_seeds'])!=set(plan['world_splits']['training']) or
                set(ck['validation_law_seeds'])!=set(plan['world_splits']['validation']) or
                set(ck['development_law_seeds'])!=set(plan['world_splits']['development']) or
                ck['best_validation']!=receipt['best_validation']):raise RuntimeError('BEST_CHECKPOINT_PLAN_IDENTITY')
            checkpoints[arm+'/'+str(seed)]={'sha256':sha(directory/'best.pt'),'epoch':ck['best_validation']['epoch'],'update':ck['optimizer_updates']}
    body={'schema':'ain.stage6.freeze.v1','source_digest':source_digest(),'plan_digest':digest(plan),'checkpoints':checkpoints,
          'selection':'minimum held-out validation total loss at completed training epochs; earliest epoch wins ties',
          'development_summary_sha256':sha(out/'development/SUMMARY.json'),'confirmation_worlds':plan['world_splits']['confirmation']}
    path=out/'FROZEN_CONFIRMATION.json'
    if path.exists():
        if read_json(path)!=body:raise RuntimeError('CONFIRMATION_FREEZE_CHANGED')
    elif create:
        if (out/'confirmation').exists():raise RuntimeError('CONFIRMATION_ALREADY_ACCESSED')
        write_json(path,body)
    else:raise RuntimeError('FREEZE_REQUIRED')
    return body

def pack(out,archive,full=False):
    from spatial_lab.core import atomic_replace,write_json
    from spatial_lab.stage6_dataset import sha
    # Snapshot only closed scientific artifacts; no live queue, locks, temp or restart files.
    files=[]
    for p in out.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(out);parts=rel.parts
        if 'CONTROL' in parts or 'partial' in parts or any(x in parts for x in ('prepare','validation_prepare')) or p.name.endswith('.partial'):continue
        if not full and ('cases' in parts or parts[0] in ('data','validation_data')):continue
        if p.name=='latest.pt':continue
        files.append(p)
    archive=Path(archive)
    if archive.resolve().is_relative_to(out.resolve()):raise RuntimeError('ARCHIVE_MUST_BE_OUTSIDE_OUTPUT')
    archive.parent.mkdir(parents=True,exist_ok=True);temp=archive.with_name(archive.name+'.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        manifest=[]
        for p in files:
            rel=p.relative_to(out).as_posix();z.write(p,rel);manifest.append({'path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2))
    atomic_replace(temp,archive);return {'archive':str(archive),'sha256':sha(archive),'files':len(files),'full':full}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','develop','data','train','development','freeze','confirmation','pack','worker'])
    ap.add_argument('--plan',default=str(ROOT/'STAGE6_SERVER/plan_12800.json'));ap.add_argument('--output',default=r'D:\LXR\SERVER_STAGE6_12800')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--queue');ap.add_argument('--slot',type=int,default=0);ap.add_argument('--gpu',action='store_true')
    ap.add_argument('--archive');ap.add_argument('--full',action='store_true');args=ap.parse_args()
    from spatial_lab.core import read_json,write_json,digest
    from spatial_lab.binding import source_digest,dependency_contract
    if args.stage=='worker':
        if not args.execute:raise RuntimeError('EXECUTE_REQUIRED')
        worker(args);return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);plan=read_json(args.plan);result={'stage':args.stage,'status':'RUNNING','optimizer_updates_in_coordinator':0,'world_advances_in_coordinator':0}
    try:
        plan_digest=check_plan(plan);result.update(source_digest=source_digest(),plan_digest=plan_digest)
        if args.stage=='pack':
            with coordinator_lock(out/'CONTROL/coordinator.lock'):result.update(pack(out,args.archive or str(out)+'_RESULTS.zip',args.full),status='COMPLETE')
            print(json.dumps(result));return
        if args.stage=='freeze':result.update(freeze=freeze(plan,out,True),status='COMPLETE');print(json.dumps(result));return
        inv=hardware(out);write_json(out/'CONTROL/HARDWARE_LATEST.json',inv)
        profile=plan['resources'];selected=[g for g in inv['gpus'] if g['index'] in profile['gpu_indices']]
        if args.stage in ('preflight','develop','train'):
            if len(selected)!=len(profile['gpu_indices']):raise RuntimeError('REQUESTED_GPUS_NOT_PRESENT')
            for g in selected:
                if g['free_mib']<profile['minimum_free_gpu_gib']*1024:raise RuntimeError('INSUFFICIENT_FREE_GPU_MEMORY:'+str(g['index']))
        if inv['free_ram_gib'] is not None and inv['free_ram_gib']<profile['minimum_free_ram_gib']:raise RuntimeError('INSUFFICIENT_FREE_RAM')
        if inv['free_disk_gib']<profile['minimum_free_disk_gib']:raise RuntimeError('INSUFFICIENT_FREE_DISK')
        binding={'plan':plan,'plan_digest':plan_digest,'source_digest':source_digest(),'dependencies':dependency_contract(),'gpu_uuids':{str(g['index']):g['uuid'] for g in selected}}
        target=out/'RUN_BINDING.json'
        if target.exists() and read_json(target)!=binding:raise RuntimeError('OUTPUT_BINDING_CHANGED_USE_NEW_DIRECTORY')
        if not target.exists():write_json(target,binding)
        if args.stage=='preflight':
            checks=[]
            for g in selected:
                env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=g['uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
                code='import sys;sys.path.insert(0,sys.argv[1]);from stage6_server import gpu_probe;gpu_probe(sys.argv[2])'
                probe=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(ROOT),str(Path(args.plan).resolve())],env=env,capture_output=True,text=True,
                                     timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if probe.returncode:raise RuntimeError('GPU_SHAPE_PROBE_FAILED:'+str(g['index'])+':'+probe.stdout+probe.stderr)
                check=json.loads(probe.stdout.strip().splitlines()[-1]);checks.append({'index':g['index'],**check})
            result.update(status='PREFLIGHT_PASSED',hardware=inv,gpu_arithmetic_checks=checks,planned_jobs={s:len(task_list(plan,s)) for s in ('data','validation_data','train','development','confirmation')})
        else:
            if not args.execute:raise RuntimeError('EXECUTE_REQUIRED')
            stages=['data','validation_data','prepare','validation_prepare','train','development'] if args.stage=='develop' else ['data','validation_data','prepare','validation_prepare'] if args.stage=='data' else [args.stage]
            with coordinator_lock(out/'CONTROL/coordinator.lock'):
                if args.stage=='confirmation':freeze(plan,out)
                result['stages']={s:run_stage(plan,Path(args.plan).resolve(),out,s,args.resume,inv) for s in stages}
            result['status']='COMPLETE'
    except BaseException as exc:result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    write_json(out/'CONTROL/driver_history'/f'{args.stage}_{time.time_ns()}.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['status']=='FAILED':sys.exit(2)

if __name__=='__main__':main()
