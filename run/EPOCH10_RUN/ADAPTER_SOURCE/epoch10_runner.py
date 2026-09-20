"""Resume nine independent Stage6 jobs on GPU0 x2 / GPU1 x3, through epoch 10.

Own operational receipts describe this shorter run. Original plan, data, source,
queue history, checkpoints and learning-rate configuration retain their identity.
"""
from pathlib import Path
from collections import Counter
import argparse,ctypes,hashlib,json,math,os,re,subprocess,sys,time,traceback,uuid

HERE=Path(__file__).resolve().parent
STOP_EPOCH=10
GPU_COUNTS={0:2,1:3}
RESERVE_GIB=14.
ALLOCATOR_LIMIT_GIB=12.5
CARD_MARGIN_GIB=2.

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def verify_package():
    profile=read(HERE/'EPOCH10_PROFILE.json')
    for name,expected in profile['files'].items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or file_sha(path)!=expected:
            raise RuntimeError('EPOCH10_PACKAGE_CONTENT_CHANGED:'+name)
    return profile

def bootstrap(project):
    profile=verify_package()
    sys.path.insert(0,str(HERE/'input_adapter'))
    import stage6_input_runner as transport
    controller,input_profile=transport.verify_adapter(project)
    return controller,transport,input_profile,profile

def load_context(project,output,planpath):
    controller,transport,input_profile,profile=bootstrap(project)
    from spatial_lab.core import digest
    from spatial_lab.stage6_dataset import Worlds
    from spatial_lab.binding import policy_configuration
    plan=read(planpath);plan_digest=controller.check_plan(plan)
    original=read(Path(output)/'RUN_BINDING.json')
    if original.get('plan_digest')!=plan_digest or original.get('plan')!=plan:
        raise RuntimeError('EXISTING_RUN_PLAN_DIFFERS')
    tr=Worlds(Path(output)/'prepare');va=Worlds(Path(output)/'validation_prepare')
    if set(tr.world_seeds)!=set(plan['world_splits']['training']) or set(va.world_seeds)!=set(plan['world_splits']['validation']):
        raise RuntimeError('PREPARED_WORLD_SPLITS_DIFFER')
    cycles={r['contexts'] for r in tr.entries+va.entries}
    if len(cycles)!=1:raise RuntimeError('PREPARED_SEQUENCE_LENGTHS_DIFFER')
    t=plan['training'];per_epoch=math.ceil(len(tr)/t['world_batch'])*math.ceil(cycles.pop()/t['bptt_steps'])
    if per_epoch!=3200 or t['epochs']!=20 or len(tr)!=12800 or len(va)!=128:
        raise RuntimeError('THIS_ADAPTER_REQUIRES_THE_EXISTING_12800_WORLD_20_EPOCH_PLAN')
    allowed={digest(policy_configuration(c,'numeric_network')):policy_configuration(c,'numeric_network')
             for split in plan['world_splits'] for _,c in controller.configurations(plan,split)}
    binding={'schema':'ain.stage6.epoch_stop.execution.v1','base_source_digest':plan['source_digest'],
             'original_plan_digest':plan_digest,'original_run_binding_digest':digest(original),
             'profile_digest':digest(profile),'input_transport_profile_digest':input_profile['execution_profile_digest'],
             'stop_epoch':STOP_EPOCH,'stop_updates':STOP_EPOCH*per_epoch,'updates_per_epoch':per_epoch,
             'original_lr_schedule_epochs':t['epochs'],'original_lr_schedule_updates':t['epochs']*per_epoch,
             'maximum_workers_per_physical_gpu':{'0':2,'1':3},'checkpoint_selection':'fixed_epoch_10',
             'prepare_index_sha256':file_sha(Path(output)/'prepare/INDEX.json'),
             'validation_index_sha256':file_sha(Path(output)/'validation_prepare/INDEX.json')}
    return controller,transport,input_profile,profile,plan,tr,va,allowed,binding

def verify_job_checkpoint(ck,job,plan,tr,va,allowed):
    from spatial_lab.binding import verify_checkpoint
    from spatial_lab.core import digest
    verify_checkpoint(ck)
    if (ck['arm']!=job['arm'] or ck['model_seed']!=job['model_seed'] or
        ck['training_configuration']!=plan['training'] or ck['training_configuration_digest']!=digest(plan['training']) or
        ck['dataset_manifest']!=tr.manifest or ck['validation_manifest']!=va.manifest or
        ck['development_law_seeds']!=plan['world_splits']['development'] or ck['allowed_policy_configurations']!=allowed):
        raise RuntimeError('JOB_CHECKPOINT_IDENTITY:'+job['arm']+'/'+str(job['model_seed']))

def model_dir(output,job):return Path(output)/'models'/job['arm']/str(job['model_seed'])

def inspect_job(output,job,plan,tr,va,allowed,binding):
    import torch
    from epoch10_snapshot import epoch_result
    dest=model_dir(output,job);latest=None;target=binding['stop_updates']
    for name in ('epoch_010.pt','latest.pt','best.pt'):
        path=dest/name
        if not path.exists():continue
        ck=torch.load(path,map_location='cpu',weights_only=True)
        verify_job_checkpoint(ck,job,plan,tr,va,allowed)
        if name=='latest.pt':latest={'updates':ck['optimizer_updates'],'cursor':ck.get('cursor')}
        if ck['optimizer_updates']==target:
            try:epoch_result(ck,STOP_EPOCH,target)
            except Exception:
                if name!='latest.pt':raise
            else:return {'status':'EXACT_EPOCH_AVAILABLE','checkpoint':str(path),'saved_updates':target}
        elif name=='epoch_010.pt':raise RuntimeError('FIXED_EPOCH_FILE_HAS_WRONG_UPDATES')
        del ck
    if latest and latest['updates']>target:
        raise RuntimeError('EXACT_EPOCH_10_NOT_RETAINED:'+str(dest)+':latest='+str(latest['updates'])+
                           '; existing later weights were preserved; no earlier weights can be reconstructed from them.')
    if latest:
        c=latest['cursor']
        if not c or c['epoch']>STOP_EPOCH:raise RuntimeError('RESUME_CURSOR_INVALID:'+str(dest))
        return {'status':'RESUME','saved_updates':latest['updates'],'cursor':c}
    if (dest/'best.pt').exists():raise RuntimeError('LATEST_RESUME_CHECKPOINT_MISSING:'+str(dest))
    return {'status':'NEW','saved_updates':0}

def slot_layout():
    # Preserve the two existing runs' usual GPU mapping for the first two slots.
    return [{'slot':i,'gpu':g} for i,g in enumerate([0,1,0,1,1])]

def check_memory(inventory,pending_jobs,original):
    gpus={g['index']:g for g in inventory['gpus']}
    old_uuids=original.get('gpu_uuids',{})
    needed=Counter(s['gpu'] for s in slot_layout()[:min(5,pending_jobs)])
    checks=[]
    for index,count in needed.items():
        if index not in gpus:raise RuntimeError('REQUESTED_GPU_MISSING:'+str(index))
        g=gpus[index];required=count*RESERVE_GIB+CARD_MARGIN_GIB
        if old_uuids.get(str(index))!=g['uuid']:raise RuntimeError('GPU_UUID_CHANGED:'+str(index))
        if g['free_mib']/1024 < required:
            raise RuntimeError(f'GPU_{index}_FREE_MEMORY_{g["free_mib"]/1024:.2f}_GiB_BELOW_{required:.2f}_GiB; stop old radar workers first or wait for other workloads to release memory.')
        checks.append({'gpu':index,'uuid':g['uuid'],'workers':count,'free_gib':g['free_mib']/1024,'required_gib':required})
    if needed and inventory['free_ram_gib'] is not None and inventory['free_ram_gib']<64:
        raise RuntimeError('TRAINING_REQUIRES_64_GIB_FREE_HOST_MEMORY')
    if needed and inventory['free_disk_gib']<10:raise RuntimeError('TRAINING_REQUIRES_10_GIB_FREE_DISK')
    return checks

def process_alive(pid):
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return ctypes.get_last_error()==5
        kernel.CloseHandle(handle);return True
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False

def command_belongs_to_run(command,output):
    command=(command or '').replace('/','\\')
    root=str(Path(output).resolve()).replace('/','\\').rstrip('\\')
    return bool(re.search(r'(?:stage6_input_runner|stage6_server|stage6|epoch10_runner)\.py(?:"|\s)',command,re.I) and
                re.search(re.escape(root)+r'(?=[\\"\s]|$)',command,re.I))

def check_old_workers(output):
    # The OS lock protects the coordinator. Inspect the recorded child PIDs
    # as well, including children of an abnormally terminated new coordinator.
    import sqlite3
    queue=Path(output)/'CONTROL/train.sqlite'
    candidates=set()
    if queue.exists():
        with sqlite3.connect(queue.as_uri()+'?mode=ro',uri=True,timeout=10) as db:
            for status,worker in db.execute("SELECT status,worker FROM tasks WHERE status='RUNNING'"):
                if worker:candidates.add(int(worker.split('_')[0]))
    statuspath=Path(output)/'EPOCH10_RUN/STATUS.json'
    if statuspath.exists():
        candidates.update(x['pid'] for x in read(statuspath).get('active',[]) if x.get('pid'))
    for path in (Path(output)/'EPOCH10_RUN/receipts').glob('*.json'):
        receipt=read(path)
        if receipt.get('status')=='RUNNING' and receipt.get('pid'):candidates.add(receipt['pid'])
    candidates.discard(os.getpid())
    live=[pid for pid in candidates if process_alive(pid)]
    if live and os.name=='nt':
        query="@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
        result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,
            text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
        if result.returncode:raise RuntimeError('CANNOT_CHECK_RECORDED_WORKER_PROCESS_IDENTITY:'+result.stderr)
        rows=json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(rows,dict):rows=[rows]
        commands={int(row['ProcessId']):row.get('CommandLine') for row in rows}
        # A recycled PID belonging to another application is not our worker.
        live=[pid for pid in live if pid in commands and command_belongs_to_run(commands[pid],output)]
    if live:raise RuntimeError('OLD_RADAR_WORKERS_STILL_ALIVE:'+str(sorted(set(live))))

def bind_execution(output,binding,profile):
    from spatial_lab.core import write_json,atomic_save
    root=Path(output)/'EPOCH10_RUN';path=root/'EXECUTION_BINDING.json'
    if path.exists() and read(path)!=binding:raise RuntimeError('EPOCH10_EXECUTION_BINDING_CHANGED')
    if not path.exists():write_json(path,binding)
    for name in [*profile['files'],'EPOCH10_PROFILE.json']:
        src=HERE/name;dest=root/'ADAPTER_SOURCE'/name
        if dest.exists() and file_sha(dest)!=file_sha(src):raise RuntimeError('EPOCH10_RECORDED_SOURCE_CHANGED:'+name)
        if not dest.exists():atomic_save(dest,lambda stream,src=src:stream.write(src.read_bytes()))

def install_epoch_loop(controller,transport,input_profile,project,binding):
    import spatial_lab.stage6_training as training
    import epoch10_training as bounded
    original_train=training.train
    def bridge(*args,**kwargs):
        # The original input adapter installs these three hooks during perform.
        bounded.loader=training.loader;bounded.write_json=training.write_json;bounded.atomic_save=training.atomic_save
        return bounded.train(*args,**kwargs,stop_epoch=STOP_EPOCH,execution=binding)
    training.train=bridge
    transport.install_adapter(controller,input_profile,project)
    return original_train

def execute_worker(specpath):
    spec=read(specpath)
    if os.environ.get('RADAR_EPOCH10_RUN_TOKEN')!=spec['run_token']:raise RuntimeError('WORKER_REQUIRES_PARENT_RUN_TOKEN')
    project=Path(spec['project']);out=Path(spec['output'])
    controller,transport,input_profile,profile,plan,tr,va,allowed,binding=load_context(project,out,Path(spec['plan']))
    from spatial_lab.core import write_json
    from spatial_lab.work_budget import enable_server,disable_server
    if binding!=spec['binding']:raise RuntimeError('WORKER_BINDING_CHANGED')
    path=Path(spec['receipt']);job=spec['job'];receipt={'status':'RUNNING','pid':os.getpid(),'job':job,'slot':spec['slot'],
        'attempt_id':spec['attempt_id'],'started_ns':time.time_ns(),'binding':binding}
    write_json(path,receipt)
    token=None
    try:
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['gpu_uuid']:raise RuntimeError('WORKER_GPU_BINDING')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('WORKER_REQUIRES_ONE_VISIBLE_GPU')
        torch.cuda.set_device(0);controller.place_worker(spec['slot']['slot'])
        total=torch.cuda.get_device_properties(0).total_memory
        # Bound each PyTorch caching allocator; CUDA context memory is separate.
        torch.cuda.set_per_process_memory_fraction(min(1.,ALLOCATOR_LIMIT_GIB*2**30/total),0)
        state=inspect_job(out,job,plan,tr,va,allowed,binding)
        if state['status']=='EXACT_EPOCH_AVAILABLE':
            from epoch10_snapshot import save_snapshot
            ck=torch.load(state['checkpoint'],map_location='cpu',weights_only=True)
            result=save_snapshot(ck,model_dir(out,job),STOP_EPOCH,binding['stop_updates'],binding)
        else:
            install_epoch_loop(controller,transport,input_profile,project,binding)
            token=enable_server(binding['original_plan_digest'])
            result=controller.perform(plan,out,job,'cuda:0',True)
        if result['status']!='COMPLETE' or result['optimizer_updates']!=binding['stop_updates']:
            raise RuntimeError('WORKER_DID_NOT_REACH_FIXED_EPOCH')
        receipt.update(status='COMPLETE',result=result)
    except BaseException as exc:
        receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if token is not None:disable_server(token)
        receipt['finished_ns']=time.time_ns();write_json(path,receipt)
    return 0 if receipt['status']=='COMPLETE' else 2

def run_pool(jobs,slots,start,finish,progress,poll_s=3.):
    """Each queued job owns one process and one unique model directory."""
    pending=list(jobs);active={};done=[];failed=[]
    try:
        while pending or active:
            for slot in slots:
                key=slot['slot']
                if key not in active and pending and not failed:
                    job=pending.pop(0);handle=start(job,slot);active[key]=(job,slot,handle)
            progress(pending,list(active.values()),done,failed)
            for key,(job,slot,handle) in list(active.items()):
                code=handle['process'].poll()
                if code is None:continue
                ok=finish(job,slot,handle,code)
                (done if ok else failed).append(job);del active[key]
            if failed and not active:break
            if pending or active:time.sleep(poll_s)
        return {'done':done,'failed':failed,'pending':pending}
    finally:
        # Only children created by this invocation are terminated.
        for _,_,handle in active.values():
            proc=handle['process']
            if proc.poll() is None:proc.terminate()
        for _,_,handle in active.values():
            proc=handle['process']
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
            for f in handle.get('streams',[]):f.close()

def run_coordinator(args,context):
    controller,transport,input_profile,profile,plan,tr,va,allowed,binding=context
    from spatial_lab.core import write_json
    from epoch10_snapshot import save_snapshot
    import torch
    out=Path(args.output);root=out/'EPOCH10_RUN';joblist=controller.task_list(plan,'train')
    with controller.coordinator_lock(out/'CONTROL/coordinator.lock'):
        check_old_workers(out)
        states=[inspect_job(out,j,plan,tr,va,allowed,binding) for j in joblist]
        pending=[(i,j) for i,(j,s) in enumerate(zip(joblist,states)) if s['status']!='EXACT_EPOCH_AVAILABLE']
        inventory=controller.hardware(out)
        memory=check_memory(inventory,len(pending),read(out/'RUN_BINDING.json'))
        bind_execution(out,binding,profile)
        write_json(root/'HARDWARE_START.json',{'inventory':inventory,'memory_admission':memory})
        run_id=uuid.uuid4().hex;run_token=uuid.uuid4().hex;started=time.monotonic();last_print=0.;completed=[]
        for i,(job,state) in enumerate(zip(joblist,states)):
            if state['status']!='EXACT_EPOCH_AVAILABLE':continue
            ck=torch.load(state['checkpoint'],map_location='cpu',weights_only=True)
            result=save_snapshot(ck,model_dir(out,job),STOP_EPOCH,binding['stop_updates'],binding);del ck
            write_json(root/'receipts'/f'{i:02d}.json',{'status':'COMPLETE','job':job,'result':result,'recovered_exact_checkpoint':True})
            completed.append((i,job))
        gpu_by_index={g['index']:g for g in inventory['gpus']}
        def start(item,slot):
            index,job=item;attempt=uuid.uuid4().hex
            attempt_path=root/'attempts'/f'{index:02d}_{attempt}.json';specpath=root/'jobs'/f'{index:02d}_{attempt}.json'
            spec={'project':str(Path(args.project).resolve()),'output':str(out.resolve()),'plan':str(Path(args.plan).resolve()),
                  'job':job,'slot':slot,'gpu_uuid':gpu_by_index[slot['gpu']]['uuid'],'receipt':str(attempt_path),
                  'attempt_id':attempt,'binding':binding,'run_token':run_token}
            write_json(specpath,spec)
            logroot=root/'logs'/run_id;logroot.mkdir(parents=True,exist_ok=True)
            streams=[(logroot/f'{index:02d}.{kind}.txt').open('wb') for kind in ('stdout','stderr')]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],'RADAR_EPOCH10_RUN_TOKEN':run_token,
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
                 'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'}
            command=[sys.executable,'-B','-X','utf8',str(HERE/'epoch10_runner.py'),'--worker-spec',str(specpath)]
            try:proc=subprocess.Popen(command,stdout=streams[0],stderr=streams[1],env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except BaseException:
                for stream in streams:stream.close()
                raise
            write_json(root/'receipts'/f'{index:02d}.json',{'status':'RUNNING','pid':proc.pid,'job':job,'slot':slot,'attempt_id':attempt})
            print(json.dumps({'event':'STARTED','job':job,'gpu':slot['gpu'],'pid':proc.pid}),flush=True)
            return {'process':proc,'streams':streams,'receipt':attempt_path,'attempt_id':attempt}
        def finish(item,slot,handle,code):
            index,job=item
            for stream in handle['streams']:stream.close()
            receipt=read(handle['receipt']) if handle['receipt'].exists() else {'status':'FAILED','error':'WORKER_RECEIPT_MISSING'}
            ok=(code==0 and receipt.get('status')=='COMPLETE' and receipt.get('attempt_id')==handle['attempt_id'] and receipt.get('binding')==binding)
            if ok:
                exact=inspect_job(out,job,plan,tr,va,allowed,binding)
                ok=exact['status']=='EXACT_EPOCH_AVAILABLE'
            receipt.update(status='COMPLETE' if ok else 'FAILED',returncode=code,job=job,slot=slot)
            write_json(root/'receipts'/f'{index:02d}.json',receipt)
            print(json.dumps({'event':receipt['status'],'job':job,'gpu':slot['gpu'],'error':receipt.get('error')}),flush=True)
            return ok
        def progress(queued,active,done,failed):
            nonlocal last_print
            status={'status':'DRAINING_AFTER_FAILURE' if failed else 'RUNNING','stage':'train_to_epoch_10',
                    'total':len(joblist),'complete':len(completed)+len(done),'running':len(active),'failed':len(failed),'pending':len(queued),
                    'stop_epoch':STOP_EPOCH,'target_updates_per_model':binding['stop_updates'],'maximum_workers_per_gpu':{'0':2,'1':3},
                    'elapsed_s':time.monotonic()-started,'updated_ns':time.time_ns(),'coordinator_pid':os.getpid(),
                    'active':[{'job_index':item[0],'job':item[1],'gpu':slot['gpu'],'pid':handle['process'].pid} for item,slot,handle in active]}
            write_json(root/'STATUS.json',status)
            if time.monotonic()-last_print>=30:print(json.dumps(status),flush=True);last_print=time.monotonic()
        try:
            outcome=run_pool(pending,slot_layout()[:min(5,len(pending))],start,finish,progress)
            success=not outcome['failed'] and not outcome['pending']
            results=[read(model_dir(out,j)/'EPOCH_010_RESULT.json') for j in joblist] if success else []
            status={'status':'COMPLETE' if success else 'FAILED','total':len(joblist),'complete':len(completed)+len(outcome['done']),
                    'failed':len(outcome['failed']),'pending':len(outcome['pending']),'running':0,'stop_epoch':STOP_EPOCH,
                    'target_updates_per_model':binding['stop_updates'],'elapsed_s':time.monotonic()-started,'updated_ns':time.time_ns()}
            write_json(root/'STATUS.json',status)
            if success:write_json(root/'EPOCH10_RESULTS.json',{'status':'COMPLETE','execution':binding,'models':results})
            print(json.dumps(status),flush=True)
            return 0 if success else 2
        except BaseException as exc:
            write_json(root/'STATUS.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                'error':repr(exc),'traceback':traceback.format_exc(),'updated_ns':time.time_ns()})
            raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker-spec')
    ap.add_argument('--stage',choices=('check','run'),default='check')
    ap.add_argument('--project',default=r'D:\LXR\radar_stage6_signal_fix')
    ap.add_argument('--output',default=r'D:\LXR\SERVER_STAGE6_SIGNAL_FIX_12800')
    ap.add_argument('--plan');ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if args.worker_spec:return execute_worker(args.worker_spec)
    args.plan=args.plan or str(Path(args.project)/'STAGE6_SERVER/plan_12800.json')
    context=load_context(args.project,args.output,args.plan)
    if args.stage=='run':
        if not args.execute:raise RuntimeError('RUN_REQUIRES_EXECUTE')
        return run_coordinator(args,context)
    controller,transport,input_profile,profile,plan,tr,va,allowed,binding=context
    jobs=controller.task_list(plan,'train')
    states=[{**j,**inspect_job(args.output,j,plan,tr,va,allowed,binding)} for j in jobs]
    print(json.dumps({'status':'EPOCH10_CHECK_PASSED','jobs':states,'stop_epoch':STOP_EPOCH,'stop_updates':binding['stop_updates'],
        'maximum_workers_per_gpu':{'0':2,'1':3},'world_advances':0,'optimizer_updates':0,
        'memory_check':'performed after old workers stop, immediately before run'},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except Exception:
        traceback.print_exc();sys.exit(2)
