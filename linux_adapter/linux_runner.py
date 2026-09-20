"""Linux execution adapter for the unchanged Stage6 epoch-10 training prefix."""
from pathlib import Path
from collections import Counter
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback,uuid

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def profile():
    p=read(HERE/'LINUX_PROFILE.json')
    if p['files']['linux_runner.py']!=file_sha(HERE/'linux_runner.py'):raise RuntimeError('LINUX_ADAPTER_CHANGED')
    return p
def context(root):
    root=Path(root).resolve();run=root/'run';project=root/'project';p=profile()
    sys.path.insert(0,str(run/'EPOCH10_RUN/ADAPTER_SOURCE'))
    import epoch10_runner as original
    ctx=original.load_context(project,run,project/'STAGE6_SERVER/plan_12800.json')
    from spatial_lab.core import digest
    binding={'schema':'ain.stage6.linux_resume.execution.v1','original_execution':ctx[-1],
             'profile':p,'profile_digest':digest(p),'source_project':str(project),'source_run':str(run),
             'target_epoch':10,'target_updates':32000,'original_lr_schedule_epochs':20}
    return original,ctx,binding
def identity(job):return job['arm']+'/'+str(job['model_seed'])
def target_gpu(job,p):return int(p['job_gpu_map'][identity(job)])
def hardware():
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in q.strip().splitlines():
        i,u,n,t,f,v=[x.strip() for x in line.split(',')]
        gpus.append({'index':int(i),'uuid':u,'name':n,'total_gib':float(t)/1024,'free_gib':float(f)/1024,'utilization':float(v)})
    return gpus
def proc_owned(pid,root):
    try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    except (FileNotFoundError,ProcessLookupError):return False
    return str(HERE/'linux_runner.py') in cmd and str(Path(root).resolve()/'run/LINUX_EPOCH10_RUN/jobs') in cmd
def completed_state(original,ctx,run,job):
    return original.inspect_job(run,job,ctx[4],ctx[5],ctx[6],ctx[7],ctx[8])
def choose(pending,slot,p):
    for index,item in enumerate(pending):
        if target_gpu(item[1],p)==slot['gpu']:return pending.pop(index)
    return None

def execute_worker(specpath):
    spec=read(specpath)
    if os.environ.get('RADAR_LINUX_TOKEN')!=spec['token']:raise RuntimeError('PARENT_RUN_TOKEN_REQUIRED')
    root=Path(spec['root']);run=root/'run';original,ctx,binding=context(root)
    if binding!=spec['binding']:raise RuntimeError('WORKER_EXECUTION_BINDING_CHANGED')
    controller,transport,input_profile,_,plan,tr,va,allowed,old_binding=ctx
    job=spec['job'];dest=run/'models'/job['arm']/str(job['model_seed']);token=None
    receipt={'status':'RUNNING','job':job,'pid':os.getpid(),'gpu':spec['gpu'],'attempt_id':spec['attempt_id'],
             'started_ns':time.time_ns(),'binding':binding,'initial_saved_updates':spec['initial_saved_updates']}
    write(spec['receipt'],receipt)
    try:
        import torch
        from spatial_lab.work_budget import enable_server,disable_server
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['gpu_uuid'] or torch.cuda.device_count()!=1:
            raise RuntimeError('GPU_ISOLATION_MISMATCH')
        torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(spec['allocator_cap_gib']*2**30/total,0)
        state=completed_state(original,ctx,run,job)
        if state['status']=='EXACT_EPOCH_AVAILABLE':
            receipt.update(status='COMPLETE',already_complete=True,saved_updates=32000)
        else:
            original.install_epoch_loop(controller,transport,input_profile,root/'project',binding)
            token=enable_server(old_binding['original_plan_digest'])
            result=controller.perform(plan,run,job,'cuda:0',True)
            if result['status']!='COMPLETE' or result['optimizer_updates']!=32000:raise RuntimeError('WRONG_COMPLETION_BOUNDARY')
            receipt.update(status='COMPLETE',result=result)
    except BaseException as e:
        receipt.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    finally:
        if token is not None:disable_server(token)
        receipt['finished_ns']=time.time_ns();write(spec['receipt'],receipt)
    return 0 if receipt['status']=='COMPLETE' else 2

def audit(root):
    root=Path(root);run=root/'run';original,ctx,binding=context(root)
    jobs=ctx[0].task_list(ctx[4],'train');rows=[]
    for job in jobs:
        state=completed_state(original,ctx,run,job)
        rows.append({'job':job,**state})
        print(json.dumps({'phase':'CHECKED','job':job,**state}),flush=True)
    return original,ctx,binding,rows

def coordinator(root):
    root=Path(root).resolve();run=root/'run';control=run/'LINUX_EPOCH10_RUN';control.mkdir(exist_ok=True)
    lock=(control/'coordinator.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('LINUX_COORDINATOR_ALREADY_RUNNING')
    for r in (control/'receipts').glob('*.json'):
        d=read(r)
        if d.get('pid') and proc_owned(d['pid'],root):raise RuntimeError('PREVIOUS_LINUX_WORKER_STILL_RUNNING')
    original,ctx,binding,rows=audit(root);p=binding['profile'];jobs=[r['job'] for r in rows]
    gpus={g['index']:g for g in hardware()}
    pending=[];complete=[];active={};failed=[]
    for index,row in enumerate(rows):
        if row['status']=='EXACT_EPOCH_AVAILABLE':complete.append(index)
        else:
            if identity(row['job']) not in p['job_gpu_map']:raise RuntimeError('UNASSIGNED_PENDING_JOB')
            pending.append((index,row['job']))
    counts=Counter(target_gpu(j,p) for _,j in pending)
    for gpu,count in counts.items():
        observed=gpus[gpu]
        if observed['uuid']!=p['gpu_uuids'][str(gpu)]:raise RuntimeError('LINUX_GPU_IDENTITY_CHANGED')
        required=count*(p['allocator_cap_gib'][str(gpu)]+p['context_reserve_gib'])
        if observed['free_gib']<required:raise RuntimeError(f'GPU_{gpu}_NEEDS_{required:.2f}_GiB_HAS_{observed["free_gib"]:.2f}')
    binding_path=control/'EXECUTION_BINDING.json'
    if binding_path.exists() and read(binding_path)!=binding:raise RuntimeError('LINUX_EXECUTION_CONFIG_CHANGED')
    write(binding_path,binding);write(control/'RECOVERY_AUDIT.json',{'rows':rows,'hardware':list(gpus.values()),'checked_ns':time.time_ns()})
    # A hard-linked baseline survives atomic replacement of a live checkpoint.
    for index,row in enumerate(rows):
        job=row['job'];dest=run/'models'/job['arm']/str(job['model_seed'])
        for name in ('latest.pt','best.pt','epoch_010.pt'):
            source=dest/name;backup=control/'IMPORTED_CHECKPOINTS'/job['arm']/str(job['model_seed'])/name
            if source.is_file() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);os.link(source,backup)
        if index in complete:write(control/'receipts'/f'{index:02d}.json',{'status':'COMPLETE','job':job,'imported_completed_epoch':10,'saved_updates':32000})
    slots=[{'slot':i,'gpu':g} for i,g in enumerate([0,0,1,1,2,2,2])]
    run_id=uuid.uuid4().hex;token=uuid.uuid4().hex;start=time.monotonic();last=0.;stopping=False
    def interrupt(signum,frame):raise KeyboardInterrupt('Coordinator stop requested')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        while pending or active:
            for slot in slots:
                if slot['slot'] in active or failed or (control/'STOP').exists():continue
                item=choose(pending,slot,p)
                if item is None:continue
                index,job=item;attempt=uuid.uuid4().hex;receipt_path=control/'attempts'/f'{index:02d}_{attempt}.json'
                spec={'root':str(root),'job':job,'gpu':slot['gpu'],'gpu_uuid':gpus[slot['gpu']]['uuid'],
                      'allocator_cap_gib':p['allocator_cap_gib'][str(slot['gpu'])],'initial_saved_updates':rows[index].get('saved_updates',0),
                      'binding':binding,'token':token,'attempt_id':attempt,'receipt':str(receipt_path)}
                specpath=control/'jobs'/f'{index:02d}_{attempt}.json';write(specpath,spec)
                logdir=control/'logs'/run_id;logdir.mkdir(parents=True,exist_ok=True)
                streams=[(logdir/f'{index:02d}.{kind}.txt').open('wb') for kind in ('stdout','stderr')]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],'RADAR_LINUX_TOKEN':token,
                     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
                     'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
                proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'linux_runner.py'),'--worker-spec',str(specpath)],env=env,stdout=streams[0],stderr=streams[1],start_new_session=True)
                active[slot['slot']]={'process':proc,'streams':streams,'item':item,'slot':slot,'receipt':receipt_path,'attempt_id':attempt}
                write(control/'receipts'/f'{index:02d}.json',{'status':'RUNNING','pid':proc.pid,'job':job,'gpu':slot['gpu'],'attempt_id':attempt,'initial_saved_updates':spec['initial_saved_updates']})
                print(json.dumps({'event':'STARTED','job':job,'gpu':slot['gpu'],'pid':proc.pid}),flush=True)
            for key,h in list(active.items()):
                code=h['process'].poll()
                if code is None:continue
                for f in h['streams']:f.close()
                receipt=read(h['receipt']) if h['receipt'].exists() else {'status':'FAILED','error':'WORKER_RECEIPT_MISSING'}
                index,job=h['item'];ok=code==0 and receipt.get('status')=='COMPLETE' and receipt.get('attempt_id')==h['attempt_id']
                if ok:ok=completed_state(original,ctx,run,job)['status']=='EXACT_EPOCH_AVAILABLE'
                receipt.update(status='COMPLETE' if ok else 'FAILED',returncode=code);write(control/'receipts'/f'{index:02d}.json',receipt)
                (complete if ok else failed).append(index);del active[key]
                print(json.dumps({'event':receipt['status'],'job':job,'error':receipt.get('error')}),flush=True)
            now=time.monotonic()
            state={'status':'DRAINING_AFTER_FAILURE' if failed else 'RUNNING','total':len(rows),'complete':len(complete),
                   'running':len(active),'failed':len(failed),'pending':len(pending),'target_updates':32000,'target_epoch':10,
                   'updated_ns':time.time_ns(),'elapsed_s':now-start,'coordinator_pid':os.getpid(),
                   'active':[{'job_index':h['item'][0],'job':h['item'][1],'gpu':h['slot']['gpu'],'pid':h['process'].pid} for h in active.values()]}
            write(control/'STATUS.json',state)
            if now-last>=30:print(json.dumps(state),flush=True);last=now
            if (failed or (control/'STOP').exists()) and not active:break
            if pending and not active and not failed:
                if (control/'STOP').exists():break
                raise RuntimeError('PENDING_JOBS_HAVE_NO_EXECUTION_SLOT')
            if pending or active:time.sleep(3)
        ok=len(complete)==9 and not failed
        state.update(status='COMPLETE' if ok else 'FAILED',running=0,updated_ns=time.time_ns());write(control/'STATUS.json',state)
        if ok:write(control/'EPOCH10_RESULTS.json',{'status':'COMPLETE','execution':binding,'models':[read(run/'models'/j['arm']/str(j['model_seed'])/'EPOCH_010_RESULT.json') for j in jobs]})
        return 0 if ok else 2
    except BaseException as e:
        write(control/'STATUS.json',{'status':'INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED','error':repr(e),'traceback':traceback.format_exc(),'updated_ns':time.time_ns()})
        raise
    finally:
        for h in active.values():
            if h['process'].poll() is None:h['process'].terminate()
        for h in active.values():
            try:h['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:h['process'].kill();h['process'].wait()
            for f in h['streams']:f.close()
        lock.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root');ap.add_argument('--worker-spec');ap.add_argument('--stage',choices=('check','run'),default='check');ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    if args.worker_spec:return execute_worker(args.worker_spec)
    if args.stage=='run':
        if not args.execute:raise RuntimeError('EXPLICIT_EXECUTION_REQUIRED')
        return coordinator(args.root)
    _,_,binding,rows=audit(args.root)
    path=Path(args.root)/'run/LINUX_EPOCH10_RUN/CHECK.json';write(path,{'status':'PASSED','binding':binding,'rows':rows,'optimizer_updates':0,'world_advances':0})
    print(json.dumps({'status':'CHECK_PASSED','jobs':len(rows),'optimizer_updates':0}),flush=True);return 0
if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except Exception:traceback.print_exc();sys.exit(2)
