"""Read-only progress and checkpoint inspection for the Linux continuation."""
from pathlib import Path
import argparse,json,os,subprocess,time

def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return default

def collect(root,checkpoints=False):
    root=Path(root);run=root/'run';control=run/'LINUX_EPOCH10_RUN'
    state=read(control/'STATUS.json',{})
    checks=read(control/'RECOVERY_AUDIT.json',read(control/'CHECK.json',{})).get('rows',[])
    result={'captured_ns':time.time_ns(),'coordinator':state,'models':[],'errors':[]}
    for index,item in enumerate(checks):
        job=item['job'];dest=run/'models'/job['arm']/str(job['model_seed'])
        worker=read(control/'receipts'/f'{index:02d}.json',{})
        receipt=read(dest/'TRAIN_RECEIPT.json',{})
        row={'arm':job['arm'],'seed':job['model_seed'],'gpu':worker.get('gpu'),
             'pid':worker.get('pid'),'status':worker.get('status',item['status']),
             'phase':receipt.get('phase'),'updates':receipt.get('optimizer_updates',item.get('saved_updates',0)),
             'initial_saved_updates':item.get('saved_updates',0),'target':32000,
             'epoch':receipt.get('cursor',{}).get('epoch'),
             'recent_contexts_per_s':receipt.get('recent_contexts_per_s'),
             'average_contexts_per_s':receipt.get('contexts_per_s'),
             'receipt_age_s':(time.time_ns()-receipt['updated_ns'])/1e9 if receipt.get('updated_ns') else None,
             'peak_allocated_gib':receipt.get('gpu_peak_allocated_bytes',0)/2**30,
             'peak_reserved_gib':receipt.get('gpu_peak_reserved_bytes',0)/2**30,
             'loss':receipt.get('loss'),'validation':receipt.get('validation'),
             'timing':receipt.get('timing'),'input_transport':receipt.get('input_transport')}
        pid=row['pid']
        row['alive']=bool(pid and Path(f'/proc/{pid}').exists())
        if worker.get('error'):result['errors'].append({'job':job,'error':worker['error'],'traceback':worker.get('traceback')})
        if receipt.get('status')=='FAILED' and receipt.get('epoch_stop_execution'):
            result['errors'].append({'job':job,'error':receipt.get('error')})
        if checkpoints and (dest/'latest.pt').is_file():
            import torch
            ck=torch.load(dest/'latest.pt',map_location='cpu',weights_only=True)
            row['checkpoint']={'updates':ck['optimizer_updates'],'cursor':ck.get('cursor'),
                 'saved_on_linux':ck.get('epoch_stop_execution',{}).get('schema')=='ain.stage6.linux_resume.execution.v1',
                 'requested_stop_epoch':ck.get('requested_stop_epoch'),
                 'has_optimizer':bool(ck.get('optimizer_state')),
                 'validation_epochs':[v['epoch'] for v in ck.get('validation_history',[])]}
            del ck
        result['models'].append(row)
    try:
        result['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    except Exception as e:result['gpu_error']=repr(e)
    for name in ('memory.current','memory.max','memory.events','cpu.stat','cpu.max'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():result['cgroup_'+name]=p.read_text().strip()
    return result

def display(body):
    s=body['coordinator'];print(time.strftime('%Y-%m-%d %H:%M:%S'),s.get('status','NOT_STARTED'),
        'complete=',s.get('complete',0),'running=',s.get('running',0),'failed=',s.get('failed',0),flush=True)
    print(f'{"Model":<18} {"Seed":>7} {"GPU":>3} {"Updates":>9} {"Epoch":>5} {"Ctx/s":>8} {"PeakGiB":>8} {"Age/s":>7} {"Status":<10}')
    for r in body['models']:
        print(f'{r["arm"]:<18} {r["seed"]:>7} {str(r["gpu"] if r["gpu"] is not None else "-"):>3} {r["updates"]:>9} {str(r["epoch"]):>5} {(r["recent_contexts_per_s"] or 0):>8.1f} {r["peak_allocated_gib"]:>8.2f} {(r["receipt_age_s"] or 0):>7.1f} {r["status"]:<10}')
    for line in body.get('gpu_csv',[]):print(line)
    for err in body['errors']:print('ERROR:',json.dumps(err,ensure_ascii=False))
    print(flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/root/autodl-tmp/radar_stage6_linux')
    ap.add_argument('--once',action='store_true');ap.add_argument('--json',action='store_true')
    ap.add_argument('--check-checkpoints',action='store_true');ap.add_argument('--interval',type=float,default=30.)
    args=ap.parse_args()
    while True:
        body=collect(args.root,args.check_checkpoints)
        if args.json:print(json.dumps(body,ensure_ascii=False),flush=True)
        else:display(body)
        if args.once:return
        time.sleep(max(3.,args.interval))
if __name__=='__main__':main()
