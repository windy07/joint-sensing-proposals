"""Read back all first Linux periodic saves without modifying a model."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/root/autodl-tmp/radar_stage6_linux');args=ap.parse_args()
    root=Path(args.root);sys.path.insert(0,str(root/'linux_adapter'))
    import linux_runner as adapter
    original,ctx,binding=adapter.context(root)
    import torch
    torch.set_num_threads(1)
    control=root/'run/LINUX_EPOCH10_RUN';audit=adapter.read(control/'RECOVERY_AUDIT.json');rows=[]
    for item in audit['rows']:
        job=item['job'];dest=root/'run/models'/job['arm']/str(job['model_seed'])
        if item['status']=='EXACT_EPOCH_AVAILABLE':
            for name in ('latest.pt','best.pt','epoch_010.pt'):
                assert adapter.file_sha(dest/name)==adapter.file_sha(control/'IMPORTED_CHECKPOINTS'/job['arm']/str(job['model_seed'])/name)
            rows.append({'arm':job['arm'],'seed':job['model_seed'],'status':'IMPORTED_COMPLETE_UNCHANGED','saved_updates':32000});continue
        with (dest/'latest.pt').open('rb') as f:
            ck=torch.load(f,map_location='cpu',weights_only=True);f.seek(0);h=hashlib.sha256()
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        original.verify_job_checkpoint(ck,job,ctx[4],ctx[5],ctx[6],ctx[7])
        done=ck['optimizer_updates'];c=ck['cursor']
        assert done>=item['saved_updates']+ctx[4]['training']['checkpoint_interval'],(job,done,'NEW_PERIODIC_SAVE_NOT_YET_AVAILABLE')
        assert ck['epoch_stop_execution']==binding
        assert ck['requested_stop_epoch']==10 and ck['requested_stop_updates']==32000 and ck['target_updates']==64000
        assert done==c['epoch']*3200+c['batch']*8+c['time']//8
        assert len(ck['cuda_rng_states'])==1 and ck['torch_rng_state'].numel()>0
        assert all(bool(torch.isfinite(t).all()) for t in ck['state'].values())
        states=ck['optimizer_state']['state'];assert states
        assert max(float(s['step']) for s in states.values())==done
        assert all(0<float(s['step'])<=done for s in states.values())
        assert all(bool(torch.isfinite(s[name]).all()) for s in states.values() for name in ('exp_avg','exp_avg_sq'))
        rows.append({'arm':job['arm'],'seed':job['model_seed'],'status':'LINUX_SAVE_VERIFIED','saved_updates':done,
            'imported_updates':item['saved_updates'],'cursor':c,'checkpoint_sha256':h.hexdigest(),
            'optimizer_and_rng_state':'VERIFIED','parameter_and_optimizer_values':'FINITE'})
        del ck
    body={'status':'PASSED','verified_ns':time.time_ns(),'models':rows,'optimizer_updates_in_verifier':0,'world_advances_in_verifier':0}
    adapter.write(control/'STARTUP_CHECKPOINT_VERIFICATION.json',body);print(json.dumps(body),flush=True)
if __name__=='__main__':main()
