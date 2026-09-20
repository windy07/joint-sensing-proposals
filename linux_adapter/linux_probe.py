"""Finite GPU memory/throughput diagnostic on existing training inputs; zero optimizer steps."""
from pathlib import Path
import argparse,gc,json,os,sys,time,traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--arm',required=True);ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args();root=Path(args.root);project=root/'project';run=root/'run'
    sys.path.insert(0,str(run/'EPOCH10_RUN/ADAPTER_SOURCE'))
    import epoch10_runner as original
    ctx=original.load_context(project,run,project/'STAGE6_SERVER/plan_12800.json')
    _,_,_,_,plan,trainset,_,_,_=ctx
    import torch
    from spatial_lab.stage6_model import make
    from spatial_lab.stage6_training import encode_step
    torch.set_num_threads(1);torch.cuda.set_device(args.gpu);device=f'cuda:{args.gpu}'
    model,_=make(args.arm,device,633433,plan['training']['model']);model.train()
    # Account for the two full-precision Adam moments without advancing an optimizer.
    moments=[torch.zeros_like(p) for _ in range(2) for p in model.parameters()]
    entries=list(range(32));worlds=[trainset[i] for i in entries]
    measurements=[]
    for start in (0,24,56):
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(args.gpu)
        memory=ids=None;loss=0.;started=time.monotonic();maximum_nodes=0
        for t in range(start,start+8):
            maximum_nodes=max(maximum_nodes,max(len(w['contexts'][t]['graph']['node_ids']) for w in worlds))
            value,parts,memory,ids=encode_step(model,worlds,t,memory,ids,device,True);loss=loss+value/8
        if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_DIAGNOSTIC_LOSS')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        torch.cuda.synchronize(args.gpu)
        measurements.append({'start':start,'maximum_nodes':maximum_nodes,'loss':float(loss),'gradient_norm':float(grad),
             'elapsed_s':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated(args.gpu)/2**30,
             'peak_reserved_gib':torch.cuda.max_memory_reserved(args.gpu)/2**30})
        del memory,ids,loss,parts,value;gc.collect()
    result={'status':'PASSED','arm':args.arm,'gpu':args.gpu,'measurements':measurements,'worlds_read':32,
            'backward_passes':3,'optimizer_updates':0,'world_advances':0,'scope':'memory and numerical readiness, not a performance comparison'}
    output=root/'PROBE';output.mkdir(exist_ok=True);(output/(args.arm+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}),flush=True);raise
