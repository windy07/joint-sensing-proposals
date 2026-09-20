"""Disaggregated tables, scientific figures and a self-contained local HTML replay."""
from __future__ import annotations
import json,csv,math,argparse
from pathlib import Path
import numpy as np
from .core import *
from .physics import SENSORS,CELLS,BEACON,MONITOR,aim,sensor_list,Task

NAMES={'random':'Random','myopic':'One-step constructive','lookahead':'Three-step rollout','feedback':'Rollout + candidate feedback','learned':'Reused graph priority (wiring)'}
NAMES.update({'coverage_feedback':'Coverage + paced feedback','no_calibration':'Rollout without calibration'})

def compact(d):
    rows=[]
    for t in d['trace']:
        n=np.asarray(t['posterior_mean']);T=[]
        for tr in t['tracks']:
            j=tr['start'];T.append({'id':tr['id'],'confirmed':tr['confirmed'],'state':n[j:j+4].tolist()})
        actions=[]
        for x in t['commit']['selected']:
            a=Task(**x);s=sensor_list(d['configuration'])[a.node];b,w=aim(a,t['causal_view'],s)
            actions.append({**x,'angle':b,'width':w})
        rows.append({'epoch':t['epoch'],'truth':t['truth_for_evaluation_only'],'tracks':T,'actions':actions,'metric':t['metric'],
         'energy':t['remaining_RF_energy_J'],'tx':t['remaining_transmit_time_s'],'packets':t['packets'],
         'feedback':t['commit']['feedback'],'scores':[{k:r[k] for k in ['schedule','proxy_m2']} for r in t['commit']['archive']],
         'horizon':t['commit']['forecast_horizon'],'status':t['commit']['status']})
    return {'method':d['method'],'seed':d['seed'],'scenario':d['configuration']['scenario'],'summary':d['summary'],'trace':rows}

HTML=r'''<!doctype html><meta charset="utf-8"><title>空间雷达调度研究回放</title>
<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;max-width:1400px;background:#fafafa;color:#222}h1{font-size:23px}button,select,input{font-size:15px;margin:5px;padding:5px}main{display:flex;gap:20px;align-items:flex-start}canvas{background:white;border:1px solid #aaa;max-width:75vw}aside{width:310px}table{border-collapse:collapse;width:100%;font-size:14px}td{padding:7px;border-bottom:1px solid #ddd}pre{font-size:12px;white-space:pre-wrap}p{line-height:1.6}.muted{font-size:13px;color:#555}</style>
<h1>二维单／多目标闭环调度 · 已记录数据回放</h1>
<p class="muted">实线为目标已发生轨迹，虚线为估计航迹；空心点为暂定航迹。图中的真实轨迹只供事后评价，未输入调度器。无网络依赖。</p>
<select id="case"></select><button id="play">播放</button><button id="reset">回到起点</button><label><input id="truth" type="checkbox" checked>显示评价真值</label><input id="time" type="range" min="0" value="0"><span id="clock"></span>
<main><canvas id="c" width="850" height="720"></canvas><aside><h3 id="method"></h3><table id="metrics"></table><h3>本周期节点任务</h3><pre id="tasks"></pre><h3>候选反馈</h3><pre id="feedback"></pre><p class="muted">辐射能量不是被发现概率。前瞻分数来自均值信念近似，不是完整未来观测树。</p></aside></main>
<script>const DATA=__DATA__, SENSORS=__SENSORS__, names=__NAMES__;const sel=document.getElementById('case'),slider=document.getElementById('time'),cv=document.getElementById('c'),ctx=cv.getContext('2d');let playing=false,timer;
DATA.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.scenario+' · '+names[d.method]+' · '+d.seed;sel.append(o)});
let d=DATA[0];const colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'];
function xy(p){return [55+(p[0]+2500)/5000*740,665-(p[1]+2000)/5000*610]}
function line(points,color,dash=[]){if(points.length<2)return;ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.lineWidth=1.8;ctx.beginPath();points.forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.setLineDash([])}
function dot(p,label,color,fill=true,r=5){let q=xy(p);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.arc(...q,r,0,2*Math.PI);fill?ctx.fill():ctx.stroke();ctx.fillText(label,q[0]+8,q[1]-8)}
function number(x,n=2){return x===null||!Number.isFinite(x)?'未定义':x.toFixed(n)}
function draw(){let k=Number(slider.value),r=d.trace[k];ctx.clearRect(0,0,850,720);ctx.font='13px sans-serif';ctx.lineWidth=.6;
for(let x=-2000;x<=2000;x+=1000){let a=xy([x,-2000]),b=xy([x,3000]);ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#555';ctx.fillText(x,a[0]-15,688)}
for(let y=-2000;y<=3000;y+=1000){let a=xy([-2500,y]),b=xy([2500,y]);ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#555';ctx.fillText(y,5,a[1]+4)}ctx.fillText('x / m',760,710);ctx.fillText('y / m',8,28);
for(let a of r.actions){let s=SENSORS[a.node],o=xy(s.position);let p=xy([s.position[0]+2500*Math.cos(a.angle),s.position[1]+2500*Math.sin(a.angle)]);ctx.strokeStyle=s.mode==='active'?'#ab5b00':'#336666';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...o);ctx.lineTo(...p);ctx.stroke();ctx.globalAlpha=.08;ctx.beginPath();ctx.moveTo(...o);for(let j=0;j<=30;j++){let ang=a.angle-a.width/2+a.width*j/30;ctx.lineTo(...xy([s.position[0]+2000*Math.cos(ang),s.position[1]+2000*Math.sin(ang)]))}ctx.closePath();ctx.fill();ctx.globalAlpha=1;}
SENSORS.forEach(s=>dot(s.position,s.name+' '+(s.mode==='active'?'主动':'无源'),'#222',true,7));dot([0,-500],'已知基准点','#555',false,6);
if(document.getElementById('truth').checked){let ids=new Set(d.trace.slice(0,k+1).flatMap(t=>Object.keys(t.truth)));ids.forEach(id=>{let pts=d.trace.slice(0,k+1).filter(t=>t.truth[id]).map(t=>t.truth[id]);line(pts,colors[Number(id)%6]);if(r.truth[id])dot(r.truth[id],'目标'+id,colors[Number(id)%6])})}
let ids=new Set(d.trace.slice(0,k+1).flatMap(t=>t.tracks.map(x=>x.id)));ids.forEach(id=>{let pts=d.trace.slice(0,k+1).flatMap(t=>t.tracks.filter(x=>x.id===id&&x.confirmed).map(x=>x.state));line(pts,colors[id%6],[5,4])});r.tracks.forEach(t=>dot(t.state,'航迹'+t.id,colors[t.id%6],t.confirmed,4));
document.getElementById('method').textContent=names[d.method];document.getElementById('clock').textContent=k+' s';let m=r.metric;let rows=[['真实／确认目标',m.truth_count+' / '+m.confirmed_track_count],['定位 RMSE / m',number(m.position_rmse_m)],['速度 RMSE / m/s',number(m.velocity_rmse_mps)],['漏跟／虚假航迹',m.missed_count+' / '+m.false_count],['距离偏差 RMSE / m',number(m.bias_rmse_m)],['剩余射频能量 / J',number(r.energy)],['剩余发射时间 / s',number(r.tx,4)],['本周期独立评分次数',m.paid_root_evaluations],['内部预测更新次数',m.forecast_updates],['前瞻周期',r.horizon]];document.getElementById('metrics').innerHTML=rows.map(x=>'<tr><td>'+x[0]+'</td><td>'+x[1]+'</td></tr>').join('');
document.getElementById('tasks').textContent=r.actions.length?r.actions.map(a=>SENSORS[a.node].name+' → '+({search:'搜索区域',track:'观测航迹',calibrate:'校准基准点'}[a.kind])+(a.kind==='calibrate'?'':a.item)).join('\n'):'全部节点不采样';document.getElementById('feedback').textContent=r.feedback.length?r.feedback.map(x=>x.reason+'：'+JSON.stringify(x.incumbent)).join('\n'):'本周期无反馈扩展';}
function choose(){d=DATA[Number(sel.value)];slider.max=d.trace.length-1;slider.value=0;draw()}sel.onchange=choose;slider.oninput=draw;document.getElementById('truth').onchange=draw;
document.getElementById('reset').onclick=()=>{slider.value=0;draw()};document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放';if(playing)timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){clearInterval(timer);playing=false;document.getElementById('play').textContent='播放';return}slider.value=Number(slider.value)+1;draw()},350);else clearInterval(timer)};choose();</script>'''

def summarize(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=Path(results);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    receipt=read_json(results/'RUN_RECEIPT.json')
    ids=receipt.get('planned_case_ids',[x['id'] for x in receipt['cases']])
    allrows=[read_json(results/'cases'/f'{tag}.json') if (results/'cases'/f'{tag}.json').exists()
             else {'status':'MISSING','id':tag} for tag in ids]
    failures=[{'id':tag,'status':d['status'],'error':d.get('error')} for tag,d in zip(ids,allrows) if d['status']!='COMPLETE']
    data=[d for d in allrows if d['status']=='COMPLETE']
    for d in data:
        if d['source_digest']!=receipt['source_digest']:raise ContractError('SUMMARY_SOURCE_MISMATCH')
    write_json(out/'failure_summary.json',{'run_status':receipt['status'],'expected':len(ids),'completed':len(data),'failures':failures})
    if not data:raise ContractError('NO_COMPLETED_TRAJECTORY')
    rows=[d['summary'] for d in data];cols=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write_json(out/'comparison.json',rows)
    (out/'replay.html').write_text(HTML.replace('__DATA__',json.dumps([compact(d) for d in data],ensure_ascii=False)).replace('__SENSORS__',json.dumps([s.doc() for s in SENSORS])).replace('__NAMES__',json.dumps(NAMES,ensure_ascii=False)),encoding='utf-8')
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    for scene in sorted({d['configuration']['scenario'] for d in data}):
        ds=[d for d in data if d['configuration']['scenario']==scene];seed=min(d['seed'] for d in ds);ds=[d for d in ds if d['seed']==seed]
        for key,ylabel,transform in [
          ('position_rmse_m','Matched-target position RMSE (m)',False),
          ('missed_count','Missed targets (count)',False),
          ('false_count','False confirmed tracks (count)',False),
          ('bias_rmse_m','Range-bias RMSE (m)',False),
          ('rf_energy_J','Cumulative radiated energy (J)',True)]:
            fig,ax=plt.subplots(figsize=(7,3.5))
            for d in ds:
                vals=np.array([np.nan if t['metric'][key] is None else t['metric'][key] for t in d['trace']]);vals=np.cumsum(vals) if transform else vals
                ax.plot([t['metric']['time_s'] for t in d['trace']],vals,label=NAMES[d['method']])
            ax.set(xlabel='Time (s)',ylabel=ylabel);ax.grid(True,alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/f'{scene}_{key}.png',dpi=170);plt.close(fig)
        for d in ds:
            fig,ax=plt.subplots(figsize=(6.5,5))
            tids=sorted({int(i) for t in d['trace'] for i in t['truth_for_evaluation_only']})
            for i in tids:
                pts=np.array([t['truth_for_evaluation_only'][str(i)][:2] for t in d['trace'] if str(i) in t['truth_for_evaluation_only']]);ax.plot(pts[:,0],pts[:,1],label=f'Target {i}')
            ids=sorted({t['id'] for r in d['trace'] for t in r['tracks'] if t['confirmed']})
            for i in ids:
                pts=[]
                for r in d['trace']:
                    for t in r['tracks']:
                        if t['id']==i and t['confirmed']:pts.append(r['posterior_mean'][t['start']:t['start']+2])
                if pts:
                    p=np.array(pts);ax.plot(p[:,0],p[:,1],'--',linewidth=1,label=f'Track {i}')
            for s in SENSORS:ax.scatter(*s.position,marker='^');ax.annotate(s.name,s.position)
            ax.scatter(*BEACON,marker='x');ax.annotate('Beacon',BEACON)
            ax.set(xlabel='East (m)',ylabel='North (m)',title=NAMES[d['method']],xlim=(-2400,2400),ylim=(-1700,2700));ax.set_aspect('equal');ax.grid(True,alpha=.2);ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(figdir/f'{scene}_{d["method"]}_tracks.png',dpi=170);plt.close(fig)
    md=['# 实际空间闭环诊断','',f'完成 {len(data)} 个轨迹；条件定位误差须与漏跟、虚假航迹一起阅读。','', '|场景|方法|定位RMSE/m|漏跟比例|平均虚假航迹|射频能量/J|距离偏差RMSE/m|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:md.append('|'+str(r['scenario'])+'|'+NAMES[r['method']]+'|'+('|'.join('未定义' if r[k] is None else f'{r[k]:.4f}' for k in ['position_rmse_m','missed_target_time_fraction','mean_false_tracks','rf_energy_J','final_bias_rmse_m']))+'|')
    (out/'SUMMARY.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'status':'PARTIAL' if failures else 'COMPLETE','trajectories':len(data),'failed_or_missing':len(failures),'html':str(out/'replay.html'),'claimed_statistical_winner':None},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args();summarize(a.results,a.output)
if __name__=='__main__':main()
