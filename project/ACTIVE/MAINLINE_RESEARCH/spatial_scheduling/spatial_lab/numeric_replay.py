"""Self-contained playback of actual numeric actions and receiver receipts."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from .core import read_json,ContractError
from .case_identity import case_identity,validate_cached
from .physics import NBIAS,SENSORS

def replay_record(r):
    fields=('epoch','truth_for_evaluation_only','posterior_mean','posterior_covariance','tracks','metric',
            'remaining_RF_energy_J','remaining_transmit_time_s','generated_packets','update_results','post_unseen_intensity','packets','pending_packet_roots')
    frames=[]
    for f in r['trace']:
        frame={k:f.get(k) for k in fields}
        frame['commit']={k:f['commit'].get(k) for k in ('selected','meter','seconds','network_raw_output','capacity_audit','context_digest')}
        frame['view_mean']=f['causal_view']['mean'];frame['view_covariance']=f['causal_view']['covariance'];frames.append(frame)
    return {'seed':r['seed'],'method':r['method'],'arm':r.get('checkpoint_arm'),'initialization':r.get('untrained_initialization',False),
       'case_digest':r['case_identity']['digest'],'configuration':r['configuration'],'summary':r['summary'],'trace':frames}

HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数值动作与统一接收回放</title><style>
body{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;margin:24px;background:#fafafa;color:#222}h1{font-size:23px;font-weight:600;margin-bottom:4px}p{max-width:1100px}label,button{margin-right:14px}select,button,input{font:inherit}button{padding:4px 15px;border:1px solid #888;background:white}header{border-bottom:1px solid #bbb;padding-bottom:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:18px}.panel{min-width:0;border:1px solid #bbb;background:white;padding:14px}.panel svg{width:100%;border-bottom:1px solid #ddd}.muted{color:#555;font-size:13px}table{font-size:12px;width:100%;border-collapse:collapse}td,th{text-align:right;border-bottom:1px solid #ddd;padding:4px;overflow-wrap:anywhere}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto;max-height:320px}pre{white-space:pre-wrap;word-break:break-all;font-size:12px}#time{width:260px}#beam{width:210px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<header><h1>数值动作、参考接收与联合后验</h1><p class="muted">实线：环境轨迹；虚线：估计轨迹；圆点：物理节点；扇形：当前执行波束。轨迹点在周期参考时刻，周期内滑块只高亮实际执行束次。未训练初始化不代表训练效果。</p>
<label>世界 <select id="world"></select></label><label>周期 <input id="time" type="range" min="0" step="1" value="0"><span id="epochLabel"></span></label><label>周期内 <input id="beam" type="range" min="0" max="1000" value="500"><span id="beamLabel"></span></label><button id="play">播放</button><label><input id="truth" type="checkbox" checked>显示环境真值</label><button id="download">导出回放数据</button></header>
<div class="grid"><section class="panel" id="left"></section><section class="panel" id="right"></section></div><p id="notice" class="muted"></p>
<script id="records" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('records').textContent);let timer=null;const $=id=>document.getElementById(id);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=(x,n=3)=>x===null||x===undefined?'未记录':Number(x).toFixed(n);const group=r=>`${r.seed} / ${r.configuration.scenario} / ${r.configuration.fixture_name||'核心同世界'}`;
const worlds=[...new Set(D.records.map(group))];worlds.forEach(w=>$('world').add(new Option(w,w)));
function title(r){return r.method+(r.arm?' / '+r.arm:'')+(r.initialization?'〔未训练〕':'');}
function options(side){const panel=$(side);panel.innerHTML=`<label>策略 <select id="${side}Select"></select></label><p class="muted" id="${side}Meta"></p><svg id="${side}Plot" viewBox="0 0 650 470"></svg><p id="${side}Metrics"></p><div class="scroll" id="${side}Table"></div><details><summary>参考、接纳和资源原始明细</summary><pre id="${side}Raw"></pre></details>`;D.records.forEach((r,i)=>{if(group(r)===$('world').value)$(side+'Select').add(new Option(title(r),i));});$(side+'Select').addEventListener('change',range);}
function range(){const a=D.records[Number($('leftSelect').value)],b=D.records[Number($('rightSelect').value)];$('time').max=Math.min(a.trace.length,b.trace.length)-1;$('time').value=Math.min(Number($('time').value),Number($('time').max));draw();}
function init(){options('left');options('right');if($('rightSelect').options.length>1)$('rightSelect').selectedIndex=1;range();}
const X=x=>325+x*.075,Y=y=>245-y*.065;function path(points){return points.map((p,i)=>`${i?'L':'M'}${X(p[0])},${Y(p[1])}`).join(' ');}
function drawPanel(side){const r=D.records[Number($(side+'Select').value)],e=Number($('time').value),z=r.trace[e],dt=r.configuration.dt_s,sub=Number($('beam').value)*dt/1000;
$(side).dataset.epoch=String(e);$(side).dataset.caseDigest=r.case_digest;let svg=`<rect x="0" y="0" width="650" height="470" fill="white"/><text x="12" y="18" font-size="12">位置 / m；周期参考时刻的联合后验</text>`;
for(let x=-3000;x<=3000;x+=1000)svg+=`<path d="M${X(x)},30 L${X(x)},435" stroke="#ddd"/><text x="${X(x)}" y="452" font-size="11">${x}</text>`;
for(let y=-2000;y<=3000;y+=1000)svg+=`<path d="M75,${Y(y)} L575,${Y(y)}" stroke="#ddd"/><text x="30" y="${Y(y)}" font-size="11">${y}</text>`;
D.sensors.forEach(s=>svg+=`<circle cx="${X(s.position[0])}" cy="${Y(s.position[1])}" r="4" fill="#333"/><text x="${X(s.position[0])+7}" y="${Y(s.position[1])-5}" font-size="12">${escape(s.name)}</text>`);
if($('truth').checked){const ids=[...new Set(r.trace.slice(0,e+1).flatMap(a=>Object.keys(a.truth_for_evaluation_only)))];ids.forEach(id=>{const pts=r.trace.slice(0,e+1).flatMap(a=>a.truth_for_evaluation_only[id]?[a.truth_for_evaluation_only[id]]:[]);svg+=`<path d="${path(pts)}" fill="none" stroke="#202020" stroke-width="2"/>`;const p=z.truth_for_evaluation_only[id];if(p)svg+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="#202020"/><text x="${X(p[0])+6}" y="${Y(p[1])-7}" font-size="12">目标 ${escape(id)}</text>`;});}
const trackids=[...new Set(r.trace.slice(0,e+1).flatMap(a=>a.tracks.map(t=>t.id)))];trackids.forEach(id=>{const pts=r.trace.slice(0,e+1).flatMap(a=>{const t=a.tracks.find(t=>t.id===id);return t?[a.posterior_mean.slice(t.start,t.start+2)]:[];});svg+=`<path d="${path(pts)}" fill="none" stroke="#446477" stroke-dasharray="5 3" stroke-width="2"/>`;const t=z.tracks.find(t=>t.id===id);if(t){const p=z.posterior_mean.slice(t.start,t.start+2);svg+=`<path d="M${X(p[0])-4},${Y(p[1])-4} l8,8 m-8,0 l8,-8" stroke="#446477"/><text x="${X(p[0])+7}" y="${Y(p[1])+12}" font-size="11">航迹 ${id} ${t.confirmed?'确认':'暂定'}</text>`;}});
const packets=z.generated_packets||[];const active=packets.filter(p=>p.task.start_s<=sub&&sub<=p.task.start_s+p.task.dwell_s);active.forEach(p=>{const s=D.sensors[p.node],a=p.task.bearing_rad,w=p.task.beam_rad/2,L=2800,pts=[[...s.position],[s.position[0]+L*Math.cos(a-w),s.position[1]+L*Math.sin(a-w)],[s.position[0]+L*Math.cos(a+w),s.position[1]+L*Math.sin(a+w)]];svg+=`<path d="${path(pts)} Z" fill="#718796" fill-opacity=".12" stroke="#647a88"/>`;});
$(side+'Plot').innerHTML=svg;$(side+'Meta').textContent=`${title(r)}；原记录 ${r.trace.length} 周期；接收层：统一量测级。比较不包含训练效果推断。`;
const m=z.metric;$(side+'Metrics').textContent=`位置误差 ${f(m.position_rmse_m,2)} m；匹配 ${m.matched_count}/${m.truth_count}；漏失 ${m.missed_count}；假航迹 ${m.false_count}；GOSPA ${f(m.gospa_m,2)} m。射频余量 ${f(z.remaining_RF_energy_J,5)} J；发射时间余量 ${f(z.remaining_transmit_time_s,6)} s。`;
let table='<table><thead><tr><th>节点 / 任务</th><th>链路 / 参考</th><th>峰值功率 W</th><th>束开始 s</th><th>驻留 s</th><th>指向 °</th><th>到达 s</th></tr></thead><tbody>';
packets.forEach(p=>{table+=`<tr><td>${escape(D.sensors[p.node].name+' / '+p.kind)}</td><td>${escape(p.task.role+' / '+(p.reference_acquired?'已取得':'未取得'))}</td><td>${f(p.task.power_W,2)}</td><td>${f(p.task.start_s,5)}</td><td>${f(p.task.dwell_s,5)}</td><td>${f(p.task.bearing_rad*180/Math.PI,2)}</td><td>${f(p.arrival_s,5)}</td></tr>`;});
$(side+'Table').innerHTML=table+'</tbody></table>';$(side+'Raw').textContent=JSON.stringify({executed_actions:z.commit.selected,network_raw_output:z.commit.network_raw_output,receivers:packets,acceptance:z.update_results,decision_costs:z.commit.meter,metrics:m},null,2);
}
function draw(){$('epochLabel').textContent=$('time').value+' / '+$('time').max;const a=D.records[Number($('leftSelect').value)];$('beamLabel').textContent=f(Number($('beam').value)*a.configuration.dt_s/1000,3)+' s';drawPanel('left');drawPanel('right');$('notice').textContent=`共同显示 ${Number($('time').max)+1} 个已记录周期。${D.note}`;}
$('world').addEventListener('change',init);$('time').addEventListener('input',draw);$('beam').addEventListener('input',draw);$('truth').addEventListener('change',draw);
$('play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';}else{$('play').textContent='暂停';timer=setInterval(()=>{let b=Number($('beam').value)+100;if(b>1000){b=0;$('time').value=(Number($('time').value)+1)%(Number($('time').max)+1);}$('beam').value=b;draw();},150);}});
$('download').addEventListener('click',()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(D)],{type:'application/json'}));a.download='numeric_replay_data.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});init();window.REPLAY_READY=true;
</script></html>'''

def build(paths,output,max_worlds=3):
    records=[];worlds=set();omitted=0
    for p in sorted(map(Path,paths)):
        r=read_json(p)
        if r.get('fixture_only') or r.get('status')!='COMPLETE':continue
        validate_cached(r,case_identity(r['seed'],r['method'],r['configuration'],r['case_identity']['checkpoint']))
        key=(r['seed'],r['configuration'].get('fixture_name','core'))
        if key not in worlds and len(worlds)>=max_worlds:omitted+=1;continue
        worlds.add(key);records.append(replay_record(r))
    if not records:raise ContractError('NO_VALIDATED_REPLAY_CASES')
    data={'schema':'ain.stage4.numeric.replay.v1','records':records,'sensors':[s.doc() for s in SENSORS],
          'note':f'已嵌入 {len(records)} 条执行记录；未嵌入 {omitted} 条。未显示的量没有补零或补真值。回放数据为原始记录的显示子集。'}
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    return {'file':str(out),'records':len(records),'world_groups':len(worlds),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--max-worlds',type=int,default=3);a=p.parse_args(argv)
    root=Path(a.results);paths=list(root.glob('*.json')) if root.name=='cases' else list(root.glob('cases/*.json'))
    print(build(paths,a.output,a.max_worlds));return 0
