"""Translate presentation labels without changing scientific records or plots."""
from pathlib import Path
import sys,json,re
p=Path(sys.argv[1]);html=p.read_text(encoding='utf-8')
names={'random':'随机优先','myopic':'单步构造','lookahead':'三步滚动','feedback':'滚动反馈',
       'coverage_feedback':'覆盖与资源节奏','learned':'图模型优先级','no_calibration':'禁用校准'}
html=re.sub(r'names=\{[^}]*\};const sel=',lambda _: 'names='+json.dumps(names,ensure_ascii=False)+';const sel=',html,count=1)
old="o.textContent=d.scenario+' · '+names[d.method]+' · '+d.seed"
new="o.textContent=({single:'单目标',late_multi:'迟到多目标',clutter_shared:'杂波与共享来源'}[d.scenario]||d.scenario)+' · '+names[d.method]+' · 世界 '+(Array.from(new Set(DATA.filter(x=>x.scenario===d.scenario).map(x=>x.seed))).indexOf(d.seed)+1)"
html=html.replace(old,new)
html=html.replace("x.reason+'：'+JSON.stringify(x.incumbent)",
 "'根据当前最好方案，补充节点替换与任务组合。原方案：'+(x.incumbent.length?x.incumbent.map(a=>SENSORS[a.node].name+' '+({search:'搜索区域',track:'观测航迹',calibrate:'校准基准点'}[a.kind])+(a.kind==='calibrate'?'':a.item)).join('；'):'全部节点不采样')")
html=html.replace('显示评价真值','显示真实目标轨迹')
p.write_text(html,encoding='utf-8')
print('Replay labels updated.')
