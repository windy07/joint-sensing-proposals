"""Predeclared functional policy. It reads only causal beliefs and public tasks.

This is an interface acceptance fixture, not a performance benchmark or a
trained controller. Every returned group still requires projection and paid
mainline scoring before execution. Unknown targets are never initialized here.
"""
from .core import Task,ContractError
from .actions import materialize

def proposal(view,config):
    epoch=view['epoch'];tracks=sorted(view['tracks'],key=lambda t:view['mean'][t['start']])
    cells=config.get('functional_search_cells',[0]);single=len(cells)==1
    tasks=[]
    for node in (0,1):
        if tracks and (single or len(tracks)>node):
            tr=tracks[0 if single else node]
            t=Task(node,'track',tr['id'],role='monostatic');u=[.8,.65,.5,0.,.5,.5,.5,0.]
        else:
            cell=cells[0 if single else node]
            t=Task(node,'search',cell,role='monostatic')
            point=.9 if node==0 else .5 if single else .1
            u=[.8,.65,point,.4,.5,.5,.5,0.]
        tasks.append(materialize(t,view,config,u))
    if epoch==2 and tracks:
        # A surveyed calibration and a real track observation share joint bias.
        tasks[0]=materialize(Task(0,'calibrate',role='monostatic'),view,config,[.8,.4,.5,0.,.5,.5,.5,0.])
    if epoch==3 and tracks:
        # Direct TDOA/FDOA/two-AOA reception uses an actual second receiver.
        tasks[1]=materialize(Task(2,'track',tracks[0]['id'],role='direct'),view,config,[0.,.65,.5,0.,.5,.5,.5,0.])
    return tuple(tasks)
