"""The one executable numeric action schema and joint device projection.

Power is peak RF watts in an LFM pulse train. Duty cycle and energy are derived
from pulse width, PRI and integer pulse slots, never separately generated.
Each search visit occupies its own interval; direct reception reserves BOTH
receivers. Projection never changes environmental properties.
"""
from __future__ import annotations
from dataclasses import replace
from fractions import Fraction
import math
import numpy as np
from .core import Task,ContractError,wrap,digest

ACTION_SCHEMA='ain.spatial.executable.pulse_scan.v4'
CONTROL_NAMES=('power','dwell','point_offset','scan_span','left_weight','centre_weight','right_weight','start')
START_CONTROL_SPEC={
 'semantics':'planned physical start within the decision cycle, before conflict projection',
 'scale':'smallest power of two seconds greater than or equal to dt_s',
 'target':'executed start_s / scale; original requested_start_s retained as proposal provenance',
 'domain':'0 <= start_s <= dt_s; the complete action must finish within dt_s',
 'projection':'preserve planned lower bounds and serialize conflicts; reject infeasible groups'}
DEFAULT_DEVICE={
 'carrier_Hz':3.e9,'bandwidth_Hz':1.e7,'pulse_width_s':2.e-6,'pri_s':1.e-4,
 'aperture_diameter_m':.6,'aperture_efficiency':.6,'noise_factor':4.,'loss':4.,'temperature_K':290.,
 'power_min_W':2.,'power_max_W':100.,'power_quantum_W':.1,
 'dwell_min_s':.006,'dwell_max_s':.18,'point_quantum_rad':math.radians(.05),
 'point_offset_limit_rad':math.radians(12.),'scan_span_max_rad':math.radians(15.),
 'turn_rate_rad_s':25.,'settle_s':.001,'rx_power_W':5.,'processing_ops_per_sample':8.,
 'processing_ops_s':2.e9,'communication_bps':1.e6,'packet_bytes':512,
 'passive_delay_window_s':2.e-4,'direct_reference_gain':10.}

def device(config,node):
 d={**DEFAULT_DEVICE,**config.get('device_defaults',{}),**config.get('devices',{}).get(str(node),{})}
 if not (0<d['pulse_width_s']<d['pri_s'] and 0<d['dwell_min_s']<=d['dwell_max_s']):raise ContractError('DEVICE_TIME_DOMAIN')
 d['beam_rad']=.886*(299792458./d['carrier_Hz'])/d['aperture_diameter_m']
 d['gain']=d['aperture_efficiency']*(math.pi*d['aperture_diameter_m']/(299792458./d['carrier_Hz']))**2
 return d

def base_bearing(t,view,config):
 from .physics import sensor_list,BEACON,CELLS,cell_list,azimuth
 s=sensor_list(config)[t.node]
 if t.kind=='calibrate':pos=BEACON
 elif t.kind=='search':pos=cell_list(config)[t.item]
 else:
  track=next((x for x in view['tracks'] if x['id']==t.item),None)
  if track is None:raise ContractError('STALE_TRACK_ID')
  pos=np.asarray(view['mean'])[track['start']:track['start']+2]
 return azimuth(pos,s)

def integer_bounds(lower,upper,quantum):
 """Exact decimal hardware bounds; integer truncation never loses an endpoint."""
 lo=Fraction(str(lower))/Fraction(str(quantum));hi=Fraction(str(upper))/Fraction(str(quantum))
 bounds=(-(-lo.numerator//lo.denominator),hi.numerator//hi.denominator)
 if bounds[0]>bounds[1]:raise ContractError('EMPTY_HARDWARE_GRID')
 return bounds

def angular_span_slots(d):
 ratio=d['scan_span_max_rad']/d['point_quantum_rad'];nearest=round(ratio)
 # Angular radians are binary approximations of degree specifications. Snap
 # only a floating representation (four ULPs), never an extra hardware step.
 if abs(ratio-nearest)<=4*math.ulp(ratio):return nearest
 return math.floor(ratio)

def pulse_bounds(d,kind):
 lo,hi=integer_bounds(d['dwell_min_s'],d['dwell_max_s'],d['pri_s'])
 lo=max(lo,6 if kind=='search' else 2)
 if lo>hi:raise ContractError('INSUFFICIENT_PULSE_DOMAIN')
 return lo,hi

def allocate_pulses(slots,weights):
 """Two reserved pulses per beam, then exact rational largest remainders.

 Zero weights are legal; the all-zero control has a uniform representative.
 Canonical inverse weights are proportional to (executed_count - 2).
 """
 w=[Fraction(str(float(x))) for x in weights]
 if len(w)!=3 or any(x<0 for x in w) or slots<6:raise ContractError('PULSE_ALLOCATION_DOMAIN')
 if not sum(w):w=[Fraction(1)]*3
 quota=[(slots-6)*x/sum(w) for x in w]
 extra=[x.numerator//x.denominator for x in quota]
 order=sorted(range(3),key=lambda i:(-(quota[i]-extra[i]),(1,0,2)[i]))
 for i in order[:slots-6-sum(extra)]:extra[i]+=1
 return [n+2 for n in extra]

def _object(t,view):
 from .physics import CELLS
 if type(t.node) is not int or not 0<=t.node<4:raise ContractError('NODE_INDEX')
 if t.kind not in ('search','track','calibrate'):raise ContractError('ACTION_KIND')
 if type(t.item) is not int:raise ContractError('ACTION_OBJECT_TYPE')
 if t.kind=='search' and not 0<=t.item<len(CELLS):raise ContractError('SEARCH_CELL')
 if t.kind=='track' and not any(x['id']==t.item for x in view['tracks']):raise ContractError('STALE_TRACK_ID')
 if t.kind=='calibrate' and t.item!=-1:raise ContractError('CALIBRATION_OBJECT')

def start_scale(config):
 """Binary scaling preserves stored physical timestamps in a label round trip."""
 dt=float(config['dt_s'])
 if not math.isfinite(dt) or dt<=0:raise ContractError('CYCLE_DURATION_DOMAIN')
 mantissa,exponent=math.frexp(dt)
 try:scale=math.ldexp(1.,exponent-1 if mantissa==.5 else exponent)
 except OverflowError:raise ContractError('CYCLE_DURATION_DOMAIN') from None
 if not math.isfinite(scale) or scale<=0:raise ContractError('CYCLE_DURATION_DOMAIN')
 return scale

def materialize(t,view,config,controls=None):
 """One normalized control domain, followed by integer hardware quantization."""
 _object(t,view)
 d=device(config,t.node);u=np.asarray(controls if controls is not None else [.45,.4,.5,.4,.5,.5,.5,0.],float)
 if u.shape!=(8,) or not np.isfinite(u).all():raise ContractError('NUMERIC_CONTROL_SCHEMA')
 u=np.clip(u,0,1);role=t.role or ('monostatic' if t.node<2 else 'bistatic')
 pmin,pmax=integer_bounds(d['power_min_W'],d['power_max_W'],d['power_quantum_W'])
 power=float(Fraction(pmin+int(round(float(u[0])*(pmax-pmin))))*Fraction(str(d['power_quantum_W']))) if role=='monostatic' else 0.
 nmin,nmax=pulse_bounds(d,t.kind);slots=nmin+int(round(float(u[1])*(nmax-nmin)))
 dwell=float(Fraction(slots)*Fraction(str(d['pri_s'])));delta=d['point_quantum_rad'];base=base_bearing(t,view,config)
 ilo=math.ceil((base-d['point_offset_limit_rad'])/delta);ihi=math.floor((base+d['point_offset_limit_rad'])/delta)
 if ilo>ihi:raise ContractError('EMPTY_STEERING_GRID')
 centre_index=min(ihi,max(ilo,round((base+(2*u[2]-1)*d['point_offset_limit_rad'])/delta)))
 centre=centre_index*delta;angles=[centre];counts=[slots]
 if t.kind=='search':
  maxspan=angular_span_slots(d)
  span_index=int(round(float(u[3])*maxspan));counts=allocate_pulses(slots,u[4:7])
  angles=[(centre_index-span_index)*delta,centre,(centre_index+span_index)*delta]
 scan=tuple((float(wrap(a)),float(Fraction(int(n))*Fraction(str(d['pri_s'])))) for a,n in zip(angles,counts))
 ref=(3 if t.node==2 else 2) if role=='direct' else -1
 requested=float(u[7]*start_scale(config))
 return replace(t,role=role,power_W=power,dwell_s=float(dwell),bearing_rad=float(wrap(centre)),scan=scan,
   start_s=requested,requested_start_s=requested,reference_node=ref,
   reference_id=('surveyed_beacon' if t.kind=='calibrate' else 'external_illuminator' if role=='bistatic' else f'receiver/{ref}' if role=='direct' else 'own_transmitter'),
   calibration='effective_delay_frequency_bearing' if t.kind=='calibrate' else '',
   carrier_Hz=d['carrier_Hz'],bandwidth_Hz=d['bandwidth_Hz'],pulse_width_s=d['pulse_width_s'],pri_s=d['pri_s'],beam_rad=d['beam_rad'],
   waveform='LFM' if role=='monostatic' else 'external_wideband',
   initial_bearing_rad=float(view.get('node_pointing_rad',[0.,0.,0.,0.])[t.node]))

def visits(t,config=None):
 """Concrete beam visits with midpoints, pulse counts, and turnaround gaps."""
 d=device(config or {},t.node);clock=t.start_s;prev=t.initial_bearing_rad;ans=[]
 for i,(bearing,duration) in enumerate(t.scan or ((t.bearing_rad,t.dwell_s),)):
  turn=d['settle_s']+(abs(float(wrap(bearing-prev)))/d['turn_rate_rad_s'] if prev is not None else 0.)
  clock+=turn
  a=replace(t,start_s=clock,dwell_s=duration,bearing_rad=bearing,scan=((bearing,duration),),initial_bearing_rad=bearing)
  ans.append({'index':i,'task':a,'start_s':clock,'end_s':clock+duration,'mid_s':clock+duration/2,
              'pulse_count':int(round(duration/t.pri_s)) if t.role=='monostatic' else 0,'turn_s':turn})
  clock+=duration;prev=bearing
 return ans

def owners(t):return (t.node,t.reference_node) if t.role=='direct' else (t.node,)

def resource(t,config=None):
 d=device(config or {},t.node);v=visits(t,config);tx=(sum(x['pulse_count'] for x in v)*t.pulse_width_s if t.role=='monostatic' else 0.)
 # Exact integer pulse/decimal power ledger is also exposed to the caller.
 tx_exact=Fraction(sum(x['pulse_count'] for x in v))*Fraction(str(t.pulse_width_s))
 e_exact=tx_exact*Fraction(str(t.power_W));listen=sum(x['task'].dwell_s for x in v)*(2 if t.role!='monostatic' else 1)
 ops=listen*t.bandwidth_Hz*d['processing_ops_per_sample'];bytes_=len(v)*d['packet_bytes']
 return {'energy_J':float(e_exact),'energy_exact_J':str(e_exact),'tx_s':float(tx_exact),'tx_exact_s':str(tx_exact),
  'listen_s':listen,'rx_energy_J':listen*d['rx_power_W'],'processing_ops':ops,'communication_bytes':int(bytes_),
  'occupied_end_s':max((x['end_s'] for x in v),default=t.start_s)}

def validate_task_numeric(t,view,config):
 _object(t,view)
 d=device(config,t.node)
 if not t.role or not t.scan or not np.isfinite([t.start_s,t.dwell_s,t.power_W,t.bearing_rad]).all():raise ContractError('NUMERIC_ACTION_REQUIRED')
 if t.kind not in ('search','track','calibrate') or t.mode not in ('full','aoa'):raise ContractError('ACTION_SCHEMA')
 if (t.role=='monostatic' or t.kind=='calibrate') and t.mode!='full':raise ContractError('UNSUPPORTED_AOA_ONLY_DEVICE_MODE')
 if t.mode not in config.get('allowed_measurement_modes',['full']):raise ContractError('MEASUREMENT_MODE_CAPABILITY')
 if t.kind not in config.get('allowed_task_kinds',['search','track','calibrate']):raise ContractError('TASK_CAPABILITY_MASK')
 if t.role not in config.get('allowed_signal_roles',['monostatic','bistatic','direct']):raise ContractError('SIGNAL_CAPABILITY_MASK')
 expected_ref='surveyed_beacon' if t.kind=='calibrate' else 'external_illuminator' if t.role=='bistatic' else f'receiver/{t.reference_node}' if t.role=='direct' else 'own_transmitter'
 if t.reference_id!=expected_ref:raise ContractError('REFERENCE_IDENTITY')
 if t.calibration!=('effective_delay_frequency_bearing' if t.kind=='calibrate' else ''):raise ContractError('CALIBRATION_OBJECT')
 if t.waveform!=('LFM' if t.role=='monostatic' else 'external_wideband'):raise ContractError('WAVEFORM_CAPABILITY')
 if not math.isclose(t.initial_bearing_rad,view.get('node_pointing_rad',[0.]*4)[t.node],abs_tol=1e-10):raise ContractError('INITIAL_POINTING_STATE')
 if t.node<2 and t.role!='monostatic' or t.node>=2 and t.role not in ('direct','bistatic'):raise ContractError('DEVICE_SIGNAL_ROLE')
 for k in ('carrier_Hz','bandwidth_Hz','pulse_width_s','pri_s','beam_rad'):
  if not math.isclose(getattr(t,k),d[k],rel_tol=1e-10,abs_tol=1e-12):raise ContractError('HARDWARE_CONSTANT_NOT_CONTROL:'+k)
 if t.role=='direct' and t.reference_node not in ({2,3}-{t.node}):raise ContractError('REFERENCE_RECEIVER')
 if t.role!='direct' and t.reference_node!=-1:raise ContractError('UNUSED_REFERENCE_RECEIVER')
 if t.role!='monostatic' and t.power_W!=0:raise ContractError('UNCONTROLLABLE_EXTERNAL_TRANSMITTER')
 if t.role=='monostatic' and not d['power_min_W']<=t.power_W<=d['power_max_W']:raise ContractError('POWER_DOMAIN')
 if abs(t.power_W/d['power_quantum_W']-round(t.power_W/d['power_quantum_W']))>1e-6:raise ContractError('POWER_QUANTIZATION')
 nmin,nmax=pulse_bounds(d,t.kind)
 total_slots=round(t.dwell_s/t.pri_s)
 if not nmin<=total_slots<=nmax or abs(t.dwell_s/t.pri_s-total_slots)>1e-7:raise ContractError('DWELL_DOMAIN')
 if len(t.scan)!=(3 if t.kind=='search' else 1):raise ContractError('BEAM_COUNT_DOMAIN')
 if abs(sum(b[1] for b in t.scan)-t.dwell_s)>1e-12 or sum(round(b[1]/t.pri_s) for b in t.scan)!=total_slots:raise ContractError('SCAN_TIME_CONSERVATION')
 for b,dur in t.scan:
  if not math.isfinite(b) or not -math.pi<=b<math.pi or abs(b/d['point_quantum_rad']-round(b/d['point_quantum_rad']))>1e-6:raise ContractError('BEAM_POINT_GRID')
  if not math.isfinite(dur) or round(dur/t.pri_s)<2 or abs(dur/t.pri_s-round(dur/t.pri_s))>1e-7:raise ContractError('PULSE_SLOT_GRID')
 if abs(float(wrap(t.bearing_rad-base_bearing(t,view,config))))>d['point_offset_limit_rad']+1e-12:raise ContractError('POINT_OFFSET_DOMAIN')
 if not -math.pi<=t.bearing_rad<math.pi:raise ContractError('CENTRE_DOMAIN')
 mid=1 if t.kind=='search' else 0
 if abs(float(wrap(t.scan[mid][0]-t.bearing_rad)))>1e-12:raise ContractError('BEAM_CENTRE_MISMATCH')
 if t.kind=='search':
  left=float(wrap(t.bearing_rad-t.scan[0][0]));right=float(wrap(t.scan[2][0]-t.bearing_rad))
  if left< -1e-12 or right< -1e-12 or max(left,right)>d['scan_span_max_rad']+1e-12:raise ContractError('SCAN_SPAN_DOMAIN')
  if abs(left-right)>1e-12:raise ContractError('SCAN_SYMMETRY_DOMAIN')
 if t.requested_start_s is not None:
  if not math.isfinite(t.requested_start_s) or not 0<=t.requested_start_s<=config['dt_s']:raise ContractError('RAW_START_DOMAIN')
  if t.start_s+1e-12<t.requested_start_s:raise ContractError('PROJECTED_START_PRECEDES_REQUEST')
 if not math.isfinite(t.start_s) or t.start_s<0:raise ContractError('CYCLE_TIME')
 return True

def validate_numeric(schedule,view,config):
 if not isinstance(schedule,tuple) or len(schedule)>config['max_simultaneous']:raise ContractError('ACTION_GROUP_CAP')
 occupied={};transmit=[];used_owners=set()
 for t in schedule:
  validate_task_numeric(t,view,config)
  end=resource(t,config)['occupied_end_s']
  if t.start_s<0 or end>config['dt_s']+1e-12:raise ContractError('CYCLE_TIME')
  for n in owners(t):
   if n in used_owners:raise ContractError('ONE_TASK_PER_NODE_PER_CYCLE')
   used_owners.add(n)
   for lo,hi in occupied.get(n,[]):
    if max(lo,t.start_s)<min(hi,end)-1e-12:raise ContractError('RX_TX_REFERENCE_CONFLICT')
   occupied.setdefault(n,[]).append((t.start_s,end))
  if t.role=='monostatic':
   for lo,hi,fc,bw in transmit:
    if abs(fc-t.carrier_Hz)<(bw+t.bandwidth_Hz)/2 and max(lo,t.start_s)<min(hi,end)-1e-12:raise ContractError('ACTIVE_SPECTRUM_CONFLICT')
   transmit.append((t.start_s,end,t.carrier_Hz,t.bandwidth_Hz))
 totals={k:sum(resource(t,config)[k] for t in schedule) for k in ('listen_s','processing_ops','communication_bytes')}
 for k,limit in [('listen_s','receive_budget_s_per_cycle'),('processing_ops','processing_budget_ops_per_cycle'),('communication_bytes','communication_budget_bytes_per_cycle')]:
  if totals[k]>config.get(limit,float('inf'))+1e-9:raise ContractError('RESOURCE_DOMAIN:'+k)
 return True

def project(schedule,view,config):
 """One deterministic joint projection, shared by every proposal mechanism.

 Device quantization occurs in materialize. Here serialize conflicting owners
 and same-frequency transmitters, then reject rather than silently drop tasks.
 """
 raw=[t.to_dict() for t in schedule];out=[];ends={};tx_end=0.;changes=[]
 try:
  for t in sorted(schedule,key=lambda x:(x.start_s,x.node,x.semantic_id())):
   validate_task_numeric(t,view,config)
   old=t.start_s;start=max([old,*[ends.get(n,0.) for n in owners(t)],tx_end if t.role=='monostatic' else 0.])
   t=replace(t,start_s=start);end=resource(t,config)['occupied_end_s']
   if start!=old:changes.append({'semantic_id':t.semantic_id(),'start_s_before':old,'start_s_after':start})
   for n in owners(t):ends[n]=end
   if t.role=='monostatic':tx_end=end
   out.append(t)
  from .physics import legal
  a=tuple(sorted(out));legal(a,view,config)
  return a,{'raw':raw,'corrections':changes,'executed':[t.to_dict() for t in a],'status':'ACCEPTED','reason':None}
 except (ValueError,OverflowError,TypeError,IndexError) as e:
  return None,{'raw':raw,'corrections':changes,'executed':None,'status':'REJECTED','reason':str(e)}


def control_target(task,view,config):
    """Canonical physical controls, including the actual projected start time."""
    validate_task_numeric(task,view,config)
    d=device(config,task.node);u=np.zeros(8);mask=np.ones(8)
    pmin,pmax=integer_bounds(d['power_min_W'],d['power_max_W'],d['power_quantum_W'])
    u[0]=(round(task.power_W/d['power_quantum_W'])-pmin)/max(1,pmax-pmin)
    lo,hi=pulse_bounds(d,task.kind);slots=round(task.dwell_s/task.pri_s)
    u[1]=(slots-lo)/max(1,hi-lo)
    u[2]=(float(wrap(task.bearing_rad-base_bearing(task,view,config)))/d['point_offset_limit_rad']+1)/2
    if task.kind=='search':
        maxspan=angular_span_slots(d)
        u[3]=round(float(wrap(task.scan[-1][0]-task.bearing_rad))/d['point_quantum_rad'])/max(1,maxspan)
        extra=np.array([round(v[1]/task.pri_s)-2 for v in task.scan],float)
        u[4:7]=extra/extra.max() if extra.max()>0 else .5
    else:mask[3:7]=0
    if not 0<=task.start_s<=config['dt_s']:raise ContractError('CYCLE_TIME')
    u[7]=task.start_s/start_scale(config)
    if task.role!='monostatic':mask[0]=0
    return np.clip(u,0,1),mask

def control_label(task,view,config):
    target,mask=control_target(task,view,config)
    reasons={CONTROL_NAMES[i]:('uncontrolled_external_power' if i==0 else 'single_beam_not_scan') for i,m in enumerate(mask) if m==0}
    return {'target':target.tolist(),'mask':mask.tolist(),'mask_reasons':reasons,
      'executable_pulses':[round(b[1]/task.pri_s) for b in task.scan],
      'requested_start_s':task.requested_start_s,'projected_start_s':task.start_s,
      'start_scale_s':start_scale(config),'start_target_semantics':START_CONTROL_SPEC['target'],
      'execution_id':task.execution_id()}

def teacher_control_labels(schedule,view,config):
    """Validate exact physical reconstruction and retain original proposal history.

    All executable fields come from materialize and project. Only the original
    request metadata is reattached from the source record; it is not an extra
    network output or a replacement for the decoded physical start time.
    The final comparison remains exact for every field, including start_s.
    """
    labels=[control_label(t,view,config) for t in schedule]
    decoded=tuple(replace(materialize(t,view,config,l['target']),requested_start_s=t.requested_start_s)
                  for t,l in zip(schedule,labels))
    reconstructed,receipt=project(decoded,view,config)
    if reconstructed is None or reconstructed!=tuple(sorted(schedule)):
        raise ContractError('TEACHER_EXECUTABLE_LABEL_ROUNDTRIP')
    return labels
