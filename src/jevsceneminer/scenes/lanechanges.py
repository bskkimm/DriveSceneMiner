"""Opt-in model/map/motion lane changes; polygon overlap is supporting evidence.

All measurements for a candidate use one window-local, continuous reference.
Unknown topology, geometry or map correspondence causes abstention.
"""
import math
from bisect import bisect_left, bisect_right

LABELS = ('lane_change_left','lane_change_right')


def number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def lane_id(value):
    return isinstance(value,int) and not isinstance(value,bool)


def _context(row):
    facts=row.get('facts')
    if not isinstance(facts,dict):return None
    observations=facts.get('ego_samples')
    lanes=facts.get('lanes')
    evidence=facts.get('lane_evidence')
    if not isinstance(observations,list) or len(observations)<5 or not isinstance(lanes,list) or not isinstance(evidence,dict):return None
    geometry=evidence.get('ego_geometry')
    if not isinstance(geometry,dict) or any(not number(geometry.get(k)) for k in ('length_m','width_m','center_forward_m')):return None
    if min(geometry['length_m'],geometry['width_m'])<=0:return None
    if not number(evidence.get('max_centerline_join_gap_m')) or not 0<=evidence['max_centerline_join_gap_m']<=.5:return None
    if any(not isinstance(lane,dict) or not lane_id(lane.get('id')) for lane in lanes):return None
    by_id={lane['id']:lane for lane in lanes}
    if len(by_id)!=len(lanes):return None
    parent={lid:lid for lid in by_id}
    def group(lid):
        while parent[lid]!=lid:lid=parent[lid]
        return lid
    for lid,lane in by_id.items():
        following=lane.get('following')
        if not isinstance(following,list) or any(not isinstance(f,dict) or not lane_id(f.get('id')) for f in following):return None
        if any(lane.get(side) is not None and not lane_id(lane[side]) for side in ('left_neighbor_id','right_neighbor_id')):return None
        if len(following)==1 and isinstance(following[0],dict):
            successor=following[0].get('id'); target=by_id.get(successor) if lane_id(successor) else None
            if (target and target.get('previous_ids')==[lid] and lane.get('intersection') is False
                    and target.get('intersection') is False):
                parent[group(successor)]=group(lid)
    for obs in observations:
        if not isinstance(obs,dict) or obs.get('ego_data_available') is not True:return None
        lid=obs.get('matched_lane_id')
        if obs.get('intersection') is not False or not lane_id(lid) or lid not in by_id:return None
        if by_id[lid].get('kind') not in ('road','highway','bus lane','bus_lane'):return None
        if by_id[lid].get('intersection') is not False:return None
        if any(not number(obs.get(key)) for key in ('time_s','speed_m_s','relative_heading_deg')):return None
        body=obs.get('body_occupancy')
        if not isinstance(body,dict) or not number(body.get('reference_offset_m')):return None
        if any(not number(body.get(key)) or not 0<=body[key]<=limit for key,limit in
               [('ambiguous_overlap_fraction',.05),('unmapped_fraction',.1)]):return None
        shares=body.get('lane_occupancy')
        if not isinstance(shares,list):return None
        if any(not isinstance(p,dict) or not lane_id(p.get('lane_id')) or not number(p.get('fraction')) or not 0<=p['fraction']<=1 or
               p.get('direction_relation')!='same-direction' for p in shares):return None
        if not isinstance(body.get('crossed_boundary_sides'),list):return None
    times=[obs['time_s'] for obs in observations]
    dt=times[1]-times[0]
    if not 0<dt<=1 or any(abs(b-a-dt)>.001 for a,b in zip(times,times[1:])):return None
    if sum(t==0 for t in times)!=1:return None
    reference=evidence.get('observed_reference_lane_ids',evidence.get('reference_lane_ids'))
    if not isinstance(reference,list) or not reference or any(not lane_id(lid) or lid not in by_id for lid in reference):return None
    original=group(reference[0])
    if any(group(lid)!=original for lid in reference):return None
    def share(obs,chain):
        # A maximum is a conservative lower bound; never sum overlapping polygons.
        return max((p['fraction'] for p in obs['body_occupancy']['lane_occupancy']
                    if p.get('lane_id') in by_id and group(p['lane_id'])==chain),default=0.)
    return observations,by_id,group,original,share,dt


def lanechange_overrides(rows,step_s,min_prob=.4):
    if not number(step_s) or step_s<=0 or not number(min_prob) or not 0<=min_prob<=1:
        raise ValueError('lane-change cadence/support must be finite and valid')
    rows=sorted(rows,key=lambda r:r['t_ns'])
    if len({r['t_ns'] for r in rows})!=len(rows):raise ValueError('duplicate lane-change timestamps')
    times=[r['t_ns'] for r in rows]
    seeds=set()
    for row in rows:
        label=row.get('lateral'); probs=row.get('lateral_probs')
        probability=probs.get(label) if isinstance(probs,dict) and isinstance(label,str) else None
        if label in LABELS and number(probability) and min_prob<=probability<=1:seeds.add((row['t_ns'],label))
    overrides={}; audit=dict(version='hybrid-lanechanges-1',accepted_samples=0,stationary_veto_samples=0,
        min_model_probability=min_prob,thresholds=dict(min_displacement_m=1.5,max_displacement_m=5.,
        max_heading_range_deg=20.,stable_share=.8,max_ambiguous_share=.05,max_unmapped_share=.1,
        min_speed_m_s=.5,max_reference_join_gap_m=.5,min_lateral_rate_m_s=.05,
        max_lateral_rate_m_s=2.,settling_tolerance_m=.15,stationary_speed_m_s=.1,
        stationary_offset_span_m=.1,min_motion_increments=2,max_table_step_s=1.),decisions=[])
    for row in rows:
        if row.get('lateral') not in ('keep_lane',*LABELS):continue
        context=_context(row)
        if context is None:continue
        obs,lanes,group,original,share,dt=context
        offsets=[s['body_occupancy']['reference_offset_m'] for s in obs]
        chains=[group(s['matched_lane_id']) for s in obs]
        now=next(i for i,s in enumerate(obs) if s['time_s']==0)
        # No lateral movement: stable same-chain stationary observations cannot
        # constitute a new lane change. Veto also raw keep-lane smoothing holes.
        if 0<now<len(obs)-1:
            local=obs[now-1:now+2]
            if (all(s['speed_m_s']<=.1 and share(s,chains[now])>=.8 for s in local)
                    and len(set(chains[now-1:now+2]))==1 and max(offsets[now-1:now+2])-min(offsets[now-1:now+2])<=.1):
                overrides[row['t_ns']]='keep_lane'; audit['stationary_veto_samples']+=1
                audit['decisions'].append(dict(t_ns=str(row['t_ns']),label='keep_lane',reason='stationary stable lane chain'))
                continue
        candidates=[]
        for i in range(1,len(obs)):
            source,target=chains[i-1],chains[i]
            if source==target or source!=original:continue
            a_lane=lanes[obs[i-1]['matched_lane_id']]; b_lane=lanes[obs[i]['matched_lane_id']]
            for direction,sign,opposite in [('left',1,'right'),('right',-1,'left')]:
                if (a_lane.get(direction+'_neighbor_id')!=b_lane['id'] or b_lane.get(opposite+'_neighbor_id')!=a_lane['id']):continue
                if a_lane['following'] or b_lane['following']:
                    # Explicit forks are ambiguous even if a neighbor edge exists.
                    if len(a_lane['following'])>1 or len(b_lane['following'])>1:continue
                a=i-1; b=i
                while a>0 and sign*(offsets[a]-offsets[a-1])>.05*dt:a-=1
                while b+1<len(obs) and sign*(offsets[b+1]-offsets[b])>.05*dt:b+=1
                if a<1 or b+2>=len(obs) or not obs[a+1]['time_s']<=0<=obs[b]['time_s']:continue
                if chains[a-1:a+1]!=[source,source] or chains[b:b+3]!=[target]*3:continue
                if abs(offsets[a]-offsets[a-1])>.15 or max(offsets[b:b+3])-min(offsets[b:b+3])>.15:continue
                if not 1.5<=sign*(offsets[b]-offsets[a])<=5:continue
                increments=[sign*(offsets[k+1]-offsets[k]) for k in range(a,b)]
                if sum(delta>.05*dt for delta in increments)<2 or any(delta>2*dt for delta in increments):continue
                if any(sign*(offsets[k+1]-offsets[k])<-.1 for k in range(a,b)):continue
                chosen=obs[a:b+3]
                if any(s['speed_m_s']<.5 for s in obs[a+1:b+1]):continue
                headings=[s['relative_heading_deg'] for s in chosen]
                if max(headings)-min(headings)>20:continue
                if not any(direction in s['body_occupancy']['crossed_boundary_sides'] for s in chosen):continue
                if any(share(s,source)<.8 or share(s,target)>.1 for s in obs[a-1:a+1]):continue
                if any(share(s,target)<.8 or share(s,source)>.1 for s in obs[b:b+3]):continue
                label='lane_change_'+direction
                lo=row['t_ns']+round(obs[a+1]['time_s']*1e9); hi=row['t_ns']+round(obs[b]['time_s']*1e9)
                interval_rows=rows[bisect_left(times,lo):bisect_right(times,hi)]
                if not any((r['t_ns'],label) in seeds for r in interval_rows):continue
                if any(r.get('lateral') not in ('keep_lane',label) for r in interval_rows):continue
                candidates.append(dict(t_ns=str(row['t_ns']),label=label,first_ns=str(lo),last_ns=str(hi),
                    source_chain=source,target_chain=target,displacement_m=offsets[b]-offsets[a],
                    reason='model + reciprocal neighbor + continuous motion + crossing + settling'))
        if len(candidates)==1:
            decision=candidates[0]; overrides[row['t_ns']]=decision['label']; audit['accepted_samples']+=1
            audit['decisions'].append(decision)
    return overrides,audit
