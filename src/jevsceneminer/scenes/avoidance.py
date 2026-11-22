"""Conservative model-supported stationary-obstacle bypass corrections."""
import math
from bisect import bisect_left, bisect_right

from .lanechanges import number


def _candidate(row):
    facts = row.get('facts')
    if not isinstance(facts, dict):
        return None
    evidence = facts.get('lane_evidence')
    ego = facts.get('ego_samples')
    tracks = facts.get('object_tracks')
    if not isinstance(evidence, dict) or not isinstance(ego, list) or len(ego) < 7 or not isinstance(tracks, list):
        return None
    geometry = evidence.get('ego_geometry')
    if not isinstance(geometry, dict) or any(not number(geometry.get(k)) for k in ('width_m', 'length_m', 'center_forward_m')):
        return None
    if min(geometry['width_m'], geometry['length_m']) <= 0:
        return None
    gap = evidence.get('max_centerline_join_gap_m')
    if not number(gap) or not 0 <= gap <= .5 or not evidence.get('reference_lane_ids'):
        return None
    for point in ego:
        if not isinstance(point, dict) or point.get('ego_data_available') is not True or point.get('intersection') is not False:
            return None
        if not number(point.get('time_s')) or not number(point.get('speed_m_s')):
            return None
        body = point.get('body_occupancy')
        if not isinstance(body, dict) or any(not number(body.get(k)) for k in
                ('reference_offset_m', 'lane_relative_heading_deg', 'reference_lane_fraction', 'ambiguous_overlap_fraction', 'unmapped_fraction')):
            return None
        if abs(body['lane_relative_heading_deg']) > 15:
            return None
        if not 0 <= body['reference_lane_fraction'] <= 1 or not 0 <= body['ambiguous_overlap_fraction'] <= .05 or not 0 <= body['unmapped_fraction'] <= .1:
            return None
    times = [point['time_s'] for point in ego]
    dt = times[1] - times[0]
    if not 0 < dt <= 1 or any(abs(b-a-dt) > .001 for a,b in zip(times,times[1:])) or times.count(0) != 1:
        return None
    offsets = [point['body_occupancy']['reference_offset_m'] for point in ego]
    peak = max(range(len(offsets)), key=lambda i: abs(offsets[i]-offsets[0]))
    displacement = offsets[peak]-offsets[0]
    if not .5 <= abs(displacement) <= 4 or peak < 2 or peak+2 >= len(ego):
        return None
    sign = 1 if displacement > 0 else -1
    if abs(offsets[1]-offsets[0]) > .15:
        return None
    end = next((i for i in range(peak+1,len(ego)-1)
                if abs(offsets[i]-offsets[0]) <= .3 and abs(offsets[i+1]-offsets[0]) <= .3
                and abs(offsets[i+1]-offsets[i]) <= .15),None)
    start = next((i for i in range(1,peak+1) if abs(offsets[i]-offsets[0]) > .15),None)
    if start is None or end is None or not times[start] <= 0 <= times[end]:
        return None
    if min(point['speed_m_s'] for point in ego[start:end+1]) < .5:
        return None
    if any(sign*(offsets[i+1]-offsets[i]) < -.15 for i in range(start-1,peak)):
        return None
    if any(sign*(offsets[i+1]-offsets[i]) > .15 for i in range(peak,end)):
        return None
    if min(ego[i]['body_occupancy']['reference_lane_fraction'] for i in (0,1,end,end+1)) < .8:
        return None
    candidates=[]
    for track in tracks:
        if not isinstance(track,dict) or not isinstance(track.get('track_id'),(str,int)) or isinstance(track.get('track_id'),bool):
            continue
        observations=track.get('observations')
        if not isinstance(observations,list) or len(observations)<3 or any(not isinstance(p,dict) for p in observations):
            continue
        keys=('time_s','ahead_m','left_m','speed_m_s','reference_lane_overlap_m2','ego_body_clearance_m')
        if any(any(not number(p.get(k)) for k in keys) for p in observations):
            continue
        ordered=sorted(observations,key=lambda p:p['time_s'])
        if any(b['time_s']-a['time_s']<=0 or b['time_s']-a['time_s']>1. for a,b in zip(ordered,ordered[1:])):
            continue
        related=[p for p in ordered if times[start]-1 <= p['time_s'] <= times[end]+1]
        if (not related or ordered[0]['time_s'] > times[start] or ordered[-1]['time_s'] < times[end]
                or any(p['speed_m_s'] < 0 or p['speed_m_s'] >= .5 for p in related)):
            continue
        headings=[math.radians(point['body_occupancy']['lane_relative_heading_deg']) for point in ego[start:end+1]]
        # Object positions are in ego coordinates. Use every observed heading as
        # a conservative bound rather than treating ego-left as lane-left.
        def opposite_reference_side(point):
            return all(sign*(point['ahead_m']*math.sin(h)+point['left_m']*math.cos(h)) < 0 for h in headings)
        ahead=[p for p in related if p['time_s'] <= times[peak] and p['ahead_m'] > 0
               and p['reference_lane_overlap_m2'] > .01 and opposite_reference_side(p)]
        behind=[p for p in related if times[peak] <= p['time_s'] <= times[end] and p['ahead_m'] < 0]
        if ahead and behind and min(p['ego_body_clearance_m'] for p in related) <= 3:
            candidates.append(track['track_id'])
    if len(set(candidates)) != 1:
        return None
    return dict(first_s=times[start],last_s=times[end],track_id=candidates[0],lateral_displacement_m=displacement)


def avoidance_overrides(rows, step_s, min_prob=.4):
    if not number(step_s) or step_s <= 0 or not number(min_prob) or not 0 <= min_prob <= 1:
        raise ValueError('invalid avoidance cadence or probability')
    rows=sorted(rows,key=lambda row:row['t_ns'])
    times=[row['t_ns'] for row in rows]
    if len(set(times)) != len(times):
        raise ValueError('duplicate timestamps')
    overrides={}
    audit=dict(version='hybrid-avoidance-1',accepted_samples=0,min_model_probability=min_prob,decisions=[])
    for row in rows:
        if row.get('lateral') not in ('keep_lane','avoidance'):
            continue
        candidate=_candidate(row)
        if candidate is None:
            continue
        first=row['t_ns']+round(candidate['first_s']*1e9)
        last=row['t_ns']+round(candidate['last_s']*1e9)
        interval=rows[bisect_left(times,first):bisect_right(times,last)]
        if not interval or any(r.get('lateral') not in ('keep_lane','avoidance') for r in interval):
            continue
        if interval[0]['t_ns']-first >= step_s*1e9 or last-interval[-1]['t_ns'] >= step_s*1e9:
            continue
        if any(abs((b['t_ns']-a['t_ns'])/1e9-step_s) > step_s*.1 for a,b in zip(interval,interval[1:])):
            continue
        def supported(r):
            probs=r.get('lateral_probs')
            p=probs.get('avoidance') if isinstance(probs,dict) else None
            return r.get('lateral')=='avoidance' and number(p) and min_prob<=p<=1
        if not any(supported(r) for r in interval):
            continue
        overrides[row['t_ns']]='avoidance'
        audit['accepted_samples']+=1
        audit['decisions'].append(dict(t_ns=str(row['t_ns']),first_ns=str(first),last_ns=str(last),**candidate))
    return overrides,audit
