"""Conservative, opt-in turn corrections from canonical NOW facts and model votes.

Rules select labels, never synthesize probabilities. Missing evidence abstains.
Only complete, bracketed rotation bouts are assessed at the inference cadence.
"""
import math

VERSION = 'hybrid-turns-1'
MIN_YAW_DEG_S = 3.
MIN_SPEED_M_S = .5
MIN_HEADING_DEG = 40.
MAX_HEADING_DEG = 135.


def _now(row):
    facts = row.get('facts')
    if not isinstance(facts, dict):
        return None
    samples = facts.get('ego_samples')
    if not isinstance(samples, list) or any(not isinstance(sample,dict) for sample in samples):
        return None
    matches = [sample for sample in samples if sample.get('time_s') == 0]
    if len(matches) != 1:
        return None
    sample = matches[0]
    if sample.get('ego_data_available') is not True:
        return None
    for key in ('speed_m_s', 'yaw_rate_rad_s'):
        value = sample.get(key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
            return None
    return sample


def turn_overrides(rows, step_s, min_prob=.4):
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError('hybrid turn cadence must be finite and positive')
    if not math.isfinite(min_prob) or not 0 <= min_prob <= 1:
        raise ValueError('hybrid model support must be between zero and one')
    rows = sorted(rows, key=lambda row: row['t_ns'])
    if len({row['t_ns'] for row in rows}) != len(rows):
        raise ValueError('hybrid turns require unique timestamps')
    facts = [_now(row) for row in rows]
    def rotation(sample):
        if sample is None or sample['speed_m_s'] < MIN_SPEED_M_S:
            return 0
        rate = math.degrees(sample['yaw_rate_rad_s'])
        return (1 if rate > 0 else -1) if abs(rate) >= MIN_YAW_DEG_S else 0
    signs = [rotation(sample) for sample in facts]
    def adjacent(a,b):
        return abs((rows[b]['t_ns']-rows[a]['t_ns'])/1e9-step_s) < step_s*.1
    def model_support(row,label):
        probabilities = row.get('lateral_probs')
        if row.get('lateral') != label or not isinstance(probabilities,dict):
            return False
        probability = probabilities.get(label)
        return (not isinstance(probability,bool) and isinstance(probability,(int,float))
                and math.isfinite(probability) and min_prob <= probability <= 1)
    overrides = {}
    audit = dict(version=VERSION, accepted_turns=0, rejected_curve_samples=0,
        yaw_threshold_deg_s=MIN_YAW_DEG_S, speed_threshold_m_s=MIN_SPEED_M_S,
        heading_range_deg=[MIN_HEADING_DEG,MAX_HEADING_DEG], min_model_probability=min_prob,
        boundary_method='inference cells of bracketed same-direction rotation', decisions=[])
    i = 0
    while i < len(rows):
        if not signs[i]:
            i += 1
            continue
        j = i
        while j+1 < len(rows) and signs[j+1] == signs[i] and adjacent(j,j+1):
            j += 1
        # Incomplete recordings/gaps cannot establish the rotation boundaries.
        complete = (i > 0 and j+1 < len(rows) and adjacent(i-1,i) and adjacent(j,j+1)
                    and facts[i-1] is not None and facts[j+1] is not None
                    and signs[i-1] == signs[j+1] == 0)
        heading = sum(abs(math.degrees(facts[k]['yaw_rate_rad_s']))*step_s for k in range(i,j+1))
        selected = rows[i:j+1]
        label = 'turn_left' if signs[i] > 0 else 'turn_right'
        supported = any(model_support(row,label) for row in selected)
        eligible = all(row.get('lateral') in ('keep_lane',label) for row in selected)
        mapped = all(isinstance(facts[k].get('intersection'), bool) for k in range(i,j+1))
        if complete and MIN_HEADING_DEG <= heading < MAX_HEADING_DEG and supported and eligible and mapped:
            direction = 'LEFT' if signs[i] > 0 else 'RIGHT'
            intersection = [facts[k] for k in range(i,j+1) if facts[k]['intersection'] is True]
            if intersection and all(str(sample.get('map_turn_direction')).upper() == direction for sample in intersection):
                overrides.update({row['t_ns']:label for row in selected})
                audit['accepted_turns'] += 1
                audit['decisions'].append(dict(first_ns=str(selected[0]['t_ns']),last_ns=str(selected[-1]['t_ns']),
                    label=label,heading_change_deg=heading*signs[i],reason='model + mapped turn + rotation'))
            elif not intersection:
                # Smoothing may have filled raw keep_lane holes with turn labels.
                # Veto the whole validated road bout, not just raw turn winners.
                overrides.update({row['t_ns']:'keep_lane' for row in selected})
                audit['rejected_curve_samples'] += sum(row.get('lateral') == label for row in selected)
                audit['decisions'].append(dict(first_ns=str(selected[0]['t_ns']),last_ns=str(selected[-1]['t_ns']),
                    label='keep_lane',reason='rotation entirely on matched non-intersection road'))
        i = j+1
    return overrides, audit
