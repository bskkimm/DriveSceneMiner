import copy
import json
import pytest

from jevsceneminer.scenes.avoidance import avoidance_overrides


def sample():
    offsets=[0,0,.2,.6,1.,1.2,.9,.5,.2,0,0]
    ego=[dict(time_s=i-5,ego_data_available=True,speed_m_s=4.,intersection=False,
        body_occupancy=dict(reference_offset_m=x,lane_relative_heading_deg=0.,reference_lane_fraction=.9,
        ambiguous_overlap_fraction=0.,unmapped_fraction=0.)) for i,x in enumerate(offsets)]
    objects=[dict(time_s=i-5,ahead_m=15-i*3,left_m=-.5,speed_m_s=0.,
        reference_lane_overlap_m2=1.,ego_body_clearance_m=1.) for i in range(11)]
    return dict(t_ns=10**10,lateral='avoidance',lateral_probs={'avoidance':.8},
        facts=dict(ego_samples=ego,lane_evidence=dict(reference_lane_ids=[1],
        ego_geometry={'width_m':2.,'length_m':4.,'center_forward_m':0.},max_centerline_join_gap_m=0.),
        object_tracks=[dict(track_id='obstacle',observations=objects)]))


def interval_rows(row):
    rows=[]
    for t in range(7,15):
        item=copy.deepcopy(row); item['t_ns']=t*10**9
        if item['facts'] is not None:
            for point in item['facts']['ego_samples']:point['time_s']+=10-t
            for point in item['facts']['object_tracks'][0]['observations']:point['time_s']+=10-t
        if t!=10 and row['lateral']=='avoidance':item.update(lateral='keep_lane',lateral_probs={'keep_lane':1.})
        rows.append(item)
    return rows


def test_supported_bypass_preserves_raw():
    row=sample(); original=copy.deepcopy(row)
    overrides,audit=avoidance_overrides(interval_rows(row),1)
    assert set(overrides)=={t*10**9 for t in range(7,15)}
    assert audit['accepted_samples']==8 and row==original


@pytest.mark.parametrize('case',['moving','missing','no_intrusion','no_return','no_pass','wrong_side','gap','no_model','ambiguous','other_maneuver'])
def test_incomplete_or_unrelated_evidence_abstains(case):
    row=sample(); facts=row['facts']; track=facts['object_tracks'][0]['observations']; ego=facts['ego_samples']
    if case=='moving':
        for p in track:p['speed_m_s']=4.
    if case=='missing':row['facts']=None
    if case=='no_intrusion':
        for p in track:p['reference_lane_overlap_m2']=0.
    if case=='no_return':
        for p in ego[7:]:p['body_occupancy']['reference_offset_m']=1.
    if case=='no_pass':
        for p in track:p['ahead_m']=5.
    if case=='wrong_side':
        for p in track:p['left_m']=2.
    if case=='gap':ego.pop(4)
    if case=='no_model':row['lateral']='keep_lane'
    if case=='ambiguous':ego[5]['body_occupancy']['ambiguous_overlap_fraction']=.7
    if case=='other_maneuver':row['lateral']='turn_left'
    assert not avoidance_overrides(interval_rows(row),1)[0]


@pytest.mark.parametrize('case',['missing_predictions','short_track','unsettled_return'])
def test_incomplete_temporal_evidence_abstains(case):
    row=sample(); rows=interval_rows(row)
    if case=='missing_predictions':rows=[rows[3]]
    if case=='short_track':
        for item in rows:item['facts']['object_tracks'][0]['observations']=item['facts']['object_tracks'][0]['observations'][4:7]
    if case=='unsettled_return':
        for item in rows:item['facts']['ego_samples'][9]['body_occupancy']['reference_offset_m']=-.29
    assert not avoidance_overrides(rows,1)[0]


def test_obstacle_side_accounts_for_ego_heading():
    row=sample()
    for point in row['facts']['ego_samples']:
        point['body_occupancy']['lane_relative_heading_deg']=15.
    for point in row['facts']['object_tracks'][0]['observations']:
        if point['ahead_m'] > 0:point['left_m']=-.5
    assert not avoidance_overrides(interval_rows(row),1)[0]


def test_restitch_preserves_raw_evidence_and_no_extra_tail(tmp_path,monkeypatch):
    from jevsceneminer import cli
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    source=tmp_path/'source'
    for folder in ('meta','steps'):(source/folder).mkdir(parents=True)
    rows=[]
    for t in range(6,16):
        row=sample();row['t_ns']=t*10**9
        for p in row['facts']['ego_samples']:p['time_s']+=10-t
        for p in row['facts']['object_tracks'][0]['observations']:p['time_s']+=10-t
        if t!=10:row.update(lateral='keep_lane',lateral_probs={'keep_lane':1.,'avoidance':.2})
        row.update(script='synthetic',longitudinal='cruising',longitudinal_probs={'cruising':1.})
        rows.append(row)
    raw=''.join(json.dumps(row)+'\n' for row in rows)
    path=source/'steps'/'d_s.jsonl';path.write_text(raw)
    (source/'meta'/'d_s.json').write_text(json.dumps(dict(date='d',name='s',start_ns=0,end_ns=20*10**9,
        step_s=1,traffic_side='right',has_indicator=False)))
    output=tmp_path/'hybrid'
    assert cli.main(['restitch',str(source),'--out',str(output),'--hybrid-avoidance','--maneuver-tail','2'])==0
    doc=json.loads((output/'scenes'/'d_s.json').read_text())
    scene=next(s for s in doc['scenes'] if s['driving_decision']=='avoidance')
    assert (int(scene['start_ns']),int(scene['end_ns']))==(6500000000,14500000000)
    assert doc['generator']['hybrid_avoidance']['accepted_samples']==8
    assert path.read_text()==raw
