import copy
import json
import pytest

from jevsceneminer.scenes.lanechanges import lanechange_overrides


def sample(direction='right'):
    sign=1 if direction=='left' else -1
    offsets=[0,0,0,.5,1.5,2.5,3.5,3.5,3.5]
    observations=[]
    for i,x in enumerate(offsets):
        origin=1 if i<4 else 2
        observations.append(dict(time_s=i-4,ego_data_available=True,matched_lane_id=origin,
            speed_m_s=4.,yaw_rate_rad_s=0.,relative_heading_deg=0.,intersection=False,
            body_occupancy=dict(reference_offset_m=x*sign,ambiguous_overlap_fraction=0.,unmapped_fraction=0.,
                crossed_boundary_sides=[direction] if i==4 else [],lane_occupancy=[
                    dict(lane_id=1,fraction=max(0,1-x/3.5),direction_relation='same-direction'),
                    dict(lane_id=2,fraction=min(1,x/3.5),direction_relation='same-direction')])) )
    return dict(t_ns=10*10**9,lateral=f'lane_change_{direction}',lateral_probs={f'lane_change_{direction}':.8},facts=dict(
        ego_samples=observations,lane_evidence=dict(ego_geometry={'width_m':2.,'length_m':4.,'center_forward_m':0.},reference_lane_ids=[1],max_centerline_join_gap_m=0.),
        lanes=[dict(id=1,kind='road',intersection=False,following=[],**{direction+'_neighbor_id':2}),
               dict(id=2,kind='road',intersection=False,following=[],**{('right' if direction=='left' else 'left')+'_neighbor_id':1})]))


@pytest.mark.parametrize('direction',['left','right'])
def test_complete_lanechange_is_supported_without_mutating_raw(direction):
    row=sample(direction); before=copy.deepcopy(row)
    overrides,audit=lanechange_overrides([row],1)
    assert overrides=={row['t_ns']:row['lateral']}
    assert audit['accepted_samples']==1 and row==before


@pytest.mark.parametrize('case',['no_neighbor','fork','intersection','no_model','ambiguous','wrong_motion','missing','no_crossing','partial','gap','stationary','reference_reset'])
def test_ambiguous_incomplete_or_wrong_evidence_abstains(case):
    row=sample(); facts=row['facts']; obs=facts['ego_samples']
    if case=='no_neighbor': facts['lanes'][0].pop('right_neighbor_id')
    if case=='fork': facts['lanes'][0]['following']=[{'id':2},{'id':3}]
    if case=='intersection': obs[4]['intersection']=True
    if case=='no_model': row['lateral']='keep_lane'
    if case=='ambiguous': obs[4]['body_occupancy']['ambiguous_overlap_fraction']=.5
    if case=='wrong_motion':
        for s in obs:s['body_occupancy']['reference_offset_m']*=-1
    if case=='missing': row['facts']=None
    if case=='no_crossing': obs[4]['body_occupancy']['crossed_boundary_sides']=[]
    if case=='partial': obs[-3]['body_occupancy']['lane_occupancy'][1]['fraction']=.5
    if case=='gap': obs.pop(4)
    if case=='stationary':
        for s in obs:s['speed_m_s']=0.
    if case=='reference_reset': facts['lane_evidence']['reference_lane_ids']=[99]
    assert lanechange_overrides([row],1)[0]=={}


def test_old_inputs_abstain_and_bad_cadence_is_rejected():
    assert lanechange_overrides([{'t_ns':0}],1)[0]=={}
    with pytest.raises(ValueError):lanechange_overrides([],0)


def test_stationary_stable_chain_veto_preserves_other_maneuvers():
    row=sample()
    for s in row['facts']['ego_samples']:
        s.update(speed_m_s=0.,matched_lane_id=1)
        s['body_occupancy'].update(reference_offset_m=0.,lane_occupancy=[dict(lane_id=1,fraction=1.,direction_relation='same-direction')])
    assert lanechange_overrides([row],1)[0]=={row['t_ns']:'keep_lane'}
    row['lateral']='avoidance'
    assert not lanechange_overrides([row],1)[0]


def test_following_segments_form_one_chain_not_a_lanechange():
    row=sample()
    for lane in row['facts']['lanes']:lane.pop('right_neighbor_id',None);lane.pop('left_neighbor_id',None)
    row['facts']['lanes'][0]['following']=[{'id':2}]
    row['facts']['lanes'][1]['previous_ids']=[1]
    assert not lanechange_overrides([row],1)[0]


@pytest.mark.parametrize('case',['geometry','kind','lane_id','overlap_id'])
def test_malformed_geometry_and_identifiers_abstain(case):
    row=sample()
    if case=='geometry':row['facts']['lane_evidence']['ego_geometry']={'width_m':-1}
    if case=='kind':row['facts']['lanes'][0]['kind']='shoulder'
    if case=='lane_id':row['facts']['ego_samples'][0]['matched_lane_id']=[]
    if case=='overlap_id':row['facts']['ego_samples'][0]['body_occupancy']['lane_occupancy'][0]['lane_id']=[]
    assert not lanechange_overrides([row],1)[0]


def test_connected_original_segments_preserve_origin_during_lanechange():
    row=sample()
    origin,target=row['facts']['lanes']
    origin['following']=[{'id':3}]
    target['left_neighbor_id']=3
    row['facts']['lanes'].append(dict(id=3,kind='road',intersection=False,following=[],previous_ids=[1],right_neighbor_id=2))
    for i,s in enumerate(row['facts']['ego_samples']):
        if i in (2,3):s['matched_lane_id']=3
    row['facts']['lane_evidence']['reference_lane_ids']=[1,3]
    assert lanechange_overrides([row],1)[0]=={row['t_ns']:'lane_change_right'}


@pytest.mark.parametrize('direction',['left','right'])
def test_offline_cli_fills_supported_interval_and_keeps_raw_file(tmp_path,monkeypatch,direction):
    from jevsceneminer import cli
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    source=tmp_path/'source'
    for folder in ('meta','steps'):(source/folder).mkdir(parents=True)
    data=[]
    for t in range(8,15):
        row=sample(direction); row['t_ns']=t*10**9
        for s in row['facts']['ego_samples']:s['time_s']+=10-t
        if t!=10:row.update(lateral='keep_lane',lateral_probs={'keep_lane':1.})
        row.update(script='synthetic',longitudinal='cruising',longitudinal_probs={'cruising':1.})
        data.append(row)
    raw=''.join(json.dumps(r)+'\n' for r in data)
    path=source/'steps'/'d_s.jsonl';path.write_text(raw)
    (source/'meta'/'d_s.json').write_text(json.dumps(dict(date='d',name='s',start_ns=0,end_ns=20*10**9,
        step_s=1,traffic_side='right',has_indicator=False)))
    output=tmp_path/'hybrid'
    assert cli.main(['restitch',str(source),'--out',str(output),'--hybrid-lanechanges','--hybrid-turns','--maneuver-tail','2'])==0
    doc=json.loads((output/'scenes'/'d_s.json').read_text())
    lanechange=next(s for s in doc['scenes'] if s['driving_decision']=='lane_change_'+direction)
    assert (int(lanechange['start_ns']),int(lanechange['end_ns']))==(8500000000,12500000000)
    assert doc['generator']['hybrid_lanechanges']['accepted_samples']==4
    assert path.read_text()==raw


def test_rule_disagreement_abstains_instead_of_selecting_priority(monkeypatch):
    from types import SimpleNamespace
    from jevsceneminer import cli
    monkeypatch.setattr(cli,'turn_overrides',lambda *args:({1:'turn_right'},{}))
    monkeypatch.setattr(cli,'lanechange_overrides',lambda *args:({1:'lane_change_right'},{}))
    overrides,turn,lane=cli._hybrid_overrides([],1,SimpleNamespace(min_prob=.4,hybrid_turns=True,hybrid_lanechanges=True))
    assert not overrides and turn['conflicting_samples']==lane['conflicting_samples']==1


def test_single_step_offset_and_occupancy_jump_is_not_sustained_motion():
    row=sample()
    for i,s in enumerate(row['facts']['ego_samples']):
        x=0 if i<4 else 3.5
        s['body_occupancy'].update(reference_offset_m=-x,lane_occupancy=[
            dict(lane_id=1,fraction=1. if i<4 else 0.,direction_relation='same-direction'),
            dict(lane_id=2,fraction=0. if i<4 else 1.,direction_relation='same-direction')])
    assert not lanechange_overrides([row],1)[0]
