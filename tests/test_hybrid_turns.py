import copy
import math
import json

import pytest

from jevsceneminer.scenes.hybrid import turn_overrides
from jevsceneminer.scenes.merge import Step, stitch
from jevsceneminer.inference.jev import Answer


def rows(direction='left', step=1.):
    sign = 1 if direction == 'left' else -1
    result = []
    for i in range(round(8/step)+1):
        t = i*step
        active = 2 <= t < 6
        label = f'turn_{direction}' if t == 3 else 'keep_lane'
        result.append(dict(t_ns=round(t*1e9), lateral=label,
            lateral_probs={label:.8}, facts={'ego_samples':[dict(time_s=0,
                ego_data_available=True, speed_m_s=4., yaw_rate_rad_s=math.radians(sign*20 if active else 0),
                intersection=active, map_turn_direction=direction.upper() if active else None)]}))
    return result


@pytest.mark.parametrize('direction', ['left','right'])
@pytest.mark.parametrize('step', [.5,1.])
def test_supported_turn_fills_motion_interval_preserving_raw_evidence(direction,step):
    data=rows(direction,step); original=copy.deepcopy(data)
    overrides,audit=turn_overrides(data,step)
    assert set(overrides)=={r['t_ns'] for r in data if 2e9<=r['t_ns']<6e9}
    assert set(overrides.values())=={f'turn_{direction}'}
    assert audit['accepted_turns']==1 and data==original


@pytest.mark.parametrize('case', ['missing','curve','wrong_direction','uturn','no_model','other_maneuver','gap','truncated','nan'])
def test_unsupported_or_ambiguous_turns(case):
    data=rows()
    if case=='missing': data[3]['facts']={}
    if case=='curve':
        for row in data:
            row['facts']['ego_samples'][0].update(intersection=False,map_turn_direction=None)
    if case=='wrong_direction':
        for row in data: row['facts']['ego_samples'][0]['map_turn_direction']='RIGHT'
    if case=='uturn':
        for row in data:
            if row['facts']['ego_samples'][0]['yaw_rate_rad_s']: row['facts']['ego_samples'][0]['yaw_rate_rad_s']=math.radians(40)
    if case=='no_model':
        for row in data: row['lateral']='keep_lane'
    if case=='other_maneuver': data[4]['lateral']='avoidance'
    if case=='gap': data.pop(4)
    if case=='truncated': data=data[2:]
    if case=='nan': data[4]['facts']['ego_samples'][0]['yaw_rate_rad_s']=float('nan')
    overrides,audit=turn_overrides(data,1.)
    if case=='curve':
        assert set(overrides)=={2*10**9,3*10**9,4*10**9,5*10**9} and audit['rejected_curve_samples']==1
    else: assert not overrides


def test_merge_uses_override_but_retains_real_probability_and_speed_phases():
    data=rows(); overrides,_=turn_overrides(data,1.)
    steps=[Step(row['t_ns'],Answer(row['lateral'],row['lateral_probs'],
        'accelerating' if i<4 else 'cruising',{'accelerating':.6,'cruising':.4})) for i,row in enumerate(data)]
    scenes=stitch(steps,1,2,0,9*10**9,.4,lateral_overrides=overrides)
    turn=next(scene for scene in scenes if scene.lateral=='turn_left')
    assert (turn.start_ns,turn.end_ns)==(1500000000,5500000000)
    assert turn.lateral_prob==pytest.approx(.2)
    assert [p.decision for p in turn.longitudinal_phases]==['accelerating','cruising']


def test_missing_facts_is_compatible_noop():
    assert turn_overrides([{'t_ns':0,'lateral':'turn_left'}],1)[0]=={}


def test_offline_cli_is_opt_in_audited_and_preserves_source(tmp_path,monkeypatch):
    from jevsceneminer import cli
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    source=tmp_path/'source'
    for folder in ('steps','meta'): (source/folder).mkdir(parents=True)
    data=rows()
    for row in data:
        row.update(script='synthetic',longitudinal='cruising',longitudinal_probs={'cruising':1.})
    raw=''.join(json.dumps(row)+'\n' for row in data)
    (source/'steps'/'d_s.jsonl').write_text(raw)
    (source/'meta'/'d_s.json').write_text(json.dumps(dict(date='d',name='s',start_ns=0,
        end_ns=9*10**9,step_s=1,traffic_side='right',has_indicator=False)))
    candidate=tmp_path/'hybrid'
    assert cli.main(['restitch',str(source),'--out',str(candidate),'--hybrid-turns','--maneuver-tail','2'])==0
    doc=json.loads((candidate/'scenes'/'d_s.json').read_text())
    turn=next(scene for scene in doc['scenes'] if scene['driving_decision']=='turn_left')
    assert int(turn['end_ns'])==5500000000
    assert doc['generator']['hybrid_turns']['accepted_turns']==1
    assert (source/'steps'/'d_s.jsonl').read_text()==raw
    ordinary=tmp_path/'ordinary'
    assert cli.main(['restitch',str(source),'--out',str(ordinary)])==0
    doc=json.loads((ordinary/'scenes'/'d_s.json').read_text())
    assert not any(scene['driving_decision']=='turn_left' for scene in doc['scenes'])


@pytest.mark.parametrize('bad', [None, {'ego_samples':None}, {'ego_samples':[None]}, {'ego_samples':'unknown'}])
def test_malformed_evidence_abstains(bad):
    data=rows(); data[3]['facts']=bad
    assert turn_overrides(data,1)[0]=={}


def test_integer_map_flag_is_unknown_not_ordinary_road():
    data=rows()
    for row in data:
        row['facts']['ego_samples'][0]['intersection']=1
    assert turn_overrides(data,1)[0]=={}


def test_curve_rejection_clears_turns_introduced_by_smoothing():
    data=[]
    for i in range(13):
        label='turn_left' if i in (3,4,5,7,8,9) else 'keep_lane'
        data.append(dict(t_ns=i*10**9,lateral=label,
            lateral_probs={'turn_left':.8 if label=='turn_left' else .45},
            facts={'ego_samples':[dict(time_s=0,ego_data_available=True,speed_m_s=4.,
                yaw_rate_rad_s=math.radians(10) if 3<=i<=9 else 0,intersection=False)]}))
    overrides,audit=turn_overrides(data,1)
    steps=[Step(r['t_ns'],Answer(r['lateral'],r['lateral_probs'],'cruising',{'cruising':1.})) for r in data]
    scenes=stitch(steps,1,2,0,13*10**9,.4,lateral_overrides=overrides)
    assert audit['rejected_curve_samples']==6
    assert all(scene.lateral=='keep_lane' for scene in scenes)


@pytest.mark.parametrize('probability', [None, True, float('inf'), 1.5])
def test_invalid_model_support_abstains(probability):
    data=rows(); data[3]['lateral_probs']['turn_left']=probability
    assert turn_overrides(data,1)[0]=={}


def test_classify_applies_hybrid_without_rewriting_raw_model_evidence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from jevsceneminer import cli
    from test_jev import LABELS
    data=rows()
    for row in data: row['script']='script '+str(row['t_ns'])
    (tmp_path/'steps').mkdir()
    path=tmp_path/'steps'/'d_s.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in data))
    class Classifier:
        def __init__(self,*args,**kwargs): pass
        def classify(self,scripts):
            return {r['t_ns']:Answer(r['lateral'],r['lateral_probs'],'cruising',{'cruising':1.}) for r in data}
    monkeypatch.setattr(cli,'JevClassifier',Classifier)
    args=SimpleNamespace(model=None,workers=1,min_scene=2,min_phase=1,min_prob=.4,
        require_indicator=False,hybrid_turns=True,maneuver_tail=2)
    meta=dict(date='d',name='s',step_s=1,start_ns=0,end_ns=9*10**9)
    cli._classify(tmp_path,meta,{r['t_ns']:r['script'] for r in data},LABELS,args)
    saved=[json.loads(line) for line in path.read_text().splitlines()]
    assert [(r['lateral'],r['lateral_probs'],r['facts']) for r in saved]==[(r['lateral'],r['lateral_probs'],r['facts']) for r in data]
    doc=json.loads((tmp_path/'scenes'/'d_s.json').read_text())
    assert doc['generator']['hybrid_turns']['accepted_turns']==1
    assert next(s for s in doc['scenes'] if s['driving_decision']=='turn_left')['end_ns']=='5500000000'
