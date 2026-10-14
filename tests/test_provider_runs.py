import hashlib
import json

import pytest

from jevsceneminer import cli
from test_public_cli import seed_run, install_classifier


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_fresh_output_retains_context_and_protects_source(tmp_path,monkeypatch):
    source=tmp_path/'source'; source.mkdir()
    _,_,rows=seed_run(source)
    before=hashes(source)
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    monkeypatch.setenv('TYPESAFE_API_KEY','synthetic-key')
    calls=install_classifier(monkeypatch)
    destination=tmp_path/'candidate'
    assert cli.main(['classify',str(source),'--out',str(destination),'--from-s','1','--to-s','3'])==0
    assert hashes(source)==before
    saved=[json.loads(line) for line in next((destination/'steps').glob('*.jsonl')).read_text().splitlines()]
    assert [row['t_ns'] for row in saved]==[2*10**9,3*10**9]
    assert [row['script'] for row in saved]==[rows[i]['script'] for i in (1,2)]
    assert [row['facts'] for row in saved]==[rows[i]['facts'] for i in (1,2)]
    assert len(calls)==1


def test_existing_destination_is_never_overwritten(tmp_path,monkeypatch):
    source=tmp_path/'source'; source.mkdir(); seed_run(source)
    destination=tmp_path/'candidate'; destination.mkdir(); (destination/'keep.txt').write_text('keep')
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    monkeypatch.setenv('TYPESAFE_API_KEY','synthetic-key')
    calls=install_classifier(monkeypatch)
    with pytest.raises(ValueError,match='exist'):
        cli.main(['classify',str(source),'--out',str(destination)])
    assert (destination/'keep.txt').read_text()=='keep' and calls==[]


def test_switching_provider_in_place_is_rejected_before_calls(tmp_path,monkeypatch):
    seed_run(tmp_path); before=hashes(tmp_path)
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-key')
    monkeypatch.setattr(cli,'_classify',lambda *a:pytest.fail('in-place switch reached API'))
    with pytest.raises(ValueError,match='--out'):
        cli.main(['classify',str(tmp_path),'--provider','openai'])
    assert hashes(tmp_path)==before


def test_compare_checks_identical_scripts_and_counts_only_reviewed_gt(tmp_path,monkeypatch):
    assert hasattr(cli,'cmd_compare')
    folders=[]
    for name in ('first','second'):
        root=tmp_path/name; root.mkdir(); seed_run(root)
        (root/'scenes').mkdir()
        sid=next((root/'meta').glob('*.json')).stem
        (root/'scenes'/f'{sid}.json').write_text(json.dumps({'scenes':[
            {'start_ns':str(10**9),'end_ns':str(6*10**9),'driving_decision':'keep_lane',
             'longitudinal_phases':[{'end_ns':str(6*10**9),'decision':'cruising'}]}]}))
        folders.append(root)
    gt=tmp_path/'gt'; gt.mkdir()
    (gt/f'{sid}.json').write_text(json.dumps({'scenes':[
        {'start_ns':str(2*10**9),'end_ns':str(3*10**9),'driving_decision':'keep_lane',
         'longitudinal_phases':[{'end_ns':str(3*10**9),'decision':'cruising'}]}]}))
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    report=tmp_path/'report'
    assert cli.main(['compare','--run',f'a={folders[0]}','--run',f'b={folders[1]}','--gt',str(gt),'--out',str(report)])==0
    doc=json.loads((report/'report.json').read_text())
    assert doc['same_scripts'] and not doc['question_definitions_verified']
    assert doc['runs']['a']['reviewed']['lateral']['reviewed_seconds']==1
    assert doc['runs']['a']['reviewed']['lateral']['agreement']==1
    path=next((folders[1]/'steps').glob('*.jsonl'))
    rows=[json.loads(line) for line in path.read_text().splitlines()]; rows[0]['script']='different evidence'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='script'):
        cli.main(['compare','--run',f'a={folders[0]}','--run',f'b={folders[1]}','--out',str(tmp_path/'bad-report')])
    assert not (tmp_path/'bad-report').exists()


def test_cache_only_can_make_a_reference_without_credentials(tmp_path,monkeypatch):
    from functools import partial
    from jevsceneminer.inference.jev import AnswerCache, Answer, JevClassifier
    source=tmp_path/'source'; source.mkdir()
    path,_,rows=seed_run(source)
    cache=AnswerCache(source/'cache'/path.name)
    for row in rows:
        cache.put(row['cache_key'],Answer(row['lateral'],row['lateral_probs'],row['longitudinal'],
                                        row['longitudinal_probs'],row['model'],row['input_tokens']))
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    def forbidden(): pytest.fail('cache-only constructed an API client')
    monkeypatch.setattr(cli,'JevClassifier',partial(JevClassifier,client_factory=forbidden))
    destination=tmp_path/'reference'
    assert cli.main(['classify',str(source),'--out',str(destination),'--cache-only'])==0
    runtime=json.loads(next((destination/'runtime').glob('*.json')).read_text())
    assert runtime['fresh_answers']==0 and runtime['estimated_cost_usd']==0


def test_comparison_rejects_overlapping_ground_truth_or_predictions():
    from jevsceneminer.scenes.compare import _reviewed
    scene={'start_ns':'0','end_ns':'1000000000','driving_decision':'keep_lane',
           'longitudinal_phases':[{'end_ns':'1000000000','decision':'cruising'}]}
    with pytest.raises(ValueError,match='overlap'):
        _reviewed({'scenes':[scene]}, {'scenes':[scene,scene]}, [[0,1]], 'lateral')
