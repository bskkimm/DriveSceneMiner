"""User-visible whole-branch review regressions; no external requests."""
import json
from types import SimpleNamespace

import httpx
import pytest

from jevsceneminer import cli
from jevsceneminer.inference.jev import AnswerCache, JevClassifier, JevError
from test_jev import FakeClient, LABELS
from test_microsoft_decisions import backend
from test_public_cli import seed_run


def test_equals_model_flag_is_not_discarded(monkeypatch):
    monkeypatch.setattr(cli,'load_dotenv',lambda:None)
    captured=[]
    monkeypatch.setattr(cli,'cmd_run',lambda args:captured.append(args) or 0)
    assert cli.main(['run','--provider','openai','--model=custom-model','--out','unused','--dry-run'])==0
    assert captured[0].model=='custom-model'


def test_jev_classifier_can_fetch_new_scripts_after_its_client_is_closed(tmp_path):
    clients=[]
    class Client(FakeClient):
        closed=False
        async def system_one(self,*args,**kwargs):
            if self.closed: raise RuntimeError('closed')
            return await super().system_one(*args,**kwargs)
        async def aclose(self): self.closed=True
    def factory():
        client=Client(); clients.append(client); return client
    classifier=JevClassifier(LABELS,AnswerCache(tmp_path/'cache.jsonl'),client_factory=factory,log=lambda _:None)
    classifier.classify({1:'first script'})
    classifier.classify({2:'new script'})
    assert len(clients)==2 and all(client.closed for client in clients)


def test_duplicate_microsoft_json_members_never_enter_cache(monkeypatch,tmp_path):
    raw='''{"model":"microsoft-decision-1","answers":{
      "lateral":{"type":"choice","choice":"keep_lane","probabilities":{"keep_lane":0.1,"keep_lane":0.8,"turn_left":0.2}},
      "longitudinal":{"type":"choice","choice":"cruising","probabilities":{"stopped":0.1,"cruising":0.9}}}}'''
    api=backend(monkeypatch,lambda _:httpx.Response(200,content=raw))
    cache=AnswerCache(tmp_path/'cache.jsonl')
    with pytest.raises(JevError):
        cli.DecisionClassifier(LABELS,cache,backend=api,model='deployment',log=lambda _:None).classify({1:'script'})
    assert not cache.written_answers


def test_compare_rejects_different_merge_thresholds(tmp_path):
    from jevsceneminer.scenes.compare import compare_runs
    runs={}
    for name,threshold in [('a',.4),('b',.8)]:
        root=tmp_path/name; root.mkdir(); seed_run(root); (root/'scenes').mkdir()
        sid=next((root/'meta').glob('*.json')).stem
        (root/'scenes'/f'{sid}.json').write_text(json.dumps({'scenes':[],
            'generator':{'questions_sha256':'same','min_prob':threshold}}))
        runs[name]=root
    with pytest.raises(ValueError,match='merge'):
        compare_runs(runs,tmp_path/'report')


def test_interrupted_and_cached_attempts_do_not_erase_recorded_cost(tmp_path,monkeypatch):
    from jevsceneminer.inference.jev import Answer
    (tmp_path/'steps').mkdir()
    scripts={10**9:'one',2*10**9:'two'}
    (tmp_path/'steps'/'d_s.jsonl').write_text(''.join(json.dumps({'t_ns':t,'script':text})+'\n' for t,text in scripts.items()))
    meta=dict(date='d',name='s',start_ns=0,end_ns=3*10**9,step_s=1)
    args=SimpleNamespace(model='jev-test',workers=1,require_indicator=False,min_scene=1,min_phase=1,min_prob=0)
    class Classifier:
        phase=0
        def __init__(self,labels,cache,**kwargs): self.cache=cache
        def classify(self,scripts):
            answer=Answer('keep_lane',{'keep_lane':1.},'cruising',{'cruising':1.},'jev-test',100)
            self.cache.written_answers.append(answer)
            if Classifier.phase==0:
                Classifier.phase=1
                raise JevError('synthetic interruption after one successful answer')
            if Classifier.phase==2: self.cache.written_answers.clear()
            return {t:answer for t in scripts}
    monkeypatch.setattr(cli,'JevClassifier',Classifier)
    with pytest.raises(JevError): cli._classify(tmp_path,meta,scripts,LABELS,args)
    runtime_path=tmp_path/'runtime'/'d_s.json'
    assert runtime_path.exists()
    first=json.loads(runtime_path.read_text())
    assert first['status']=='failed' and first['fresh_answers']==1
    cli._classify(tmp_path,meta,scripts,LABELS,args)
    second=json.loads(runtime_path.read_text())
    assert second['run_total']['fresh_answers']==2
    assert second['run_total']['estimated_cost_usd']>first['estimated_cost_usd']
    Classifier.phase=2
    cli._classify(tmp_path,meta,scripts,LABELS,args)
    third=json.loads(runtime_path.read_text())
    assert third['estimated_cost_usd']==0
    assert third['run_total']['estimated_cost_usd']==second['run_total']['estimated_cost_usd']
    assert len((runtime_path.parent/'d_s.attempts.jsonl').read_text().splitlines())==3


def test_legacy_usage_is_not_presented_as_complete_run_total(tmp_path):
    from jevsceneminer.inference.usage import record_attempt
    (tmp_path/'runtime').mkdir()
    (tmp_path/'runtime'/'s.json').write_text(json.dumps({'estimated_cost_usd':1.}))
    result=record_attempt(tmp_path,'s',dict(status='completed',fresh_answers=0,
        inference_seconds=.1,input_tokens=0,output_tokens=0,estimated_cost_usd=0))
    assert not result['run_total']['history_complete']
    assert result['run_total']['estimated_cost_usd'] is None
    assert result['recorded_total']['estimated_cost_usd']==0


def test_compare_separates_latest_invocation_from_run_history(tmp_path):
    from jevsceneminer.inference.usage import record_attempt
    from jevsceneminer.scenes.compare import compare_runs
    runs={}
    for name in ('a','b'):
        root=tmp_path/name; root.mkdir(); seed_run(root); (root/'scenes').mkdir()
        sid=next((root/'meta').glob('*.json')).stem
        (root/'scenes'/f'{sid}.json').write_text(json.dumps({'scenes':[],
            'generator':{'questions_sha256':'same','min_scene_s':2,'min_phase_s':1,
                         'min_prob':.4,'maneuver_tail_s':2,'require_indicator':False}}))
        record_attempt(root,sid,dict(fresh_answers=3,inference_seconds=2,
            input_tokens=300,output_tokens=0,estimated_cost_usd=.2))
        record_attempt(root,sid,dict(fresh_answers=0,inference_seconds=.1,
            input_tokens=0,output_tokens=0,estimated_cost_usd=0))
        runs[name]=root
    report=compare_runs(runs,tmp_path/'report')
    assert report['merge_configuration_verified']
    assert report['runs']['a']['estimated_cost_usd']==.2
    assert report['runs']['a']['inference_seconds']==2.1
    assert report['runs']['a']['latest_invocation']['estimated_cost_usd']==0
    assert report['runs']['a']['recorded_attempts']==2
