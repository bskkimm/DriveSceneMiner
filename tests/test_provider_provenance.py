import json
from types import SimpleNamespace

from jevsceneminer import cli
from jevsceneminer.inference.jev import Answer
from test_jev import LABELS


def test_microsoft_results_record_provider_without_jev_pricing(tmp_path, monkeypatch):
    assert hasattr(cli, 'DecisionClassifier') and hasattr(cli, 'create_backend')
    (tmp_path/'steps').mkdir()
    scripts = {10**9: 'same readable evidence'}
    (tmp_path/'steps'/'d_s.jsonl').write_text(json.dumps({'t_ns':10**9, 'script':scripts[10**9],
                                                       'facts':{'preserve':True}})+'\n')
    meta = dict(date='d', name='s', start_ns=0, end_ns=2*10**9, step_s=1,
                inference_model='jev-test')
    args = SimpleNamespace(provider='microsoft', endpoint='https://example.test', model='deployment',
                           workers=1, require_indicator=False, min_scene=1, min_phase=1, min_prob=0)
    class Classifier:
        def __init__(self, labels, cache, **kwargs):
            self.cache = cache
        def classify(self, scripts):
            answer = Answer('keep_lane', {'keep_lane':1.}, 'cruising', {'cruising':1.},
                            'microsoft-decision-1', 50, provider='microsoft')
            self.cache.written_answers.append(answer)
            return {t:answer for t in scripts}
    monkeypatch.setattr(cli, 'DecisionClassifier', Classifier)
    monkeypatch.setattr(cli, 'create_backend', lambda settings: object())
    cli._classify(tmp_path, meta, scripts, LABELS, args)
    row = json.loads((tmp_path/'steps'/'d_s.jsonl').read_text())
    assert row['provider']=='microsoft' and row['model']=='microsoft-decision-1'
    assert row['requested_model']=='deployment' and row['facts']=={'preserve':True}
    runtime=json.loads((tmp_path/'runtime'/'d_s.json').read_text())
    assert runtime['provider']=='microsoft' and runtime['input_tokens']==50
    assert runtime['estimated_cost_usd'] is None


def test_missing_usage_is_not_reported_as_zero_cost():
    assert hasattr(cli, 'runtime_usage')
    config = cli.resolve_settings(SimpleNamespace(provider='openai', model=None, endpoint=None))
    answer = Answer('keep_lane', {'keep_lane':1.}, None, {}, input_tokens=None, provider='openai')
    assert cli.runtime_usage([answer], config)['estimated_cost_usd'] is None
