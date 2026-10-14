"""Both source adapters through actual REST decoders, common merging and saved runs."""
from collections import Counter
import json
from types import SimpleNamespace

import httpx
import pytest

from jevsceneminer import cli
from test_public_demo import demo


@pytest.mark.parametrize('source', ['rosbag', 'nuplan'])
@pytest.mark.parametrize('provider', ['microsoft', 'openai'])
def test_prepared_facts_and_scripts_survive_each_provider(tmp_path, monkeypatch, source, provider):
    out=demo.build_demo(tmp_path/'run',source)
    meta_path=next((out/'meta').glob('*.json'))
    meta=json.loads(meta_path.read_text()); sid=meta_path.stem
    rows=[json.loads(line) for line in (out/'steps'/f'{sid}.jsonl').read_text().splitlines()]
    scripts={row['t_ns']:row['script'] for row in rows}
    labels=cli.load_labels(cli.DEFAULT_LABELS,traffic_side='right',has_indicator=meta['has_indicator'])
    captured=[]
    def handler(request):
        body=json.loads(request.content)
        captured.append(body['state'] if provider=='microsoft' else body['input'])
        answers={name:{'type':'choice','choice':'keep_lane' if name=='lateral' else 'cruising',
            'probabilities':{label:float(label==('keep_lane' if name=='lateral' else 'cruising'))
                             for label in spec['criteria']}}
                 for name,spec in labels.choice_specs().items()}
        if provider=='openai':
            answers=[dict(answer,name=name,probabilities=[dict(value=k,probability=v)
                for k,v in answer['probabilities'].items()]) for name,answer in answers.items()]
        return httpx.Response(200,json={'model':'gpt-6-luna' if provider=='openai' else 'microsoft-decision-1',
            'answers':answers,'usage':{'input_tokens':100,'output_tokens':0}})
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-key')
    monkeypatch.setenv('MICROSOFT_DECISION_API_KEY','synthetic-key')
    def create(settings):
        from jevsceneminer.inference.openai import OpenAIBackend
        from jevsceneminer.inference.microsoft import MicrosoftBackend
        cls=OpenAIBackend if provider=='openai' else MicrosoftBackend
        return cls(settings,client_factory=lambda:httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(cli,'create_backend',create)
    args=SimpleNamespace(provider=provider,model='gpt-6-luna' if provider=='openai' else 'deployment',
        endpoint=None if provider=='openai' else 'https://example.test',workers=3,
        require_indicator=False,min_scene=2,min_phase=1,min_prob=.4,maneuver_tail=0)
    cli._classify(out,meta,scripts,labels,args)
    after=[json.loads(line) for line in (out/'steps'/f'{sid}.jsonl').read_text().splitlines()]
    assert [row['facts'] for row in after]==[row['facts'] for row in rows]
    assert [row['script'] for row in after]==[row['script'] for row in rows]
    assert Counter(captured)==Counter(scripts.values())
    assert all(row['provider']==provider and row['lateral']=='keep_lane' for row in after)
    assert json.loads(meta_path.read_text())['inference_provider']==provider
    scenes=json.loads((out/'scenes'/f'{sid}.json').read_text())
    assert scenes['generator']['provider']==provider
    assert all(scene['longitudinal_phases'][-1]['end_ns']==scene['end_ns'] for scene in scenes['scenes'])
    captured.clear()
    cli._classify(out,json.loads(meta_path.read_text()),scripts,labels,args)
    assert captured==[]  # Repeat reads this provider's answers, not the illustrative Jev cache.
