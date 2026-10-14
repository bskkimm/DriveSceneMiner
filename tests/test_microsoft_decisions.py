"""Documented Foundry REST contract exercised through an in-memory HTTP transport."""
import asyncio
import importlib.util
import json
from types import SimpleNamespace

import httpx
import pytest

from jevsceneminer.inference.contracts import AnswerCache, DecisionError
from jevsceneminer.inference.config import resolve_settings
from jevsceneminer.inference.engine import DecisionClassifier
from test_jev import LABELS


def microsoft_response():
    return {'model':'microsoft-decision-1', 'usage':{'input_tokens':100, 'output_tokens':0},
            'answers':{
                'lateral':{'type':'choice','choice':'keep_lane','probabilities':{'keep_lane':.8,'turn_left':.2}},
                'longitudinal':{'type':'choice','choice':'cruising','probabilities':{'stopped':.1,'cruising':.9}},
            }}


def backend(monkeypatch, handler, **kwargs):
    assert importlib.util.find_spec('jevsceneminer.inference.microsoft') is not None
    from jevsceneminer.inference.microsoft import MicrosoftBackend
    monkeypatch.setenv('MICROSOFT_DECISION_API_KEY', 'synthetic-key-never-sent-to-a-network')
    settings=resolve_settings(SimpleNamespace(provider='microsoft',model='deployment-name',endpoint='https://example.test'))
    return MicrosoftBackend(settings, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                            **kwargs)


def test_foundry_payload_and_resolved_model_preserve_the_evidence(monkeypatch):
    captured=[]
    def handler(request):
        captured.append(request)
        return httpx.Response(200,json=microsoft_response())
    api=backend(monkeypatch,handler)
    async def run():
        try:
            return await api.predict('exact readable evidence\nNOW = 0', LABELS, 'deployment-name')
        finally:
            await api.aclose()
    answer=asyncio.run(run())
    request=captured[0]
    assert str(request.url)=='https://example.test/providers/microsoft/v1/systemone'
    assert json.loads(request.content)=={'model':'deployment-name','state':'exact readable evidence\nNOW = 0',
        'questions':{name:dict(type='choice',**question) for name,question in LABELS.choice_specs().items()}}
    assert answer.provider=='microsoft' and answer.model=='microsoft-decision-1'
    assert answer.lateral_probs=={'keep_lane':.8,'turn_left':.2}
    assert answer.input_tokens==100 and answer.output_tokens==0


@pytest.mark.parametrize('fault', ['refusal','missing_option','negative','wrong_winner','missing_question'])
def test_invalid_answers_are_not_cached(monkeypatch,tmp_path,fault):
    response=microsoft_response()
    lateral=response['answers']['lateral']
    if fault=='refusal': lateral['type']='refusal'
    elif fault=='missing_option': del lateral['probabilities']['turn_left']
    elif fault=='negative': lateral['probabilities']={'keep_lane':1.1,'turn_left':-.1}
    elif fault=='wrong_winner': lateral['choice']='turn_left'
    else: del response['answers']['longitudinal']
    api=backend(monkeypatch,lambda _:httpx.Response(200,json=response))
    cache=AnswerCache(tmp_path/'cache.jsonl')
    with pytest.raises(DecisionError):
        DecisionClassifier(LABELS,cache,backend=api,model='deployment-name',log=lambda _:None).classify({1:'evidence'})
    assert cache.written_answers==[] and not cache.path.exists()


def test_retryable_status_is_retried_without_losing_answers(monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(429 if len(requests)==1 else 200,json=microsoft_response())
    api=backend(monkeypatch,handler,retry_delay=0)
    async def run():
        try: return await api.predict('evidence',LABELS,'deployment-name')
        finally: await api.aclose()
    assert asyncio.run(run()).lateral=='keep_lane'
    assert len(requests)==2


def test_redirects_do_not_forward_credentials_or_echo_response_body(monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(302,headers={'Location':'https://other.test'},text='private request and credential')
    api=backend(monkeypatch,handler)
    async def run():
        try: await api.predict('private evidence',LABELS,'deployment-name')
        finally: await api.aclose()
    with pytest.raises(DecisionError) as error:
        asyncio.run(run())
    assert len(requests)==1 and '302' in str(error.value)
    assert 'private' not in str(error.value) and 'synthetic-key' not in str(error.value)
