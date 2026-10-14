import asyncio
import importlib.util
import json
from types import SimpleNamespace

import httpx
import pytest

from jevsceneminer.inference.config import resolve_settings
from jevsceneminer.inference.contracts import DecisionError
from test_jev import LABELS


def openai_response():
    return {'model':'gpt-6-luna','usage':{'input_tokens':50,'output_tokens':0},'answers':[
        {'name':'longitudinal','type':'choice','choice':'cruising','probabilities':[
            {'value':'stopped','probability':.1},{'value':'cruising','probability':.9}]},
        {'name':'lateral','type':'choice','choice':'keep_lane','probabilities':[
            {'value':'turn_left','probability':.2},{'value':'keep_lane','probability':.8}]},
    ]}


def backend(monkeypatch,response):
    assert importlib.util.find_spec('jevsceneminer.inference.openai') is not None
    from jevsceneminer.inference.openai import OpenAIBackend
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-key')
    settings=resolve_settings(SimpleNamespace(provider='openai',model=None,endpoint=None))
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,content=json.dumps(response),headers={'Content-Type':'application/json'})
    return OpenAIBackend(settings,client_factory=lambda:httpx.AsyncClient(transport=httpx.MockTransport(handler))),requests


def predict(api, labels=LABELS):
    async def run():
        try: return await api.predict('exact script\nNOW = 0',labels,'gpt-6-luna')
        finally: await api.aclose()
    return asyncio.run(run())


def test_openai_translates_named_arrays_without_changing_evidence(monkeypatch):
    api,requests=backend(monkeypatch,openai_response())
    answer=predict(api)
    payload=json.loads(requests[0].content)
    assert payload['input']=='exact script\nNOW = 0' and payload['model']=='gpt-6-luna'
    assert payload['questions']==[
        {'type':'choice','name':name,'instructions':question['instructions'],
         'choices':[{'value':key,'description':text} for key,text in question['criteria'].items()]}
        for name,question in LABELS.choice_specs().items()]
    assert answer.lateral_probs=={'turn_left':.2,'keep_lane':.8}
    assert answer.longitudinal=='cruising' and answer.provider=='openai'


@pytest.mark.parametrize('fault',['refusal','duplicate_question','duplicate_option','missing_option','nan','unknown_option'])
def test_openai_rejects_refusals_and_malformed_probability_arrays(monkeypatch,fault):
    response=openai_response()
    question=response['answers'][1]
    if fault=='refusal': question['type']='refusal'
    elif fault=='duplicate_question': response['answers'].append(question)
    elif fault=='duplicate_option': question['probabilities'].append(question['probabilities'][0])
    elif fault=='missing_option': question['probabilities'].pop()
    elif fault=='nan': question['probabilities'][0]['probability']=float('nan')
    else: question['probabilities'][0]['value']='unrequested'
    api,_=backend(monkeypatch,response)
    with pytest.raises(DecisionError): predict(api)


def test_lateral_only_does_not_require_a_longitudinal_answer(monkeypatch):
    from dataclasses import replace
    response=openai_response(); response['answers']=response['answers'][1:]
    api,_=backend(monkeypatch,response)
    answer=predict(api,replace(LABELS,lateral_only=True))
    assert answer.longitudinal is None and answer.longitudinal_probs=={}
