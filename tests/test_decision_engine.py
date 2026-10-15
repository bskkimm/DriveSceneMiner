"""Shared decision contracts and resumable classification, without network access."""
import asyncio
import json

import pytest

from jevsceneminer.inference import jev
from test_jev import LABELS, FakeClient


def test_questions_are_plain_data_with_unchanged_definitions():
    questions = LABELS.choice_specs()
    assert questions == {
        'lateral': {'instructions': 'lateral?', 'criteria': {'keep_lane': 'stay', 'turn_left': 'turn'}},
        'longitudinal': {'instructions': 'longitudinal?', 'criteria': {'stopped': 'stopped', 'cruising': 'steady'}},
    }
    questions['lateral']['criteria']['keep_lane'] = 'changed'
    assert LABELS.lateral['keep_lane'] == 'stay'


def test_shared_types_keep_legacy_identity():
    assert hasattr(jev, 'DecisionClassifier')
    from jevsceneminer.inference.contracts import Answer, Labels
    from jevsceneminer.jev import Answer as LegacyAnswer
    assert Answer is jev.Answer is LegacyAnswer
    assert Labels is jev.Labels


class Backend:
    provider = 'microsoft'
    endpoint = 'https://example.test/providers/microsoft/v1/systemone'

    def __init__(self):
        self.calls = []
        self.closed = False

    async def predict(self, script, labels, model):
        self.calls.append((script, labels.choice_specs(), model))
        if script == 'bad':
            await asyncio.sleep(.01)
            raise RuntimeError('synthetic failure')
        return jev.Answer('keep_lane', {'keep_lane': 1., 'turn_left': 0.},
                          'cruising', {'stopped': 0., 'cruising': 1.}, 'resolved-model', 100)

    async def aclose(self):
        self.closed = True


def test_generic_classifier_resumes_without_replaying_success(tmp_path):
    assert hasattr(jev, 'DecisionClassifier')
    backend = Backend()
    cache = jev.AnswerCache(tmp_path/'cache.jsonl')
    classifier = jev.DecisionClassifier(LABELS, cache, model='deployment',
                                      backend=backend, workers=1, log=lambda _: None)
    with pytest.raises(jev.JevError):
        classifier.classify({1: 'good', 2: 'bad'})
    assert len(cache.written_answers) == 1 and backend.closed
    retry = Backend()
    answers = jev.DecisionClassifier(LABELS, jev.AnswerCache(cache.path), model='deployment',
                                    backend=retry, log=lambda _: None).classify({1: 'good'})
    assert not retry.calls
    assert answers[1].provider == 'microsoft'
    assert answers[1].model == 'resolved-model'


def test_provider_and_endpoint_caches_do_not_collide():
    original = jev.cache_key('same', LABELS, 'script')
    assert original == jev.cache_key('same', LABELS, 'script', provider='jev')
    microsoft = jev.cache_key('same', LABELS, 'script', provider='microsoft', endpoint='https://one.test')
    assert microsoft != original
    assert microsoft != jev.cache_key('same', LABELS, 'script', provider='openai', endpoint='https://one.test')
    assert microsoft != jev.cache_key('same', LABELS, 'script', provider='microsoft', endpoint='https://two.test')


def test_jev_classifier_fetches_new_scripts_after_closing_its_previous_client(tmp_path):
    clients = []

    class Client(FakeClient):
        closed = False

        async def system_one(self, *args, **kwargs):
            if self.closed:
                raise RuntimeError('closed')
            return await super().system_one(*args, **kwargs)

        async def aclose(self):
            self.closed = True

    def factory():
        client = Client()
        clients.append(client)
        return client

    classifier = jev.JevClassifier(LABELS, jev.AnswerCache(tmp_path/'cache.jsonl'),
                                   client_factory=factory, log=lambda _: None)
    first = classifier.classify({1: 'first script'})
    second = classifier.classify({2: 'new script'})
    assert first[1].lateral == second[2].lateral == 'keep_lane'
    assert len(clients) == 2 and all(client.closed for client in clients)


def test_legacy_jev_cache_is_reused_without_constructing_a_client(tmp_path):
    # Golden key and answer shape from before the shared-contract extraction.
    key = '4c9dc574214dc6b2119ad5b996eb575b1399c35344890671a146e1b2526ba923'
    path = tmp_path/'legacy.jsonl'
    path.write_text(json.dumps({'key': key, 'answer': {
        'lateral': 'keep_lane', 'lateral_probs': {'keep_lane': 1.},
        'longitudinal': 'cruising', 'longitudinal_probs': {'cruising': 1.},
        'model': 'jev-test', 'input_tokens': 100}}) + '\n')

    def forbidden():
        pytest.fail('a compatible cache hit must not construct an API client')

    assert jev.cache_key(None, LABELS, 'legacy script') == key
    result = jev.JevClassifier(LABELS, jev.AnswerCache(path), client_factory=forbidden,
                               log=lambda _: None).classify({7: 'legacy script'})
    assert result[7].lateral == 'keep_lane' and result[7].provider == 'jev'
    assert result[7].model == 'jev-test' and result[7].input_tokens == 100
