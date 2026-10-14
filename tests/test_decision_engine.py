"""Shared decision contracts and resumable classification, without network access."""
import asyncio

import pytest

from jevsceneminer.inference import jev
from test_jev import LABELS


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
