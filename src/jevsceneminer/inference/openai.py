"""OpenAI Decisions text Choice adapter; no generated-label fallback."""
from .contracts import DecisionError
from .http import HTTPBackend, parse_choices


class OpenAIBackend(HTTPBackend):
    def payload(self, script, labels, model):
        return {'model': model, 'input': script, 'questions': [
            {'type': 'choice', 'name': name, 'instructions': question['instructions'],
             'choices': [{'value': key, 'description': text} for key, text in question['criteria'].items()]}
            for name, question in labels.choice_specs().items()]}

    def decode(self, payload, labels):
        raw = payload.get('answers')
        if not isinstance(raw, list):
            raise DecisionError('OpenAI response must contain a named answers array')
        answers = {}
        for answer in raw:
            if not isinstance(answer, dict):
                raise DecisionError('invalid OpenAI answer')
            name = answer.get('name')
            if not isinstance(name, str) or name in answers:
                raise DecisionError('missing or duplicated OpenAI question name')
            entries = answer.get('probabilities')
            if not isinstance(entries, list):
                raise DecisionError('OpenAI refusal or missing probability array')
            probabilities = {}
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get('value'), str):
                    raise DecisionError('invalid OpenAI probability option')
                value = entry['value']
                if value in probabilities:
                    raise DecisionError('duplicated OpenAI probability option')
                probabilities[value] = entry.get('probability')
            answers[name] = dict(answer, probabilities=probabilities)
        return parse_choices(answers, labels, self.provider, payload.get('model'), payload.get('usage'))
