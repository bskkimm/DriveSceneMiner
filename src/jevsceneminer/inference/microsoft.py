"""Microsoft Decision-1 via the documented Microsoft Foundry system-one endpoint."""
from .http import HTTPBackend, parse_choices


class MicrosoftBackend(HTTPBackend):
    def payload(self, script, labels, model):
        return {'model': model, 'state': script,
                'questions': {name: dict(type='choice', **question)
                              for name, question in labels.choice_specs().items()}}

    def decode(self, payload, labels):
        return parse_choices(payload.get('answers'), labels, self.provider,
                             payload.get('model'), payload.get('usage'))
