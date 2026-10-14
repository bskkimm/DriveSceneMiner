"""Optional REST transport; response contents and credentials never enter errors."""
from __future__ import annotations

import asyncio
import json
import math

from .config import auth_headers
from .contracts import Answer, DecisionError


def _unique_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON member')
        result[key] = value
    return result


def parse_choices(answers, labels, provider, model, usage):
    specs = labels.choice_specs()
    if not isinstance(answers, dict) or set(answers) != set(specs):
        raise DecisionError('response must contain exactly the requested questions')
    parsed = {}
    for name, spec in specs.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get('type') != 'choice':
            raise DecisionError(f'{name}: refusal or invalid choice answer')
        probabilities, winner = answer.get('probabilities'), answer.get('choice')
        if not isinstance(probabilities, dict) or set(probabilities) != set(spec['criteria']):
            raise DecisionError(f'{name}: missing, duplicated or unsupported probability options')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise DecisionError(f'{name}: probabilities must be finite values between 0 and 1')
        if abs(sum(probabilities.values()) - 1) > .001:
            raise DecisionError(f'{name}: probability distribution does not sum to 1')
        if not isinstance(winner, str) or winner not in probabilities or probabilities[winner] + .000001 < max(probabilities.values()):
            raise DecisionError(f'{name}: invalid or inconsistent selected option')
        parsed[name] = (winner, {key: float(value) for key, value in probabilities.items()})
    if not isinstance(model, str) or not model.strip():
        raise DecisionError('response did not identify its model')
    usage = usage if isinstance(usage, dict) else {}
    tokens = {}
    for key in ('input_tokens', 'output_tokens'):
        value = usage.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise DecisionError('invalid token usage')
        tokens[key] = value
    lateral = parsed['lateral']
    longitudinal = parsed.get('longitudinal', (None, {}))
    return Answer(*lateral, *longitudinal, model=model, provider=provider, **tokens)


class HTTPBackend:
    def __init__(self, settings, *, client_factory=None, max_retries=6, retry_delay=1.):
        self.settings = settings
        self.provider, self.endpoint = settings.provider, settings.endpoint
        self._factory, self._client = client_factory, None
        self.max_retries, self.retry_delay = max_retries, retry_delay

    async def predict(self, script, labels, model):
        try:
            import httpx
        except ImportError:
            raise DecisionError('Install the optional transport with uv sync --extra decisions') from None
        if self._client is None:
            self._client = self._factory() if self._factory else httpx.AsyncClient(timeout=30., follow_redirects=False)
        headers = auth_headers(self.settings)
        body = self.payload(script, labels, model)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(self.endpoint, json=body, headers=headers,
                                                   timeout=30., follow_redirects=False)
            except httpx.RequestError:
                if attempt == self.max_retries:
                    raise DecisionError(f'{self.provider}: request failed after retries') from None
            else:
                if response.status_code == 200:
                    try:
                        payload = json.loads(response.content, object_pairs_hook=_unique_members)
                    except ValueError:
                        raise DecisionError(f'{self.provider}: invalid JSON response') from None
                    if not isinstance(payload, dict):
                        raise DecisionError(f'{self.provider}: invalid response object')
                    return self.decode(payload, labels)
                if response.status_code not in {429, 500, 502, 503, 504, 529} or attempt == self.max_retries:
                    raise DecisionError(f'{self.provider}: HTTP {response.status_code}; check endpoint, access, request and input limits')
            await asyncio.sleep(min(30., self.retry_delay * 2 ** attempt))

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
