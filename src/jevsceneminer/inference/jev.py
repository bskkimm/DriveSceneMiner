"""Legacy Jev API backed by the common resumable decision classifier."""
from __future__ import annotations

from typing import Callable

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .contracts import Answer, AnswerCache, JevError, Labels, cache_key, load_labels
from .engine import DecisionClassifier

RETRY = RetryPolicy(max_retries=6, http_statuses={429, 500, 502, 503, 504, 529},
                    backoff_initial=1.0, backoff_max=30.0, timeout=30.0)

def parse_response(response) -> Answer:
    lat, lon = response.answers["lateral"], response.answers.get("longitudinal")
    usage = getattr(response, "usage", None)
    return Answer(
        lateral=lat.choice,
        lateral_probs={k: float(v) for k, v in lat.probabilities.items()},
        longitudinal=lon.choice if lon is not None else None,
        longitudinal_probs={k: float(v) for k, v in lon.probabilities.items()} if lon is not None else {},
        model=getattr(response, "model", None),
        input_tokens=getattr(usage, "input_tokens", None) if usage is not None else None,
    )


class JevBackend:
    provider = "jev"
    endpoint = None

    def __init__(self, client_factory=None):
        self._factory = client_factory or (lambda: AsyncTypeSafeClient(retry=RETRY, timeout=30.0))
        self._client = None

    async def predict(self, script, labels, model):
        if self._client is None:
            self._client = self._factory()
        response = await self._client.system_one(state=script, questions=labels.questions(), model=model)
        return parse_response(response)

    async def aclose(self):
        close = getattr(self._client, "aclose", None)
        try:
            if close is not None:
                await close()
        finally:
            self._client = None


class JevClassifier(DecisionClassifier):
    def __init__(self, labels: Labels, cache: AnswerCache, model: str | None = None, workers: int = 8,
                 client_factory: Callable[[], object] | None = None, log: Callable[[str], None] = print):
        super().__init__(labels, cache, model=model, workers=workers,
                         backend=JevBackend(client_factory), log=log)
