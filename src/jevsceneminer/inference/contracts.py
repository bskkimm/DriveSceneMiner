"""Shared question, answer and cache contracts; no client is constructed here."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

INDICATOR_CODES = {"left": 2, "right": 3}   # TurnIndicatorsReport: 2 = LEFT, 3 = RIGHT


@dataclass(frozen=True)
class Labels:
    version: str
    lateral_instructions: str
    lateral: dict[str, str]
    longitudinal_instructions: str
    longitudinal: dict[str, str]
    lateral_only: bool = False   # ask only the lateral question
    indicator: dict = field(default_factory=dict)   # lateral label -> indicator code it needs (2 | 3)
    traffic_side: str = "left"
    has_indicator: bool = True   # definitions written for logs that record the turn indicator

    def spec(self) -> dict:
        """Plain-JSON form of the questions, used in the cache key."""
        spec = {"lateral": [self.lateral_instructions, self.lateral]}
        if not self.lateral_only:
            spec["longitudinal"] = [self.longitudinal_instructions, self.longitudinal]
        return spec

    def choice_specs(self) -> dict:
        questions = {"lateral": {"instructions": self.lateral_instructions, "criteria": dict(self.lateral)}}
        if not self.lateral_only:
            questions["longitudinal"] = {"instructions": self.longitudinal_instructions,
                                         "criteria": dict(self.longitudinal)}
        return questions

    def questions(self) -> dict:
        from typesafe_sdk import Choice

        qs = {"lateral": Choice(instructions=self.lateral_instructions, criteria=self.lateral)}
        if not self.lateral_only:
            qs["longitudinal"] = Choice(instructions=self.longitudinal_instructions, criteria=self.longitudinal)
        return qs


def _sides(traffic_side: str) -> dict[str, str]:
    if traffic_side not in ("left", "right"):
        raise ValueError(f"traffic_side must be left or right, not {traffic_side!r}")
    far = "right" if traffic_side == "left" else "left"
    return {"curb": traffic_side, "far": far, "CURB": traffic_side.upper(), "FAR": far.upper()}


_VARIANT = re.compile(r"\[\[(.*?)(?:\|\|(.*?))?\]\]", re.S)


def _text(text: str, sides: dict, has_indicator: bool) -> str:
    """Fill in the traffic side and pick the [[with||without indicator]] variants."""
    text = _VARIANT.sub(lambda m: m.group(1) if has_indicator else (m.group(2) or ""), text)
    return " ".join(text.format(**sides).split())


def _definitions(entries: dict, sides: dict, has_indicator: bool) -> tuple[dict[str, str], dict[str, int]]:
    """Label -> definition text, and label -> indicator code it needs."""
    texts, needs = {}, {}
    for name, entry in entries.items():
        if isinstance(entry, dict):
            texts[name] = _text(entry["definition"], sides, has_indicator)
            if entry.get("indicator"):
                needs[name] = INDICATOR_CODES[sides.get(entry["indicator"], entry["indicator"])]
        else:
            texts[name] = _text(entry, sides, has_indicator)
    return texts, needs


def load_labels(path: Path, lateral_only: bool = False, traffic_side: str | None = None,
                has_indicator: bool = True) -> Labels:
    doc = yaml.safe_load(Path(path).read_text())
    side = traffic_side or doc.get("traffic_side", "left")
    sides = _sides(side)
    lateral, needs = _definitions(doc["lateral"]["labels"], sides, has_indicator)
    longitudinal, _ = _definitions(doc["longitudinal"]["labels"], sides, has_indicator)
    return Labels(
        lateral_only=lateral_only,
        version=str(doc["version"]),
        lateral_instructions=doc["lateral"]["instructions"],
        lateral=lateral,
        longitudinal_instructions=doc["longitudinal"]["instructions"],
        longitudinal=longitudinal,
        indicator=needs if has_indicator else {},
        traffic_side=side,
        has_indicator=has_indicator,
    )


@dataclass(frozen=True)
class Answer:
    lateral: str
    lateral_probs: dict[str, float]
    longitudinal: str | None          # None when only the lateral question was asked
    longitudinal_probs: dict[str, float]
    model: str | None = None
    input_tokens: int | None = None
    provider: str = "jev"


class JevError(RuntimeError):
    """Jev calls failed after retries."""


def cache_key(model: str | None, labels: Labels, script: str, *, provider: str = "jev",
              endpoint: str | None = None) -> str:
    identity = {"model": model or "default", "questions": labels.spec(), "state": script}
    if provider != "jev":
        identity.update(provider=provider, endpoint=endpoint, adapter_version=1)
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._answers: dict[str, Answer] = {}
        # Successful responses written by this instance, including repeated keys.
        self.written_answers: list[Answer] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._answers[row["key"]] = Answer(**row["answer"])

    def get(self, key: str) -> Answer | None:
        return self._answers.get(key)

    def put(self, key: str, answer: Answer) -> None:
        self._answers[key] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"key": key, "answer": asdict(answer)}, ensure_ascii=False) + "\n")
        self.written_answers.append(answer)



DecisionError = JevError
