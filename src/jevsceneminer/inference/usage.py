"""Reported usage and explicit, dated input/output pricing estimates."""
import json
import math
import os
from datetime import datetime, timezone
from uuid import uuid4

USAGE_TOTAL_FIELDS = ('fresh_answers', 'inference_seconds', 'input_tokens',
                      'output_tokens', 'estimated_cost_usd')


def record_attempt(out, sid, attempt, *, inherited_cache=False, existing_cache=False):
    """Keep failed/completed invocation usage; cached reruns cannot erase history."""
    directory = out / 'runtime'
    directory.mkdir(exist_ok=True)
    latest = directory / f'{sid}.json'
    journal = directory / f'{sid}.attempts.jsonl'
    attempts = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    complete = all(row['history_complete'] for row in attempts) if attempts else (
        not latest.exists() and (inherited_cache or not existing_cache))
    attempt = dict(attempt, attempt_id=uuid4().hex, recorded_at=datetime.now(timezone.utc).isoformat(),
                   history_complete=complete)
    with journal.open('a') as stream:
        stream.write(json.dumps(attempt, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    attempts.append(attempt)
    recorded = {}
    for field in USAGE_TOTAL_FIELDS:
        values = [row.get(field) for row in attempts]
        recorded[field] = sum(values) if all(value is not None for value in values) else None
    run_total = dict(recorded if complete else {field: None for field in USAGE_TOTAL_FIELDS},
                     attempts=len(attempts), history_complete=complete,
                     scope='attempts in this output folder; inherited source-cache costs excluded')
    document = dict(attempt, run_total=run_total, recorded_total=recorded)
    temporary = latest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(latest)
    return document


def runtime_usage(fresh, settings, input_price=None, output_price=None):
    known_input = all(answer.input_tokens is not None for answer in fresh)
    known_output = all(getattr(answer, 'output_tokens', None) is not None for answer in fresh)
    inputs = sum(answer.input_tokens for answer in fresh) if known_input else None
    outputs = sum(answer.output_tokens for answer in fresh) if known_output else None
    source = None
    if input_price is None and output_price is None:
        if settings.provider == 'jev' and (settings.model is None or settings.model.startswith('jev-')):
            input_price, output_price, source = .042, 0., 'https://docs.typesafe.ai/models'
        elif settings.provider == 'openai' and settings.model == 'gpt-6-luna':
            input_price, output_price, source = .10, 0., 'https://developers.openai.com/api/docs/guides/decisions'
    else:
        if input_price is None or output_price is None or any(
                not math.isfinite(v) or v < 0 for v in (input_price, output_price)):
            raise ValueError('provide both finite nonnegative --input-price and --output-price')
        source = 'explicit run configuration'
    cost = None
    if not fresh:
        cost = 0.
    elif input_price is not None and inputs is not None and (output_price == 0 or outputs is not None):
        cost = (inputs * input_price + (outputs or 0) * output_price) / 1e6
    return dict(provider=settings.provider, requested_model=settings.model, input_tokens=inputs,
                output_tokens=outputs, estimated_cost_usd=cost,
                price_per_million_input_tokens_usd=input_price,
                price_per_million_output_tokens_usd=output_price, pricing_source=source,
                pricing_as_of='2026-10-14', billing_scope='successful fresh responses; no API cache discount assumed')
