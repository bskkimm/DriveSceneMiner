"""Reported usage and explicit, dated input/output pricing estimates."""
import math


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
