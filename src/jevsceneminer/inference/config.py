"""Explicit provider selection; credentials are read only for an inference request."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

PROVIDERS = ('jev', 'microsoft', 'openai')
OPENAI_ENDPOINT = 'https://api.openai.com/v1/decisions'
MICROSOFT_PATH = '/providers/microsoft/v1/systemone'


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str | None
    endpoint: str | None = None


def resolve_settings(args, meta=None):
    meta = meta or {}
    recorded_provider = meta.get('inference_provider', 'jev')
    provider = getattr(args, 'provider', None) or recorded_provider
    if provider not in PROVIDERS:
        raise ValueError('unsupported decision provider')
    same = provider == recorded_provider
    model = getattr(args, 'model', None) or (meta.get('inference_model') if same else None)
    endpoint = getattr(args, 'endpoint', None) or (meta.get('inference_endpoint') if same else None)
    if provider == 'openai':
        if endpoint and endpoint != OPENAI_ENDPOINT:
            raise ValueError('OpenAI Decisions uses its fixed HTTPS endpoint')
        return Settings(provider, model or 'gpt-6-luna', OPENAI_ENDPOINT)
    if provider == 'jev':
        if endpoint:
            raise ValueError('--endpoint is supported only for Microsoft Foundry')
        return Settings(provider, model)
    model = model or os.environ.get('MICROSOFT_DECISION_DEPLOYMENT')
    endpoint = endpoint or os.environ.get('MICROSOFT_DECISION_ENDPOINT')
    if not model or not endpoint:
        raise ValueError('Microsoft requires --model DEPLOYMENT and --endpoint HTTPS_RESOURCE '
                         '(or MICROSOFT_DECISION_DEPLOYMENT/ENDPOINT)')
    parts = urlsplit(endpoint)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Microsoft endpoint must be HTTPS without credentials, query or fragment')
    try:
        parts.port
    except ValueError:
        raise ValueError('invalid Microsoft endpoint port') from None
    endpoint = endpoint.rstrip('/')
    if not endpoint.endswith(MICROSOFT_PATH):
        endpoint += MICROSOFT_PATH
    return Settings(provider, model, endpoint)


def auth_headers(settings):
    if settings.provider == 'microsoft':
        key = os.environ.get('MICROSOFT_DECISION_API_KEY')
        if key:
            return {'api-key': key}
        token = os.environ.get('MICROSOFT_DECISION_ENTRA_TOKEN')
        if token:
            return {'Authorization': 'Bearer ' + token}
        raise ValueError('Set MICROSOFT_DECISION_API_KEY or MICROSOFT_DECISION_ENTRA_TOKEN')
    variable = 'OPENAI_API_KEY' if settings.provider == 'openai' else 'TYPESAFE_API_KEY'
    key = os.environ.get(variable)
    if not key:
        raise ValueError(f'{variable} is not set; configure it locally or use --dry-run')
    return {'Authorization': 'Bearer ' + key}
