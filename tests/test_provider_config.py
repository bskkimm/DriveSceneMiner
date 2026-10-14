import json
from types import SimpleNamespace

import pytest

from jevsceneminer import cli


def test_switching_provider_does_not_inherit_jev_model(monkeypatch):
    assert hasattr(cli, 'resolve_settings')
    settings = cli.resolve_settings(SimpleNamespace(provider='openai', model=None, endpoint=None),
                                    {'inference_model': 'jev-1.13.0'})
    assert settings.provider == 'openai' and settings.model == 'gpt-6-luna'


@pytest.mark.parametrize('endpoint', ['http://example.test', 'https://key:secret@example.test',
                                      'https://example.test?key=secret', 'https://example.test#x'])
def test_microsoft_rejects_unsafe_endpoints(endpoint):
    assert hasattr(cli, 'resolve_settings')
    with pytest.raises(ValueError, match='HTTPS|endpoint'):
        cli.resolve_settings(SimpleNamespace(provider='microsoft', model='deployment', endpoint=endpoint))


def test_cli_dry_run_needs_no_credentials_or_client(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'load_dotenv', lambda: None)
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    for folder in ('meta', 'steps'):
        (tmp_path/folder).mkdir()
    (tmp_path/'meta'/'d_s.json').write_text(json.dumps({'date': 'd', 'name': 's'}))
    (tmp_path/'steps'/'d_s.jsonl').write_text(json.dumps({'t_ns': 1, 'script': 'readable evidence'})+'\n')
    monkeypatch.setattr(cli, '_classify', lambda *a: pytest.fail('dry run reached inference'))
    assert cli.main(['classify', str(tmp_path), '--provider', 'openai', '--dry-run']) == 0
    assert 'openai' in capsys.readouterr().out


def test_legacy_classify_does_not_require_openai_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'load_dotenv', lambda: None)
    monkeypatch.setenv('TYPESAFE_API_KEY', 'synthetic-test-key')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    called = []
    monkeypatch.setattr(cli, 'cmd_classify', lambda args: called.append(args) or 0)
    assert cli.main(['classify', str(tmp_path)]) == 0
    assert called[0].provider is None


def test_openai_requires_its_own_credential_without_echoing_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'load_dotenv', lambda: None)
    monkeypatch.setenv('TYPESAFE_API_KEY', 'do-not-print-this')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(cli, 'cmd_classify', lambda _: pytest.fail('missing key reached classifier'))
    assert cli.main(['classify', str(tmp_path), '--provider', 'openai']) == 2
    message = capsys.readouterr().err
    assert 'OPENAI_API_KEY' in message and 'do-not-print-this' not in message
