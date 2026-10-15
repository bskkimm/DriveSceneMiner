"""Test outbound API protection using a captured connector, never real sockets."""
import socket
from types import SimpleNamespace

import conftest as fixtures
import pytest


@pytest.mark.parametrize('method', ['connect', 'connect_ex'])
@pytest.mark.parametrize('family,address', [
    (socket.AF_INET, ('198.51.100.1', 443)),
    (socket.AF_INET, ('api.example.test', 443)),
    (socket.AF_INET6, ('2001:db8::1', 443, 0, 0)),
])
def test_external_connections_are_rejected_before_reaching_connector(monkeypatch, method, family, address):
    assert hasattr(fixtures, 'block_external_network')
    calls = []
    monkeypatch.setattr(socket.socket, method, lambda *args: calls.append(args))
    fixtures.block_external_network.__wrapped__(monkeypatch)
    with pytest.raises(AssertionError, match='External network is blocked'):
        getattr(socket.socket, method)(SimpleNamespace(family=family), address)
    assert not calls


@pytest.mark.parametrize('method', ['connect', 'connect_ex'])
@pytest.mark.parametrize('family,address', [
    (socket.AF_INET, ('127.0.0.1', 8650)),
    (socket.AF_INET6, ('::1', 8650, 0, 0)),
    (socket.AF_UNIX, '/tmp/jev-synthetic.sock'),
])
def test_local_connections_preserve_connector_result(monkeypatch, method, family, address):
    assert hasattr(fixtures, 'block_external_network')
    calls = []

    def connector(sock, target):
        calls.append((sock.family, target))
        return 123

    monkeypatch.setattr(socket.socket, method, connector)
    fixtures.block_external_network.__wrapped__(monkeypatch)
    assert getattr(socket.socket, method)(SimpleNamespace(family=family), address) == 123
    assert calls == [(family, address)]
