"""The HTTPS client: request shape, error mapping and connection reuse, against fake
connections (no network)."""

from __future__ import annotations

import http.client
import json
import ssl
import threading
from typing import Any

import pytest

from polyglot.adapters.http import HttpsClient
from polyglot.core.errors import ProviderError, ProviderTimeout, ProviderUnavailable


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes = b'{"ok": true}', close: bool = False):
        self.status, self._body, self.will_close = status, body, close

    def read(self) -> bytes:
        return self._body


class FakeConn:
    """Plays back ``script``: one entry per request, a response or an exception."""

    def __init__(self, script: list[Any], log: list[dict[str, Any]]) -> None:
        self.script, self.log = script, log
        self.timeout: float = 0
        self.sock: Any = None
        self.closed = False

    def request(self, method: str, target: str, body: bytes | None, headers: dict[str, str]):
        self.log.append(
            {"conn": id(self), "method": method, "target": target, "body": body, "headers": headers}
        )

    def getresponse(self) -> FakeResponse:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[no-any-return]

    def close(self) -> None:
        self.closed = True


def client(*conns_scripts: list[Any]) -> tuple[HttpsClient, list[dict[str, Any]], list[FakeConn]]:
    log: list[dict[str, Any]] = []
    made: list[FakeConn] = []
    scripts = list(conns_scripts)

    def connect(host: str, port: int, timeout: float) -> FakeConn:
        conn = FakeConn(scripts.pop(0), log)
        made.append(conn)
        return conn

    return HttpsClient(connect), log, made  # type: ignore[arg-type]


def test_refuses_plain_http() -> None:
    with pytest.raises(ProviderError):
        HttpsClient().post_json("http://example.com", {}, timeout=1)
    with pytest.raises(ProviderError):
        HttpsClient().get_json("http://example.com", timeout=1)


def test_post_sends_json_with_query_and_headers() -> None:
    c, log, _ = client([FakeResponse()])
    resp = c.post_json(
        "https://h/x", {"a": 1}, headers={"Authorization": "Bearer k"}, params={"key": "q"},
        timeout=2.5,
    )  # fmt: skip
    assert resp.json() == {"ok": True}
    (req,) = log
    assert req["method"] == "POST" and req["target"] == "/x?key=q"
    assert json.loads(req["body"]) == {"a": 1}
    assert req["headers"]["Authorization"] == "Bearer k"
    assert req["headers"]["Content-Type"] == "application/json"


def test_get_sends_no_body() -> None:
    c, log, _ = client([FakeResponse(200, b'{"data": []}')])
    resp = c.get_json(
        "https://api.openai.com/v1/models", headers={"Authorization": "Bearer k"}, timeout=2
    )
    assert resp.status == 200
    assert log[0]["method"] == "GET" and log[0]["body"] is None
    assert log[0]["target"] == "/v1/models"


def test_error_status_is_returned_not_raised() -> None:
    c, _, _ = client([FakeResponse(401, b'{"error":"bad key"}')])
    resp = c.post_json("https://h/x", {}, timeout=1)
    assert resp.status == 401 and "bad key" in resp.body


def test_connection_is_reused_on_the_same_thread() -> None:
    c, log, made = client([FakeResponse(), FakeResponse()])
    c.post_json("https://h/a", {}, timeout=1)
    c.post_json("https://h/b", {}, timeout=1)
    assert len(made) == 1 and log[0]["conn"] == log[1]["conn"]


def test_each_thread_gets_its_own_connection() -> None:
    c, _, made = client([FakeResponse()], [FakeResponse()])
    c.post_json("https://h/a", {}, timeout=1)
    t = threading.Thread(target=lambda: c.post_json("https://h/b", {}, timeout=1))
    t.start()
    t.join()
    assert len(made) == 2


def test_other_hosts_get_other_connections() -> None:
    c, _, made = client([FakeResponse()], [FakeResponse()])
    c.post_json("https://one/a", {}, timeout=1)
    c.post_json("https://two/a", {}, timeout=1)
    assert len(made) == 2


def test_connection_is_dropped_when_the_server_says_close() -> None:
    c, _, made = client([FakeResponse(close=True)], [FakeResponse()])
    c.post_json("https://h/a", {}, timeout=1)
    c.post_json("https://h/b", {}, timeout=1)
    assert len(made) == 2 and made[0].closed


def test_stale_connection_is_replaced_and_the_request_resent() -> None:
    stale = http.client.RemoteDisconnected("closed while idle")
    c, log, made = client([FakeResponse(), stale], [FakeResponse(200, b'{"n": 2}')])
    c.post_json("https://h/a", {}, timeout=1)
    resp = c.post_json("https://h/b", {}, timeout=1)
    assert resp.json() == {"n": 2} and len(made) == 2 and made[0].closed
    assert [r["target"] for r in log] == ["/a", "/b", "/b"]


def test_a_fresh_connection_failure_is_not_retried() -> None:
    c, log, made = client([http.client.RemoteDisconnected("no")], [FakeResponse()])
    with pytest.raises(ProviderUnavailable):
        c.post_json("https://h/a", {}, timeout=1)
    assert len(made) == 1 and len(log) == 1


def test_a_timeout_is_never_retried_and_drops_the_connection() -> None:
    c, log, made = client([FakeResponse(), TimeoutError("slow")], [FakeResponse()])
    c.post_json("https://h/a", {}, timeout=1)
    with pytest.raises(ProviderTimeout):
        c.post_json("https://h/b", {}, timeout=1)
    assert len(log) == 2 and made[0].closed
    c.post_json("https://h/c", {}, timeout=1)  # next request starts a new connection
    assert len(made) == 2


@pytest.mark.parametrize(
    ("raised", "expected", "retryable"),
    [
        (TimeoutError("t"), ProviderTimeout, True),
        (ConnectionRefusedError("no"), ProviderUnavailable, True),
        (OSError("reset"), ProviderUnavailable, True),
        (http.client.IncompleteRead(b"partial"), ProviderUnavailable, True),
        (ssl.SSLCertVerificationError("bad cert"), ProviderUnavailable, False),
    ],
)
def test_transport_errors_are_typed(raised: Exception, expected: type, retryable: bool) -> None:
    c, _, _ = client([raised])
    with pytest.raises(expected) as info:
        c.post_json("https://h/x", {}, timeout=1)
    assert info.value.retryable is retryable
