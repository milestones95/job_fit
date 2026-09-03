"""The verification gate: pass only on 2xx + JSON body + at least one
job-like object. Rejections return a reason, never raise, and — through the
whole register path — never touch sources.json (byte-identical)."""
import json

import pytest
import requests

import source_registry as sr


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


ASHBY_CANDIDATE = {"endpoint": "https://api.ashbyhq.com/posting-api/job-board/acme"}


def patch_http(monkeypatch, response=None, exc=None, calls=None):
    def fake(method, url, headers=None, timeout=None):
        if calls is not None:
            calls.append((method, url))
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(sr, "_http_request", fake)


# --- passing cases ---------------------------------------------------------

def test_pass_on_2xx_json_with_jobs(monkeypatch):
    body = {"jobs": [{"title": "Backend Engineer"}, {"title": "Founding Engineer"}]}
    patch_http(monkeypatch, response=FakeResponse(200, body))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is True
    assert result.reason is None
    assert len(result.jobs) == 2


def test_pass_on_lever_style_top_level_list(monkeypatch):
    body = [{"text": "Engineer", "hostedUrl": "https://jobs.lever.co/acme/1"}]
    patch_http(monkeypatch, response=FakeResponse(200, body))
    assert sr.verify_source(ASHBY_CANDIDATE).ok is True


def test_pass_on_smartrecruiters_content_key(monkeypatch):
    body = {"content": [{"name": "Engineer", "id": "abc"}], "totalFound": 1}
    patch_http(monkeypatch, response=FakeResponse(200, body))
    assert sr.verify_source(ASHBY_CANDIDATE).ok is True


def test_pass_records_jobs_not_written(monkeypatch, seeded_registry):
    body = {"jobs": [{"title": "Engineer"}]}
    patch_http(monkeypatch, response=FakeResponse(200, body))
    before = seeded_registry.read_bytes()
    sr.verify_source(ASHBY_CANDIDATE)
    # A passing verify also writes nothing — persistence is register_known_source's call.
    assert seeded_registry.read_bytes() == before


# --- rejections ------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (404, "http_404"),
    (401, "http_401"),
    (500, "http_500"),
])
def test_non_2xx_rejected(monkeypatch, status, expected):
    patch_http(monkeypatch, response=FakeResponse(status, body={}))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is False
    assert result.reason == expected


def test_200_with_html_body_rejected_as_non_json(monkeypatch):
    patch_http(monkeypatch, response=FakeResponse(200, body=None, text="<html>login</html>"))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is False
    assert result.reason == "non_json"


@pytest.mark.parametrize("body", [
    [],                      # empty top-level list
    {"jobs": []},            # empty collection key
    [{"foo": "bar"}],        # objects with no job-like keys
    {"jobs": [{"foo": "bar"}]},
    "ok",                    # JSON scalar
])
def test_200_without_job_like_objects_rejected_as_empty(monkeypatch, body):
    patch_http(monkeypatch, response=FakeResponse(200, body=body))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is False
    assert result.reason == "empty"


def test_unsafe_scheme_rejected_before_any_network_call(monkeypatch):
    calls = []
    patch_http(monkeypatch, response=FakeResponse(200, body={"jobs": [{"title": "x"}]}), calls=calls)
    result = sr.verify_source({"endpoint": "http://jobs.ashbyhq.com/acme"})
    assert result.ok is False
    assert result.reason == "unsafe_scheme"
    assert calls == []  # the gate never dialed out


def test_network_error_rejected_not_raised(monkeypatch):
    patch_http(monkeypatch, exc=requests.ConnectionError("refused"))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is False
    assert result.reason == "network_error"


def test_unexpected_http_error_rejected_not_raised(monkeypatch):
    patch_http(monkeypatch, exc=RuntimeError("boom"))
    result = sr.verify_source(ASHBY_CANDIDATE)
    assert result.ok is False
    assert result.reason == "verify_error"


# --- rejections leave sources.json byte-identical --------------------------

REJECTION_CASES = [
    ("http_404", dict(response=FakeResponse(404, body={}))),
    ("http_401", dict(response=FakeResponse(401, body={}))),
    ("non_json", dict(response=FakeResponse(200, body=None, text="<html>oops</html>"))),
    ("empty", dict(response=FakeResponse(200, body={"jobs": []}))),
]


@pytest.mark.parametrize("expected,_kwargs", REJECTION_CASES, ids=[c[0] for c in REJECTION_CASES])
def test_rejected_verify_and_register_leave_registry_byte_identical(
    monkeypatch, seeded_registry, expected, _kwargs
):
    patch_http(monkeypatch, **_kwargs)
    before = seeded_registry.read_bytes()

    gate = sr.verify_source(ASHBY_CANDIDATE)
    registered = sr.register_known_source("https://jobs.ashbyhq.com/acme")

    assert gate.ok is False and gate.reason == expected
    assert registered["status"] == "rejected" and registered["reason"] == expected
    assert seeded_registry.read_bytes() == before
    # no tmp residue from atomic writes either
    assert list(seeded_registry.parent.glob(".sources-*")) == []


def test_rejected_register_result_shape(monkeypatch, seeded_registry):
    patch_http(monkeypatch, response=FakeResponse(404, body={}))
    out = sr.register_known_source("https://jobs.ashbyhq.com/acme")
    assert out == {
        "status": "rejected",
        "url": "https://jobs.ashbyhq.com/acme",
        "ats": "ashby",
        "reason": "http_404",
    }
    assert json.loads(seeded_registry.read_text())["version"] == 1  # seeded, intact
