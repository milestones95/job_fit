"""register_known_source: the detect -> verify -> persist flow, incl.
research_pending for unknown boards and idempotent repeat registration."""
import json

import source_registry as sr


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def patch_http(monkeypatch, response):
    monkeypatch.setattr(sr, "_http_request", lambda *a, **k: response)


ASHBY_BODY = {"jobs": [{"title": "Engineer"}, {"title": "Product Engineer"},
                       {"title": "Full-Stack Engineer"}]}


def test_unknown_url_returns_research_pending(seeded_registry):
    out = sr.register_known_source("https://careers.acme.com/jobs")
    assert out["status"] == "research_pending"
    assert "message" in out
    # nothing was written
    data = json.loads(seeded_registry.read_text())
    assert [s["id"] for s in data["sources"]] == ["eliseai", "browserbase", "tamarindbio", "decagon"]


def test_known_board_registers_and_verifies(monkeypatch, seeded_registry):
    patch_http(monkeypatch, FakeResponse(200, ASHBY_BODY))
    out = sr.register_known_source("https://jobs.ashbyhq.com/acme-corp")

    assert out["status"] == "registered"
    assert out["ats"] == "ashby"
    assert out["job_count"] == 3
    # The popup analyzes native lanes with the exact verified token (ids are
    # lowercased; tokens keep their original case).
    assert out["board_token"] == "acme-corp"

    data = json.loads(seeded_registry.read_text())
    entry = next(s for s in data["sources"] if s["id"] == "acme-corp")
    assert entry["company"] == "Acme Corp"  # derived from the token, popup-style
    assert entry["board_token"] == "acme-corp"
    assert entry["added_via"] == "extension"
    assert entry["verification"] == {
        "status": "passed", "job_count": 3, "checked_at": entry["verification"]["checked_at"],
    }
    assert entry["verification"]["status"] == "passed"


def test_registering_existing_builtin_refreshes_without_duplicating(monkeypatch, seeded_registry):
    before = json.loads(seeded_registry.read_text())
    eliseai_before = next(s for s in before["sources"] if s["id"] == "eliseai")
    assert eliseai_before["added_via"] == "builtin"

    patch_http(monkeypatch, FakeResponse(200, ASHBY_BODY))
    out = sr.register_known_source("https://jobs.ashbyhq.com/eliseai")

    assert out["status"] == "registered"
    assert out["source_id"] == "eliseai"
    after = json.loads(seeded_registry.read_text())
    assert len(after["sources"]) == 4  # no duplicate
    eliseai_after = next(s for s in after["sources"] if s["id"] == "eliseai")
    assert eliseai_after["added_via"] == "builtin"          # original identity kept
    assert eliseai_after["verification"]["status"] == "passed"  # verification refreshed


def test_repeat_call_is_idempotent(monkeypatch, seeded_registry):
    patch_http(monkeypatch, FakeResponse(200, ASHBY_BODY))
    first = sr.register_known_source("https://jobs.ashbyhq.com/acme-corp")
    second = sr.register_known_source("https://jobs.ashbyhq.com/acme-corp")

    assert first == second  # same response
    data = json.loads(seeded_registry.read_text())
    assert len([s for s in data["sources"] if s["board_token"] == "acme-corp"]) == 1
