"""The full adapter registration flow (spec §3): confirm-gated persistence,
round-trip through start/confirm, and runtime dispatch of stored snippets.

The sandbox tests here run real subprocesses but never touch the network —
research and the LLM are mocked at their seams.
"""

import json

import adapter_researcher as ar
import job_fit_finder as jf

# An offline snippet: exercises the runner end-to-end without network.
SNIPPET = (
    "def fetch_jobs():\n"
    "    return [\n"
    '        {"title": "Backend Engineer", "location": "Remote",\n'
    '         "url": "https://careers.acme.com/1", "department": "Platform"},\n'
    '        {"title": "Product Engineer", "location": "NYC",\n'
    '         "url": "https://careers.acme.com/2"},\n'
    "    ]\n"
)

BAD_SNIPPETS = {
    "raises": (
        "def fetch_jobs():\n    raise RuntimeError(\"boom\")\n"
    ),
    "junk_objects": (
        'def fetch_jobs():\n    return [{"foo": "bar"}, {"baz": 1}]\n'
    ),
    "empty": ('def fetch_jobs():\n    return []\n'),
    "not_a_list": ('def fetch_jobs():\n    return {"jobs": []}\n'),
    "http_egress": (
        'def fetch_jobs():\n    return [http_get("http://example.com/jobs")]\n'
    ),
}


def _research(endpoint="https://careers.acme.com/api/jobs"):
    return ar.Research(
        platform="Acme",
        endpoint=endpoint,
        method="GET",
        docs_url="https://careers.acme.com/developers",
        response_shape="list of {name, location, absolute_url}",
        field_map={"title": "name", "url": "absolute_url"},
        discovered_via="user_hints",
    )


def _registry_ids(path):
    data = json.loads(path.read_text())
    return [s["id"] for s in data["sources"]]


# --------------------------------------------------------------------------
# register_source — the confirm gate
# --------------------------------------------------------------------------

def test_happy_path_persists_a_runnable_snippet(seeded_registry):
    out = ar.register_source("https://careers.acme.com/jobs", SNIPPET, _research(), confirmed=True)
    assert out["status"] == "registered"
    assert out["company"] == "Acme"

    data = json.loads(seeded_registry.read_text())
    entry = next(s for s in data["sources"] if s["id"] == "custom-careers.acme.com")
    # The snippet is stored as a plain string — the registry never imports it.
    assert isinstance(entry["snippet"], str)
    assert entry["ats"] == "custom"
    assert entry["adapter"] is True
    assert entry["endpoint"] == "https://careers.acme.com/api/jobs"
    assert entry["research"]["docs_url"] == "https://careers.acme.com/developers"
    assert entry["research"]["field_map"] == {"title": "name", "url": "absolute_url"}
    assert entry["verification"]["status"] == "passed"
    assert entry["verification"]["via"] == "sandbox"


def test_registry_stores_snippet_as_string_and_module_never_imports_it(seeded_registry):
    ar.register_source("https://careers.acme.com/jobs", SNIPPET, _research(), confirmed=True)
    data = json.loads(seeded_registry.read_text())
    entry = next(s for s in data["sources"] if s["id"] == "custom-careers.acme.com")
    assert isinstance(entry["snippet"], str)
    assert not hasattr(ar, "fetch_jobs")
    assert not hasattr(jf, "fetch_jobs")


def test_failing_snippet_is_never_persisted(seeded_registry):
    before = _registry_ids(seeded_registry)
    for name, bad in BAD_SNIPPETS.items():
        out = ar.register_source("https://careers.acme.com/jobs", bad, _research(), confirmed=True)
        assert out["status"] == "rejected", f"{name} must not register"
        assert out["reason"]
    assert _registry_ids(seeded_registry) == before


def test_unconfirmed_pass_is_never_persisted(seeded_registry):
    out = ar.register_source("https://careers.acme.com/jobs", SNIPPET, _research(), confirmed=False)
    assert out["status"] == "needs_confirmation"
    assert out["preview_jobs"]  # the sandbox test ran and produced a preview
    assert _registry_ids(seeded_registry) == [
        "eliseai", "browserbase", "tamarindbio", "decagon",
    ]


def test_http_egress_snippet_fails_fast_and_never_persists(seeded_registry):
    import time

    before = _registry_ids(seeded_registry)
    t0 = time.time()
    out = ar.register_source(
        "https://careers.acme.com/jobs", BAD_SNIPPETS["http_egress"], _research(), confirmed=True
    )
    assert out["status"] == "rejected"
    assert "https" in out["reason"]
    assert time.time() - t0 < 5.0  # refused before any network activity
    assert _registry_ids(seeded_registry) == before


# --------------------------------------------------------------------------
# Round-trip — start_registration -> confirm_registration
# --------------------------------------------------------------------------

def _mock_agent(monkeypatch, snippet=SNIPPET):
    monkeypatch.setattr(ar, "research_platform", lambda url, page_html="", user_hints=None: _research())
    monkeypatch.setattr(ar, "generate_snippet", lambda research: snippet)


def test_round_trip_persists_only_on_confirm(monkeypatch, seeded_registry):
    _mock_agent(monkeypatch)
    before = _registry_ids(seeded_registry)

    start = ar.start_registration("https://careers.acme.com/jobs", page_html="<html></html>")
    assert start["status"] == "research_pending"
    assert start["test_passed"] is True
    assert start["token"]
    assert start["preview_jobs"]
    assert start["research"]["endpoint"] == "https://careers.acme.com/api/jobs"
    # Nothing persisted by the research phase.
    assert _registry_ids(seeded_registry) == before

    out = ar.confirm_registration(start["token"])
    assert out["status"] == "registered"
    assert "custom-careers.acme.com" in _registry_ids(seeded_registry)


def test_unknown_token_is_rejected(seeded_registry):
    out = ar.confirm_registration("no-such-token")
    assert out["status"] == "rejected"
    assert out["reason"] == "unknown_token"


def test_research_failure_reports_unavailable_and_persists_nothing(monkeypatch, seeded_registry):
    monkeypatch.setattr(ar, "research_platform", lambda url, page_html="", user_hints=None: None)
    before = _registry_ids(seeded_registry)
    out = ar.start_registration("https://careers.acme.com/jobs")
    assert out["status"] == "research_unavailable"
    assert _registry_ids(seeded_registry) == before


def test_codegen_failure_reports_and_persists_nothing(monkeypatch, seeded_registry):
    monkeypatch.setattr(ar, "research_platform", lambda url, page_html="", user_hints=None: _research())
    monkeypatch.setattr(ar, "generate_snippet", lambda research: None)
    before = _registry_ids(seeded_registry)
    out = ar.start_registration("https://careers.acme.com/jobs")
    assert out["status"] == "codegen_failed"
    assert _registry_ids(seeded_registry) == before


# --------------------------------------------------------------------------
# Runtime dispatch — stored snippets execute at fetch time
# --------------------------------------------------------------------------

def _register_confirmed():
    return ar.register_source("https://careers.acme.com/jobs", SNIPPET, _research(), confirmed=True)


def test_adapter_sources_appear_in_the_source_table(seeded_registry):
    _register_confirmed()
    sources = jf.get_sources()
    entry = sources["custom-careers.acme.com"]
    assert entry["adapter"] is True
    assert entry["name"] == "Acme"


def test_stored_snippet_executes_at_dispatch(seeded_registry):
    _register_confirmed()
    jobs = jf.fetch_adapter_source({"id": "custom-careers.acme.com", "name": "Acme"})
    assert len(jobs) == 2
    assert jobs[0] == {
        "company": "Acme",
        "title": "Backend Engineer",
        "location": "Remote",
        "url": "https://careers.acme.com/1",
        "description": "",
        "department": "Platform",
        "workplace_type": "",
        "compensation": "",
        "job_id": "",
    }


def test_dispatch_with_broken_snippet_yields_no_postings(seeded_registry, monkeypatch):
    _register_confirmed()
    # A snippet that worked at registration can rot (board changed) — the
    # fetch must degrade to zero postings for that source, not crash.
    monkeypatch.setattr(
        ar, "run_stored_snippet", lambda snippet, timeout=30.0: (False, [], "timeout")
    )
    jobs = jf.fetch_adapter_source({"id": "custom-careers.acme.com", "name": "Acme"})
    assert jobs == []


def test_dispatch_coerces_non_string_fields(seeded_registry):
    _register_confirmed()
    numeric = (
        "def fetch_jobs():\n"
        "    return [{\"title\": \"Engineer\", \"url\": \"https://a/1\", "
        "\"job_id\": 12345, \"location\": None}]\n"
    )
    ar.register_source("https://beta.acme.com/jobs", numeric, _research(), confirmed=True)
    jobs = jf.fetch_adapter_source({"id": "custom-beta.acme.com", "name": "Beta"})
    assert jobs[0]["job_id"] == ""  # non-strings coerced, pipeline stays string-safe
    assert jobs[0]["location"] == ""
    assert jobs[0]["title"] == "Engineer"
