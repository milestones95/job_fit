"""Tests for snippet generation and the sandbox test runner (slice 2).

Codegen is tested against a mocked OpenAI client; the sandbox tests run real
subprocesses but never touch the network.
"""

import time

import adapter_researcher as ar

# Offline happy-path snippet — the sandbox network path is covered by the
# egress/timeout tests; happy-path tests must not touch the network.
GOOD_SNIPPET = (
    'def fetch_jobs():\n'
    '    rows = [{"name": "Engineer", "location": {"city": "SF"}, "absolute_url": "https://a/1"}]\n'
    '    return [\n'
    '        {\n'
    '            "title": r.get("name"),\n'
    '            "location": r.get("location", {}).get("city"),\n'
    '            "url": r.get("absolute_url"),\n'
    '        }\n'
    '        for r in rows\n'
    '    ]\n'
)


def test_sandbox_runs_snippet_and_returns_jobs():
    result = ar.test_snippet(GOOD_SNIPPET)
    assert result.ok
    assert len(result.jobs) == 1
    assert result.jobs[0]["title"] == "Engineer"
    assert result.jobs[0]["location"] == "SF"


def test_http_get_wrapper_fails_gracefully_offline():
    # The wrapper is present in the sandbox and network failures surface as
    # snippet errors with a reason — never a crash or a hang.
    result = ar.test_snippet(
        'def fetch_jobs():\n    return [http_get("https://localhost:9/jobs")]',
        timeout=15,
    )
    assert not result.ok
    assert result.reason.startswith("snippet_error")


def test_sandbox_rejects_import_snippets_statically():
    # Import snippets are refused before any subprocess is spawned — this
    # must fail fast, not after a process round-trip.
    t0 = time.time()
    result = ar.test_snippet("import os\ndef fetch_jobs():\n    return []")
    assert not result.ok
    assert result.reason == "import_blocked"
    assert time.time() - t0 < 1.0


def test_sandbox_blocks_http_egress_before_any_call():
    # http_get must refuse http:// before opening any connection — the
    # failure is immediate, not a 10s timeout.
    t0 = time.time()
    result = ar.test_snippet(
        'def fetch_jobs():\n    return [http_get("http://example.com/jobs")]'
    )
    assert not result.ok
    assert "https" in result.reason
    assert time.time() - t0 < 5.0


def test_sandbox_timeout_enforced():
    t0 = time.time()
    result = ar.test_snippet(
        "def fetch_jobs():\n    while True:\n        pass", timeout=1.5
    )
    assert not result.ok
    assert result.reason == "timeout"
    assert time.time() - t0 < 10.0


def test_sandbox_blocks_file_access():
    # open/exec/eval/__import__ are not in the vetted builtins — the snippet
    # cannot read, write, or reach the environment.
    result = ar.test_snippet('def fetch_jobs():\n    return [open("/tmp/x", "w")]')
    assert not result.ok
    assert "NameError" in result.reason


def test_sandbox_blocks_env_access():
    result = ar.test_snippet(
        "def fetch_jobs():\n    return [{\"k\": str(len(__import__))}]"
    )
    assert not result.ok


def test_sandbox_rejects_non_list_output():
    result = ar.test_snippet('def fetch_jobs():\n    return {"jobs": []}')
    assert not result.ok
    assert result.reason.startswith("bad_shape")


def test_sandbox_empty_jobs_fails_the_gate():
    result = ar.test_snippet('def fetch_jobs():\n    return []')
    assert not result.ok
    assert result.reason == "empty"


def test_sandbox_junk_objects_fail_the_gate():
    # Dicts without any job-like key do not pass the verification gate.
    result = ar.test_snippet('def fetch_jobs():\n    return [{"foo": "bar"}]')
    assert not result.ok
    assert result.reason == "empty"


def test_snippet_runner_never_raises():
    # Garbage in every shape — the runner answers with a reason, never an
    # exception.
    for bad in (None, "", "   ", "def fetch_jobs(:", "x" * 60_000):
        result = ar.test_snippet(bad)
        assert result.ok is False
        assert result.reason


# --------------------------------------------------------------------------
# Codegen — mocked OpenAI client
# --------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return type("Resp", (), {"choices": [_FakeChoice(self._content)]})()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


def _mock_llm(monkeypatch, content):
    monkeypatch.setattr(ar, "_get_openai_client", lambda: _FakeClient(content))


def test_generate_snippet_strips_fences_and_prose(monkeypatch):
    _mock_llm(
        monkeypatch,
        "Here is your snippet:\n```python\n"
        'def fetch_jobs():\n    return [{"title": "x", "url": "https://a"}]\n'
        "```\nLet me know if you need anything else!",
    )
    # generate_snippet with a real Research-shaped object:
    research = type("R", (), {
        "endpoint": "https://api.example.com/jobs",
        "response_shape": "list",
        "field_map": {},
    })()
    snippet = ar.generate_snippet(research)
    assert isinstance(snippet, str)
    assert "```" not in snippet
    assert "Here is your snippet" not in snippet
    assert snippet.startswith("def fetch_jobs")
    # The generated code is data — a string the app never imports.
    assert not hasattr(ar, "fetch_jobs")


def test_generate_snippet_rejects_imports(monkeypatch):
    _mock_llm(monkeypatch, "import requests\ndef fetch_jobs():\n    return []")
    research = type("R", (), {
        "endpoint": "https://api.example.com/jobs",
        "response_shape": "list",
        "field_map": {},
    })()
    assert ar.generate_snippet(research) is None


def test_generate_snippet_rejects_missing_fetch_jobs(monkeypatch):
    _mock_llm(monkeypatch, "I cannot write that snippet, sorry.")
    research = type("R", (), {
        "endpoint": "https://api.example.com/jobs",
        "response_shape": "list",
        "field_map": {},
    })()
    assert ar.generate_snippet(research) is None


def test_generate_snippet_none_research():
    assert ar.generate_snippet(None) is None


def test_generated_snippet_round_trips_through_the_sandbox(monkeypatch):
    _mock_llm(
        monkeypatch,
        'def fetch_jobs():\n    return [{"title": "Engineer", "url": "https://a/1"}]',
    )
    research = type("R", (), {
        "endpoint": "https://api.example.com/jobs",
        "response_shape": "list",
        "field_map": {},
    })()
    snippet = ar.generate_snippet(research)
    result = ar.test_snippet(snippet)
    assert result.ok
    assert result.jobs[0]["title"] == "Engineer"


# --------------------------------------------------------------------------
# Registry contract — snippets are stored as strings, never imported
# --------------------------------------------------------------------------

def test_module_never_imports_generated_code():
    # After generating and running snippets, the module namespace must not
    # contain anything from generated code.
    result = ar.test_snippet(GOOD_SNIPPET)
    assert result.ok
    assert not hasattr(ar, "fetch_jobs")


def test_driver_substitutes_knobs():
    driver = ar._driver_code()
    assert '"ADAPTER_RESULT:"' in driver
    assert "__RESULT_PREFIX__" not in driver
    assert "__MAX_JOBS__" not in driver
    assert "__SAFE_BUILTINS__" not in driver
