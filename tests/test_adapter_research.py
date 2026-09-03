"""research_platform: candidate-endpoint discovery and proof for unknown
boards. All HTTP goes through the _http_get seam — no live network here
(the live end-to-end probe is live_adapter_probe.py, run manually)."""
import adapter_researcher as ar


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


REMOTIVE_BODY = {
    "job_count": 2,
    "jobs": [
        {"id": 1, "title": "Backend Engineer", "company_name": "Acme",
         "url": "https://remotive.com/1", "candidate_required_location": "Remote",
         "description": "Build things"},
        {"id": 2, "title": "Full-Stack Engineer", "company_name": "Beta",
         "url": "https://remotive.com/2", "candidate_required_location": "Remote",
         "description": "Build more things"},
    ],
}


def patch_http(monkeypatch, responder, calls=None):
    """responder(url) -> FakeResponse | raises. Records every probed URL so
    tests can assert what did (and did not) touch the network."""
    def fake_get(url, timeout=10.0):
        if calls is not None:
            calls.append(url)
        return responder(url)

    monkeypatch.setattr(ar, "_http_get", fake_get)


# --- not researchable (no network) -----------------------------------------

def test_workday_excluded_before_any_network(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    research = ar.research_platform(
        "https://acme.wd3.myworkdayjobs.com/en-US/careers"
    )
    assert research is None
    assert calls == []  # excluded before any call


def test_known_ats_url_rejected_native_lane(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    assert ar.research_platform("https://jobs.ashbyhq.com/acme") is None
    assert calls == []


def test_unsafe_or_bare_url_rejected_without_network(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    assert ar.research_platform("") is None
    assert ar.research_platform("not a url") is None
    assert ar.research_platform(None) is None
    assert calls == []


def test_no_candidate_proves_itself(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(404, {}), calls)
    assert ar.research_platform("https://careers.acme.com/jobs") is None


def test_endpoint_without_job_like_objects_rejected(monkeypatch):
    # 2xx + JSON but no recognizable posting objects -> not researchable.
    patch_http(monkeypatch, lambda url: FakeResponse(200, {"status": "ok", "items": [{"foo": 1}]}))
    assert ar.research_platform("https://careers.acme.com/jobs") is None


# --- hints path (devtools-pasted endpoint details) --------------------------

def test_hint_endpoint_wins_and_records_provenance(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    research = ar.research_platform(
        "https://careers.acme.com/jobs",
        user_hints={"endpoint": "https://careers.acme.com/api/jobs", "docs_url": "https://careers.acme.com/docs"},
    )
    assert research is not None
    assert research.endpoint == "https://careers.acme.com/api/jobs"
    assert research.discovered_via == "user_hints"
    assert research.docs_url == "https://careers.acme.com/docs"
    assert research.method == "GET"
    assert calls == ["https://careers.acme.com/api/jobs"]  # hint first, no wasted probes
    prov = research.to_dict()
    assert prov["endpoint"] == research.endpoint
    assert prov["field_map"]["title"] == "title"
    assert "title" in prov["response_shape"]


def test_http_hint_never_reaches_the_wire(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    research = ar.research_platform(
        "https://careers.acme.com/jobs",
        user_hints={"endpoint": "http://careers.acme.com/api/jobs"},
    )
    # The http:// hint is rejected before any network call; research then
    # falls through to the https convention probes (by design).
    assert all(u.startswith("https://") for u in calls), f"unsafe URL probed: {calls}"
    assert research is None or research.endpoint.startswith("https://")


def test_post_hint_endpoint_never_called(monkeypatch):
    calls = []
    patch_http(monkeypatch, lambda url: FakeResponse(200, REMOTIVE_BODY), calls)
    ar.research_platform(
        "https://careers.acme.com/jobs",
        user_hints={"endpoint": "https://careers.acme.com/api/search", "method": "POST"},
    )
    # v1 wrapper is GET-only: the POST endpoint is rejected pre-network.
    assert "https://careers.acme.com/api/search" not in calls


def test_hint_falls_through_to_conventions_when_hint_fails(monkeypatch):
    # Hint 404s, first convention path validates -> research still succeeds.
    def responder(url):
        if url == "https://careers.acme.com/api/jobs":
            return FakeResponse(404, {})
        return FakeResponse(200, REMOTIVE_BODY)

    calls = []
    patch_http(monkeypatch, responder, calls)
    research = ar.research_platform(
        "https://careers.acme.com/jobs",
        user_hints={"endpoint": "https://careers.acme.com/api/jobs"},
    )
    assert research is not None
    assert research.discovered_via == "convention_probe"
    assert research.endpoint == "https://careers.acme.com/api/jobs.json"
    assert calls[0] == "https://careers.acme.com/api/jobs"


# --- page-scan path ---------------------------------------------------------

def test_api_url_harvested_from_page_html(monkeypatch):
    page_html = """
    <html><head><title>Acme Careers</title></head><body>
    <script>fetch("/api/v2/jobs.json").then(r => r.json())</script>
    </body></html>
    """
    def responder(url):
        if url == "https://careers.acme.com/api/v2/jobs.json":
            return FakeResponse(200, REMOTIVE_BODY)
        return FakeResponse(404, {})

    patch_http(monkeypatch, responder)
    research = ar.research_platform("https://careers.acme.com/jobs", page_html=page_html)
    assert research is not None
    assert research.discovered_via == "page_scan"
    assert research.endpoint == "https://careers.acme.com/api/v2/jobs.json"
    assert research.platform == "Acme Careers"  # from the page <title>


# --- field mapping ----------------------------------------------------------

def test_field_map_infers_custom_key_names(monkeypatch):
    # "name" passes the job-like gate; the other fields use custom raw keys
    # the field map should discover.
    body = {"jobs": [{"name": "Engineer", "jobUrl": "https://x/1",
                      "candidate_required_location": "Remote", "jobId": "9",
                      "jobDescription": "Build things"}]}
    patch_http(monkeypatch, lambda url: FakeResponse(200, body))
    research = ar.research_platform("https://careers.acme.com/jobs")
    assert research.field_map == {
        "title": "name",
        "location": "candidate_required_location",
        "url": "jobUrl",
        "description": "jobDescription",
        "job_id": "jobId",
    }
