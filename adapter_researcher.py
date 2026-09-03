"""
Adapter agent for unknown boards — research, generate, test, run.

Spec: Blueprint art_FLK9djM7, work stream §3 ("Adapter agent for unknown
boards — research, generate, test"). For careers pages no fingerprint
matches (see source_registry.detect_ats), this module:

  Researches — research_platform() identifies the platform and finds its
      public jobs-list endpoint (GET preferred) by validating candidate
      URLs over HTTP: devtools-pasted endpoint details first (user hints),
      then API-looking URLs harvested from the page source, then common
      convention paths. Every accepted candidate must prove itself live:
      2xx + JSON + at least one job-like object (same gate as
      source_registry.verify_source). Provenance (docs_url, endpoint,
      method, response_shape, field_map) is recorded on the Research.

  Generates — generate_snippet() asks the LLM to write a self-contained
      fetch_jobs() -> list[dict] Python function allowed ONLY the provided
      http_get wrapper. The snippet is a plain string; the application
      never imports generated code.

  Tests/runs — test_snippet() executes a snippet in a fresh subprocess
      with a restricted namespace (no imports, no file/env access, network
      solely via an https-enforcing http_get wrapper), a hard timeout, and
      shape-validated output. It NEVER raises; failures return a reason.
      run_stored_snippet() is the dispatch-time twin of the same sandbox.

  Registers — register_source() persists snippet + research provenance to
      sources.json ONLY when the sandbox test passed AND the user
      explicitly confirmed. begin_registration()/confirm_registration()
      implement the server's research -> preview -> confirm round-trip.

Threat model, stated honestly: the subprocess boundary, the restricted
namespace, and the https-only egress wrapper are hard guarantees; the
builtins whitelist narrows what generated code can name, but CPython is
not a hostile-code sandbox, so generated snippets are treated as
semi-trusted (they run as the local user, like any dependency would).
This module imports cleanly with OPENAI_API_KEY unset — the OpenAI client
is lazy and only generate_snippet() touches it.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import source_registry as sr

# ---------------------------------------------------------------------------
# Research — find and prove the platform's public jobs-list endpoint.
# ---------------------------------------------------------------------------

# Workday boards are excluded by spec §3: their SOAP/GraphQL gateways are
# tenant-specific, undocumented per-company, and unstable to generate for.
_WORKDAY_HOST_PATTERN = re.compile(
    r"(?:^|\.)myworkday(?:jobs|login)\.com$|workday\.", re.IGNORECASE
)


@dataclass
class Research:
    """Provenance-carrying result of researching one unknown board."""

    platform: str
    endpoint: str
    method: str = "GET"
    docs_url: str | None = None
    response_shape: str = ""
    field_map: dict = field(default_factory=dict)
    discovered_via: str = ""  # "user_hints" | "page_scan" | "convention_probe"

    def to_dict(self):
        """The provenance block persisted alongside the snippet — the five
        keys the spec names, plus the platform name and how the endpoint was
        found."""
        return {
            "platform": self.platform,
            "docs_url": self.docs_url,
            "endpoint": self.endpoint,
            "method": self.method,
            "response_shape": self.response_shape,
            "field_map": dict(self.field_map),
            "discovered_via": self.discovered_via,
        }


# Normalized field -> raw keys we look for on a posting object, in priority
# order. Superset of the four known ATSes plus shapes seen on custom boards.
_FIELD_CANDIDATES = {
    "title": ("title", "name", "text", "jobTitle", "position", "opening"),
    "location": ("location", "candidate_required_location", "jobLocation", "locationName", "address"),
    "url": ("url", "link", "jobUrl", "hostedUrl", "absolute_url", "applyUrl", "job_url", "share_url"),
    "description": ("description", "descriptionPlain", "content", "body", "jobDescription", "snippet"),
    "department": ("department", "category", "team", "departmentName"),
    "workplace_type": ("workplace_type", "jobType", "employment_type", "type"),
    "compensation": ("compensation", "salary", "salary_min", "pay"),
    "job_id": ("id", "jobId", "job_id", "uuid", "externalId"),
}


def _infer_field_map(job):
    """Map normalized posting fields onto the raw keys a platform actually
    uses, based on one real posting object. Fields with no recognizable key
    are simply absent from the map."""
    field_map = {}
    for normalized, candidates in _FIELD_CANDIDATES.items():
        for key in candidates:
            value = job.get(key)
            if isinstance(value, (str, int, float)) and value != "":
                field_map[normalized] = key
                break
    return field_map


def _describe_shape(body, jobs):
    """Short human-readable response-shape string for the provenance record."""
    if isinstance(body, list):
        shape = f"list[{len(body)}] of dict"
    elif isinstance(body, dict):
        list_key = next((k for k, v in body.items() if isinstance(v, list)), None)
        shape = f"dict{'{' + repr(list_key) + ': list}' if list_key else ''}"
    else:
        shape = type(body).__name__
    keys = sorted({str(k) for j in jobs[:3] for k in j})[:12]
    return f"{shape}; job keys: {','.join(keys)}"


def _platform_name(url, page_html=""):
    """Best-effort platform name: og:site_name, then <title>, then the host
    minus www./careers./jobs.-style prefixes."""
    if page_html:
        m = re.search(
            r"""<meta[^>]+property=["']og:site_name["'][^>]+content=["']([^"']+)""", page_html
        ) or re.search(r"<title[^>]*>([^<]+)</title>", page_html, re.IGNORECASE)
        if m:
            name = m.group(1).strip().split(" — ")[0].split(" | ")[0].strip()
            if name:
                return name[:60]
    host = (urlsplit(url).netloc or "").lower()
    for prefix in ("www.", "careers.", "jobs.", "join.", "work.", "boards."):
        host = host.removeprefix(prefix)
    return host.split(".")[0].capitalize() or "unknown"


def _derive_id(url):
    """Stable registry id for an unknown board: cleaned host core, lowercased."""
    host = (urlsplit(url).netloc or "custom").lower()
    for prefix in ("www.", "careers.", "jobs.", "join.", "work.", "boards."):
        host = host.removeprefix(prefix)
    core = host.split(".")[0]
    return re.sub(r"[^a-z0-9_-]", "-", core) or "custom"


# API-looking URLs harvestable from the page source the popup sends up.
_PAGE_ENDPOINT_PATTERN = re.compile(
    r"""["'\(]((?:https?:)?//[^"'\s<>()]+|/(?:api/)?[^"'\s<>()]*\.json[^"'\s<>()]*|/api/[^"'\s<>()]+)["'\)]""",
    re.IGNORECASE,
)
_JOB_WORD = re.compile(r"jobs?|careers?|postings?|openings?|vacanc|positions?", re.IGNORECASE)


def _candidates_from_page(page_html, base_url):
    """Up to three https API-looking job URLs harvested from page source."""
    found = []
    for match in _PAGE_ENDPOINT_PATTERN.findall(page_html or ""):
        candidate = urljoin(base_url, match.strip())
        parts = urlsplit(candidate)
        if parts.scheme != "https" or not _JOB_WORD.search(parts.path + parts.netloc):
            continue
        if candidate not in found and "sw.js" not in candidate:
            found.append(candidate)
        if len(found) >= 3:
            break
    return found


# Common convention paths probed when hints and page source come up empty.
_CONVENTION_PATHS = (
    "/api/jobs", "/api/jobs.json", "/api/careers", "/api/careers.json",
    "/api/v1/jobs", "/api/v1/jobs.json", "/api/postings", "/api/openings",
    "/jobs.json", "/careers.json",
)

# Bounded research: never probe more than this many URLs per call.
_MAX_PROBES = 12


def _http_get(url, timeout=10.0):
    # Seam for tests: monkeypatch this instead of the requests module.
    import requests

    return requests.get(url, timeout=timeout)


def _validate_candidate_endpoint(endpoint, method):
    """Pre-network validation of a candidate endpoint. Returns None when the
    candidate may be probed, else the reason it is rejected — checked BEFORE
    any network call (notably: http:// endpoints never reach the wire)."""
    if not endpoint or not isinstance(endpoint, str):
        return "no endpoint"
    parts = urlsplit(endpoint.strip())
    if parts.scheme.lower() != "https" or not parts.netloc:
        return "unsafe_scheme"
    if (method or "GET").upper() != "GET":
        # The sandbox wrapper is GET-only in v1 (spec: GET preferred).
        return f"method_not_supported_v1:{method}"
    return None


def _probe_endpoint(candidate):
    """One live GET against a candidate endpoint. Returns (body, jobs, reason):
    jobs is a non-empty list of raw posting dicts on success, body is the
    parsed JSON the jobs came from (for the provenance shape record)."""
    try:
        resp = _http_get(candidate, timeout=10.0)
    except Exception as e:
        print(f"[research_platform] probe failed for {candidate}: {e}")
        return None, None, "network_error"
    if not (200 <= resp.status_code < 300):
        return None, None, f"http_{resp.status_code}"
    try:
        body = resp.json()
    except (ValueError, TypeError):
        return None, None, "non_json"
    jobs = sr._extract_jobs(body)
    if not any(sr._looks_like_job(j) for j in jobs):
        return None, None, "no_job_like_objects"
    return body, jobs, None


def research_platform(url, page_html="", user_hints=None):
    """Research one unknown careers URL. Returns a Research (with provenance)
    or None when the board is not researchable — never a silent guess.

    Order of candidates: user-pasted devtools endpoint details (hints) first,
    then API-looking URLs harvested from the page source, then common
    convention paths. Every candidate must prove itself live before it is
    trusted; the first that does wins. Excludes Workday boards by spec."""
    if not url or not isinstance(url, str) or not url.strip():
        print("[research_platform] not researchable: no url")
        return None
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        print(f"[research_platform] not researchable: not a http(s) URL: {url}")
        return None
    if _WORKDAY_HOST_PATTERN.search(parts.netloc):
        print(f"[research_platform] not researchable: {parts.netloc} is a Workday board (excluded by spec)")
        return None
    detected = sr.detect_ats(url)
    if detected and detected[0] in sr.ENDPOINT_TEMPLATES:
        print(f"[research_platform] not researchable here: {url} is a known {detected[0]} board — native lane")
        return None

    hints = user_hints if isinstance(user_hints, dict) else {}
    # Candidates: (via, url, method) — hint first, then page-scan URLs, then
    # convention paths. Each is validated BEFORE any network call.
    candidates: list[tuple[str, str, str]] = []
    hint_endpoint = (hints.get("endpoint") or "").strip()
    hint_method = ((hints.get("method") or "GET").strip() or "GET").upper()
    if hint_endpoint:
        reason = _validate_candidate_endpoint(hint_endpoint, hint_method)
        if reason:
            print(f"[research_platform] hint endpoint rejected: {reason}")
        else:
            candidates.append(("user_hints", hint_endpoint, hint_method))
    candidates.extend(("page_scan", c, "GET") for c in _candidates_from_page(page_html, url))
    origin = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
    candidates.extend(("convention_probe", origin + path, "GET") for path in _CONVENTION_PATHS)

    attempts = 0
    for via, candidate, method in candidates:
        if attempts >= _MAX_PROBES:
            break
        attempts += 1
        body, jobs, probe_reason = _probe_endpoint(candidate)
        if jobs is None:
            print(f"[research_platform] candidate {candidate} rejected: {probe_reason}")
            continue
        first_job = next(j for j in jobs if sr._looks_like_job(j))
        research = Research(
            platform=_platform_name(url, page_html),
            endpoint=candidate,
            method=method,
            docs_url=hints.get("docs_url") or url,
            response_shape=_describe_shape(body, jobs),
            field_map=_infer_field_map(first_job),
            discovered_via=via,
        )
        print(f"[research_platform] {url} -> {candidate} ({via}, {len(jobs)} postings)")
        return research
    print(f"[research_platform] not researchable: no candidate endpoint proved itself for {url}")
    return None
