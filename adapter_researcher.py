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

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
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
    except Exception as e:  # noqa: BLE001 — a probe failure is a data point, not an error
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

# ---------------------------------------------------------------------------
# Codegen — LLM writes a self-contained fetch_jobs() snippet (a string; the
# application never imports generated code).
# ---------------------------------------------------------------------------

ADAPTER_CODEGEN_MODEL = os.environ.get("JOB_FIT_ADAPTER_MODEL", "gpt-4o-mini")
SNIPPET_MAX_CHARS = 50_000

# The OpenAI client is lazy — constructed on the first codegen call, not at
# import (same pattern as job_fit_finder) so this module imports cleanly
# without OPENAI_API_KEY.
_client = None


def _get_openai_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — put it in .env next to this "
                "script or export it in your shell."
            )
        from openai import OpenAI

        _client = OpenAI(api_key=api_key)
    return _client


_CODEGEN_PROMPT = """You write tiny, safe Python snippets that fetch a job board's public JSON API.

Write ONE function:

def fetch_jobs():

that returns a list of dicts, one per job posting, with as many of these keys
as the API provides: title, location, url, description, department,
workplace_type, compensation, job_id.

The ONLY capability your function has is a function named http_get, already
provided in scope:

    data = http_get(url)   # GETs url, returns the parsed JSON (dict or list).
                           # Raises RuntimeError on any failure (non-https,
                           # network error, non-JSON, oversized response).

Hard rules:
- No import statements. No open(). No os/sys/env access. No network calls
  except through http_get. No infinite loops.
- Use .get() chains defensively — the API shape is known but may vary.
- Return [] (empty list) if nothing is found. Never print.
- Output ONLY the function, no explanations, no markdown fences.

The endpoint to fetch and the shape it returns (from live research):

endpoint: {endpoint}
response_shape: {response_shape}
field_map (normalized -> the API's raw key): {field_map}"""


def _clean_snippet(text):
    """Extract and statically validate the fetch_jobs snippet from a model
    response. Returns the cleaned code string, or None (with a printed
    reason) — never a guess."""
    code = text.strip()
    # Strip markdown fences when present.
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```\s*$", "", code)
    # Keep from the function definition onward, dropping any prose above it.
    match = re.search(r"^(def fetch_jobs\s*\(.*)", code, re.DOTALL | re.MULTILINE)
    if not match:
        print("[generate_snippet] model response has no def fetch_jobs")
        return None
    # An import ABOVE the function means the model is ignoring the contract —
    # reject rather than silently trimming it away.
    if re.search(r"^\s*(?:import|from)\s+\w", code[:match.start()], re.MULTILINE):
        print("[generate_snippet] model snippet rejected: import_blocked")
        return None
    # Trim trailing prose: take the largest line-prefix starting at the
    # function definition that parses as Python. Prose after the function
    # breaks the parse, so back off line by line until it doesn't.
    lines = code[match.start():].splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        code = candidate
        break
    else:
        print("[generate_snippet] model snippet rejected: unparsable")
        return None
    reason = _validate_snippet(code)
    if reason:
        print(f"[generate_snippet] model snippet rejected: {reason}")
        return None
    return code


def _validate_snippet(code):
    """Static checks every snippet must pass before it is ever run. Returns
    None when the snippet is acceptable, else the reason."""
    if not code or not isinstance(code, str) or not code.strip():
        return "empty"
    if len(code) > SNIPPET_MAX_CHARS:
        return "snippet_too_large"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax_error:{e.lineno}"
    has_fetch_jobs = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import_blocked"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global_blocked"
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_jobs":
            has_fetch_jobs = True
    if not has_fetch_jobs:
        return "no_fetch_jobs"
    return None


def generate_snippet(research):
    """Ask the LLM for a self-contained fetch_jobs() snippet for this
    research. Returns the snippet as a string, or None on any failure —
    never raises, never guesses. The snippet is data: the application never
    imports it; it only ever runs inside the sandbox below."""
    if research is None:
        return None
    try:
        resp = _get_openai_client().chat.completions.create(
            model=ADAPTER_CODEGEN_MODEL,
            temperature=0,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": "You output raw Python code only. No markdown fences, no prose."},
                {"role": "user", "content": _CODEGEN_PROMPT.format(
                    endpoint=research.endpoint,
                    response_shape=research.response_shape,
                    field_map=json.dumps(research.field_map, ensure_ascii=False),
                )},
            ],
        )
    except Exception as e:  # noqa: BLE001 — never let codegen break the caller
        print(f"[generate_snippet] LLM call failed: {e}")
        return None
    return _clean_snippet(resp.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Sandbox — run a snippet in a fresh subprocess with a restricted namespace,
# https-only network via the provided wrapper, and a hard timeout. Never
# raises; every failure carries a reason.
# ---------------------------------------------------------------------------

_RESULT_PREFIX = "ADAPTER_RESULT:"
# Cap on the JSON body a single http_get may return (5 MB — no jobs API needs
# more, and it bounds memory even where resource limits are unavailable).
_HTTP_GET_MAX_BYTES = 5 * 1024 * 1024
# Hard cap on jobs a snippet may return (bounds memory for pathological
# snippets; dispatch consumes the list as-is).
_MAX_JOBS = 2000

# Vetted builtins the snippet's namespace exposes; anything not listed
# (open, exec, eval, __import__, globals, ...) is unavailable by name.
_SAFE_BUILTIN_NAMES = (
    "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "isinstance", "sorted", "range", "enumerate", "zip", "min", "max",
    "sum", "abs", "round", "any", "all", "repr", "map", "filter",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "AttributeError", "Exception",
)

_SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

# The child-process driver. Runs as `python -I -c <driver>` with a minimal
# environment: reads the snippet from stdin, executes it under the restricted
# namespace, prints exactly one ADAPTER_RESULT: JSON envelope line. Tokens
# (__RESULT_PREFIX__ etc.) are substituted with plain .replace() — the driver
# is full of braces, so str.format is not an option here.
_DRIVER = r'''
import ast
import builtins as _builtins_mod
import json
import sys
import urllib.error  # noqa: F401  (imported so urllib.request is complete)
import urllib.request

# Resource limits: no file writes, bounded memory, bounded CPU. Best-effort
# (POSIX only) — the parent's hard timeout stays the primary guard.
try:
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
except Exception:
    pass

_RESULT_PREFIX = "__RESULT_PREFIX__"
_MAX_JOBS = __MAX_JOBS__
_HTTP_GET_MAX_BYTES = __MAX_BYTES__
_SAFE_BUILTIN_NAMES = __SAFE_BUILTINS__


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirects to non-https URLs are refused — the https-only rule holds
    across the whole redirect chain, not just the first hop."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not str(newurl).lower().startswith("https://"):
            raise RuntimeError(f"http_get: insecure redirect blocked ({newurl})")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())


def _http_get(url, timeout=10):
    """The wrapper generated snippets are allowed to use: https-only GET
    returning parsed JSON. The scheme check happens BEFORE any network call."""
    if not isinstance(url, str) or not url.strip().lower().startswith("https://"):
        raise RuntimeError("http_get: only https:// URLs are allowed")
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "job-fit-adapter/1.0"}
    )
    with _opener.open(req, timeout=timeout) as resp:
        raw = resp.read(_HTTP_GET_MAX_BYTES + 1)
    if len(raw) > _HTTP_GET_MAX_BYTES:
        raise RuntimeError("http_get: response too large")
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        raise RuntimeError("http_get: endpoint did not return valid JSON")


def _fail(reason):
    print(_RESULT_PREFIX + json.dumps({"ok": False, "reason": reason}))
    sys.exit(0)


def _main():
    code = sys.stdin.read()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        _fail(f"syntax_error:{e.lineno}")
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _fail("import_blocked")
            return

    # Defense in depth: the parent validated too, but the child re-checks —
    # the snippet arrives over a pipe and the parent could be misconfigured.
    namespace = {
        "http_get": _http_get,
        "__builtins__": {name: getattr(_builtins_mod, name) for name in _SAFE_BUILTIN_NAMES},
    }
    try:
        exec(compile(code, "<adapter-snippet>", "exec"), namespace)
    except Exception as e:
        _fail(f"snippet_error: {type(e).__name__}: {e}")
        return

    fetch_jobs = namespace.get("fetch_jobs")
    if not callable(fetch_jobs):
        _fail("no_fetch_jobs")
        return
    try:
        jobs = fetch_jobs()
    except Exception as e:
        _fail(f"snippet_error: {type(e).__name__}: {e}")
        return

    if not isinstance(jobs, list):
        _fail(f"bad_shape: fetch_jobs returned {type(jobs).__name__}, expected list")
        return
    jobs = [j for j in jobs if isinstance(j, dict)][:_MAX_JOBS]
    print(_RESULT_PREFIX + json.dumps({"ok": True, "jobs": jobs}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    _main()
'''


def _driver_code():
    """The driver with its knobs substituted (tokens, not str.format — the
    driver text is full of braces)."""
    return (
        _DRIVER
        .replace("__RESULT_PREFIX__", _RESULT_PREFIX)
        .replace("__MAX_JOBS__", str(_MAX_JOBS))
        .replace("__MAX_BYTES__", str(_HTTP_GET_MAX_BYTES))
        .replace("__SAFE_BUILTINS__", repr(_SAFE_BUILTIN_NAMES))
    )


def _execute_sandboxed(code, timeout=20.0):
    """Run one snippet in the sandbox. Returns (ok, jobs, reason) and NEVER
    raises. The subprocess is fresh, isolated (-I), minimally env'd, and
    killed hard on timeout."""
    reason = _validate_snippet(code)
    if reason:
        return False, [], reason
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _driver_code()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_SANDBOX_ENV, cwd=tempfile.gettempdir(),
        )
    except OSError as e:
        return False, [], f"spawn_error: {e}"
    try:
        out, err = proc.communicate(input=code.encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False, [], "timeout"
    except Exception as e:  # noqa: BLE001 — never raise for a bad snippet
        proc.kill()
        proc.wait()
        return False, [], f"runner_error: {e}"

    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", errors="replace").strip().splitlines()
        return False, [], f"runner_error[exit {proc.returncode}]: {tail[-1] if tail else 'no stderr'}"

    envelope = next(
        (line for line in reversed((out or b"").decode("utf-8", errors="replace").splitlines())
         if line.startswith(_RESULT_PREFIX)),
        None,
    )
    if envelope is None:
        return False, [], "runner_error: no result envelope"
    try:
        payload = json.loads(envelope[len(_RESULT_PREFIX):])
    except ValueError:
        return False, [], "runner_error: malformed result envelope"
    if not payload.get("ok"):
        return False, [], str(payload.get("reason") or "unknown")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return False, [], "bad_shape: expected a list of jobs"
    return True, jobs, None


# Preview cap: how many verified postings the registration preview carries.
_PREVIEW_CAP = 50


def test_snippet(code, timeout=20.0):
    """Test one candidate snippet in the sandbox. Pass = it runs cleanly and
    returns at least one job-like object. Returns a VerifyResult (the same
    shape the verification gate uses); NEVER raises and NEVER persists."""
    try:
        ok, jobs, reason = _execute_sandboxed(code, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — absolute backstop, the runner never raises
        return sr.VerifyResult(ok=False, reason=f"runner_error: {e}")
    if not ok:
        return sr.VerifyResult(ok=False, reason=reason)
    if not any(sr._looks_like_job(j) for j in jobs):
        return sr.VerifyResult(ok=False, reason="empty")
    return sr.VerifyResult(ok=True, jobs=jobs[:_PREVIEW_CAP])
