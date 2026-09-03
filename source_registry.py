"""
Source registry, ATS detection, and the verification gate.

Spec: Blueprint art_FLK9djM7 ("Spec — dynamic job-source registration"),
work stream 1. Three halves live here:

  Detection — a fingerprint table over careers-page URLs says which ATS a
      board runs on and what its board token is (detect_ats).

  Registry — a versioned, gitignored sources.json (user-local config, like
      .env) replaces the hardcoded COMPANIES table as the source of truth
      for tracked boards. Built-ins seed it on first run; writes are atomic
      (tmp file + rename).

  Gate — verify_source() makes one live call against a candidate endpoint
      and passes ONLY on 2xx + JSON body + at least one job-like object.
      It never raises for a rejected source and never persists anything —
      the rejection reason is returned for the caller to surface. Nothing
      is written unless a verify passed (register_known_source).

Nothing here touches OpenAI: all modules in this repo must import cleanly
with OPENAI_API_KEY unset.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

# ---------------------------------------------------------------------------
# Built-ins — the companies hardcoded in job_fit_finder.py today. They seed
# sources.json on first run; after that the registry is the source of truth
# (a registry entry overrides a built-in with the same id).
# ---------------------------------------------------------------------------

# NOTE: the " Tamarind Bio" leading space is a known pre-existing bug,
# deliberately preserved — the spec flags it as an explicit follow-up,
# not v1 scope.
BUILTIN_COMPANIES = [
    {"name": "EliseAI", "ats": "ashby", "token": "eliseai"},
    {"name": "Browserbase", "ats": "ashby", "token": "browserbase"},
    {"name": " Tamarind Bio", "ats": "ashby", "token": "tamarindbio"},
    {"name": "Decagon", "ats": "ashby", "token": "decagon"},
]

# Public job-list endpoint per ATS — the same entrypoints the native
# fetchers in job_fit_finder.py already call.
ENDPOINT_TEMPLATES = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true",
    "lever": "https://api.lever.co/v0/postings/{token}?mode=json",
    "smartrecruiters": "https://api.smartrecruiters.com/v1/companies/{token}/postings",
}

# Fingerprint table over careers-page URLs (same patterns the add-job-source
# skill documents; verified live against Greenhouse/Ashby boards, 2026-09-03 —
# spec §2). Matched against the lowercased host + path so a token smuggled
# into a query string or a foreign host's path can't fake a match.
FINGERPRINTS = [
    ("greenhouse", re.compile(r"boards\.greenhouse\.io/([A-Za-z0-9_-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.lever\.co/([A-Za-z0-9_-]+)")),
    ("smartrecruiters", re.compile(r"jobs\.smartrecruiters\.com/([A-Za-z0-9_-]+)")),
]


def detect_ats(url):
    """(ats, board_token) from a careers URL, or None if no fingerprint matches.

    Scheme-agnostic on purpose: detection is about identity, and the
    https-only safety rule is enforced by verify_source (unsafe_scheme)."""
    if not url or not isinstance(url, str):
        return None
    parts = urlsplit(url.strip())
    host_path = f"{parts.netloc.lower()}{parts.path}"
    for ats, pattern in FINGERPRINTS:
        match = pattern.match(host_path)  # anchored: host must start with the fingerprint
        if match:
            return ats, match.group(1)
    return None


# ---------------------------------------------------------------------------
# Registry — versioned sources.json, seeded from built-ins, atomic writes.
# ---------------------------------------------------------------------------

REGISTRY_VERSION = 1


def registry_path():
    """Where sources.json lives: repo root by default, overridable via
    JOB_FIT_SOURCES_PATH so tests (and multi-copy installs) can relocate it."""
    override = os.environ.get("JOB_FIT_SOURCES_PATH")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources.json")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path, data):
    """Write JSON atomically: tmp file in the same directory, fsync, rename.
    A crash mid-write leaves the previous version intact — never a truncated
    or half-written registry."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".sources-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tmp file; the registry itself is untouched.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _builtin_entry(company):
    ats, token = company["ats"], company["token"]
    return {
        "id": token.lower(),
        "company": company["name"],
        "ats": ats,
        "board_token": token,
        "endpoint": ENDPOINT_TEMPLATES[ats].format(token=token),
        "method": "GET",
        "headers": {},
        "added_via": "builtin",
        "added_at": _now_iso(),
        # Built-ins are trusted defaults, not probed — they have not been
        # through the verify gate, so they don't claim a "passed" status.
        "verification": {"status": "builtin", "job_count": None, "checked_at": None},
    }


def load_registry():
    """Load sources.json, seeding it from the built-ins on first run.

    A file that exists but doesn't parse raises — silently re-seeding would
    destroy user data, and a corrupt registry is a fix-it-now problem."""
    path = registry_path()
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        if not content.strip():
            # Empty file (e.g. touch'd) — treat as absent and seed it.
            seed = {"version": REGISTRY_VERSION,
                    "sources": [_builtin_entry(c) for c in BUILTIN_COMPANIES]}
            _atomic_write_json(path, seed)
            return seed
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"sources.json at {path} is not valid JSON — fix or delete it "
                f"(refusing to overwrite user data): {e}"
            ) from e
        if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
            raise RuntimeError(
                f"sources.json at {path} has an unexpected shape "
                f"(expected {{\"version\": ..., \"sources\": [...]}})"
            )
        return data
    seed = {"version": REGISTRY_VERSION,
            "sources": [_builtin_entry(c) for c in BUILTIN_COMPANIES]}
    _atomic_write_json(path, seed)
    return seed


def save_registry(data):
    _atomic_write_json(registry_path(), data)


def upsert_source(entry):
    """Insert or update one registry entry and persist atomically.

    Dedupe key is (ats, board_token): re-registering a known board refreshes
    it in place instead of duplicating it, preserving the original
    company/added_via/added_at. Ids stay unique — a colliding id for a
    different board gains an -<ats> suffix."""
    data = load_registry()
    sources = data["sources"]
    key = (entry.get("ats"), entry.get("board_token"))
    for i, existing in enumerate(sources):
        if (existing.get("ats"), existing.get("board_token")) == key:
            merged = {
                **existing,
                **entry,
                "id": existing.get("id") or entry.get("id"),
                "company": existing.get("company") or entry.get("company"),
                "added_via": existing.get("added_via") or entry.get("added_via"),
                "added_at": existing.get("added_at") or entry.get("added_at"),
            }
            sources[i] = merged
            save_registry(data)
            return merged
    new_entry = dict(entry)
    ids = {s.get("id") for s in sources}
    if new_entry.get("id") in ids:
        new_entry["id"] = f"{new_entry['id']}-{new_entry.get('ats')}"
    sources.append(new_entry)
    save_registry(data)
    return new_entry
