---
name: local-dev
---

# Local dev environment — job_fit

Durable record of the LOCAL-DEV onboarding run on 2026-09-03. Sandbox snapshot
`9m3n77vv9qqyg1uvtt3f:default` already contains this environment with the dev
server running on port 8765 — resume it instead of rebuilding when possible.

## What's installed

- Python 3.13.14 (system), pip 26.2.1
- `requests` 2.33.0, `openai` 3.7.0
  (`pip install openai requests --break-system-packages`)
- Playwright + Chromium headless shell (browser-automation evidence)
- ruff (informational static checks; the repo has no lint config)

## Setup from a cold sandbox

1. `pip install openai requests --break-system-packages`
2. Create `.env` in the repo root with `OPENAI_API_KEY=...` — any non-empty
   value lets the server start (the key is read at import time); a real key is
   only needed for the LLM flows (`/api/analyze`, `job_fit_finder.py` scoring).
3. `python feedback_server.py` — builds the dashboard once, then serves
   http://localhost:8765/jobs_dashboard.html

## Verified flows (2026-09-03)

- `GET /jobs_dashboard.html` → 200
- `POST /api/show_all` → 200, 266 postings from EliseAI, Browserbase,
  Tamarind Bio, Decagon (public ATS APIs, no auth, no LLM)
- Browser: cards render, search + department filters and reset work, zero
  console errors / page errors
- `POST /api/analyze` → 400 on missing `titles` / missing `ideal_role`; with a
  dummy key → 200 but every score is 0% with an inline 401 note (graceful by
  design — `score_job` never raises)
- `python -m compileall` clean; no test suite exists in the repo

## Gotchas

- `jobs_dashboard.html` is a tracked, generated file — every analyze /
  show_all / build rewrites it. Run
  `git checkout -- jobs_dashboard.html` before committing unless you intend to
  commit regenerated output.
- Caches `target_titles_cache.json` / `job_score_cache.json` are gitignored;
  delete the former to re-enter search titles.
- The server binds `localhost:8765` (`PORT` in `feedback_server.py`).
- No requirements file exists — deps are documented only in the
  `job_fit_finder.py` docstring.
