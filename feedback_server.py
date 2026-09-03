"""
Local server for the job dashboard — serves the static dashboard files and
handles Analyze / Show All requests from the page.

On startup this builds jobs_dashboard.html once (reusing the last cached
title search if any, otherwise the empty state) so the dashboard always has
something to serve — no need to run build_dashboard.py manually first.

Usage:
  python feedback_server.py
  open http://localhost:8765/jobs_dashboard.html

Endpoints:
  POST /api/analyze              {"titles": "...", "ideal_role": "..."}  -> build_dashboard.build(titles, ideal_role)
  POST /api/show_all             (no body)                                -> build_dashboard.build_show_all()
  POST /api/extension/analyze    {"ats": "ashby"|"greenhouse", "company_token": "...",
                                   "company_name": "...", "titles": "...", "ideal_role": "..."}
                                  -> ranked jobs (JSON) for one company on one ATS, for the
                                     Chrome extension popup (see extension/). CORS-enabled
                                     since the caller is a chrome-extension:// origin.
  POST /api/extension/register_source  {"url": "https://jobs.ashbyhq.com/...", "page_html": "..."}
                                  -> detect ATS from the URL, verify the board live, and
                                     register it in sources.json on pass ({"status":
                                     "registered"|"rejected"|"research_pending"}). Unknown
                                     boards are the adapter agent's lane (follow-on PR);
                                     page_html is accepted and unused until then.
  GET  /api/extension/list_sources -> current registry summary for the popup.
"""

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import adapter_researcher as ar
import build_dashboard
import job_fit_finder as jf
import source_registry as sr

# JOB_FIT_PORT lets smoke tests (and parallel installs) avoid clashing with
# a server already on the default port.
PORT = int(os.environ.get("JOB_FIT_PORT", "8765"))


class Handler(SimpleHTTPRequestHandler):
    def _cors_headers(self):
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _handle_analyze(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        titles = (body.get("titles") or "").strip()
        ideal_role = (body.get("ideal_role") or "").strip()
        if not titles:
            self._send_json(400, {"error": "missing titles"})
            return
        if not ideal_role:
            self._send_json(400, {"error": "missing ideal_role"})
            return

        try:
            ranked = build_dashboard.build(user_titles_raw=titles, ideal_role_text=ideal_role)
        except Exception as e:
            self._send_json(500, {"error": f"analyze failed: {e}"})
            return
        self._send_json(200, {"status": "ok", "count": len(ranked)})

    def _handle_show_all(self):
        try:
            jobs = build_dashboard.build_show_all()
        except Exception as e:
            self._send_json(500, {"error": f"show_all failed: {e}"})
            return
        self._send_json(200, {"status": "ok", "count": len(jobs)})

    def _handle_extension_analyze(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        company_token = (body.get("company_token") or "").strip()
        company_name = (body.get("company_name") or company_token).strip()
        ats = (body.get("ats") or "").strip().lower()
        titles = (body.get("titles") or "").strip()
        ideal_role = (body.get("ideal_role") or "").strip()
        if not company_token:
            self._send_json(400, {"error": "missing company_token"})
            return
        if ats not in jf.FETCHERS:
            self._send_json(400, {"error": f"unsupported ats '{ats}'"})
            return
        if not titles:
            self._send_json(400, {"error": "missing titles"})
            return
        if not ideal_role:
            self._send_json(400, {"error": "missing ideal_role"})
            return

        try:
            _, keywords = jf.set_target_title_keywords(titles, ideal_role)
        except Exception as e:
            self._send_json(500, {"error": f"title expansion failed: {e}"})
            return

        try:
            jobs = jf.FETCHERS[ats](company_name, company_token)
        except Exception as e:
            self._send_json(502, {"error": f"could not fetch postings for '{company_token}': {e}"})
            return

        matched = [j for j in jobs if jf.title_matches(j["title"], keywords)]
        if not matched:
            self._send_json(200, {"status": "ok", "count": 0, "jobs": []})
            return

        if ats == "greenhouse":
            try:
                jf.enrich_greenhouse_compensation(matched, company_token)
            except Exception as e:
                print(f"[extension_analyze] compensation enrichment failed, continuing without it: {e}")

        try:
            ranked = jf.rank_jobs_by_llm(matched, ideal_role_text=ideal_role)
        except Exception as e:
            self._send_json(500, {"error": f"scoring failed: {e}"})
            return

        self._send_json(200, {"status": "ok", "count": len(ranked), "jobs": ranked})

    def _handle_extension_register_source(self):
        """Add-source flow, both lanes (spec §3–§4). Known boards: detect,
        verify, register. Unknown boards: the adapter agent — research the
        platform, generate a snippet, sandbox-test it, and return the
        research provenance + preview + a token. Nothing persists until the
        popup posts the token back with confirmed=true (the confirm
        round-trip), and only if the sandbox test passed."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        url = (body.get("url") or "").strip()
        if not url:
            self._send_json(400, {"error": "missing url"})
            return

        try:
            if body.get("confirmed"):
                result = ar.confirm_registration(body.get("token") or "")
            else:
                known = sr.register_known_source(url)
                if known.get("status") != "research_pending":
                    result = known
                else:
                    result = ar.start_registration(
                        url,
                        page_html=body.get("page_html") or "",
                        hints=body.get("hints") or {},
                    )
        except Exception as e:
            self._send_json(500, {"error": f"registration failed: {e}"})
            return
        self._send_json(200, result)

    def _handle_extension_list_sources(self):
        """Registry summary for the popup (spec §4)."""
        try:
            registry = sr.load_registry()
        except Exception as e:
            self._send_json(500, {"error": f"could not load sources.json: {e}"})
            return
        sources = [
            {
                "id": entry.get("id"),
                "company": entry.get("company"),
                "ats": entry.get("ats"),
                "endpoint": entry.get("endpoint"),
                "verification": entry.get("verification"),
            }
            for entry in registry.get("sources", [])
        ]
        self._send_json(200, {"status": "ok", "count": len(sources), "sources": sources})

    def do_GET(self):
        if self.path == "/api/extension/list_sources":
            self._handle_extension_list_sources()
        else:
            super().do_GET()  # static dashboard files

    def do_POST(self):
        if self.path == "/api/analyze":
            self._handle_analyze()
        elif self.path == "/api/show_all":
            self._handle_show_all()
        elif self.path == "/api/extension/analyze":
            self._handle_extension_analyze()
        elif self.path == "/api/extension/register_source":
            self._handle_extension_register_source()
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        if self.command == "POST":
            super().log_message(fmt, *args)
        # silence the noisy GET/static-file logging


if __name__ == "__main__":
    print("Preparing dashboard...")
    build_dashboard.build()

    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"\nServing dashboard + feedback API on http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}/jobs_dashboard.html")
    server.serve_forever()
