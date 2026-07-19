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
"""

import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import build_dashboard
import job_fit_finder as jf

PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def _cors_headers(self):
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
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

    def do_POST(self):
        if self.path == "/api/analyze":
            self._handle_analyze()
        elif self.path == "/api/show_all":
            self._handle_show_all()
        elif self.path == "/api/extension/analyze":
            self._handle_extension_analyze()
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
