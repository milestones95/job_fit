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
  POST /api/analyze    {"titles": "...", "ideal_role": "..."}  -> build_dashboard.build(titles, ideal_role)
  POST /api/show_all   (no body)                                -> build_dashboard.build_show_all()
"""

import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import build_dashboard

PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
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

    def do_POST(self):
        if self.path == "/api/analyze":
            self._handle_analyze()
        elif self.path == "/api/show_all":
            self._handle_show_all()
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
