"""
Local server for the job dashboard — serves the static dashboard files and
handles Analyze / keep / dismiss requests from the page.

On startup this builds jobs_dashboard.html once (reusing the last cached
title search if any, otherwise the empty state) so the dashboard always has
something to serve — no need to run build_dashboard.py manually first.

Usage:
  python feedback_server.py
  open http://localhost:8765/jobs_dashboard.html

Endpoints:
  POST /api/analyze    {"titles": "...", "ideal_role": "..."}  -> build_dashboard.build(titles, ideal_role)
  POST /api/show_all    (no body)                               -> build_dashboard.build_show_all()
  POST /api/keep         {"job_id": "<url>"}                    -> record_keep
  POST /api/dismiss      {"job_id": "<url>", "reason": "..."}    -> record_dismissal
                        (reason is optional but preferred — see feedback_scoring.record_dismissal)
  POST /api/reset        (no body)                               -> reset_feedback + rebuild dashboard
"""

import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np

import job_fit_finder as jf
import feedback_scoring as fs
import build_dashboard

PORT = 8765
EMBEDDING_CACHE_PATH = "job_embeddings_cache.json"


def load_embedding_cache():
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        return {}
    with open(EMBEDDING_CACHE_PATH, "r") as f:
        return json.load(f)


class Handler(SimpleHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_label(self, action):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        job_id = body.get("job_id")
        if not job_id:
            self._send_json(400, {"error": "missing job_id"})
            return

        cache = load_embedding_cache()
        entry = cache.get(job_id)
        if not entry:
            self._send_json(404, {"error": f"no cached embedding for job_id {job_id!r} — rerun build_dashboard.py"})
            return

        embedding = np.array(entry["embedding"])
        if action == "keep":
            feedback = fs.record_keep(job_id, embedding)
        else:
            reason = (body.get("reason") or "").strip()
            reason_embedding = jf.embed(reason) if reason else None
            feedback = fs.record_dismissal(job_id, embedding, reason=reason or None, reason_embedding=reason_embedding)

        self._send_json(200, {
            "status": "ok",
            "kept": len(feedback["kept"]),
            "dismissed": len(feedback["dismissed"]),
        })

    def _handle_reset(self):
        fs.reset_feedback()
        try:
            build_dashboard.main()
        except Exception as e:
            self._send_json(500, {"error": f"feedback cleared but rebuild failed: {e}"})
            return
        self._send_json(200, {"status": "ok", "kept": 0, "dismissed": 0})

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
        if self.path == "/api/keep":
            self._handle_label("keep")
        elif self.path == "/api/dismiss":
            self._handle_label("dismiss")
        elif self.path == "/api/reset":
            self._handle_reset()
        elif self.path == "/api/analyze":
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
