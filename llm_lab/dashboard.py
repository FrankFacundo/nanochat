"""Dependency-free local dashboard for experiment comparison."""

from __future__ import annotations

import json
import mimetypes
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from llm_lab.learning import LearningError, evaluate_submission, learning_snapshot
from llm_lab.runner import default_runs_dir, list_runs, resolve_run


STATIC_DIR = Path(__file__).with_name("static")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), LabRequestHandler)
    url = f"http://{host}:{port}"
    print(f"Nanochat Mastery Lab: {url}")
    print(f"Reading runs from: {default_runs_dir()}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


class LabRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path == "/api/learning":
            self._json(learning_snapshot())
            return
        if parsed.path == "/api/learning/export":
            payload = json.dumps(learning_snapshot(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="nanochat-mastery-progress.json"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/runs":
            self._json(list_runs())
            return
        if parsed.path == "/api/metrics":
            query = parse_qs(parsed.query)
            run_id = query.get("run", [""])[0]
            try:
                run_dir = resolve_run(run_id)
                metrics_path = run_dir / "metrics.jsonl"
                metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
                self._json(metrics)
            except (OSError, ValueError, RuntimeError) as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path in {"/", "/index.html"}:
            self._file(STATIC_DIR / "index.html")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path != "/api/evaluate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise LearningError("Submission body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise LearningError("Submission must be a JSON object")
            self._json(evaluate_submission(payload))
        except (json.JSONDecodeError, LearningError, OSError, ValueError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _file(self, path: Path) -> None:
        payload = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
