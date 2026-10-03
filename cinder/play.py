"""Local page for the annealing shed.

The page is the environment: a tablet, a question, a reply, a reward.
The batch number is revealed only after the reply is scored.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cinder.script import VIEW_SIZE
from cinder.sim import CinderSim

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "static" / "index.html"
WEIGHTS = ROOT / "weights" / "policy.pt"
METRICS = ROOT / "weights" / "metrics.json"

_sim = CinderSim()
_lock = threading.Lock()
_model = None
_model_error = ""


def _public(outcome, extra: dict | None = None) -> dict:
    body = {
        "prompt": outcome.prompt,
        "task": outcome.task,
        "image": outcome.image_base64,
        "legal": list(outcome.legal_actions),
        "done": outcome.done,
        "reward": outcome.reward,
    }
    if extra:
        body.update(extra)
    return body


def _ensure_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if not WEIGHTS.is_file():
        _model_error = "Train the policy first: python -m cinder.grpo"
        return None
    import torch

    from cinder.policy import load_policy

    torch.set_num_threads(4)
    _model, _ = load_policy(WEIGHTS, torch.device("cpu"))
    return _model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/metrics":
            if METRICS.is_file():
                self._send(200, json.loads(METRICS.read_text(encoding="utf-8")))
            else:
                self._send(200, {})
            return
        if path not in ("/", "/index.html"):
            self.send_error(404)
            return
        raw = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read()
        except json.JSONDecodeError:
            self._send(400, {"error": "Expected JSON."})
            return
        if path == "/api/reset":
            task = body.get("task", "batch")
            with _lock:
                try:
                    outcome = _sim.reset(task=task, size=VIEW_SIZE)
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, _public(outcome))
            return
        if path == "/api/step":
            with _lock:
                try:
                    outcome = _sim.step(str(body.get("text", "")))
                except RuntimeError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, _public(outcome, {"reading": _sim.answer}))
            return
        if path == "/api/model":
            model = _ensure_model()
            if model is None:
                self._send(503, {"error": _model_error})
                return
            from cinder.grpo import greedy_reply

            with _lock:
                if _sim.task == "pair":
                    self._send(400, {"error": "The policy reads one tablet. Pair is for you."})
                    return
                if _sim.tablet is None:
                    self._send(400, {"error": "Call reset() first."})
                    return
                tablet = _sim.tablet
                task = _sim.task
                outcome = _sim.reset(tablet=tablet, task=task, size=VIEW_SIZE)
                text = greedy_reply(
                    model,
                    _sim.model_image(),
                    outcome.prompt,
                    task,
                    next(model.parameters()).device,
                )
                outcome = _sim.step(text)
                self._send(200, _public(outcome, {"text": text, "reading": _sim.answer}))
            return
        self.send_error(404)


def main(host: str = "127.0.0.1", port: int = 8091) -> None:
    _sim.reset(task="batch", size=VIEW_SIZE)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Cinder is at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
