from __future__ import annotations

import http
import logging
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from vla_scratch.robomme_eval.codec import packb, unpackb

logger = logging.getLogger(__name__)


class RoboMMEHTTPServer(ThreadingHTTPServer):
    policy: Any
    metadata: Dict[str, Any]


class RoboMMEHTTPHandler(BaseHTTPRequestHandler):
    server_version = "VlaScratchRoboMMEHTTP/0.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send(http.HTTPStatus.OK, b"OK\n", "text/plain")
            return
        if self.path == "/metadata":
            self._send(http.HTTPStatus.OK, packb(self.server.metadata), "application/msgpack")
            return
        self._send(http.HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            inputs = unpackb(payload) if payload else {}
            if self.path == "/reset":
                self.server.policy.reset()
                self._send(http.HTTPStatus.OK, packb({"reset_finished": True}), "application/msgpack")
                return
            if self.path == "/infer":
                t0 = time.monotonic()
                outputs = self.server.policy.infer(inputs)
                outputs["server_timing"] = {"infer_s": time.monotonic() - t0}
                self._send(http.HTTPStatus.OK, packb(outputs), "application/msgpack")
                return
            self._send(http.HTTPStatus.NOT_FOUND, b"Not found\n", "text/plain")
        except Exception:  # noqa: BLE001
            body = traceback.format_exc().encode("utf-8", errors="replace")
            self._send(http.HTTPStatus.INTERNAL_SERVER_ERROR, body, "text/plain")

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def make_server(host: str, port: int, policy: Any, metadata: Dict[str, Any]) -> RoboMMEHTTPServer:
    server = RoboMMEHTTPServer((host, int(port)), RoboMMEHTTPHandler)
    server.policy = policy
    server.metadata = metadata
    return server

