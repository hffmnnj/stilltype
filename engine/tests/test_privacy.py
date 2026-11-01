"""Exercise the actual Ollama HTTP adapter without any external service."""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


class LoopbackPrivacy(unittest.TestCase):
    def test_cleanup_ignores_proxies_and_does_not_follow_redirects(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                requests.append((self.path, json.loads(body)))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"response":"Local reply."}')

            def do_GET(self):
                requests.append((self.path, None))
                if self.path == "/api/tags":
                    self.send_response(302)
                    self.send_header("Location", "/unexpected-redirect")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"models":[{"name":"qwen3.5:4b"}]}')

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                with patch.object(inference, "OLLAMA_URL", origin), patch.dict(
                    os.environ,
                    {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1",
                     "no_proxy": "", "NO_PROXY": ""},
                ):
                    self.assertEqual(
                        inference._ollama_generate("Keep these words private.", "qwen3.5:4b"),
                        "Local reply.",
                    )
                    self.assertFalse(inference._ollama_available("qwen3.5:4b"))
            finally:
                server.shutdown()
                thread.join(timeout=2)

        self.assertEqual([path for path, _ in requests], ["/api/generate", "/api/tags"])
        self.assertEqual(requests[0][1]["prompt"], "Keep these words private.")


if __name__ == "__main__":
    unittest.main()
