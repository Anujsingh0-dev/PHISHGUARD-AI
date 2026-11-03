"""
api/index.py
────────────
Vercel Serverless Function Handler for PHISHGUARD AI.
Handles API requests for URL predictions and metrics.
Uses standard Python http.server for 100% native Vercel serverless compatibility.
"""

import sys
import json
import pathlib
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add root directory to sys.path so src/ imports work
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_url
from src.utils import load_metrics


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        response_body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.endswith("/api/predict"):
            url_param = query.get("url", [""])[0]
            if not url_param:
                self._send_json(400, {"error": "Missing 'url' query parameter"})
                return
            result = predict_url(url_param)
            self._send_json(200, result)

        elif path.endswith("/api/metrics"):
            try:
                full_m = load_metrics("full_model_metrics.json")
                url_m = load_metrics("url_model_metrics.json")
                self._send_json(200, {"full_model": full_m, "url_model": url_m})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif path.endswith("/api/health"):
            self._send_json(200, {"status": "ok", "app": "PHISHGUARD AI"})

        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.endswith("/api/predict"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
                target_url = data.get("url", "")
                if not target_url:
                    self._send_json(400, {"error": "Missing 'url' in JSON body"})
                    return
                result = predict_url(target_url)
                self._send_json(200, result)
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
        else:
            self._send_json(404, {"error": "Endpoint not found"})
