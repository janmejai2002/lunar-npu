"""Unit and integration tests for Lunar Studio HUD and static Command Deck routes."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from lunar_core.studio import LunarStudioHandler


class ReusableThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def test_studio_static_and_api_routes():
    server = ReusableThreadingServer(("127.0.0.1", 0), LunarStudioHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Root / index.html
        req = urllib.request.Request(f"{base_url}/", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8")
            assert "LUNARNPU SOVEREIGN COMMAND DECK" in body
            assert "screen-perception-canvas" in body
            assert "geodesic-radar-canvas" in body

        # 2. style.css
        req = urllib.request.Request(f"{base_url}/style.css", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            assert "text/css" in resp.headers.get("Content-Type", "")
            css = resp.read().decode("utf-8")
            assert "--moss" in css
            assert "--bg-deck" in css

        # 3. app.js
        req = urllib.request.Request(f"{base_url}/app.js", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            assert "javascript" in resp.headers.get("Content-Type", "")
            js = resp.read().decode("utf-8")
            assert "engageSwarm" in js
            assert "captureScreenGlance" in js

        # 4. Health endpoint
        req = urllib.request.Request(f"{base_url}/api/health", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["npu"] is True

        # 5. Circuit Breaker audit: Safe command
        req = urllib.request.Request(f"{base_url}/api/audit?cmd=git+status", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["verdict"] == "ALLOWED"

        # 6. Circuit Breaker audit: Dangerous command
        req = urllib.request.Request(f"{base_url}/api/audit?cmd=rm+-rf+/", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["verdict"] == "BLOCKED"

        # 7. Screen perception
        req = urllib.request.Request(f"{base_url}/api/screen", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "elements" in data
            assert "image_width" in data
            assert "image_height" in data

        # 8. Geodesic Route
        req = urllib.request.Request(f"{base_url}/api/route?prompt=implement+ring+buffer", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["target_agent"] == "CODER"
            assert "geodesic_distances" in data

        # 9. Swarm: Cyclic mode
        req = urllib.request.Request(f"{base_url}/api/swarm?prompt=implement+quicksort&cyclic=true", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "task" in data
            assert "trajectory" in data
            assert "converged" in data

        # 10. Swarm: Standard mode
        req = urllib.request.Request(f"{base_url}/api/swarm?prompt=audit+security+token", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "lead_persona" in data
            assert data["lead_persona"] == "SECURITY_AUDITOR"

        # 11. Audio transcription
        req = urllib.request.Request(f"{base_url}/api/audio", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "text" in data
            assert "latency_ms" in data

    finally:
        server.shutdown()
        server.server_close()
