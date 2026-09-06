#!/usr/bin/env python3
"""Run a dependency-free MCP demo against an isolated fake Academic API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


JOB_ID = "fresh-demo-job"


class DemoApi(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith(f"/{JOB_ID}/result"):
            self._send(
                {
                    "schema_version": "academic-evaluation.v1",
                    "job_id": JOB_ID,
                    "paper_id": "fresh:paper",
                    "status": "completed",
                    "completed_at": "2026-08-26T10:00:00",
                    "result": {
                        "overall_score": 84,
                        "assessment": "strong",
                        "evidence_coverage": 1.0,
                        "missing_dimensions": [],
                        "dimensions": [
                            {
                                "dimension": "methodological_rigor",
                                "score": 4,
                                "status": "scored",
                                "rationale": "Evidence-backed demo rationale.",
                                "evidence_chunk_ids": ["fresh-p1-c1"],
                                "verification": {"status": "verified", "reasons": []},
                                "concerns": [],
                            }
                        ],
                    },
                }
            )
            return
        if self.path.endswith(f"/{JOB_ID}"):
            self._send(
                {
                    "schema_version": "academic-evaluation.v1",
                    "job_id": JOB_ID,
                    "paper_id": "fresh:paper",
                    "status": "completed",
                    "progress": 1.0,
                    "result": {"large": "omitted by status tool"},
                }
            )
            return
        self._send({"detail": "Not found."}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.endswith("/academic/evaluations"):
            self._send(
                {
                    "schema_version": "academic-evaluation.v1",
                    "job_id": JOB_ID,
                    "paper_id": "fresh:paper",
                    "status": "queued",
                },
                201,
            )
        elif self.path.endswith(f"/{JOB_ID}/retry"):
            self._send({"job_id": JOB_ID, "status": "queued", "retry_count": 1})
        elif self.path.endswith(f"/{JOB_ID}/cancel"):
            self._send({"job_id": JOB_ID, "status": "cancel_requested"})
        else:
            self._send({"detail": "Not found."}, 404)


def request(process, request_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def tool_call(process, request_id, name, arguments):
    response = request(
        process,
        request_id,
        "tools/call",
        {"name": name, "arguments": arguments},
    )
    result = response["result"]
    assert result["isError"] is False, result
    return result["structuredContent"]


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    api = ThreadingHTTPServer(("127.0.0.1", 0), DemoApi)
    thread = threading.Thread(target=api.serve_forever, daemon=True)
    thread.start()
    environment = os.environ.copy()
    environment["ACADEMIC_API_BASE_URL"] = (
        f"http://127.0.0.1:{api.server_port}/api/v1"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.academic_mcp"],
        cwd=repository,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        initialized = request(
            process,
            1,
            "initialize",
            {"protocolVersion": "2025-06-18", "clientInfo": {"name": "fresh-demo", "version": "1"}},
        )
        listed = request(process, 2, "tools/list", {})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert len(names) == 7
        submitted = tool_call(
            process,
            3,
            "submit_evaluation",
            {"paper_id": "fresh:paper", "pdf_path": "/allowed/fresh.pdf"},
        )
        status = tool_call(process, 4, "get_evaluation_status", {"job_id": JOB_ID})
        dimensions = tool_call(process, 5, "get_evaluation_dimensions", {"job_id": JOB_ID})
        evidence = tool_call(process, 6, "get_evaluation_evidence", {"job_id": JOB_ID})
        retried = tool_call(process, 7, "retry_evaluation", {"job_id": JOB_ID})
        cancelled = tool_call(process, 8, "cancel_evaluation", {"job_id": JOB_ID})
        assert initialized["result"]["serverInfo"]["name"] == "academic-evaluation"
        assert submitted["status"] == "queued"
        assert status["status"] == "completed" and "result" not in status
        assert dimensions["overall_score"] == 84
        assert evidence["evidence"][0]["evidence_chunk_ids"] == ["fresh-p1-c1"]
        assert retried["retry_count"] == 1
        assert cancelled["status"] == "cancel_requested"
        print(
            json.dumps(
                {
                    "demo": "academic-mcp-fresh-process",
                    "protocol": initialized["result"]["protocolVersion"],
                    "tools": sorted(names),
                    "job_id": JOB_ID,
                    "overall_score": dimensions["overall_score"],
                    "evidence_refs": len(evidence["evidence"][0]["evidence_chunk_ids"]),
                    "result": "passed",
                },
                indent=2,
            )
        )
        return 0
    finally:
        process.terminate()
        process.wait(timeout=5)
        api.shutdown()
        api.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
