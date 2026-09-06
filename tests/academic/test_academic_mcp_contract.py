from __future__ import annotations

import io
import json
from pathlib import Path

from backend.academic_api.mcp_policy import AcademicMcpToolPolicy
from backend.academic_mcp.server import AcademicMcpServer, serve_stdio
from backend.academic_mcp.tools import AcademicMcpTools


class StubClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def submit_evaluation(self, arguments):
        self.calls.append(("submit", arguments))
        return {"schema_version": "academic-evaluation.v1", "job_id": "job-1", "status": "queued"}

    def get_evaluation(self, job_id):
        self.calls.append(("status", job_id))
        return {
            "schema_version": "academic-evaluation.v1",
            "job_id": job_id,
            "status": "completed",
            "progress": 1.0,
            "result": {"must_not": "appear in status"},
        }

    def get_result(self, job_id):
        self.calls.append(("result", job_id))
        return {
            "schema_version": "academic-evaluation.v1",
            "job_id": job_id,
            "paper_id": "paper-1",
            "status": "completed",
            "result": {
                "overall_score": 82,
                "assessment": "strong",
                "evidence_coverage": 0.75,
                "missing_dimensions": [],
                "external_evidence": {
                    "status": "available",
                    "provider_used": "openalex",
                    "records": [],
                },
                "dimensions": [
                    {
                        "dimension": "methodological_rigor",
                        "score": 4,
                        "status": "scored",
                        "rationale": "Supported by the method section.",
                        "evidence_chunk_ids": ["paper-1-p2-c3"],
                        "external_evidence_ids": ["external:prior"],
                        "verification": {"status": "verified", "reasons": []},
                        "concerns": [],
                    }
                ],
            },
        }

    def retry_evaluation(self, job_id):
        self.calls.append(("retry", job_id))
        return {"job_id": job_id, "status": "queued", "retry_count": 1}

    def cancel_evaluation(self, job_id):
        self.calls.append(("cancel", job_id))
        return {"job_id": job_id, "status": "cancel_requested"}


def server(policy=None):
    return AcademicMcpServer(AcademicMcpTools(StubClient(), policy=policy))


def call(request_id, name, arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


def test_initialize_and_tool_contract() -> None:
    instance = server()
    initialized = instance.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
    )
    assert initialized["result"]["protocolVersion"] == "2024-11-05"
    assert initialized["result"]["capabilities"] == {"tools": {"listChanged": False}}

    listed = instance.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    tools = listed["result"]["tools"]
    assert {tool["name"] for tool in tools} == {
        "submit_evaluation",
        "get_evaluation_status",
        "get_evaluation_result",
        "get_evaluation_evidence",
        "get_evaluation_dimensions",
        "retry_evaluation",
        "cancel_evaluation",
    }
    read_only = {
        tool["name"]
        for tool in tools
        if tool.get("annotations", {}).get("readOnlyHint")
    }
    assert read_only == {
        "get_evaluation_status",
        "get_evaluation_result",
        "get_evaluation_evidence",
        "get_evaluation_dimensions",
    }
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in tools)


def test_status_result_evidence_and_dimension_shapes() -> None:
    instance = server()
    status = instance.handle(call(1, "get_evaluation_status", {"job_id": "job-1"}))
    assert status["result"]["structuredContent"]["status"] == "completed"
    assert "result" not in status["result"]["structuredContent"]

    result = instance.handle(call(2, "get_evaluation_result", {"job_id": "job-1"}))
    assert result["result"]["structuredContent"]["result"]["overall_score"] == 82

    evidence = instance.handle(call(3, "get_evaluation_evidence", {"job_id": "job-1"}))
    evidence_item = evidence["result"]["structuredContent"]["evidence"][0]
    assert evidence_item["dimension"] == "methodological_rigor"
    assert evidence_item["evidence_chunk_ids"] == ["paper-1-p2-c3"]
    assert evidence_item["external_evidence_ids"] == ["external:prior"]
    assert evidence["result"]["structuredContent"]["external_evidence"][
        "provider_used"
    ] == "openalex"

    dimensions = instance.handle(call(4, "get_evaluation_dimensions", {"job_id": "job-1"}))
    payload = dimensions["result"]["structuredContent"]
    assert payload["overall_score"] == 82
    assert payload["dimensions"] == [
        {"dimension": "methodological_rigor", "score": 4, "status": "scored"}
    ]


def test_submit_retry_cancel_and_default_deny() -> None:
    instance = server()
    submitted = instance.handle(
        call(1, "submit_evaluation", {"paper_id": "paper-1", "pdf_path": "/data/paper.pdf"})
    )
    assert submitted["result"]["structuredContent"]["status"] == "queued"
    assert instance.handle(call(2, "retry_evaluation", {"job_id": "job-1"}))["result"]["isError"] is False
    assert instance.handle(call(3, "cancel_evaluation", {"job_id": "job-1"}))["result"]["isError"] is False

    restricted = server(AcademicMcpToolPolicy({"get_evaluation_status"}))
    denied = restricted.handle(call(4, "cancel_evaluation", {"job_id": "job-1"}))
    assert denied["result"]["isError"] is True
    assert "not allowed" in denied["result"]["content"][0]["text"]


def test_stdio_transport_handles_notifications_and_parse_errors() -> None:
    requests = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            "not-json",
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
        ]
    )
    output = io.StringIO()
    serve_stdio(server(), io.StringIO(requests), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32700
    assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_project_mcp_configuration_launches_academic_server() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = json.loads((repository / ".mcp.json").read_text())
    academic = config["mcpServers"]["academic-evaluation"]
    assert academic["command"] == "python3"
    assert academic["args"] == ["-m", "backend.academic_mcp"]
    assert academic["env"]["ACADEMIC_API_BASE_URL"].endswith("/api/v1")
