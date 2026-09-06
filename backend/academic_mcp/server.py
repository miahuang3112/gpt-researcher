from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from backend.academic_api.mcp_policy import McpToolDeniedError
from backend.academic_mcp.client import AcademicApiClient, AcademicApiError
from backend.academic_mcp.tools import AcademicMcpTools


JSONRPC_VERSION = "2.0"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2024-11-05")
SERVER_INFO = {"name": "academic-evaluation", "version": "1.0.0"}


@dataclass
class AcademicMcpServer:
    tools: AcademicMcpTools

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        if request.get("jsonrpc") != JSONRPC_VERSION or not isinstance(
            request.get("method"), str
        ):
            return self._error(request_id, -32600, "Invalid Request")

        method = request["method"]
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            params = request.get("params") or {}
            requested = params.get("protocolVersion")
            protocol = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else SUPPORTED_PROTOCOL_VERSIONS[0]
            )
            return self._result(
                request_id,
                {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": (
                        "Use status while a job is active. Read result, evidence "
                        "or dimensions only after completion. Treat every score "
                        "as automated assessment, not peer review."
                    ),
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": self.tools.list_tools()})
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str):
                return self._error(request_id, -32602, "Tool name is required.")
            try:
                output = self.tools.call(name, arguments)
            except (AcademicApiError, McpToolDeniedError, ValueError) as exc:
                return self._result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )
            return self._result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(output, ensure_ascii=False),
                        }
                    ],
                    "structuredContent": output,
                    "isError": False,
                },
            )
        return self._error(request_id, -32601, "Method not found")

    @staticmethod
    def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _error(
        request_id: Any,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def serve_stdio(
    server: AcademicMcpServer,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError
            response = server.handle(request)
        except (json.JSONDecodeError, ValueError):
            response = AcademicMcpServer._error(None, -32700, "Parse error")
        if response is not None:
            output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            output_stream.flush()


def main() -> None:
    client = AcademicApiClient.from_environment()
    serve_stdio(AcademicMcpServer(AcademicMcpTools(client)))


if __name__ == "__main__":
    main()
