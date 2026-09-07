from __future__ import annotations

from typing import Any, Callable

from backend.academic_api.mcp_policy import AcademicMcpToolPolicy
from backend.academic_mcp.client import AcademicApiClient


JOB_ID_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {
            "type": "string",
            "minLength": 1,
            "description": "Public evaluation job identifier.",
        }
    },
    "required": ["job_id"],
    "additionalProperties": False,
}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "submit_evaluation",
        "description": (
            "Submit a locally accessible PDF for asynchronous academic "
            "evaluation. Returns job_id, paper_id, status and schema_version."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "minLength": 1, "maxLength": 255},
                "pdf_path": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Path visible to the Academic API and allowed by its "
                        "configured import roots."
                    ),
                },
                "title": {"type": "string", "maxLength": 500},
                "source_url": {"type": "string", "maxLength": 1000},
                "model_name": {"type": "string", "minLength": 1, "maxLength": 255},
                "max_chunks": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["pdf_path"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Submit academic evaluation",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "get_evaluation_status",
        "description": (
            "Get compact job state, progress, retry count and public errors. "
            "The large evaluation result is omitted."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "get_evaluation_result",
        "description": (
            "Get the complete sanitized result for a completed job. Returns a "
            "conflict error while the result is unavailable."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "get_evaluation_evidence",
        "description": (
            "Get per-dimension evidence references, rationales, verification "
            "notes and concerns for a completed evaluation."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "get_evaluation_dimensions",
        "description": (
            "Get the nine dimension names, scores and statuses plus the overall "
            "score and evidence coverage for a completed evaluation."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "retry_evaluation",
        "description": (
            "Idempotently requeue a failed or cancelled evaluation. Repeated "
            "calls do not create duplicate queue work."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "cancel_evaluation",
        "description": (
            "Idempotently request cancellation of a queued or running "
            "evaluation and return its persisted state."
        ),
        "inputSchema": JOB_ID_SCHEMA,
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]


class AcademicMcpTools:
    def __init__(
        self,
        client: AcademicApiClient,
        policy: AcademicMcpToolPolicy | None = None,
    ) -> None:
        self.client = client
        self.policy = policy or AcademicMcpToolPolicy()
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "submit_evaluation": self._submit,
            "get_evaluation_status": self._status,
            "get_evaluation_result": self._result,
            "get_evaluation_evidence": self._evidence,
            "get_evaluation_dimensions": self._dimensions,
            "retry_evaluation": self._retry,
            "cancel_evaluation": self._cancel,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            definition
            for definition in TOOL_DEFINITIONS
            if definition["name"] in self.policy.allowed_tools
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.policy.authorize(name)
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError("Unknown Academic MCP tool.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        return handler(arguments)

    @staticmethod
    def _job_id(arguments: dict[str, Any]) -> str:
        job_id = arguments.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id is required.")
        return job_id.strip()

    def _submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paper_id = arguments.get("paper_id")
        pdf_path = arguments.get("pdf_path")
        if paper_id is not None and (
            not isinstance(paper_id, str) or not paper_id.strip()
        ):
            raise ValueError("paper_id must be a non-empty string when supplied.")
        if not isinstance(pdf_path, str) or not pdf_path.strip():
            raise ValueError("pdf_path is required.")
        return self.client.submit_evaluation(arguments)

    def _status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = dict(self.client.get_evaluation(self._job_id(arguments)))
        payload.pop("result", None)
        return payload

    def _result(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.get_result(self._job_id(arguments))

    def _evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.client.get_result(self._job_id(arguments))
        result = payload.get("result") or {}
        dimensions = result.get("dimensions") or []
        return {
            "schema_version": payload.get("schema_version"),
            "job_id": payload.get("job_id"),
            "paper_id": payload.get("paper_id"),
            "external_evidence": result.get("external_evidence"),
            "evidence": [
                {
                    "dimension": item.get("dimension"),
                    "rationale": item.get("rationale"),
                    "evidence_chunk_ids": item.get("evidence_chunk_ids") or [],
                    "external_evidence_ids": (
                        item.get("external_evidence_ids") or []
                    ),
                    "verification": item.get("verification"),
                    "concerns": item.get("concerns") or [],
                }
                for item in dimensions
                if isinstance(item, dict)
            ],
        }

    def _dimensions(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.client.get_result(self._job_id(arguments))
        result = payload.get("result") or {}
        dimensions = result.get("dimensions") or []
        return {
            "schema_version": payload.get("schema_version"),
            "job_id": payload.get("job_id"),
            "paper_id": payload.get("paper_id"),
            "overall_score": result.get("overall_score"),
            "assessment": result.get("assessment"),
            "evidence_coverage": result.get("evidence_coverage"),
            "missing_dimensions": result.get("missing_dimensions") or [],
            "dimensions": [
                {
                    "dimension": item.get("dimension"),
                    "score": item.get("score"),
                    "status": item.get("status"),
                }
                for item in dimensions
                if isinstance(item, dict)
            ],
        }

    def _retry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.retry_evaluation(self._job_id(arguments))

    def _cancel(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.client.cancel_evaluation(self._job_id(arguments))
