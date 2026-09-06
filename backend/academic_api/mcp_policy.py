from __future__ import annotations


DEFAULT_ACADEMIC_MCP_TOOLS = frozenset(
    {
        "submit_evaluation",
        "get_evaluation_status",
        "get_evaluation_result",
        "get_evaluation_evidence",
        "get_evaluation_dimensions",
        "retry_evaluation",
        "cancel_evaluation",
    }
)


class McpToolDeniedError(PermissionError):
    pass


class AcademicMcpToolPolicy:
    """Default-deny authorization boundary for the future Academic MCP server."""

    def __init__(self, allowed_tools=frozenset(DEFAULT_ACADEMIC_MCP_TOOLS)):
        requested = frozenset(allowed_tools)
        unknown = requested - DEFAULT_ACADEMIC_MCP_TOOLS
        if unknown:
            raise ValueError(
                "Unknown Academic MCP tools cannot be allowlisted: "
                f"{sorted(unknown)}"
            )
        self.allowed_tools = requested

    def authorize(self, tool_name: str) -> None:
        normalized = tool_name.strip()
        if not normalized or normalized not in self.allowed_tools:
            raise McpToolDeniedError(
                "Academic MCP tool is not allowed by policy."
            )
