import pytest

from backend.academic_api.mcp_policy import (
    AcademicMcpToolPolicy,
    McpToolDeniedError,
)


def test_default_allowlist_contains_supported_mcp_tools() -> None:
    policy = AcademicMcpToolPolicy()

    policy.authorize("submit_evaluation")
    policy.authorize("get_evaluation_result")
    policy.authorize("retry_evaluation")
    policy.authorize("cancel_evaluation")

    with pytest.raises(McpToolDeniedError):
        policy.authorize("delete_all_papers")


def test_policy_rejects_unknown_tools_during_configuration() -> None:
    with pytest.raises(ValueError, match="Unknown Academic MCP tools"):
        AcademicMcpToolPolicy({"submit_evaluation", "shell"})


def test_policy_can_reduce_but_not_expand_the_allowlist() -> None:
    policy = AcademicMcpToolPolicy({"get_evaluation_status"})
    policy.authorize("get_evaluation_status")

    with pytest.raises(McpToolDeniedError):
        policy.authorize("submit_evaluation")
