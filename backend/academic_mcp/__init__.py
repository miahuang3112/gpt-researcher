"""Model Context Protocol adapter for the Academic Evaluation API."""

from backend.academic_mcp.client import AcademicApiClient, AcademicApiError

__all__ = ["AcademicApiClient", "AcademicApiError"]
