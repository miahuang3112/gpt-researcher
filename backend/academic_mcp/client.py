from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class AcademicApiError(RuntimeError):
    """Safe error returned to an MCP client without leaking transport details."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AcademicApiClient:
    base_url: str = "http://127.0.0.1:8001/api/v1"
    api_key: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "AcademicApiClient":
        return cls(
            base_url=os.getenv(
                "ACADEMIC_API_BASE_URL",
                "http://127.0.0.1:8001/api/v1",
            ).rstrip("/"),
            api_key=os.getenv("ACADEMIC_API_KEY") or None,
            timeout_seconds=float(os.getenv("ACADEMIC_MCP_TIMEOUT_SECONDS", "30")),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            detail = f"Academic API request failed ({exc.code})."
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
                    detail = parsed["detail"]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            raise AcademicApiError(detail, status_code=exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AcademicApiError("Academic API is unavailable.") from exc

        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcademicApiError("Academic API returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise AcademicApiError("Academic API returned an invalid response shape.")
        return parsed

    def submit_evaluation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = {
            key: arguments[key]
            for key in (
                "paper_id",
                "pdf_path",
                "title",
                "source_url",
                "model_name",
                "max_chunks",
            )
            if key in arguments and arguments[key] is not None
        }
        return self._request("POST", "/academic/evaluations", payload)

    def get_evaluation(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/academic/evaluations/{quote(job_id, safe='')}",
        )

    def get_result(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/academic/evaluations/{quote(job_id, safe='')}/result",
        )

    def retry_evaluation(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/academic/evaluations/{quote(job_id, safe='')}/retry",
        )

    def cancel_evaluation(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/academic/evaluations/{quote(job_id, safe='')}/cancel",
        )
