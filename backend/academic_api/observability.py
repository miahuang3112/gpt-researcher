from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from backend.academic_api.config import settings


correlation_id_context: ContextVar[str | None] = ContextVar(
    "academic_correlation_id",
    default=None,
)


class AcademicJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": getattr(record, "event_name", record.getMessage()),
        }
        correlation_id = correlation_id_context.get()
        if correlation_id:
            payload["correlation_id"] = correlation_id

        payload.update(getattr(record, "structured_fields", {}))

        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("academic")
    if not any(getattr(handler, "academic_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AcademicJsonFormatter())
        handler.academic_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"academic.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields,
) -> None:
    logger.log(
        level,
        event,
        extra={
            "event_name": event,
            "structured_fields": fields,
        },
    )


def normalize_correlation_id(raw_value: str | None) -> str:
    if raw_value:
        normalized = "".join(
            character
            for character in raw_value.strip()
            if character.isalnum() or character in "-_.:"
        )[:128]
        if normalized:
            return normalized
    return uuid4().hex


def get_correlation_id() -> str | None:
    return correlation_id_context.get()


def set_correlation_id(value: str) -> Token:
    return correlation_id_context.set(value)


def reset_correlation_id(token: Token) -> None:
    correlation_id_context.reset(token)


@contextmanager
def correlation_context(value: str):
    token = set_correlation_id(value)
    try:
        yield
    finally:
        reset_correlation_id(token)


def elapsed_milliseconds(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
