from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

from gpt_researcher.academic.retrieval_policy import FROZEN_RETRIEVAL_POLICY


class Settings(BaseSettings):
    app_name: str = "Academic Evaluation API"
    log_level: str = "INFO"
    academic_auth_mode: Literal["trusted_local", "api_key"] = "trusted_local"
    academic_api_key: SecretStr | None = None
    academic_pdf_import_roots: str = ""
    academic_pdf_storage_dir: str = "/data/academic/pdfs"
    academic_checkpoint_storage_dir: str = "/data/academic/checkpoints"
    academic_pdf_max_bytes: int = 50 * 1024 * 1024
    academic_pdf_max_pages: int = 200
    academic_pdf_max_extracted_characters: int = 5_000_000
    academic_pdf_retention_days: int = 30
    academic_checkpoint_retention_days: int = 7
    academic_result_retention_days: int = 90
    academic_sse_poll_seconds: float = 1.0
    academic_sse_heartbeat_seconds: float = 15.0
    academic_hybrid_retrieval_enabled: bool = (
        FROZEN_RETRIEVAL_POLICY.enabled_by_default
    )
    academic_hybrid_candidate_top_k: int = (
        FROZEN_RETRIEVAL_POLICY.candidate_top_k
    )
    academic_hybrid_rrf_k: int = FROZEN_RETRIEVAL_POLICY.rrf_k
    academic_embedding_model: str = FROZEN_RETRIEVAL_POLICY.embedding_model
    academic_external_evidence_enabled: bool = False
    academic_external_evidence_limit: int = 5

    mysql_user: str = "academic"
    mysql_password: str = "academic_dev"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_database: str = "academic_evaluation"

    redis_url: str = "redis://127.0.0.1:6380/0"
    ollama_base_url: str = "http://127.0.0.1:11434"
    healthcheck_timeout_seconds: float = 3.0

    stream_block_milliseconds: int = 2000
    stream_claim_idle_milliseconds: int = 30000
    stream_max_delivery_attempts: int = 3
    stream_error_retry_seconds: float = 1.0

    outbox_poll_interval_seconds: float = 1.0
    outbox_batch_size: int = 50
    outbox_max_retry_seconds: int = 60
    outbox_error_max_length: int = 1000

    @model_validator(mode="after")
    def validate_authentication(self) -> "Settings":
        if (
            self.academic_auth_mode == "api_key"
            and (
                self.academic_api_key is None
                or not self.academic_api_key.get_secret_value().strip()
            )
        ):
            raise ValueError(
                "ACADEMIC_API_KEY is required when "
                "ACADEMIC_AUTH_MODE=api_key."
            )
        if self.academic_hybrid_candidate_top_k < 1:
            raise ValueError(
                "ACADEMIC_HYBRID_CANDIDATE_TOP_K must be at least 1."
            )
        if self.academic_hybrid_rrf_k < 1:
            raise ValueError(
                "ACADEMIC_HYBRID_RRF_K must be at least 1."
            )
        if not self.academic_embedding_model.strip():
            raise ValueError(
                "ACADEMIC_EMBEDDING_MODEL must not be blank."
            )
        if not 1 <= self.academic_external_evidence_limit <= 20:
            raise ValueError(
                "ACADEMIC_EXTERNAL_EVIDENCE_LIMIT must be between 1 and 20."
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env.academic",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="mysql+asyncmy",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )

    @property
    def pdf_import_roots(self) -> tuple[str, ...]:
        return tuple(
            value.strip()
            for value in self.academic_pdf_import_roots.split(",")
            if value.strip()
        )


settings = Settings()
