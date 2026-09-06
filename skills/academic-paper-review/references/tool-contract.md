# Academic MCP tool contract

All responses are sanitized public data. Evaluation API payloads use `academic-evaluation.v1`.

## Tools

- `submit_evaluation`: requires `paper_id` and `pdf_path`; optionally accepts `title`, `source_url`, `model_name`, and `max_chunks`. Returns a resumable `job_id` and initial state.
- `get_evaluation_status`: requires `job_id`; returns compact persisted state, progress, retry count, timestamps, and public errors without the large result.
- `get_evaluation_result`: requires `job_id`; returns the full completed result.
- `get_evaluation_dimensions`: requires `job_id`; returns overall assessment, evidence coverage, missing dimensions, and dimension name/score/status records.
- `get_evaluation_evidence`: requires `job_id`; returns per-dimension rationale,
  internal PDF evidence chunk IDs, external scholarly evidence IDs,
  verification information, and concerns. The top-level `external_evidence`
  object reports retrieval status, provider, target year, warnings, and the
  frozen records used by novelty and citation-quality evaluation. External
  records are discovery context rather than proof that a citation exists or
  that an omitted work should reduce the score.
- `retry_evaluation`: requires `job_id`; valid for failed or cancelled jobs and idempotent after requeue.
- `cancel_evaluation`: requires `job_id`; idempotently cancels queued work or requests cooperative cancellation for running work.

## States

`queued → running → completed|failed` is the normal path. Cancellation adds `cancel_requested → cancelled`; a queued job may transition directly to `cancelled`. Retry moves `failed|cancelled → queued` without duplicating an already accepted retry.

Treat HTTP conflict as a state/precondition error, not a transport retry signal. Treat authentication and availability failures as resumable configuration failures and retain the existing job ID.
