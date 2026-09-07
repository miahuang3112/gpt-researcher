---
name: academic-paper-review
description: Submit and inspect automated computer-science paper evaluations through the Academic MCP tools. Use for PDF scoring, job resumption, nine-dimension results, evidence inspection, retry, or cancellation; do not present the output as human peer review.
---

# Academic Paper Review

Use the Academic MCP tools to run and interpret an evidence-grounded automated assessment. Preserve the distinction between automated screening and human peer review in every result-facing response.

## Start or resume

- For a new evaluation, require only a PDF path visible to the Academic API. Preserve any supplied paper ID, title, source URL, model, and chunk limit. When no `paper_id` is supplied, let the API generate a `local:<uuid>` identifier. Call `submit_evaluation` once and retain its `job_id`.
- When the user supplies a `job_id`, resume with `get_evaluation_status`. Never resubmit merely because a conversation or polling loop was interrupted.
- State the `job_id` whenever work remains queued or running so another session can resume it.

## Poll and retrieve

Poll `get_evaluation_status` at a moderate interval until the job reaches `completed`, `failed`, or `cancelled`. Do not call result tools while the job is non-terminal; a conflict response means polling should continue, not that a new job should be submitted.

After completion:

1. Use `get_evaluation_dimensions` for the score overview.
2. Use `get_evaluation_evidence` when explaining scores, verification gaps, or concerns.
3. Use `get_evaluation_result` only when the full result is necessary.

Read [references/tool-contract.md](references/tool-contract.md) when exact tool inputs, states, or return boundaries are needed.

## Failure, retry, and cancellation

- On API authentication or availability failure, report the configuration problem and preserve the current `job_id`; do not create a replacement job.
- On `failed` or `cancelled`, summarize the public error and retry count. Call `retry_evaluation` only when the user requested a retry or approved it after seeing the failure.
- Call `cancel_evaluation` only for an explicit cancellation request. Treat `cancel_requested` as in progress and continue status polling until `cancelled` or another terminal state.
- Stop repeated polling when the API remains unavailable and give the user the resumable `job_id`.

## Human-review boundary

Always label the output **Automated academic assessment — not peer review**. Recommend human review when a decision affects acceptance, hiring, funding, publication, or reputation, or when any dimension is missing, evidence coverage is incomplete, verification says `review_required`, or the evidence conflicts with the rationale. Do not invent missing evidence or turn a numeric score into an acceptance recommendation.
