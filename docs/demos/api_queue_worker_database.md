# Demo Record: API → Queue → Worker → Database

## Scope

This record demonstrates the implemented asynchronous boundary rather than
claiming model quality. The path is:

```text
POST API → MySQL Job + Outbox → Redis Stream → Worker → Checkpoint/Pipeline
→ MySQL terminal result → GET API
```

## Reproduce

Start the stack and confirm readiness:

```bash
docker compose -f compose.academic.yml up -d --wait
curl http://127.0.0.1:8001/health/ready
```

Run the opt-in real integration test with a PDF path allowed by
`ACADEMIC_PDF_IMPORT_ROOTS`:

```bash
RUN_ACADEMIC_E2E=1 \
ACADEMIC_E2E_PDF_PATH=/absolute/path/to/paper.pdf \
ACADEMIC_E2E_QUEUE_TIMEOUT_SECONDS=300 \
ACADEMIC_E2E_RUN_TIMEOUT_SECONDS=300 \
python -m pytest tests/academic/test_academic_api_e2e.py -v -s
```

Queue wait and model execution have separate deadlines, so a slow job ahead of
the submitted job no longer consumes its model-runtime allowance.

While it runs, inspect only public state through the API:

```bash
curl http://127.0.0.1:8001/api/v1/academic/evaluations/JOB_ID
```

Operational verification can inspect the local development stack:

```bash
docker compose -f compose.academic.yml exec -T redis \
  redis-cli XPENDING academic:evaluation:stream academic-workers
docker compose -f compose.academic.yml logs academic-api academic-worker
```

## Recorded verification

On 2026-08-26 the deployed stack reported API, worker, MySQL and Redis healthy,
with Alembic head `b7d2f6c8a104`. The established real completed sample
`59c18c8435fa4d54ad36665603cc311d` persisted `completed`, `9/9`, score `85.0`,
and timestamps `2026-08-23 12:52:50` → `12:52:52` in MySQL; its public result
contains all nine dimensions.

Earlier reliability acceptance runs, recorded in the canonical TODO, verified:

- a fresh migration volume accepted a real job and reached `completed`;
- stopping Redis after API commit left the outbox event pending, then recovery
  published it and the job completed;
- killing a worker at 8/9 caused `XAUTOCLAIM` redelivery, checkpoint resume and
  one terminal result, with PEL returning to zero;
- a poison job reached delivery three, moved to the DLQ and left the PEL.

## Regression found while recording this demo

A fresh 2026-08-26 submission exposed a post-commit SQLAlchemy
`MissingGreenlet`: the job/outbox transaction committed and the worker received
the job, but response serialization attempted to lazy-load database-generated
timestamps and returned HTTP 500. Retry/cancel responses had the same risk.

The fix explicitly flushes and refreshes the row, freezes the public response
before commit, and performs no post-commit lazy IO. Regression tests assert that
refresh occurs before commit and that create/retry/cancel return persisted
timestamps.
The two temporary demo jobs were cooperatively cancelled after diagnosis; no
existing user jobs were changed.

## Interpretation boundary

This demo proves migration order, durable publication, acknowledged delivery,
checkpoint recovery and result persistence. It does not prove scoring accuracy,
retrieval recall, human agreement, or production throughput.
