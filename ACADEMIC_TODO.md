# Academic Paper Evaluation Assistant — Canonical TODO

This file is the single source of truth for implementation order. Historical
handoff notes and `academic_review_modification_log.md` are context only.

## Status and execution policy

- `[x]` completed and verified
- `[~]` current engineering task
- `[ ]` not started
- `[!]` blocked on human input or an external dependency

Rules:

1. Work follows explicit dependencies instead of one global serial queue.
2. Only one engineering item may be `[~]`; one independent human task may stay
   `[!]` without blocking engineering work.
3. An item is complete only when its acceptance command, test, migration, or
   versioned report is reproducible.
4. MySQL is authoritative for job lifecycle and final results; checkpoint
   artifacts are authoritative for partial-evaluation recovery; Redis is an
   at-least-once delivery transport.
5. Public claims and resume metrics must come from versioned reports, never
   example data or an unrecorded local run.

## Verified baseline

- [x] **BASE-01** Nine-dimension evaluation, structured schemas and weighted score.
- [x] **BASE-02** Evidence selection, citation validation, verification policy,
  retries, checkpoint and resume logic.
- [x] **BASE-03** Academic API/worker source implementation with persisted job
  status, progress, timestamps, result and errors; unit tests pass.
- [x] **BASE-04** One local Worker run progressed from 0/9 to 9/9.
  - Fresh-volume deployment is verified by DBQ-02. The older, currently running
    pre-Alembic volume was deliberately not modified during isolated testing.

## Track RET — Retrieval quality and evidence selection

- [x] **RET-01** PDF chunking with stable page/section/chunk metadata.
- [x] **RET-02** BM25 and Ollama vector retrieval.
- [x] **RET-03** RRF Hybrid Retrieval with configurable depths, deterministic
  ordering and single-channel failure fallback.
- [x] **RET-04** Recall@K, Precision@K, MRR, nDCG@K and comparison runner.
- [x] **RET-05** Frozen real-PDF corpus/template export with provenance.
- [x] **RET-06** Completed-annotation compiler for validated qrels JSONL.
- [x] **RET-07** Human-label the first retrieval benchmark.
  - Acceptance: protocol, annotator/version metadata, adjudication notes and at
    least one positive judgement per query.
  - Verified: 810/810 judgements use valid 0/1/2 grades, all nine queries have
    positive evidence, and project-owner review/adjudication is recorded with
    the original ChatGPT pre-annotation lineage retained. This was not an
    independent blind inter-annotator study.
- [x] **RET-07B** Expand the benchmark to at least 5–10 papers before splitting.
  - Dependencies: RET-07 annotation protocol.
  - Verified (2026-09-01): the frozen collection contains 5 computer-science
    papers, 265 chunks, 45 dimension queries and 725 positive judgements. Each
    member records PDF/corpus/query/annotation hashes and provenance; collection
    construction and retrieval-data tests pass (`40 passed`). Four expansion
    papers use ChatGPT annotation and ChatGPT review, not independent human
    annotation or inter-annotator agreement.
- [x] **RET-08** Run BM25/Vector/Hybrid baselines and save a versioned report.
  - Dependencies: RET-07.
  - Provisional dry-run (2026-09-01): the full runner completed with zero
    retrieval failures against explicitly identified ChatGPT AI pre-annotations;
    the versioned report records corpus/query/source hashes, chunking, embedding
    model and label provenance.
  - Verified (2026-09-01): the human-reviewed qrels and
    `qwen3-embedding:4b` completed with zero failures. BM25 remained strongest
    on Recall@10 (0.2849) and nDCG@10 (0.3719); Hybrid reached Recall@10 0.2000
    and nDCG@10 0.3026. The 4B vector model did not improve recall over 0.6B,
    so model size alone is not treated as evidence for a better default.
- [x] **RET-09** Create paper-disjoint development/test splits and tune only on
  development data.
  - Dependencies: RET-07B, RET-08.
  - Implemented: a versioned fixed-paper split specification and reproducible
    split builder that validates parent hashes, assigns every paper exactly
    once, rejects paper overlap, freezes split hashes and marks test data as
    ineligible for tuning. The tuning entry point accepts only a verified
    development manifest, caches embeddings across trials and rejects any trial
    containing retrieval failures.
  - Verified (2026-09-01): development contains Transformer, BERT and RAG (3
    papers, 108 chunks, 27 queries, 293 positive judgements); held-out test
    contains MBR and LoRA (2 papers, 157 chunks, 18 queries, 432 positive
    judgements). A zero-failure 9-trial grid search with
    `qwen3-embedding:4b`, optimising Hybrid Recall@10 then nDCG@10, selected
    `candidate_top_k=20` and `rrf_k=100`; development Recall@10 is 0.4388 and
    nDCG@10 is 0.4426. The report records `selection_scope=development-only`
    and `test_split_accessed=false`; held-out test metrics were not run or used.
    Retrieval split/tuning and existing benchmark tests pass (`47 passed`).
- [x] **RET-10A** Integrate configurable Hybrid retrieval behind a feature flag,
  reusing one vector index per paper.
  - Dependencies: RET-08.
  - Implemented: the production nine-dimension scoring pipeline can select
    evidence through BM25/Vector/RRF Hybrid queries when
    `ACADEMIC_HYBRID_RETRIEVAL_ENABLED=true`. Candidate depth, RRF constant and
    Ollama embedding model are environment-configurable and passed from the
    Worker to the evaluation runner. At the RET-10A integration checkpoint the
    flag remained false; RET-10B subsequently activated the frozen v1 default.
  - Index reuse: one BM25/Vector/Hybrid set is constructed for the paper and
    shared by all unfinished dimensions; VectorRetriever builds the document
    matrix on the first query and reuses it for subsequent dimension queries.
    Production and benchmark query text now share one canonical mapping.
  - Verified (2026-09-01): integration tests prove one document-embedding batch
    and nine query batches for a complete evaluation, no embedding call while
    disabled, configuration propagation through the Worker, and candidate-depth
    validation. Full Academic suite passes (`299 passed, 1 skipped`); the skip
    is the existing opt-in real-Ollama E2E test.
- [x] **RET-10B** Freeze the measured default retriever and parameters.
  - Dependencies: RET-09, RET-10A.
  - Frozen policy: `ret10b-hybrid-v1` selects Hybrid using only the predeclared
    development objective, with `qwen3-embedding:4b`, `candidate_top_k=20`,
    `rrf_k=100` and production evidence depth 4. Python settings, Compose and a
    versioned policy artifact share these defaults; drift tests pin the policy
    and canonical dimension-query hash. Section-only remains an explicit
    rollback/ablation flag, not the default.
  - Held-out verification (2026-09-01): the immutable runner opened the 2-paper,
    18-query test split once and completed with zero failures. Frozen Hybrid v1
    reached Recall@10 0.1824 and nDCG@10 0.3462; BM25 in the same report reached
    0.2555 and 0.4338. This negative generalisation result is retained honestly:
    test was not used to change v1, and a future default change requires new
    development data plus a separately versioned policy. Labels retain the
    mixed ChatGPT/project-owner provenance warning.
  - Verified: frozen-policy, deployment-default drift, test-manifest integrity,
    no-retuning and full Academic regression tests pass (`308 passed, 1
    skipped`); the skip is the existing opt-in real-Ollama API E2E test.
- [x] **RET-11A** Compare section-only and Hybrid selection using automatic
  evidence/citation/latency/failure metrics.
  - Dependencies: RET-10A.
  - Implemented: a hash-bound held-out comparison runner evaluates section-only
    and frozen Hybrid at the same production evidence depth (`top_k=4`). It
    records per-query selected chunk IDs, Recall/Precision/nDCG, qrels-based
    relevant/highly-relevant citation-candidate proxies, query coverage, cold
    per-paper index cost, warm-query latency and selector/index failures.
  - Verified (2026-09-02): over 2 papers and 18 queries, section-only reached
    Recall@4 0.1846, nDCG@4 0.6066 and relevant-candidate rate 0.7361, versus
    frozen Hybrid at 0.0755, 0.3930 and 0.4861. Query coverage was 1.0000 versus
    0.7222. Both had zero query failures and valid-ID rate 1.0. Hybrid incurred
    364.3s total cold index construction and 292.0ms mean warm-query latency;
    section sorting averaged 1.3ms.
  - Boundary: citation values are qrels-based readiness proxies, not actual LLM
    citation choice or claim support. Labels retain mixed ChatGPT/project-owner
    provenance, and rubric-aligned section preferences may advantage the section
    selector. Frozen v1 was not changed; RET-11B separately evaluates
    downstream agreement with the ChatGPT paper-quality reference.
  - Verified: selector comparison, policy integrity and full Academic regression
    tests pass (`309 passed, 1 skipped`); the skip is the existing opt-in real
    Ollama API E2E test.
- [x] **RET-11B** Compare their effect on the versioned paper-quality reference.
  - Dependencies satisfied: RET-11A and EVAL-01.
  - Held-out protocol: two papers, 18 dimensions, four evidence chunks,
    `qwen3.5:9b`, temperature 0; the exact RET-11A selections were reused and
    the frozen Hybrid policy was not retuned.
  - Both selectors generated 18 results with zero call failures; each returned
    insufficient evidence for LoRA limitations, leaving 17 scorable pairs.
  - Section-only beat Hybrid on ChatGPT-reference dimension MAE (`0.471` vs
    `0.588`), Pearson (`0.270` vs `-0.236`), exact match (`58.8%` vs `47.1%`),
    and mean weighted overall-score absolute error (`5.61` vs `12.34` points).
    Both were within one point on `94.1%` of scorable dimensions.
  - Boundary: only two held-out papers and one AI reference annotator were
    measured. This is descriptive agreement with the ChatGPT reference, not
    human agreement, statistical significance, or ground-truth paper quality;
    test results cannot be used to retune the policy.
  - Verified: report/checkpoint hashes and focused quality tests pass; full
    Academic regression suite passes (`316 passed, 1 skipped`).

## Wave 1 — Persistence, queue safety and fast feedback

### Track DBQ — Database and queue reliability

- [x] **DBQ-01** Complete Alembic migrations for papers and evaluation jobs.
  - Dependencies: BASE-03.
  - Acceptance: upgrade succeeds on a fresh schema; downgrade is tested; ORM
    columns, indexes, foreign keys and defaults match the migration.
  - Verified: SQLite fresh-schema upgrade/downgrade tests, MySQL offline DDL
    generation, and the full Academic suite (`198 passed, 1 skipped`).
- [x] **DBQ-02** Add a one-shot deployment migration service and verify a clean
  Docker volume.
  - Dependencies: DBQ-01.
  - Acceptance: migration runs once before API/worker startup; failure blocks
    startup; a fresh checkout accepts and completes a test job.
  - Verified: isolated `academic-dbq02` containers and new MySQL/Redis volumes;
    Alembic exited 0 before API/Worker startup; real E2E job reached `completed`;
    temporary containers and volumes were removed afterward.
  - Operational note: pre-Alembic volumes must be explicitly validated and
    stamped or recreated before adopting this startup path.
- [x] **DBQ-03** Make database job creation and queue publication recoverable
  with an outbox or equivalent reconciler.
  - Dependencies: DBQ-01.
  - Acceptance: delivery is at least once; duplicate publication is safe; no
    committed queued row remains permanently unpublished.
  - Implemented: transactional `academic_outbox_events`, an API lifecycle
    dispatcher with row locking and bounded exponential backoff, transactional
    interrupted-job recovery, and atomic/terminal-safe Worker job claiming.
  - Verified: full Academic suite (`205 passed, 1 skipped`); with Redis stopped,
    POST still committed a queued job and pending event; after Redis recovery the
    event published and the job completed 9/9; two duplicate messages were
    ignored without changing the completed result.
  - Observation: the current `BLPOP` Worker process restarts when Redis becomes
    unreachable. Compose recovery succeeded; DBQ-04/05 must make this explicit
    and testable without relying on process restart.
- [x] **DBQ-04** Replace `BLPOP` with acknowledged delivery, bounded retry,
  stale-message recovery and a dead-letter queue.
  - Dependencies: DBQ-03.
  - Preferred implementation: Redis Streams consumer groups.
  - Acceptance: worker termination causes redelivery; job processing is
    idempotent; exactly one terminal result is committed; poison jobs reach DLQ.
  - Implemented: Redis Streams consumer group, explicit ACK, PEL stale recovery
    with `XAUTOCLAIM`, three-attempt retry, atomic DLQ+ACK, Redis reconnect, and
    row-locked recovery of only the reclaimed running job.
  - Verified: full Academic suite (`208 passed, 1 skipped`); Worker was killed at
    8/9 and its replacement reclaimed delivery 2, resumed the checkpoint and
    completed once; PEL returned to 0. A missing-PDF poison job failed on delivery
    3, produced one DLQ entry, and was removed from PEL.
- [x] **DBQ-05** Separate liveness and per-service readiness checks.
  - Dependencies: DBQ-01.
  - Acceptance: API readiness reports MySQL/Redis separately; Worker readiness
    also reports Ollama; temporary dependency failure does not kill liveness.
  - Implemented: `/health/live`, dependency-aware `/health/ready`, a Worker
    readiness CLI for MySQL/Redis/Ollama, Compose healthchecks, bounded check
    timeouts, and backward-compatible `/health`.
  - Verified: full Academic suite (`213 passed, 1 skipped`); Redis outage kept
    API liveness at HTTP 200 while readiness returned 503 with only Redis failed;
    Worker reported MySQL/Ollama healthy and Redis failed, stayed running with
    restart count 0, and returned ready after Redis recovery in the same process.

### Track OPS — Early observability and CI

- [x] **OPS-01** Add structured logs, correlation IDs and per-stage timing.
  - Dependencies: BASE-03.
  - Implemented: JSON application logs, sanitized/generated `X-Request-ID`
    propagation through the transactional outbox and Redis Stream, and Worker
    prepare/evaluate/persist/total duration measurements.
  - Verified: full Academic suite (`216 passed, 1 skipped`); an isolated Docker
    job completed 9/9 with correlation ID `resume-ops01-e2e` present in API job
    creation, outbox publication and every Worker stage log. Temporary
    containers, network and volumes were removed afterward.
- [x] **OPS-02** Add an Academic CI job for unit/API/worker/migration tests.
  - Dependencies: DBQ-01; expand after DBQ-03/04.
  - Acceptance: clean services are created in CI; the fast gate does not require
    a long-running real Ollama request.
  - Implemented: a path-filtered GitHub Actions fast gate with fresh MySQL 8.4
    and Redis 7.4 services, pinned Academic/test dependencies, an Alembic
    upgrade/downgrade/upgrade cycle, and the complete Academic test suite with
    real Ollama E2E explicitly disabled. Workflow contract tests protect the
    service and command requirements.
  - Verified: CI-equivalent isolated MySQL/Redis services completed the real
    migration cycle; the full fast gate passed (`218 passed, 1 skipped`). The
    skipped test is the opt-in real Ollama E2E. Validation also fixed MySQL-safe
    migration downgrades and made the SQLAlchemy `greenlet` dependency explicit.

## Wave 2 — Data quality and scoring validity

### Track DATA — External scholarly evidence

- [x] **DATA-01** OpenAlex and Crossref adapters and fallback.
- [x] **DATA-02** Source scoring, claim extraction and contradiction components.
- [x] **DATA-03** Add DOI/title deduplication and canonical paper identity.
  - Implemented: shared DOI and Unicode-title normalization, deterministic
    DOI-first/title-hash identities, source-record provenance, conservative
    duplicate grouping, and deterministic metadata merging in both provider
    adapters and fallback results. Conflicting DOI records are never collapsed
    solely because they share a title.
  - Verified: identity, OpenAlex, Crossref, fallback and source-scoring tests
    pass (`27 passed`).
- [x] **DATA-04** Connect external retrieval to novelty and citation quality.
  - Dependencies: DATA-03, RET-10A.
  - Implemented: opt-in OpenAlex-first/Crossref-fallback retrieval is executed
    once by the Worker and frozen in the resumable checkpoint. Canonicalized
    records exclude the target paper and only publication-time-eligible records
    enter the `novelty` and `citation_quality` prompts; later, undated, or
    target-year-unverifiable records remain visible for audit but cannot affect
    scoring. External records use a separate ID namespace, are treated as
    untrusted discovery context, and are exposed through API results, MCP, Web,
    JSON, and PDF without being confused with PDF evidence chunks.
  - Failure boundary: disabled retrieval, absent credentials, no results, and
    provider errors never fail the evaluation or lower a score. Status and
    sanitized warnings are persisted, while internal-evidence scoring continues.
  - Verified: retrieval/temporal filtering, prompt isolation and injection
    resistance, invented-ID rejection, checkpoint resume, Worker/config, MCP,
    export, and frontend integration tests pass. Full Academic suite: `322
    passed, 1 skipped`; Next.js production build and `git diff --check` pass.
- [x] **DATA-05A** Integrate contradiction warnings with evidence and graceful
  failure behaviour.
  - Dependencies: DATA-02.
  - Implemented: optional cross-paper comparison claims in the scoring pipeline,
    evidence-linked contradiction/partial-contradiction warnings, per-dimension
    unresolved contradiction metrics, and explicit non-fatal analysis warnings
    when detection is absent or becomes unavailable. Detector failures stop
    further comparison calls but never discard the completed paper score.
  - Calibration note: warnings retain model confidence without applying a
    decision threshold; threshold selection remains DATA-05B and must use
    development labels.
  - Verified: warning evidence/count, unavailable-detector, unconfigured-detector
    and existing contradiction tests pass; full Academic suite (`227 passed, 1
    skipped`).
- [x] **DATA-05B** Calibrate contradiction thresholds on development labels.
  - Dependencies: DATA-05A, EVAL-01.
  - Implemented: a versioned 24-pair ChatGPT-labelled development challenge
    set covering all six relation classes, explicit AI/provenance and held-out
    exclusions, resumable temperature-zero `qwen3.5:9b` inference, deterministic
    threshold selection/reporting, and a frozen production warning policy.
    `contradiction` and `partial_contradiction` results now create unresolved
    review warnings only at confidence `>= 0.95`; lower-confidence results do
    not affect verification counts.
  - Calibration result: with a predeclared precision floor of `0.80`, threshold
    `0.95` produced development warning precision/recall/F1 of `1.0/1.0/1.0`
    on 8 positive and 16 negative pairs. Six-class relation accuracy was
    `0.875` because three partial contradictions were classified as full
    contradictions. These are development-set engineering measurements, not
    held-out accuracy or independent human agreement; EVAL-02B remains the
    independent accuracy task.
  - Verified: dataset integrity, no held-out leakage, report hashes, threshold
    selection and production boundary tests pass. Full Academic suite: `331
    passed, 1 skipped`; `git diff --check` passes.

### Track EVAL — Human scoring and calibration

- [x] **EVAL-01** Create a nine-dimension paper-quality reference set.
  - Acceptance amendment: the project owner explicitly required ChatGPT to
    perform all manual-review tasks. The resulting five-paper/45-label set is
    versioned under `evals/academic_quality/` with per-score evidence,
    rationale, confidence, source hashes, validation, and tests.
  - Provenance boundary: this is an AI-scored ChatGPT reference, not an
    independent human-scored benchmark or peer review. Downstream results must
    say "agreement with the ChatGPT reference" and cannot claim human agreement
    or ground-truth paper quality.
  - Paper-quality labels remain separate from RET relevance labels; RET grades
    were not converted into quality scores and chunk IDs are evidence pointers
    only.
- [x] **EVAL-02A** Measure per-dimension score correlation, MAE and run-to-run
  stability.
  - Dependencies: EVAL-01, RET-10A.
  - Implemented a resumable five-paper/45-dimension measurement using the
    exact frozen Hybrid selections, local temperature-zero `qwen3.5:9b`, and
    the versioned ChatGPT reference. The report includes aggregate and
    per-dimension MAE/Pearson/Spearman, exact/within-one agreement, terminal
    failure auditing, and a hashed versioned summary.
  - Result: 42 dimensions were scorable, three returned explicit
    `insufficient_evidence`, and no terminal failures remained.
    ChatGPT-reference MAE was `0.500`; Pearson/Spearman were `0.149/0.169`;
    exact and within-one agreement were `0.571/0.929`.
  - Stability: the independent RET-11B frozen-Hybrid run supplied a second
    held-out run. Sixteen comparable pairs had exact agreement, zero mean
    within-item standard deviation and Pearson/Spearman `1.0/1.0`; one pair
    was excluded for selection mismatch and one was not jointly scorable.
  - Failure recovery: corrected a prompt/JSON-Schema `confidence` mismatch,
    bounded free-text output, raised the paper evaluator output budget, added
    safe failure codes, and added terminal-only compact recovery. All 15 items
    that experienced a failed model call ultimately recovered; full attempt
    history remains versioned for auditability.
  - Claim boundary: these are descriptive results for five papers and one AI
    reference annotator. Per-dimension correlations have only four or five
    observations; no human-agreement, significance, or generalisation claim
    is made, and held-out results were not used for retuning.
  - Verified: summary/full-report hash checks and recovery regression tests
    pass. Full Academic suite: `342 passed, 1 skipped`; `git diff --check`
    passes.
- [x] **EVAL-02B** Measure citation validation and contradiction accuracy.
  - Dependencies: EVAL-01, DATA-05A.
  - Implemented two balanced, versioned ChatGPT-labelled held-out challenge
    sets sourced only from the frozen LoRA and MBR papers: 16 citation-support
    cases across four statuses and 12 contradiction pairs across six relation
    classes. Development papers are explicitly excluded, provenance and input
    hashes are pinned, and test labels were not used for tuning.
  - Citation result: all 16 temperature-zero `qwen3.5:9b` calls completed with
    no terminal failure. Four-class accuracy/macro F1 were `0.8125/0.781818`;
    the production supported/not-supported decision achieved precision,
    recall and F1 of `1.0/1.0/1.0`. Three insufficient-evidence cases were
    called contradicted, identifying the main class-granularity weakness.
  - Contradiction result: all 12 calls completed with no terminal failure.
    Six-class accuracy/macro F1 were `0.833333/0.777778`; with the frozen
    DATA-05B threshold `0.95`, warning precision/recall/F1 were
    `1.0/1.0/1.0`. Both partial contradictions were called full
    contradictions, preserving warning decisions while losing relation detail.
  - Added a resumable evaluator, frozen-data validation, confusion matrices,
    per-class and production-binary metrics, safe failure records,
    retry-failures support, a hash-pinned summary, and regression tests.
  - Claim boundary: these are 28 controlled examples from two papers and one
    AI annotator, not human labels, independent peer review, statistical
    significance, scientific correctness, or evidence of generalisation.
    Held-out observations were recorded without prompt or threshold retuning.
  - Verified: EVAL-02B tests `7 passed`; full Academic suite `349 passed, 1
    skipped`; `git diff --check` passes.
- [x] **EVAL-03** Run prompt/retriever/policy ablations and calibrate on dev data.
  - Dependencies: EVAL-02A, EVAL-02B.
  - Froze a development-only protocol before inference and evaluated three
    one-factor configurations across 3 papers and 27 dimensions each with
    `qwen3.5:9b` at temperature zero; held-out LoRA/MBR data was not used.
  - Retained the anchored prompt: Hybrid anchored MAE was `0.304348` versus
    `0.521739` without anchors, with 23 scorable items in each configuration.
  - Retained Hybrid retrieval: section-only improved development Recall@4 from
    `0.222359` to `0.313365` and MAE from `0.304348` to `0.28`, but its MAE
    gain (`0.024348`) missed the predeclared `0.1` adoption minimum.
  - Retained the DATA-05B contradiction-warning threshold `0.95`; development
    precision, recall, and F1 remain `1.0` on 24 controlled examples.
  - All 81 evaluations ended with zero terminal failures. Four invalid-JSON
    items used an audited compact recovery that preserved model, temperature,
    rubric, dimension, and evidence chunk IDs.
  - Versioned the full report, frozen protocol, hash-pinned summary, resumable
    runner, recovery audit trail, and regression tests. Measurements use one AI
    annotator and three development papers; no generalisation claim is made.
- [x] **EVAL-04** Add deterministic benchmark smoke checks to CI and a manual
  full benchmark run.
  - Dependencies: RET-09, EVAL-02A.
  - Added a fail-closed, network-free CI smoke command that verifies 11 pinned
    artifact hashes and 15 zero-failure, split-isolation, no-retuning and frozen
    policy invariants before the Academic test suite.
  - Added a manually dispatched six-hour full benchmark workflow with pinned Ollama
    `0.32.13`, frozen `qwen3-embedding:4b`/`qwen3.5:9b` models, eight retrieval
    and quality reports, checkpoint/recovery, post-run validation, SHA-256 run
    manifest, and 30-day artifact retention.
  - The weekly schedule runs only the public, deterministic smoke gate because
    copyrighted PDF corpora are deliberately excluded from the repository.
  - Added a reusable local full-suite runner, full-output validator and
    workflow regression tests. Verified: EVAL-04/EVAL-03 tests `11 passed`, both
    workflow YAML files parse, shell syntax passes, and `git diff --check`
    passes.

## Wave 3 — Stable product and Agent interfaces

### Track SEC — Security and data lifecycle

- [x] **SEC-01** Define authentication or a documented trusted-local boundary.
  - Implemented: explicit `trusted_local` and `api_key` modes, constant-time
    `X-API-Key` verification, startup rejection for missing configured secrets,
    unauthenticated orchestration health endpoints, loopback-only Compose port
    bindings, and deployment-boundary documentation.
  - Verified: security boundary, secret-redaction, Compose and health tests pass
    (`14 passed`).
- [x] **SEC-02** Validate PDF size, signature, page limits and randomized storage.
  - Implemented: allowlisted import roots; streamed byte caps; PDF signature,
    parseability, encryption and page-count checks; private random filenames;
    managed checkpoint paths; and a shared API/Worker secure-data volume.
  - Verified: valid-copy, randomized-name, permissions, root escape, fake,
    malformed, encrypted, oversized and excessive-page tests pass (`11 passed`).
- [x] **SEC-03** Remove server paths, secrets and internal tracebacks from public
  responses and stored public errors.
  - Implemented: public checkpoint paths removed; stable error codes and generic
    messages stored on jobs; internal errors represented by log fingerprints;
    recursive legacy-result sanitization removes private fields, filesystem
    paths, secrets and tracebacks; DLQ messages receive only safe job errors.
  - Verified: public-error, response-schema, Worker, migration and Redis Stream
    tests pass (`13 passed`).
- [x] **SEC-04** Add prompt-injection/malicious-PDF tests and MCP tool allowlists.
  - Implemented: active-content/embedded-payload and extracted-text PDF guards;
    prompt-injection pattern annotations preserved on Evidence Chunks; explicit
    untrusted-document instructions in every scoring prompt; and a default-deny
    Academic MCP policy limited to the planned submit/read MVP tools.
  - Verified: malicious PDF, prompt isolation, normal-text, MCP deny/allow,
    chunker and evaluation-pipeline tests pass (`39 passed`).
- [x] **SEC-05** Define PDF/checkpoint/result retention and deletion semantics.
  - Implemented: independently configurable 30-day PDF, 7-day checkpoint and
    90-day result defaults; terminal-only purge CLI; retained minimal audit
    rows; job-owned PDF paths; shared-legacy-file reference protection; and an
    authenticated terminal-job DELETE endpoint returning 409 for active jobs.
    Physical deletion is restricted to managed roots and is documented as
    irreversible without external backups.
  - Verified: retention policy tests pass; full Academic suite (`251 passed, 1
    skipped`); an isolated MySQL/Redis/API deployment migrated cleanly, copied a
    real PDF to a random mode-600 shared-volume path, returned no server path,
    deleted the terminal job with HTTP 204, and left no DB/file residue.

### Track API/WEB — Usable paper-evaluation product

- [x] **WEB-01** Add secure PDF upload.
  - Dependencies: DBQ-01, SEC-02.
  - Completed: added an authenticated multipart upload-and-submit endpoint that
    streams into bounded managed storage, validates MIME/signature/structure,
    encryption, pages, extracted text and active content, uses random mode-600
    filenames, creates the Paper/Job/Outbox transaction, and removes the file
    when persistence fails. The legacy trusted-local path endpoint remains for
    compatibility and the public response exposes no server paths.
  - Verified: upload/service/API tests cover success, hostile filenames, wrong
    MIME, oversize input and database failure cleanup; full Academic suite
    passes (`257 passed, 1 skipped`).
- [x] **WEB-02A** Define and test the persisted job state machine.
  - Dependencies: DBQ-04.
  - Completed: added persisted queued/running/cancel-requested/completed/failed/
    cancelled states, an explicit transition table, monotonic state versions,
    retry/cancellation timestamps, and migration `b7d2f6c8a104`.
  - Verified: transition tests cover allowed, rejected, idempotent and terminal
    transitions; the migration was applied to the deployed MySQL stack.
- [x] **WEB-02B** Add paginated job listing and result retrieval.
  - Dependencies: DBQ-01, SEC-03.
  - Completed: added status-filtered paginated job metadata and separate
    completed-result retrieval endpoints; public payloads are sanitized and do
    not expose large results in list responses.
  - Verified: API tests cover paging metadata, filters, completed-only result
    access and private-field removal; live API returned the existing 17 jobs.
- [x] **WEB-02C** Add idempotent retry.
  - Dependencies: WEB-02A, DBQ-04.
  - Completed: failed/cancelled jobs can be atomically re-queued with one outbox
    event, reset result/error fields and an incremented retry counter; repeated
    requests cannot enqueue duplicate work.
  - Verified: route tests cover accepted and repeated retry requests.
- [x] **WEB-02D** Add cooperative cancellation.
  - Dependencies: WEB-02A, DBQ-04.
  - Completed: queued jobs cancel immediately, running jobs persist a
    cancellation request, and the worker cooperatively stops the evaluation
    task before committing the terminal cancelled state.
  - Verified: API/worker tests cover both paths and idempotency; a real job was
    cancelled from running state and its uploaded PDF/checkpoint were cleaned.
- [x] **WEB-03** Add SSE progress with stable event IDs and reconnect fallback.
  - Dependencies: WEB-02A, WEB-02B.
  - Completed: added authenticated SSE snapshots keyed by persisted state
    version, `Last-Event-ID` replay, heartbeat frames and terminal closure; the
    web client falls back to polling when streaming is unavailable.
  - Verified: stream tests cover replay, monotonic IDs and heartbeats; a live
    reconnect replayed the terminal cancellation event with the same stable ID.
- [x] **WEB-04** Build submission and job-history pages.
  - Dependencies: WEB-01, WEB-02B.
  - Completed: added the responsive `/academic` workspace with secure PDF
    submission, optional API-key storage, paginated history, live status,
    progress, retry and cancellation controls.
  - Verified: the optimized Next.js build succeeds and browser regression at
    desktop and 390px mobile sizes shows history and all controls without
    horizontal overflow.
- [x] **WEB-05** Build the nine-dimension report and evidence viewer.
  - Dependencies: WEB-02B.
  - Requirement: label results as automated assessment, not peer review.
  - Completed: rendered score, coverage and assessment summaries, all nine
    dimension cards, rationales, evidence chunk references, verification notes
    and concerns; the UI explicitly says `Automated assessment · Not peer
    review`.
  - Verified: browser regression confirms nine evidence actions and the
    disclaimer at desktop and mobile widths using a completed real evaluation.
- [x] **WEB-06** Add sanitized, versioned JSON/PDF export.
  - Dependencies: WEB-05, SEC-03.
  - Completed: added sanitized `academic-evaluation-export.v1` JSON and generated
    PDF downloads containing scores and evidence but no PDF/checkpoint paths;
    the web client downloads both through authenticated requests.
  - Verified: export unit tests and live downloads validate schema, PDF
    structure, disclaimer and absence of private paths; browser regression
    confirms both download controls.
- [x] **API-01** Version public evaluation/result schemas.
  - Dependencies: WEB-02B.
  - Completed: every public evaluation response declares
    `academic-evaluation.v1`; canonical routes are exposed under `/api/v1` while
    the existing `/api` routes remain compatible and hidden from OpenAPI.
  - Verified: versioning/OpenAPI tests pass; both legacy and v1 live endpoints
    returned the same versioned job data. Full Academic suite: `275 passed, 1
    skipped`; final Next.js production build and desktop/mobile browser checks
    pass.

### Track AGENT — MCP and Skill

- [x] **AGENT-01** Implement an Academic MCP MVP backed by the same API/jobs.
  - Dependencies: DBQ-02, DBQ-03, stable create/get API.
  - Completed: added a dependency-free stdio JSON-RPC MCP server and sanitized
    API client backed exclusively by the versioned Academic API and persisted
    jobs. The server supports MCP initialize, ping, tool discovery/calls,
    structured content and safe protocol/tool errors.
  - Verified: MCP protocol negotiation succeeds for `2025-06-18` and
    `2024-11-05`; project `.mcp.json` exposes the `academic-evaluation` server.
- [x] **AGENT-02** Expose submit/status/result/evidence/dimensions tools.
  - Dependencies: AGENT-01; polling is sufficient for the MVP.
  - Completed: exposed `submit_evaluation`, `get_evaluation_status`,
    `get_evaluation_result`, `get_evaluation_evidence` and
    `get_evaluation_dimensions` with bounded JSON schemas, structured outputs
    and read-only annotations. Status omits the large result; evidence and
    dimension tools return purpose-specific projections.
  - Verified: contract tests cover tool discovery, annotations and every output
    boundary; a real MCP subprocess read a completed job's status, score and all
    nine dimensions from the deployed API.
- [x] **AGENT-03** Create `academic-paper-review` Skill with input, polling,
  resume, failure and human-review rules.
  - Dependencies: AGENT-02.
  - Completed: created the repository Skill with concise workflow instructions,
    an exact tool-contract reference, MCP dependency metadata, resumable job
    handling, bounded polling, failure behavior and explicit human-review
    escalation rules. Results must be labelled automated assessment, not peer
    review.
  - Verified: the bundled Skill validator reports `Skill is valid!`.
- [x] **AGENT-04** Add MCP contract tests and one fresh-environment demo.
  - Dependencies: AGENT-03.
  - Completed: added MCP initialize/list/call/stdio contract coverage plus a
    dependency-free demo that starts an isolated API and a fresh MCP subprocess,
    then exercises the complete tool lifecycle.
  - Verified: fresh-process demo negotiated MCP `2025-06-18`, discovered seven
    tools and passed submit/status/dimensions/evidence/retry/cancel assertions;
    current full Academic suite passes (`286 passed, 1 skipped`).
- [x] **AGENT-05** Add retry/cancel tools after reliable queue semantics exist.
  - Dependencies: AGENT-02, WEB-02C, WEB-02D.
  - Completed: exposed idempotent `retry_evaluation` and `cancel_evaluation`
    tools backed by the persisted API transitions; both are marked as writes,
    retry/cancel are marked idempotent, and cancellation is marked destructive
    for client approval handling. The default-deny policy rejects unknown or
    non-allowlisted tools.
  - Verified: policy/contract tests cover allowed, denied and repeated-safe
    semantics; the isolated fresh-process demo executes both tools successfully.

## Wave 4 — Portfolio and optional production polish

### Track PORT — Documentation and evidence

- [x] **PORT-01A** Update README now with verified baseline, setup, tests and
  known limitations.
  - Dependencies: DBQ-02, OPS-02.
- [x] **PORT-01B** Maintain the modification log each wave; add measured results
  after EVAL-03.
  - Engineering verification and the completed EVAL-03 measured results are
    recorded in `academic_review_modification_log.md`.
- [x] **PORT-02A** Add current architecture/job/checkpoint diagrams.
  - Dependencies: DBQ-04.
- [x] **PORT-02B** Update diagrams after MCP/Web contracts stabilize.
  - Dependencies: AGENT-02, WEB-02B.
- [x] **PORT-03A** Record an API→queue→worker→database MVP demo.
  - Dependencies: DBQ-04, DBQ-05.
- [x] **PORT-03B** Record the final Web/MCP/Skill demo.
  - Dependencies: AGENT-04, WEB-06.
- [x] **PORT-04** Write resume bullets using versioned measured facts only.
  - Dependencies: RET-11B, EVAL-03, PORT-03B.
  - Added a versioned evidence registry that pins every resume source with
    SHA-256 and records only measured facts plus prohibited overclaims.
  - Added recommended Chinese and English Agent-role bullets covering the
    asynchronous job architecture, MCP/Skill/Web surfaces, retrieval and model
    evaluation, failure recovery, and benchmark automation.
  - Preserved the required boundaries: ChatGPT reference rather than human
    gold, automated assessment rather than peer review, small-set denominators,
    and the negative held-out Hybrid result without test-set retuning.
  - Verified: PORT-04/EVAL-04 tests `7 passed`; full Academic suite `363 passed,
    1 skipped`; `git diff --check` passes.

### Optional stretch goals

- [x] **STRETCH-01** Add Prometheus/Grafana dashboards and alerts for queue depth,
  latency, retry/DLQ and model failures.
- [x] **STRETCH-02** Add per-source diversity constraints if a future multi-paper
  evidence benchmark demonstrates a need.
  - Implemented as an opt-in `max_chunks_per_source` constraint with deterministic
    tests. It remains disabled by default; benchmark activation evidence is still
    required and no recall improvement is claimed.

## Immediate execution

1. Retrieval: RET-07 through RET-11B are complete. RET-11A and RET-11B both
   favour section-only on the two-paper held-out comparison, but no frozen
   policy retuning is permitted from test results.
2. DATA-04, DATA-05B, EVAL-02A, EVAL-02B and EVAL-03 are complete. The frozen
   anchored-Hybrid prompt/retriever and `0.95` contradiction threshold remain
   unchanged after development-only calibration.
3. EVAL-04 and PORT-04 are complete. No canonical TODO tasks remain.
