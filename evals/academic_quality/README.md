# Academic paper-quality reference set

This directory contains the EVAL-01 paper-quality reference set: five public
computer-science papers scored on the same nine dimensions used by the
production evaluator. It contains 45 dimension labels on the 1–5 production
scale, with rationale, strengths, concerns, confidence, and frozen-corpus
evidence pointers for every score.

## Provenance and claim boundary

The project owner explicitly chose ChatGPT to perform the manual-review step.
These labels are therefore an **AI-scored reference set**, not independent
human peer review or ground-truth paper quality. Future metrics must be named
"agreement with the ChatGPT reference" (for example, ChatGPT-reference MAE),
not human agreement. Publication, hiring, funding, or reputation decisions
still require qualified independent reviewers.

Paper-quality labels are deliberately separate from RET relevance labels.
RET grades were not converted into scores. Existing frozen chunk IDs appear
only as auditable pointers to the paper passages used in the assessment.

## Files

- `rubric.v1.json`: frozen score anchors and dimension definitions.
- `papers.v1.json`: paper identity, development/test role, and source hashes.
- `labels.chatgpt.v1.jsonl`: one record per paper and nine labels per record.
- `provenance.chatgpt.v1.json`: annotator identity and permitted claim scope.
- `validate_dataset.py`: integrity, separation, evidence, and PDF-hash checks.

The split roles match the frozen RET-09 split: Transformer, BERT, and RAG are
development papers; LoRA and MBR are held-out test papers. Test-paper scores
must not be used to tune prompts, policies, thresholds, or weights.

## Validate

From the repository root:

```bash
python3 -m evals.academic_quality.validate_dataset
pytest -q tests/academic/test_academic_quality_dataset.py
```

## RET-11B selector-quality comparison

`compare_selector_quality.py` reuses the exact four-chunk selections persisted
by RET-11A, scores both selector inputs with one fixed local model, and compares
the outputs with the held-out ChatGPT reference labels. It writes a checkpoint
after every model call. The held-out report is descriptive only and must never
be used to retune the frozen retrieval policy.

The completed held-out run is versioned in
`ret11b_selector_quality_summary.json` with its full-report hash. Section-only
had lower dimension MAE (`0.471` versus `0.588`) and lower mean weighted
overall-score absolute error (`5.61` versus `12.34` points). Both selectors had
one `insufficient_evidence` result and no model-call failures. With only two
test papers, these are descriptive engineering measurements, not inferential
evidence.

```bash
python3 -m evals.academic_quality.compare_selector_quality \
  --selector-report evals/academic_retrieval/results/ret11a-section-vs-hybrid-heldout.json \
  --test-manifest evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --checkpoint evals/academic_quality/results/ret11b-checkpoint.json \
  --output evals/academic_quality/results/ret11b-section-vs-hybrid-heldout.json \
  --model qwen3.5:9b
```

## DATA-05B contradiction-warning calibration

`contradiction_labels.chatgpt.dev.v1.jsonl` contains 24 controlled claim pairs
covering all six relation classes. ChatGPT supplied these development labels;
they are not independent human annotations. Only Transformer, BERT, and RAG
development-paper propositions were used. LoRA and MBR remain excluded from
threshold selection.

The production `qwen3.5:9b` detector was run at temperature zero. With a
predeclared minimum precision of 0.80 and warning-class F1 as the primary
objective, the selected confidence threshold was `0.95`. Development warning
precision, recall, and F1 were each 1.0 (8 positive, 16 negative). Six-class
relation accuracy was 0.875 because three partial contradictions were called
full contradictions. These are calibration-set measurements, not held-out
accuracy or evidence of real-world generalisation.

The frozen policy and complete predictions are versioned in
`data05b_frozen_contradiction_policy.json` and
`results/data05b-contradiction-calibration.json`. Re-running the command resumes
from its checkpoint and reproduces the same report hash:

```bash
python3 -m evals.academic_quality.calibrate_contradiction_threshold \
  --model qwen3.5:9b
```

## EVAL-02A score reliability

`measure_score_reliability.py` evaluates the exact frozen Hybrid selections
for all five papers and compares dimension scores with the versioned ChatGPT
reference. The completed run produced 42 scorable results and three explicit
`insufficient_evidence` results with no terminal failures. ChatGPT-reference
MAE was `0.500`; Pearson and Spearman correlations were `0.149` and `0.169`.

Run-to-run stability reuses the independent RET-11B Hybrid run for the held-out
subset. Sixteen directly comparable dimension pairs had exact agreement, zero
mean within-item standard deviation, and Pearson/Spearman of `1.0`. One pair
was excluded because its frozen selections differed and one was not jointly
scorable. With only two held-out papers, these are descriptive diagnostics,
not evidence of generalisation. The versioned summary and full-report hash are
in `eval02a_score_reliability_summary.json`.

The runner checkpoints every item. `--retry-terminal` retries only failed
items; its bounded recovery mode preserves selected chunk IDs, limits each
chunk to 1,600 characters, and requests a compact response. Historical failed
calls and superseded terminal records remain in the report for auditability.

```bash
python3 -m evals.academic_quality.measure_score_reliability \
  --run-count 1 \
  --concurrency 1
```

## EVAL-02B validation accuracy

EVAL-02B uses two balanced ChatGPT-labelled held-out challenge sets derived
only from the frozen LoRA and MBR papers. Citation validation contains 16 cases
across `supported`, `partially_supported`, `contradicted`, and
`insufficient_evidence`. Contradiction detection contains 12 pairs balanced
across all six relation classes. These test labels were created after DATA-05B
threshold selection and were not used to change prompts, models, or policy.

The temperature-zero `qwen3.5:9b` run completed all 28 cases without terminal
failure. Citation four-class accuracy was `0.8125` with macro F1 `0.781818`;
the production binary supported/not-supported decision had precision, recall,
and F1 of `1.0`. Three insufficient-evidence cases were called contradicted.
Contradiction six-class accuracy was `0.833333` with macro F1 `0.777778`; at
the frozen `0.95` threshold, warning precision, recall, and F1 were `1.0`.
Both partial contradictions were called full contradictions, preserving the
warning decision while losing relation granularity.

The complete predictions, confusion matrices, per-class metrics and provenance
are hash-pinned by `eval02b_validation_accuracy_summary.json`. Results are
small-sample descriptive engineering measurements against one AI annotator,
not human agreement or evidence of generalisation.

```bash
python3 -m evals.academic_quality.measure_validation_accuracy
```

## EVAL-03 development-only ablations

EVAL-03 froze its protocol before inference and used only the three development
papers (Transformer, BERT, and RAG). It compared the production
Hybrid-plus-anchor baseline with two one-factor challengers: the same Hybrid
evidence without numeric scoring anchors, and section-only evidence with the
production scoring anchors. The two LoRA/MBR held-out papers were not read or
used for selection.

All three configurations completed 27 dimension evaluations with zero terminal
failures. Four initially invalid JSON responses were recovered with the same
model, temperature, dimension, evidence chunk IDs, and rubric; recovery only
truncated each chunk to 1,600 characters and requested a compact JSON response.
The superseded errors and recovery flags remain in the report.

The production Hybrid-plus-anchor baseline produced 23 scorable results, MAE
`0.304348`, and a within-one-point rate of `0.956522` against the ChatGPT
reference. Removing anchors worsened MAE to `0.521739`, so the anchored prompt
was retained. Section-only raised development Recall@4 from `0.222359` to
`0.313365`, produced 25 scorable results, and reduced MAE to `0.28`, but its
`0.024348` MAE improvement did not reach the predeclared `0.1` adoption
minimum. Hybrid retrieval was therefore retained. The contradiction-warning
threshold remains `0.95`, with development precision, recall, and F1 all `1.0`.

The report is hash-pinned by `eval03_development_ablation_summary.json`. These
small, AI-referenced development measurements support configuration calibration
only; they are not held-out accuracy or evidence of generalisation.

```bash
python3 -m evals.academic_quality.run_development_ablations
python3 -m evals.academic_quality.run_development_ablations --retry-failures
```

## EVAL-04 CI benchmark gates

Every Academic pull request and main-branch push runs a deterministic benchmark
smoke check before the test suite. It verifies 11 SHA-256-pinned policies,
datasets and reports plus 15 zero-failure, split-isolation and frozen-decision
invariants without calling Ollama or the network:

```bash
python -m evals.verify_academic_benchmarks
```

`.github/workflows/academic-benchmarks.yml` runs the deterministic public gate
weekly. The complete eight-report Ollama job is manual because the source PDF
corpora are intentionally not committed. With that protected corpus supplied,
the job re-runs development tuning, held-out retrieval, selector comparison,
contradiction calibration, score reliability, held-out validation, and
development ablations. Reports, checkpoints, and a SHA-256 run manifest are
retained as one CI artifact for 30 days. Missing corpus data, terminal model
failure, or a development/held-out boundary violation fails the full run.

For the same complete run against an existing local Ollama service:

```bash
./scripts/run_academic_benchmarks.sh artifacts/academic-benchmarks
```
