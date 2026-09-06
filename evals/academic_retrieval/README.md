# Academic Retrieval Benchmark

This directory compares BM25, vector and RRF hybrid retrieval on the same
academic evidence queries.

The `example_*.jsonl` files are synthetic examples for validating the runner.
They are not research results and must not be reported as benchmark evidence.

## Data format

Each corpus line is an `EvidenceChunk` JSON object. Each query line contains a
query and human relevance judgements:

```json
{
  "query_id": "paper-001-reproducibility",
  "paper_id": "paper-001",
  "query": "Does the paper report random seeds and hardware?",
  "dimension": "reproducibility",
  "relevance_by_chunk_id": {
    "paper-001-p4-c12": 2,
    "paper-001-p8-c27": 1
  }
}
```

Recommended grades are `0=irrelevant`, `1=partially relevant`, and
`2=highly relevant`. Recall and precision treat every grade above zero as
relevant; nDCG preserves the graded values.

## Run

The command below uses the configured local Ollama embedding model:

```bash
python -m evals.academic_retrieval.run_benchmark \
  --corpus evals/academic_retrieval/example_corpus.jsonl \
  --queries evals/academic_retrieval/example_queries.jsonl \
  --cutoffs 1 2 4 \
  --output evals/academic_retrieval/results/example.json
```

For a versioned real-data report, also pass `--dataset-manifest` and
`--label-provenance`. The report then records input hashes, PDF/chunking
provenance, the embedding model, and whether the qrels are human labels or AI
pre-annotations. AI pre-annotations may be used for pipeline dry-runs, but must
not be reported as an independently human-labelled retrieval benchmark.

For a real evaluation, replace the example files with frozen paper chunks and
human relevance labels. Tune retrieval settings on a development split and
report final metrics only on a held-out test split.

## Export a real PDF for annotation

Create a frozen corpus, nine dimension-specific queries, and a TSV containing
every query/chunk pair:

```bash
python -m evals.academic_retrieval.export_pdf_benchmark \
  --pdf /absolute/path/to/paper.pdf \
  --paper-id arxiv:2410.02902 \
  --source-url https://arxiv.org/abs/2410.02902 \
  --output-dir evals/academic_retrieval/datasets/paper-001
```

Fill `relevance_grade` in `annotation_tasks.tsv` using only `0`, `1`, or `2`.
Do not tune a retriever while assigning labels: relevance is a property of the
query and evidence, not of which retrieval system returned the evidence.

The exporter refuses to replace existing artifacts unless `--overwrite` is
explicitly supplied. The manifest records the source PDF hash and chunking
configuration so benchmark results can be reproduced.

After every row has been labelled, validate and compile the annotations:

```bash
python -m evals.academic_retrieval.compile_annotations \
  --annotations evals/academic_retrieval/datasets/paper-001/annotation_tasks.tsv \
  --corpus evals/academic_retrieval/datasets/paper-001/corpus.jsonl \
  --output evals/academic_retrieval/datasets/paper-001/queries.jsonl
```

Compilation fails if any grade is blank, outside 0/1/2, duplicated, references
another paper, or leaves a query without positive evidence. This validation is
intentional: incomplete qrels would make Recall@K misleading.

## Build a multi-paper collection

After compiling each paper independently, freeze a validated collection from a
versioned member spec:

```bash
python -m evals.academic_retrieval.build_collection \
  --spec evals/academic_retrieval/ret07b_five_paper_spec.json \
  --output-dir evals/academic_retrieval/datasets/ret07b-five-paper
```

The collection manifest records every source PDF hash, corpus/query hash,
annotation-provenance hash and label origin. The current RET-07B expansion uses
ChatGPT annotation and ChatGPT review for four papers; it must not be described
as an independent human-labelled or inter-annotator-agreement dataset.

## Freeze paper-disjoint development/test splits

Create the versioned RET-09 split from the frozen five-paper collection:

```bash
python -m evals.academic_retrieval.split_collection \
  --collection-dir evals/academic_retrieval/datasets/ret07b-five-paper \
  --spec evals/academic_retrieval/ret09_split_spec.json \
  --output-dir evals/academic_retrieval/datasets/ret09-five-paper-v1
```

The split builder verifies the parent corpus/query hashes, assigns every paper
exactly once, rejects development/test overlap, and records independent hashes
for both splits. The development manifest permits tuning; the test manifest
explicitly forbids it.

Tune Hybrid parameters using only the development manifest:

```bash
python -m evals.academic_retrieval.tune_retrieval \
  --development-manifest \
    evals/academic_retrieval/datasets/ret09-five-paper-v1/development/manifest.json \
  --candidate-top-k 10 20 40 \
  --rrf-k 20 60 100 \
  --cutoffs 4 10 \
  --primary-metric recall_at_k \
  --embedding-model qwen3-embedding:4b \
  --output \
    evals/academic_retrieval/results/ret09-development-tuning-qwen3-embedding-4b.json
```

The tuning command refuses a test manifest and refuses to select parameters if
any retrieval system fails. Embeddings are cached across grid-search trials.
Do not inspect held-out test metrics until the retriever implementation and
parameters are frozen under RET-10B; test results must never trigger retuning.

## Frozen policy and held-out evaluation

RET-10B freezes `ret10b-hybrid-v1` in
`ret10b_frozen_policy.json`. Its operational values are Hybrid retrieval,
`qwen3-embedding:4b`, `candidate_top_k=20`, `rrf_k=100`, and a production
evidence-depth default of four. Production, Compose and policy-file drift is
covered by tests.

The held-out runner accepts no parameter overrides and requires a frozen policy
plus a test-only manifest:

```bash
python -m evals.academic_retrieval.evaluate_frozen_policy \
  --policy evals/academic_retrieval/ret10b_frozen_policy.json \
  --test-manifest \
    evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --output \
    evals/academic_retrieval/results/ret10b-heldout-test-qwen3-embedding-4b.json
```

The zero-failure held-out run produced Hybrid Recall@10 `0.1824` and nDCG@10
`0.3462`, below BM25 at `0.2555` and `0.4338`. This is a negative generalisation
result, not permission to select BM25 on the test set. Hybrid v1 remains frozen;
changing the default requires a new development set and a new policy version.
The tracked `ret10b_heldout_summary.json` records hashes and the interpretation
boundary without treating mixed-provenance labels as independent human gold.

## Section-only versus frozen Hybrid

RET-11A compares the two production selectors at the frozen evidence depth of
four without invoking the paper-scoring model:

```bash
python -m evals.academic_retrieval.compare_selectors \
  --policy evals/academic_retrieval/ret10b_frozen_policy.json \
  --test-manifest \
    evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --output \
    evals/academic_retrieval/results/ret11a-section-vs-hybrid-heldout.json
```

On the 18 held-out queries, section-only achieved Recall@4 `0.1846`, nDCG@4
`0.6066` and relevant-candidate rate `0.7361`, compared with frozen Hybrid at
`0.0755`, `0.3930` and `0.4861`. Query coverage was `1.0` versus `0.7222`; both
selectors had zero query failures and valid chunk-ID rate `1.0`. Hybrid required
`364.3s` total cold index construction for two papers and averaged `292.0ms` per
warm query, while section sorting averaged `1.3ms`.

Relevant/highly-relevant candidate rates are citation-readiness proxies derived
from qrels. They do not measure which chunks a scoring model actually cites or
whether its claims are supported. The tracked
`ret11a_selector_comparison_summary.json` preserves this boundary. The report
does not change frozen policy v1. Downstream selector-quality impact is reported
separately by RET-11B under `../academic_quality/`: on
the same two held-out papers, section-only had lower ChatGPT-reference score
MAE than frozen Hybrid (`0.471` versus `0.588`). This descriptive two-paper AI
reference result does not authorize test-set retuning or human-quality claims.
