#!/usr/bin/env bash
set -euo pipefail

benchmark_output_dir="${1:-artifacts/academic-benchmarks}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${benchmark_output_dir}"

"${python_bin}" -m evals.verify_academic_benchmarks \
  > "${benchmark_output_dir}/versioned-smoke.json"

"${python_bin}" -m evals.academic_retrieval.tune_retrieval \
  --development-manifest evals/academic_retrieval/datasets/ret09-five-paper-v1/development/manifest.json \
  --candidate-top-k 10 20 40 \
  --rrf-k 20 60 100 \
  --cutoffs 4 10 \
  --primary-metric recall_at_k \
  --embedding-model qwen3-embedding:4b \
  --output "${benchmark_output_dir}/ret09-development.json"

"${python_bin}" -m evals.academic_retrieval.evaluate_frozen_policy \
  --policy evals/academic_retrieval/ret10b_frozen_policy.json \
  --test-manifest evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --output "${benchmark_output_dir}/ret10b-heldout.json"

"${python_bin}" -m evals.academic_retrieval.compare_selectors \
  --policy evals/academic_retrieval/ret10b_frozen_policy.json \
  --test-manifest evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --output "${benchmark_output_dir}/ret11a-selectors.json"

"${python_bin}" -m evals.academic_quality.compare_selector_quality \
  --selector-report "${benchmark_output_dir}/ret11a-selectors.json" \
  --test-manifest evals/academic_retrieval/datasets/ret09-five-paper-v1/test/manifest.json \
  --checkpoint "${benchmark_output_dir}/ret11b-checkpoint.json" \
  --output "${benchmark_output_dir}/ret11b-selector-quality.json" \
  --model qwen3.5:9b \
  --concurrency 1

"${python_bin}" -m evals.academic_quality.calibrate_contradiction_threshold \
  --checkpoint "${benchmark_output_dir}/data05b-checkpoint.json" \
  --output "${benchmark_output_dir}/data05b-calibration.json" \
  --model qwen3.5:9b

"${python_bin}" -m evals.academic_quality.measure_score_reliability \
  --checkpoint "${benchmark_output_dir}/eval02a-checkpoint.json" \
  --output "${benchmark_output_dir}/eval02a-reliability.json" \
  --run-count 1 \
  --model qwen3.5:9b \
  --concurrency 1
"${python_bin}" -m evals.academic_quality.measure_score_reliability \
  --checkpoint "${benchmark_output_dir}/eval02a-checkpoint.json" \
  --output "${benchmark_output_dir}/eval02a-reliability.json" \
  --run-count 1 \
  --model qwen3.5:9b \
  --concurrency 1 \
  --retry-terminal

"${python_bin}" -m evals.academic_quality.measure_validation_accuracy \
  --checkpoint "${benchmark_output_dir}/eval02b-checkpoint.json" \
  --output "${benchmark_output_dir}/eval02b-validation.json" \
  --model qwen3.5:9b
"${python_bin}" -m evals.academic_quality.measure_validation_accuracy \
  --checkpoint "${benchmark_output_dir}/eval02b-checkpoint.json" \
  --output "${benchmark_output_dir}/eval02b-validation.json" \
  --model qwen3.5:9b \
  --retry-failures

"${python_bin}" -m evals.academic_quality.run_development_ablations \
  --checkpoint "${benchmark_output_dir}/eval03-checkpoint.json" \
  --output "${benchmark_output_dir}/eval03-ablations.json" \
  --model qwen3.5:9b
"${python_bin}" -m evals.academic_quality.run_development_ablations \
  --checkpoint "${benchmark_output_dir}/eval03-checkpoint.json" \
  --output "${benchmark_output_dir}/eval03-ablations.json" \
  --model qwen3.5:9b \
  --retry-failures

"${python_bin}" -m evals.validate_scheduled_benchmark_run \
  "${benchmark_output_dir}"
