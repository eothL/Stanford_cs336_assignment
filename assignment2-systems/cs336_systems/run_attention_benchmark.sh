#!/bin/bash
# Benchmark PyTorch attention at different scales (section 1.2.1)
# Usage: bash run_attention_benchmark.sh [output_dir]

OUT_DIR="${1:-attention_benchmark_results}"
RESULTS_FILE="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

uv run python -m cs336_systems.benchmark_attention \
    --device cuda \
    --results-file "$RESULTS_FILE" \
2>&1 | tee "${OUT_DIR}/benchmark.log"

echo "=== Generating markdown table ==="
uv run python -m cs336_systems.results_to_markdown "$RESULTS_FILE" -o "${OUT_DIR}/results.md"

echo "=== Done! Results in ${OUT_DIR}/ ==="
