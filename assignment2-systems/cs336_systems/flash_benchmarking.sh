#!/bin/bash
# Problem (flash_benchmarking): Triton FlashAttention-2 vs regular PyTorch attention.
# Run on a single H100.
#
# Usage:
#   bash cs336_systems/flash_benchmarking.sh [output_dir]
#
# Debug a small grid first with the CLI flags, e.g.:
#   uv run python -m cs336_systems.flash_benchmarking --seq-lens 128 256 --d-heads 64 --dtypes bfloat16

set -euo pipefail

OUT_DIR="${1:-flash_benchmarking_results}"
mkdir -p "$OUT_DIR"

uv run python -m cs336_systems.flash_benchmarking \
    --results-file "${OUT_DIR}/results.md" \
2>&1 | tee "${OUT_DIR}/benchmark.log"

echo "=== Done! Table in ${OUT_DIR}/results.md ==="
