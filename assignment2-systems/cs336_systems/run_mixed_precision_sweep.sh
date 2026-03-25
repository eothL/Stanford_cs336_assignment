#!/bin/bash
# Sweep: mixed precision vs full precision across all model sizes
# Runs forward + backward (full mode) for each config, with and without --mixed-precision
#
# Usage: bash run_mixed_precision_sweep.sh [output_dir]

set -e

OUT_DIR="${1:-mixed_precision_results}"
RESULTS_FILE="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

> "$RESULTS_FILE"

# Table 1 model configs: name d_model d_ff num_layers num_heads
MODELS=(
    "small   768  3072  12 12"
    "medium  1024 4096  24 16"
    "large   1280 5120  36 20"
    "xl      1600 6400  48 25"
    "2.7B    2560 10240 32 32"
)

CTX_LENGTHS=(128 256 512 1024)

for model_spec in "${MODELS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$model_spec"

    for ctx_len in "${CTX_LENGTHS[@]}"; do
        # BF16 mixed precision
        tag="${name}_ctx${ctx_len}_bf16"
        echo "=== $tag ==="
        uv run python -m cs336_systems.benchmark \
            --device cuda \
            --d-model "$d_model" \
            --d-ff "$d_ff" \
            --num-layers "$num_layers" \
            --num-heads "$num_heads" \
            --context-length "$ctx_len" \
            --mode full \
            --warmup-step 5 \
            --rep 10 \
            --mixed-precision \
            --results-file "$RESULTS_FILE" \
        2>&1 | tee "${OUT_DIR}/${tag}.log"

        echo ""
    done
done

# generate markdown table
echo "=== Generating markdown table ==="
uv run python -m cs336_systems.results_to_markdown "$RESULTS_FILE" -o "${OUT_DIR}/results.md"

echo "=== All done! Results in ${OUT_DIR}/ ==="
