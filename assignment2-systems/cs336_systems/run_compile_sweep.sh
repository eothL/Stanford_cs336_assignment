#!/bin/bash
# Sweep: torch.compile across all model sizes (section 1.3 - torch_compile problem b)
# Usage: bash run_compile_sweep.sh [output_dir]

OUT_DIR="${1:-compile_results}"
RESULTS_FILE="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

> "$RESULTS_FILE"

MODELS=(
    "small   768  3072  12 12"
    "medium  1024 4096  24 16"
    "large   1280 5120  36 20"
    "xl      1600 6400  48 25"
    "2.7B    2560 10240 32 32"
)

CTX_LENGTHS=(128 256 512 1024)
MODES=("forward" "full" "train")

for model_spec in "${MODELS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$model_spec"

    for ctx_len in "${CTX_LENGTHS[@]}"; do
        for mode in "${MODES[@]}"; do
            tag="${name}_ctx${ctx_len}_${mode}_compiled"
            echo "=== $tag ==="
            if uv run python -m cs336_systems.benchmark \
                --device cuda \
                --d-model "$d_model" \
                --d-ff "$d_ff" \
                --num-layers "$num_layers" \
                --num-heads "$num_heads" \
                --context-length "$ctx_len" \
                --mode "$mode" \
                --warmup-step 5 \
                --rep 10 \
                --compile \
                --results-file "$RESULTS_FILE" \
            2>&1 | tee "${OUT_DIR}/${tag}.log"; then
                echo ""
            else
                echo "!!! FAILED (likely OOM): $tag !!!"
            fi
        done
    done
done

echo "=== Generating markdown table ==="
uv run python -m cs336_systems.results_to_markdown "$RESULTS_FILE" -o "${OUT_DIR}/results.md"

echo "=== Done! Results in ${OUT_DIR}/ ==="
