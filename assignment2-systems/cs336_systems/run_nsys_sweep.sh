#!/bin/bash
# Sweep script for nsys_profile problem (section 1.1.4)
# Profiles all model sizes x context lengths x modes
#
# Usage: bash run_nsys_sweep.sh [output_dir]

set -e

OUT_DIR="${1:-nsys_results}"
RESULTS_FILE="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

# clear previous results
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
MODES=("forward" "full" "train")

for model_spec in "${MODELS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$model_spec"

    for ctx_len in "${CTX_LENGTHS[@]}"; do
        for mode in "${MODES[@]}"; do
            tag="${name}_ctx${ctx_len}_${mode}"
            echo "=== Profiling: $tag ==="

            # Run with nsys profiling
            uv run nsys profile \
                -o "${OUT_DIR}/${tag}" \
                --force-overwrite true \
                python -m cs336_systems.benchmark \
                    --device cuda \
                    --d-model "$d_model" \
                    --d-ff "$d_ff" \
                    --num-layers "$num_layers" \
                    --num-heads "$num_heads" \
                    --context-length "$ctx_len" \
                    --mode "$mode" \
                    --warmup-step 5 \
                    --rep 10 \
                    --results-file "$RESULTS_FILE" \
                2>&1 | tee "${OUT_DIR}/${tag}.log"

            echo ""
        done
    done
done

# Separate run with NVTX annotations for question (e): softmax vs matmul
echo "=== Profiling: small model with NVTX annotations ==="
uv run nsys profile \
    -o "${OUT_DIR}/small_ctx128_forward_annotated" \
    --force-overwrite true \
    python -m cs336_systems.benchmark \
        --device cuda \
        --d-model 768 --d-ff 3072 --num-layers 12 --num-heads 12 \
        --context-length 128 \
        --mode forward \
        --warmup-step 5 \
        --rep 10 \
        --annotate \
    2>&1 | tee "${OUT_DIR}/small_ctx128_forward_annotated.log"

# convert results to markdown table
echo "=== Generating markdown table ==="
uv run python -m cs336_systems.results_to_markdown "$RESULTS_FILE" -o "${OUT_DIR}/results.md"

echo "=== All done! Results in ${OUT_DIR}/ ==="
