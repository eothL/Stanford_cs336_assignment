#!/bin/bash
# Memory profiling sweep (section 1.1.6)
# Usage: bash run_memory_profiling.sh [output_dir]

OUT_DIR="${1:-memory_profiling_results}"
RESULTS_FILE="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

> "$RESULTS_FILE"

CTX_LENGTHS=(128 256 512)

# ── (a) 2.7B model: memory snapshots for forward and train ──────────
echo "=== 2.7B forward pass (memory snapshot) ==="
uv run python -m cs336_systems.benchmark \
    --device cuda \
    --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
    --context-length 256 \
    --mode forward \
    --warmup-step 2 --rep 1 \
    --memory-profiling \
    --results-file "$RESULTS_FILE" \
2>&1 | tee "${OUT_DIR}/2.7B_forward_snapshot.log"
mv memory_snapshot.pickle "${OUT_DIR}/2.7B_forward_snapshot.pickle" 2>/dev/null

echo "=== 2.7B train step (memory snapshot) ==="
uv run python -m cs336_systems.benchmark \
    --device cuda \
    --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
    --context-length 256 \
    --mode train \
    --warmup-step 2 --rep 1 \
    --memory-profiling \
    --results-file "$RESULTS_FILE" \
2>&1 | tee "${OUT_DIR}/2.7B_train_snapshot.log"
mv memory_snapshot.pickle "${OUT_DIR}/2.7B_train_snapshot.pickle" 2>/dev/null

# ── (b) 2.7B model: peak memory per context length (forward + train) ──
for ctx_len in "${CTX_LENGTHS[@]}"; do
    echo "=== 2.7B forward ctx=${ctx_len} ==="
    uv run python -m cs336_systems.benchmark \
        --device cuda \
        --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
        --context-length "$ctx_len" \
        --mode forward \
        --warmup-step 2 --rep 1 \
        --memory-profiling \
        --results-file "$RESULTS_FILE" \
    2>&1 | tee "${OUT_DIR}/2.7B_ctx${ctx_len}_forward.log"
    mv memory_snapshot.pickle "${OUT_DIR}/2.7B_ctx${ctx_len}_forward.pickle" 2>/dev/null

    echo "=== 2.7B train ctx=${ctx_len} ==="
    uv run python -m cs336_systems.benchmark \
        --device cuda \
        --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
        --context-length "$ctx_len" \
        --mode train \
        --warmup-step 2 --rep 1 \
        --memory-profiling \
        --results-file "$RESULTS_FILE" \
    2>&1 | tee "${OUT_DIR}/2.7B_ctx${ctx_len}_train.log"
    mv memory_snapshot.pickle "${OUT_DIR}/2.7B_ctx${ctx_len}_train.pickle" 2>/dev/null
done

# ── (c) 2.7B model: mixed precision (forward + train) ───────────────
for ctx_len in "${CTX_LENGTHS[@]}"; do
    echo "=== 2.7B BF16 forward ctx=${ctx_len} ==="
    uv run python -m cs336_systems.benchmark \
        --device cuda \
        --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
        --context-length "$ctx_len" \
        --mode forward \
        --warmup-step 2 --rep 1 \
        --mixed-precision \
        --memory-profiling \
        --results-file "$RESULTS_FILE" \
    2>&1 | tee "${OUT_DIR}/2.7B_ctx${ctx_len}_forward_bf16.log"
    mv memory_snapshot.pickle "${OUT_DIR}/2.7B_ctx${ctx_len}_forward_bf16.pickle" 2>/dev/null

    echo "=== 2.7B BF16 train ctx=${ctx_len} ==="
    uv run python -m cs336_systems.benchmark \
        --device cuda \
        --d-model 2560 --d-ff 10240 --num-layers 32 --num-heads 32 \
        --context-length "$ctx_len" \
        --mode train \
        --warmup-step 2 --rep 1 \
        --mixed-precision \
        --memory-profiling \
        --results-file "$RESULTS_FILE" \
    2>&1 | tee "${OUT_DIR}/2.7B_ctx${ctx_len}_train_bf16.log"
    mv memory_snapshot.pickle "${OUT_DIR}/2.7B_ctx${ctx_len}_train_bf16.pickle" 2>/dev/null
done

# generate markdown table
echo "=== Generating markdown table ==="
uv run python -m cs336_systems.results_to_markdown "$RESULTS_FILE" -o "${OUT_DIR}/results.md"

echo "=== All done! Results in ${OUT_DIR}/ ==="
echo "Load .pickle files at https://pytorch.org/memory_viz"
