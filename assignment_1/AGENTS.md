This project CS336 Assignment 1 (Basics) implements the core building blocks of modern transformer language models from scratch, including NN layers, attention mechanism, optimization algorithm, and tokenization. The assignment is structured around implementing various functions in the `adapters.py` file to pass comprehensive test suites. 

## Key Learning Objectives
1. **Deep Understanding**: Implement core ML components from scratch
2. **Numerical Stability**: Handle edge cases and overflow scenarios  
3. **Efficiency**: Optimize for both speed and memory usage
4. **Architecture Knowledge**: Understand transformer internals
5. **Training Pipeline**: Complete ML training infrastructure

This assignment provides hands-on experience with the fundamental building blocks of modern language models, requiring both theoretical understanding and practical implementation skills.

## Coding information
### Environment:
- prefer `python` (not `python3`) and use `uv` for package management.

### Current execution context
- Model training is performed in a remote session.
- Current checkpoint naming in `basic/train.py`: `result_<run_name>_<run_number>_<epoch>.pth` under `basic/artifacts/experiment_<run_name>/`.

### Key code:
- `basic/model.py`: core NN/transformer implementations (Linear, Embedding, RMSNorm, RoPE, SDPA, MHA, transformer block/LM, loss, optimizer pieces).
- `basic/train.py`: training-loop support utilities (`data_loading`/`run_get_batch` path, checkpoint save/load helpers, and training-script wiring surface).
- `tests/adapters.py`: required adapter entry points for the assignment tests (main implementation targets).
- `basic/Tokenizer.py`: BPE tokenizer class (`encode`/`decode`, merge application, special token handling).
- `basic/train_bpe.py`: main BPE training pipeline (heap-based merge selection and vocab/merge construction).
- `basic/pretokenization.py`: corpus chunking + multiprocessing pretokenization/counting.
- `basic/bytes_utils.py`: byte-level helper utilities used in tokenizer/BPE workflow.
- `basic/assignment_question.py`: assignment-specific analysis/answers and related experiments.
- `tests/test_data.py` and `tests/test_serialization.py`: data-loading and checkpointing validation.
- `tests/test_model.py`, `tests/test_nn_utils.py`, `tests/test_optimizer.py`, `tests/test_tokenizer.py`, `tests/test_train_bpe.py`: primary architecture/optimizer/tokenizer test suites.
- `tests/fixtures/` and `tests/_snapshots/`: reference assets and expected outputs for tests.
- `local/`: scratch space for experiments and debugging scripts.

### Progress so far (Theo + prior agents)
- 2026-02-13: completed data batch sampling and checkpoint save/load paths (`run_get_batch`, `run_save_checkpoint`, `run_load_checkpoint`) with targeted data/serialization tests.
- 2026-02-13: completed cosine LR scheduling and gradient clipping integration.
- 2026-02-12: focused on transformer-core completion (RoPE, SDPA, MHA, transformer block, LM wiring, and cross-entropy reasoning); resolved several adapter/model weight-loading and boundary-contract issues.
- 2026-02-11 to 2026-02-10: resolved key correctness issues in attention and masking logic (Q/K/V handling, causal masking, shape contracts).
- Remaining adapter `NotImplementedError` stubs in assignment entry points: none.
- Current risk: integration drift from naming/key mismatches across adapters, model modules, training configs, and checkpoints.


## Core Tasks
✅: done / 🟡: currently doing / ⬜: not started

### 1. Architecture
- ✅ Core neural layers in `basic/model.py`: `Linear`, `Embedding`, `RMSNorm`, `SiLU`, `SwiGLU`.
- ✅ Attention stack in `basic/model.py` and adapters: SDPA, MHA, RoPE, MHA+RoPE.
- ✅ End-to-end transformer structure: `transformer_block`, `TransformerLM`, LM head/logits path.
- ✅ Tokenization/BPE path: `basic/Tokenizer.py`, `basic/train_bpe.py`, `basic/pretokenization.py`.
- 🟡 Ongoing architecture reliability: keep adapter/model key conventions aligned to avoid silent load or eval mismatches.

### 2. Training
- ✅ Training data pipeline and batch sampling (`load_tokens`, `get_batch`, `run_get_batch`).
- ✅ Loss and optimization primitives: softmax, cross-entropy, gradient clipping, AdamW, cosine LR schedule.
- ✅ Checkpoint serialization/resume (`save_checkpoint`, `load_checkpoint`) and periodic checkpoint writes from `basic/train.py`.
- ✅ Config-driven training runs via `configs/*.yaml` and CLI override support.
- 🟡 Remote-first training operations: continue standardizing run naming, artifact directories, and sync workflow to local.

### 3. Inference
- ⬜ Add a dedicated inference entrypoint in `local/` (recommended: `local/generate.py`) that loads tokenizer + checkpoint + model config.
- ⬜ Define a stable prompt-to-generation contract (inputs: prompt, max tokens, temperature/top-k; outputs: text + metadata).
- ⬜ Add local smoke tests for checkpoint load and short generation after artifact sync.
- ⬜ Add a minimal evaluation workflow (quick perplexity/sample quality pass on validation snippets).

