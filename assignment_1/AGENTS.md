
This project CS336 Assignment 1 (Basics) implements the core building blocks of modern
transformer language models from scratch, including NN layers, attention mechanism, optimization algorithm, and tokenization. The assignment is structured around implementing various functions in the `adapters.py` file to pass comprehensive test suites. 

Environment:
- A dedicated env exists for the agent; prefer `python` (not `python3`) and use `uv` for package management.

Key code:
- tests/adapters.py: required functions (test targets).
- basic/Tokenizer.py: tokenizer class (currently stubs).
- basic/train_bpe.py: main BPE training logic (heap-based).
- basic/pretokenization.py: chunking + multiprocessing pretokenization.
- basic/assigment_question.py: assignment answers / training script.


Progress so far (Theo + prior agents):
- Implemented split-special-token pretokenization in `basic/pretokenization.py`:
  - chunk boundaries avoid splitting `<|endoftext|>` across workers
  - pattern excludes the split token; token counted explicitly.
- Contains debug scripts in `local/` or can be used to vibe test anything quickly. No restriction folder in term of implementation.
- Heap performance experiments done in `basic/train_bpe_copy.py` / `local/train_bpe_test.py`
  (stale-node rebuild heuristics, `heapq` variant, profiling).

Remaining work (use ✅:done/🟡: currently doing/⏳: not started):

1) Basic Neural Network Components
- ⏳ Linear (`run_linear`)
- ⏳ Embedding (`run_embedding`)
- ⏳ RMSNorm (`run_rmsnorm`)
- ⏳ SiLU (`run_silu`)

1) Attention Mechanisms
- ⏳ Scaled dot‑product attention (`run_scaled_dot_product_attention`)
- ⏳ Multi‑head self‑attention (`run_multihead_self_attention`)
- ⏳ RoPE (`run_rope`)
- ⏳ Multi‑head self‑attention w/ RoPE (`run_multihead_self_attention_with_rope`)

1) Feed‑Forward Networks
- ⏳ SwiGLU (`run_swiglu`)

1) Transformer Architecture
- ⏳ Transformer block (`run_transformer_block`)
- ⏳ Transformer LM (`run_transformer_lm`)

1) Training Infrastructure
- ⏳ Batch sampling (`run_get_batch`)
- ⏳ Softmax (`run_softmax`)
- ⏳ Cross‑entropy (`run_cross_entropy`)
- ⏳ Gradient clipping (`run_gradient_clipping`)

1) Optimization
- ⏳ AdamW (`get_adamw_cls`)
- ⏳ Cosine LR schedule (`run_get_lr_cosine_schedule`)

1) Model Serialization
- ⏳ Save checkpoint (`run_save_checkpoint`)
- ⏳ Load checkpoint (`run_load_checkpoint`)

1) Tokenization
- 🟡 Tokenizer class (`basic/Tokenizer.py`):
  - `from_files`, `encode`, `encode_iterable`, `decode`
- 🟡 BPE training (heap‑based) exists but needs final TinyStories run + output files

1) BPE Training Deliverables (TinyStories)
- 🟡 Train on TinyStories, vocab=10,000, include `<|endoftext|>`
- 🟡 Serialize vocab/merges (GPT‑2 bytes→unicode)
- 🟡 Report training time + peak memory
- 🟡 Identify longest token + comment if it makes sense
- 🟡 Provide 1–2 sentence answer for profiling (part b)

