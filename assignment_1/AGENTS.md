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

### Key code:
- `basic/` folder contains all the code wrote from scratch by the USER
- `basic/Tokenizer.py`: tokenizer class (currently stubs).
- `basic/train_bpe.py`: main BPE training logic (heap-based).
- `basic/pretokenization.py`: chunking + multiprocessing pretokenization.
- `basic/assigment_question.py`: script to answer assignment question.
- `tests/`: contains all the test file
- `tests/adapters.py`: required functions (test targets).
- `local/`: for more experiment

### Progress so far (Theo + prior agents):
- Implemented pretokenization for the BPE algorithm in `basic/pretokenization.py`:
  - Using parallelism to speed up the process.
  - pattern excludes the split token; token counted explicitly.
- Implemented and tested the train_bpe algorithm in `basic/train_bpe` with the function `train_bpe_heap()`:
  - Leveraging heap structure to find the best pair of bytes instead of max() function. Not using heapq library. 
  - Missing how to save the vocab and merge list to reuse after.
- Finishing `basic/assignment_question.py` to answer question.
- Started implementing the Tokenizer class in.

## Core Tasks
✅:done/🟡: currently doing

### 1. **Basic Neural Network Components**

#### Linear Layer (`run_linear`)
- **Task**: Implement a linear transformation layer
- **Parameters**: 
  - `d_in`: Input dimension size
  - `d_out`: Output dimension size  
  - `weights`: Weight matrix of shape `(d_out, d_in)`
  - `in_features`: Input tensor of shape `(..., d_in)`
- **Goal**: Apply matrix multiplication to transform input features

#### Embedding Layer (`run_embedding`)
- **Task**: Implement token embedding lookup
- **Parameters**:
  - `vocab_size`: Number of vocabulary items
  - `d_model`: Embedding dimension
  - `weights`: Embedding matrix of shape `(vocab_size, d_model)`
  - `token_ids`: Token indices to look up
- **Goal**: Retrieve embeddings for given token IDs

#### RMSNorm (`run_rmsnorm`)
- **Task**: Implement Root Mean Square Layer Normalization
- **Parameters**:
  - `d_model`: Input dimension
  - `eps`: Numerical stability constant
  - `weights`: RMSNorm weights
  - `in_features`: Input tensor
- **Goal**: Normalize inputs using RMS normalization

#### SiLU Activation (`run_silu`)
- **Task**: Implement Swish/SiLU activation function
- **Formula**: `x * sigmoid(x)`
- **Goal**: Apply element-wise SiLU activation

### 2. **Attention Mechanisms**

#### Scaled Dot Product Attention (`run_scaled_dot_product_attention`)
- **Task**: Implement the core attention mechanism
- **Parameters**:
  - `Q`: Query tensor
  - `K`: Key tensor  
  - `V`: Value tensor
  - `mask`: Optional attention mask
- **Goal**: Compute attention weights and apply to values

#### Multi-Head Self-Attention (`run_multihead_self_attention`)
- **Task**: Implement multi-head attention without RoPE
- **Parameters**:
  - `d_model`: Model dimension
  - `num_heads`: Number of attention heads
  - Projection weights for Q, K, V, and output
- **Goal**: Apply multi-head attention with batched operations

#### Multi-Head Self-Attention with RoPE (`run_multihead_self_attention_with_rope`)
- **Task**: Implement multi-head attention with Rotary Position Embedding
- **Additional Parameters**:
  - `max_seq_len`: Maximum sequence length
  - `theta`: RoPE parameter
  - `token_positions`: Position indices
- **Goal**: Apply RoPE to queries and keys before attention

#### RoPE Implementation (`run_rope`)
- **Task**: Implement Rotary Position Embedding
- **Parameters**:
  - `d_k`: Key/query dimension
  - `theta`: RoPE parameter
  - `max_seq_len`: Maximum sequence length
  - `in_query_or_key`: Input tensor
  - `token_positions`: Position indices
- **Goal**: Apply rotary position encoding to input tensors

### 3. **Feed-Forward Networks**

#### SwiGLU (`run_swiglu`)
- **Task**: Implement SwiGLU feed-forward network
- **Parameters**:
  - `d_model`: Input/output dimension
  - `d_ff`: Hidden dimension
  - `w1_weight`, `w2_weight`, `w3_weight`: Weight matrices
- **Formula**: `w2 @ (SiLU(w1 @ x) * w3 @ x)`
- **Goal**: Apply SwiGLU transformation

### 4. **Transformer Architecture**

#### Transformer Block (`run_transformer_block`)
- **Task**: Implement a complete transformer block
- **Components**:
  - Multi-head self-attention with RoPE
  - SwiGLU feed-forward network
  - RMSNorm layers
- **Goal**: Process input through one transformer layer

#### Transformer Language Model (`run_transformer_lm`)
- **Task**: Implement full transformer language model
- **Components**:
  - Token embeddings
  - Multiple transformer blocks
  - Final layer norm
  - Language modeling head
- **Goal**: Generate logits for next token prediction

### 5. **Training Infrastructure**

#### Data Loading (`run_get_batch`)
- **Task**: Sample training batches from dataset
- **Parameters**:
  - `dataset`: 1D array of token IDs
  - `batch_size`: Batch size
  - `context_length`: Sequence length
  - `device`: PyTorch device
- **Goal**: Create input sequences and corresponding labels

#### Loss Functions
- **Softmax** (`run_softmax`): Implement softmax with numerical stability
- **Cross-Entropy** (`run_cross_entropy`): Compute cross-entropy loss
- **Gradient Clipping** (`run_gradient_clipping`): Clip gradients by L2 norm

### 6. **Optimization**

#### AdamW Optimizer (`get_adamw_cls`)
- **Task**: Implement AdamW optimizer from scratch
- **Features**:
  - Momentum (β1) and RMSprop (β2) terms
  - Weight decay
  - Bias correction
- **Goal**: Provide efficient optimization for transformer training

#### Learning Rate Scheduling (`run_get_lr_cosine_schedule`)
- **Task**: Implement cosine learning rate schedule with warmup
- **Parameters**:
  - `max_learning_rate`: Peak learning rate
  - `min_learning_rate`: Minimum learning rate
  - `warmup_iters`: Warmup iterations
  - `cosine_cycle_iters`: Cosine cycle length
- **Goal**: Provide learning rate schedule for training

### 7. **Model Serialization**

#### Checkpointing
- **Save** (`run_save_checkpoint`): Serialize model, optimizer, and iteration
- **Load** (`run_load_checkpoint`): Restore model state from checkpoint
- **Goal**: Enable training resumption and model persistence

### 8. **Tokenization** ✅

#### BPE Tokenizer (`get_tokenizer`) ✅
- **Task**: Implement Byte Pair Encoding tokenizer
- **Features**:
  - Vocabulary lookup
  - Merge operations
  - Special token handling
  - Encode/decode functionality
- **Goal**: Convert text to token IDs and back

#### BPE Training (`run_train_bpe`) ✅
- **Task**: Train BPE tokenizer from corpus
- **Parameters**:
  - `input_path`: Training corpus path
  - `vocab_size`: Target vocabulary size
  - `special_tokens`: Special tokens to preserve
- **Goal**: Learn vocabulary and merge rules from data
