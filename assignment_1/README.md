This project is an end-to-end implementation pipeline from training to inference of a GPT-2 like model for language modeling using Transformer architecture.

We obtained a loss function of 3.3 on the openwebtext using the following architecture :
12 layers with 6 heads and 768 for the hidden dimension.
Transformer blocks: 
- Causal Multi-head attention with QKNorm and learnable temperature tau 
- FFN with ramp_relu activation function, as custom function that use ReLU and square ReLU with alpha following a cosine schedule.

$$
\text{activation function}=\alpha \text{ReLU(x)}+(1-\alpha)\text{ReLU(x)}^{2}
$$
- tied the embedding and LM head layer. 
- 512 for the context length 
- using AdamW as an optimizer 
- torch.compile to accelerate the training

for more information look at the `ablation_study.md`

using this config to reproduce the result and seed 93
```bash
uv run -m basic.train --config configs/train_owt_small_tied_qk_norm_compiled.yaml --device cuda:0 \
--optimizer-mode adamw \
--activation-fcn ramp_relu \
--cosine-cycle-iters 24000 \
--lr-max 0.1 \
--lr-min 0.00001 \
--betas 0.9 0.95 \
--batch-size 84 \
--use-x0-mixing \
--num-value-embeddings 1 \
--value-embedding-pattern cycle \
```