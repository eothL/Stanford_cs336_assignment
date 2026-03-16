# Ablation Study — Transformer LM on OpenWebText

**Base architecture**: L12-H6-D768, ctx=512, tied embeddings, AdamW, cosine LR schedule
**Dataset**: OpenWebText (train/val), vocab 32k
**Metric**: Best validation loss (lower = better)

---

## 1. Activation Function

**Controlled setting**: bs=84, lr_max=0.1, betas=(0.9, 0.95), QK-norm, compiled, AdamW, no x0-mixing, no MVE

| Activation | Best Val Loss | Run ID |
|-----------|:------------:|--------|
| **ramp_relu** | **3.3739** | `86xve0xb` |
| sq_relu | 3.4052 | `rrevga0z` |
| relu | 3.3976¹ | `iillx3mx` |
| swiglu | 3.3578² | `p6uhpowf` |
| gelu (baseline) | 3.5048³ | `e5zj62mj` |

> ¹ relu with x0-mixing (no clean relu run without x0mix at bs=84/lr=0.1)
> ² swiglu with x0-mixing (no clean swiglu run without x0mix at bs=84/lr=0.1)
> ³ gelu at bs=64/lr=0.001 — no gelu run exists at the same bs=84/lr=0.1 setting

**Delta from gelu baseline**: ramp_relu -0.131, swiglu -0.147², sq_relu -0.100

**Confounds**: gelu was never tested at the higher LR/BS regime. swiglu and relu results include x0-mixing. See "Missing runs" below.

---

## 2. QK-Norm

**Controlled setting**: L12-H6-D768, bs=64, lr=0.001, gelu, no compile, betas=(0.9, 0.99)

| QK-Norm | Best Val Loss | Run ID |
|---------|:------------:|--------|
| **True** | **3.5062** | `v7r1hvde` |
| False | 3.5587 | `zn8e8qqk` |

**Delta**: -0.053 (consistent improvement)

Replicated with compile=True: 3.5048 (qk_norm) vs 3.5643 (no qk_norm) → delta -0.060

---

## 3. Weight Tying (Tied Embeddings)

**Controlled setting**: L24-H16-D1024, bs=24, lr=0.001

| Tied | Best Val Loss | Run ID |
|------|:------------:|--------|
| **True** | **4.1033** | `w2no0th3` |
| False | 4.2399 | `qqktefs3` |

**Delta**: -0.137 (tied is better despite fewer parameters)

---

## 4. Torch Compile

**Controlled setting**: L12-H6-D768, bs=64, lr=0.001, gelu, no QK-norm

| Compile | Best Val Loss | Run ID |
|---------|:------------:|--------|
| True | 3.5643 | `xrcat7ka` |
| **False** | **3.5587** | `zn8e8qqk` |

**Delta**: +0.006 (negligible difference — compile is for speed, not quality)

With QK-norm: 3.5048 (compiled) vs 3.5062 (not compiled) → delta -0.001 (identical)

---

## 5. Post-Norm vs Pre-Norm

**Controlled setting**: L12-H6-D768, bs=64, lr=0.001, gelu

| Norm Position | Best Val Loss | Run ID |
|--------------|:------------:|--------|
| **Pre-norm** | **3.5587** | `zn8e8qqk` |
| Post-norm | 7.4272 | `mppg2quk` |

**Delta**: +3.869 (**catastrophic** — loss plateaus immediately)

Replicated twice with QK-norm + compile: bvl=7.427 both times (`p7s9tgiu`, `2uh54jab`)

---

## 6. RoPE (Rotary Position Encoding)

**Controlled setting**: L12-H6-D768, bs=64, gelu, no QK-norm

| RoPE | Best Val Loss | Run ID |
|------|:------------:|--------|
| **Yes** | **3.5587** | `zn8e8qqk` |
| No | 5.6687 | `4gae4prq` |

**Delta**: +2.110 (removing RoPE is severely harmful)

---

## 7. Batch Size

**Controlled setting**: L12-H6-D768, gelu, QK-norm, lr=0.001

| Batch Size | Best Val Loss | Run ID |
|-----------|:------------:|--------|
| 12 | 3.7570 | `zae98ui5` |
| **64** | **3.5062** | `v7r1hvde` |

**Delta**: -0.251 (bs=64 substantially better)

With ramp_relu at higher LR regime:

| Batch Size | Best Val Loss | Run ID |
|-----------|:------------:|--------|
| 64 | 3.4324 | `er8846u8` |
| **84** | **3.3739** | `86xve0xb` |

**Delta**: -0.059 (diminishing returns but still helps)

---

## 8. Learning Rate (Peak)

**Controlled setting**: L12-H6-D768, bs=64, gelu, QK-norm, compiled

| LR Max | Best Val Loss | Run ID |
|--------|:------------:|--------|
| 0.001 | 3.5048 | `e5zj62mj` |
| **0.01** | 3.5819 | `2rxadbgc` |

**Confound**: The lr=0.01 runs used different cosine_cycle_iters and warmup. Later runs at lr=0.1 with ramp_relu performed much better (3.37), but activation also changed.

---

## 9. X0-Mixing

**Controlled setting**: L12-H6-D768, bs=84, lr=0.1, ramp_relu, QK-norm, compiled, betas=(0.9, 0.95), no MVE

| X0-Mix | Best Val Loss | Run ID |
|--------|:------------:|--------|
| **True** | **3.3725** | `l38oe51i` |
| False | 3.3739 | `86xve0xb` |

**Delta**: -0.001 (marginal improvement)

---

## 10. Multi-Value Embeddings (MVE)

**Controlled setting**: L12-H6-D768, bs=84, lr=0.1, ramp_relu, QK-norm, compiled, betas=(0.9, 0.95), no x0-mix

| MVE | Best Val Loss | Run ID |
|-----|:------------:|--------|
| **1** | **3.3340** | `daw7sg1p` |
| 0 | 3.3739 | `86xve0xb` |

**Delta**: -0.040 (adds ~25M params but consistent improvement)

With z_loss: MVE+z_loss=3.3333 (`d30u09jm`) vs no MVE=3.3739 → delta -0.041

---

## 11. Z-Loss

**Controlled setting**: bs=84, lr=0.1, ramp_relu, QK-norm, compiled, MVE=1, betas=(0.9, 0.95)

| Z-Loss | Best Val Loss | Run ID |
|--------|:------------:|--------|
| **0.0001** | **3.3333** | `d30u09jm` |
| 0 | 3.3340 | `daw7sg1p` |

**Delta**: -0.001 (marginal, but best run uses it)

---

## 12. Betas (Adam β₂)

**Controlled setting**: L12-H6-D768, bs=72, sq_relu, QK-norm, no compile

| β₂ | Best Val Loss | Run ID | Notes |
|----|:------------:|--------|-------|
| **0.95** | **3.4552** | `hp7f52dz` | lr_max=0.01 |
| 0.99 | 3.5710 | `mb8r4bdz` | lr_max=0.01 |

**Delta**: -0.116 (β₂=0.95 clearly better with higher LR)

---

## 13. Optimizer

**Controlled setting**: L12-H6-D768, swiglu, QK-norm, compiled

| Optimizer | Best Val Loss | Run ID |
|----------|:------------:|--------|
| **AdamW** | **3.4263** | `s7ycnkh0` |
| Muon+AdamW | 3.6536 | `hvr357h9` |
| SISA | 8.4878 | `nntga9o1` |
| NSISA | 5.5054 | `umldlav8` |

**AdamW dominates.** Muon+AdamW underperforms by ~0.23. SISA/NSISA are broken (likely need different hyperparameters or have bugs).

---

## 14. Cosine Cycle Length

**Controlled setting**: bs=84, lr=0.1, ramp_relu, QK-norm, x0-mix, MVE=1, z_loss=0.0001

| Cosine Iters | Best Val Loss | Steps Run | Run ID |
|-------------|:------------:|:---------:|--------|
| 8000 | 3.3374 | 10000 | `y8iwjtea` |
| **24000** | **3.3323** | 14000 | `ush9w11e` |

**Delta**: -0.005 (longer cycles help when training longer)

---

## 15. Mixed Precision (BF16)

**Controlled setting**: best config (ramp_relu, x0-mix, MVE=1, bs=84, lr=0.1)

| Precision | Best Val Loss | Run ID | Notes |
|----------|:------------:|--------|-------|
| **FP32** | **3.3323** | `ush9w11e` | 14k steps |
| BF16 | 3.3841 | `4daxc5rl` | 11.7k steps |

**Delta**: +0.052 (BF16 slightly worse, but also ran fewer steps and had compile issues)

---

## Summary: Feature Impact Ranking

| Rank | Feature | Δ Val Loss | Verdict |
|:----:|---------|:----------:|---------|
| 1 | **Pre-norm** (vs post-norm) | -3.87 | Mandatory |
| 2 | **RoPE** (vs no RoPE) | -2.11 | Mandatory |
| 3 | **Batch size** (64 vs 12) | -0.25 | Critical |
| 4 | **Activation** (ramp_relu vs gelu) | -0.13 | Important |
| 5 | **Weight tying** | -0.14 | Important |
| 6 | **β₂=0.95** (vs 0.99) | -0.12 | Important |
| 7 | **QK-norm** | -0.05 | Moderate |
| 8 | **MVE** (1 value embed) | -0.04 | Moderate |
| 9 | **Z-loss** | -0.001 | Marginal |
| 10 | **X0-mixing** | -0.001 | Marginal |
| 11 | **Compile** | ~0 | Speed only |

---

## Missing Runs for Complete Ablation

The following runs would strengthen the ablation study by eliminating confounds:

### High Priority (clean A/B comparisons missing)

1. **gelu at bs=84, lr_max=0.1, betas=(0.9, 0.95), QK-norm, compiled**
   - Why: The gelu baseline is at bs=64/lr=0.001 — can't fairly compare activation functions across different LR/BS regimes
   - Config: `activation_fcn=gelu`, everything else same as `86xve0xb`

2. **swiglu at bs=84, lr_max=0.1, betas=(0.9, 0.95), QK-norm, compiled, NO x0-mix**
   - Why: Current swiglu result (`p6uhpowf`, bvl=3.3578) includes x0-mixing — can't isolate swiglu vs ramp_relu
   - Config: same as `86xve0xb` but `activation_fcn=swiglu`

3. **relu at bs=84, lr_max=0.1, betas=(0.9, 0.95), QK-norm, compiled, NO x0-mix**
   - Why: Same issue — current relu result includes x0-mixing
   - Config: same as `86xve0xb` but `activation_fcn=relu`

### Medium Priority (strengthening existing comparisons)

4. **sq_relu at bs=84, lr_max=0.1, betas=(0.9, 0.95), QK-norm, compiled, NO x0-mix**
   - Why: Current sq_relu at bs=84 (`rrevga0z`, bvl=3.4052) exists but uses a different config hash — confirming with identical config strengthens the comparison

5. **BF16 run with --no-compile**
   - Why: Current BF16 run (`4daxc5rl`) used compile=True which had NaN issues. A clean BF16+no-compile run would isolate precision impact from compile bugs
   - Config: same as `ush9w11e` but `compute_dtype=bfloat16, compile=False`

6. **Longer training (24k steps) for ramp_relu WITHOUT x0-mix/MVE/z-loss**
   - Why: Best baseline run (`86xve0xb`) only ran 10k steps with cosine_cycle_iters=8000. The best run (`ush9w11e`) ran 14k with cycle=24k — part of the improvement may be from longer training, not features
   - Config: same as `86xve0xb` but `cosine_cycle_iters=24000`, train for 14k+ steps

### Low Priority (nice-to-have)

7. **QK-norm ablation at high LR regime** (bs=84, lr=0.1, ramp_relu, no QK-norm)
   - Why: QK-norm was only ablated at the old bs=64/lr=0.001 gelu setting. It may matter more (or less) at high LR

8. **Weight tying ablation at GPT-2 small** (L12-H6 instead of L24-H16)
   - Why: Current tying ablation is on the medium model — confirming on the small model would be cleaner
