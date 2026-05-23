# H100 vs RTX 3090 — FlashAttention-2 Triton vs naive PyTorch

Cross-GPU comparison of the `flash_benchmarking` sweep. Raw results: `benckmark_h100.md` (H100) and `benchmark.md` (RTX 3090).
Grid: dtype ∈ {bf16, fp32}, `d_head` ∈ {16, 32, 64, 128}, `seq_len` 128 → 65536, batch=1, causal. Times in ms (lower = better); `peak` in MB; `full` = fwd + bwd.

## TL;DR

- **H100 is ~16× faster than the 3090 for naive PyTorch attention, but only ~10× faster for Flash** at large `seq_len`. The weaker GPU benefits *more* from Flash.
- **Flash's speedup over PyTorch is ~3.4× on the 3090 but a flat ~2× on the H100.** The H100's bandwidth papers over naive attention's inefficiency.
- **The OOM cliff just moved, it didn't disappear.** 3090 PyTorch dies after `seq=16384`; H100 PyTorch bf16 survives all the way to `65536` (61.8 GB peak), fp32 to `32768`. Flash OOMs on *neither* GPU.
- **Peak memory is identical on both GPUs** — it's a property of the algorithm.

## 1. Raw speed: how much does the H100 buy you?

`full` time (fwd + bwd), `d_head=64`, bf16:

| seq_len | 3090 PyTorch | H100 PyTorch | H100 gain | 3090 Flash | H100 Flash | H100 gain |
|---|---|---|---|---|---|---|
| 1024 | 1.14 | 0.557 | 2.0× | 0.496 | 0.270 | 1.8× |
| 4096 | 13.74 | 0.972 | 14.1× | 5.20 | 0.547 | 9.5× |
| 8192 | 54.09 | 3.38 | 16.0× | 16.09 | 1.59 | 10.1× |
| 16384 | 213.4 | 12.70 | 16.8× | 69.30 | 6.11 | 11.3× |

Two things to read here:

- **Below `seq ≈ 2048` the cross-GPU gap collapses to ~2×.** At small `seq` you're measuring kernel-launch + harness overhead, not GPU compute — the H100's muscle has nothing to flex. The real hardware gap only appears at `seq ≥ 4096`.
- **The H100 gain is bigger for PyTorch (~16×) than for Flash (~10×).** Naive attention is exactly the 3090's worst-case workload: streaming the O(N²) score matrix through 936 GB/s GDDR6X and a tiny 6 MB L2. The H100 (3.35 TB/s HBM3, 50 MB L2, far stronger tensor cores) fixes precisely that. Flash is *already* efficient on the 3090 — it runs close to that card's roofline — so moving to the H100 gives roughly the raw hardware ratio and nothing extra to "recover".

## 2. Flash vs PyTorch — does the better GPU change the verdict?

Flash speedup over PyTorch (`full`, `d_head=64`, bf16):

| seq_len | 3090 | H100 |
|---|---|---|
| 1024 | 2.3× | 2.1× |
| 4096 | 2.6× | 1.8× |
| 8192 | 3.4× | 2.1× |
| 16384 | 3.1× | 2.1× |

On the 3090 Flash's lead **widens** with `seq_len` (→ 3.4×); on the H100 it's a **flat ~2×**.

Mechanism: PyTorch's penalty *is* the repeated HBM round-trips of the full N×N matrix. As N grows, the 3090 pays an ever-larger bandwidth tax — Flash, which never materializes that matrix, is immune, so the gap opens up. The H100's bandwidth and huge L2 absorb that traffic, so PyTorch degrades far less relative to Flash, and the ratio settles at Flash's "pure algorithmic" margin (fused kernels, fewer passes — but both still do O(N²) tensor-core matmuls).

**Takeaway: Flash helps the weaker, bandwidth-starved GPU the most.** A fast GPU partially hides naive attention's sloppiness; a slow one does not.

## 3. Memory and the OOM cliff

PyTorch peak memory (`d_head=64`, bf16) and survival:

| seq_len | 3090 (6 GB) | H100 (80 GB) |
|---|---|---|
| 16384 | 4126 MB ✓ | 4174 MB ✓ |
| 32768 | **OOM** | 15708 MB ✓ |
| 65536 | **OOM** | 61817 MB ✓ |

- **Peak memory is hardware-independent.** At `seq=16384` both cards report ~4.1 GB (1% apart); Flash reports 276 MB on both, *exactly*. The allocation is the algorithm's — the GPU only determines whether it fits.
- **80 GB vs 6 GB only buys ~2× sequence length, not 4×.** Attention memory is O(N²): 4× the RAM → only 2× the `seq_len`. The H100 has 3.3× the 3090's memory → √3.3 ≈ 1.8× more sequence before OOM. That's why H100 PyTorch fp32 survives `32768` (30 GB) but still **OOMs at `65536`** (would need ~120 GB). Throwing hardware at an O(N²) problem gives sharply diminishing returns.
- **Flash never OOMs on either GPU.** Peak stays 256 → 576 MB across the whole sweep. At `seq=65536` that's a **~220× memory gap** vs PyTorch's 61.8 GB. This is the structural fix: O(N) memory changes the asymptote; more VRAM only slides the cliff.

## 4. Data-quality note

The H100 run is **much cleaner** than the 3090 run. The 3090 had two artifacts its own writeup flagged:

- a fp32-backward blow-up at `seq=16384` (~600 ms)
- a `flash fp32 d_head=128 seq=65536` row where `full < bwd` (impossible — a measurement glitch)

The H100 numbers are monotonic and self-consistent (e.g. fp32 `d_head=128 seq=65536`: fwd 69.7 + bwd 298.4 ≈ full 368.2 ✓). Trust the H100 table as-is; treat the 3090's large-fp32 rows with caution.

---

## Punchline

The H100 makes naive attention *tolerable* (16× faster, runs to 65K in bf16) but doesn't make it *correct engineering* — it still burns 62 GB and still OOMs in fp32. Flash wins on both GPUs, wins biggest on the weaker one, and is the only implementation whose memory doesn't depend on which GPU you happened to get.
