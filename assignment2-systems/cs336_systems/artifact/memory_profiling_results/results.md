for a model with d_model = 2560, d_ff=10240, num_layers= 32, number_heads= 32

| mode    | mixed_precision   | ctx_len |   peak_memory_mb | time (s)        |
|:--------|:------------------|----------:|-----------------:|:----------------|
| forward | False    |       256 |          13334.9 | 0.1752 ± 0.0000 |
| train   | False|       256 |          65436.9 | 0.6617 ± 0.0000 |
| forward | False|       128 |          1  36.7 | 0.0898 ± 0.0000 |
| train   | False|128 |          65576.2 | 0.4074 ± 0.0000 |
| forward | False |       256 |          13334.9 | 0.1748 ± 0.0000 |
| train   | False|       256 |          65436.9 | 0.6620 ± 0.0000 |
| forward | False|       512 |          13769.3 | 0.3389 ± 0.0000 |
| train   | False|       512 |          67169   | 1.1831 ± 0.0000 |
| forward | True|       128 |          19623   | 0.0665 ± 0.0000 |
| train   | True|       128 |          65500.9 | 0.2535 ± 0.0000 |
| forward | True|       256 |          19640.1 | 0.0678 ± 0.0000 |
| train   | True|       256 |          65444.9 | 0.2836 ± 0.0000 |
| forward | True|       512 |          19871.3 | 0.0797 ± 0.0000 |
| train   | True|       512 |          65296.7 | 0.3582 ± 0.0000 |
