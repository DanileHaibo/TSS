# Vicuna-7B EAGLE search-method comparison

Held-out test=64. Baselines searched on train=4 (fast); TSS Breadth Search reused from existing train=16 results.

## mmlu

| Method | Accept Len. | Metric | Skip Layers | Throughput | Search Evaluations |
|---|---:|---:|---|---:|---:|
| SLEB | 3.488 | 0.2969 | `[2, 3, 4, 6]` | 84.0 | 114 |
| Random Search | 2.948 | 0.2969 | `[6, 10, 13, 18]` | 76.0 | 48 |
| Greedy Search | 2.889 | 0.3281 | `[2, 3, 4, 5]` | 70.4 | 114 |
| Accept-only | 4.023 | 0.2188 | `[4, 6, 16]` | 88.0 | 114 |
| Metric-only | 3.001 | 0.2656 | `[3]` | 69.3 | 30 |
| TSS Breadth Search | 3.955 | 0.3281 | `[3, 7, 9, 14, 20]` | 90.8 | 548 |

## translation

| Method | Accept Len. | Metric | Skip Layers | Throughput | Search Evaluations |
|---|---:|---:|---|---:|---:|
| SLEB | 3.477 | 0.1298 | `[2, 3, 4, 9]` | 101.5 | 114 |
| Random Search | 3.084 | 0.0986 | `[7, 11]` | 92.9 | 48 |
| Greedy Search | 2.995 | 0.1089 | `[2, 3, 4, 5]` | 91.1 | 114 |
| Accept-only | 3.769 | 0.1264 | `[3, 15]` | 107.9 | 87 |
| Metric-only | 3.029 | 0.1398 | `[12, 18, 28]` | 88.2 | 114 |
| TSS Breadth Search | 4.528 | 0.2370 | `[3, 25, 30]` | 127.3 | 334 |

