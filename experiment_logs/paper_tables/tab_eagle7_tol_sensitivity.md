# Vicuna-7B EAGLE TSS tol sensitivity (real layer search)

Each tol (except `-5%`) runs breadth skip-layer search with joint floor on accept & metric.
`-5%` = published triple-win TSS (not re-run).

| Domain | Tol | Floor | Accept | Metric | Skip | Sparsity | Tok/s | Evals | Source |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| translation | 0 | 1.00 | 3.810 | 0.1540 | `[7, 11, 12, 14, 29]` | 15.6% | 116.2 | 214 | floor_breadth_search |
| translation | +5% | 1.05 | 3.810 | 0.1540 | `[7, 11, 12, 14, 29]` | 15.6% | 116.2 | 0 | floor_breadth_search |
| translation | -5% | 0.95 | 4.528 | 0.2370 | `[3, 25, 30]` | 9.4% | 127.3 | — | published_triple_win |
| translation | -10% | 0.90 | 3.810 | 0.1540 | `[7, 11, 12, 14, 29]` | 15.6% | 116.2 | 45 | floor_breadth_search |
| translation | -20% | 0.80 | 3.810 | 0.1540 | `[7, 11, 12, 14, 29]` | 15.6% | 116.2 | 0 | floor_breadth_search |
| qa | 0 | 1.00 | 3.127 | 0.0434 | `[7, 12, 21, 28, 30]` | 15.6% | 91.3 | 212 | floor_breadth_search |
| qa | +5% | 1.05 | 3.127 | 0.0434 | `[7, 12, 21, 28, 30]` | 15.6% | 91.3 | 0 | floor_breadth_search |
| qa | -5% | 0.95 | 3.970 | 0.0640 | `[6, 8, 19, 21, 27]` | 15.6% | 113.7 | — | published_triple_win |
| qa | -10% | 0.90 | 3.127 | 0.0434 | `[7, 12, 21, 28, 30]` | 15.6% | 91.3 | 0 | floor_breadth_search |
| qa | -20% | 0.80 | 3.127 | 0.0434 | `[7, 12, 21, 28, 30]` | 15.6% | 91.3 | 0 | floor_breadth_search |
| rag | 0 | 1.00 | 3.547 | 0.1061 | `[21, 28, 30]` | 9.4% | 87.0 | 213 | floor_breadth_search |
| rag | +5% | 1.05 | 3.547 | 0.1061 | `[21, 28, 30]` | 9.4% | 87.0 | 0 | floor_breadth_search |
| rag | -5% | 0.95 | 4.027 | 0.1091 | `[3, 6, 14, 25, 30]` | 15.6% | 97.1 | — | published_triple_win |
| rag | -10% | 0.90 | 3.547 | 0.1061 | `[21, 28, 30]` | 9.4% | 87.0 | 0 | floor_breadth_search |
| rag | -20% | 0.80 | 3.547 | 0.1061 | `[21, 28, 30]` | 9.4% | 87.0 | 0 | floor_breadth_search |
| mmlu | 0 | 1.00 | 3.405 | 0.2812 | `[12, 15, 16, 28]` | 12.5% | 81.0 | 215 | floor_breadth_search |
| mmlu | +5% | 1.05 | 3.115 | 0.2812 | `[]` | 0.0% | 74.4 | 0 | floor_breadth_search |
| mmlu | -5% | 0.95 | 3.955 | 0.3281 | `[3, 7, 9, 14, 20]` | 15.6% | 90.8 | — | published_triple_win |
| mmlu | -10% | 0.90 | 3.405 | 0.2812 | `[12, 15, 16, 28]` | 12.5% | 81.0 | 0 | floor_breadth_search |
| mmlu | -20% | 0.80 | 3.405 | 0.2812 | `[12, 15, 16, 28]` | 12.5% | 81.0 | 0 | floor_breadth_search |
