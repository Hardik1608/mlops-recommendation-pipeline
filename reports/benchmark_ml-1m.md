# Offline benchmark - ml-1m

Temporal split, fit on train+val, evaluated on test (K=10, relevant = rating >= 4.0).

| Model | RMSE | P@10 | R@10 | NDCG@10 | MAP@10 | NDCG@10 warm | NDCG@10 cold | Coverage | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 0.9238 | 0.1853 | 0.0519 | 0.2070 | 0.1198 | 0.2007 | 0.4588 | 0.040 | 0.12 |
| explicit_als | 0.8951 | 0.0510 | 0.0169 | 0.0548 | 0.0159 | 0.0446 | 0.4588 | 0.083 | 10.05 |
| implicit_als | n/a | 0.1987 | 0.0648 | 0.2254 | 0.1306 | 0.2196 | 0.4588 | 0.118 | 4.51 |
| hybrid_als | n/a | 0.2104 | 0.0730 | 0.2383 | 0.1380 | 0.2328 | 0.4588 | 0.047 | 4.49 |
| sasrec | n/a | 0.1068 | 0.0458 | 0.1269 | 0.0596 | 0.1186 | 0.4588 | 0.450 | 54.56 |
| sasrec_shuffled | n/a | 0.1338 | 0.0498 | 0.1536 | 0.0763 | 0.1460 | 0.4588 | 0.490 | 52.74 |

P/R/MAP and 'warm' columns: 1159 users seen in training; 'cold': 29 users first seen after the cutoff; NDCG 'all': 1188 users.

## Paired bootstrap over returning test users (NDCG@10)

| Comparison | Lift | 95% CI | p (two-sided) | Users |
|---|---|---|---|---|
| implicit_als vs popularity | +9.4% | [+4.5%, +14.9%] | 0.0005 | 1159 |
| hybrid_als vs popularity | +16.0% | [+12.3%, +20.1%] | 0.0005 | 1159 |
| hybrid_als vs implicit_als | +6.0% | [+2.8%, +9.5%] | 0.0005 | 1159 |
| sasrec vs popularity | -40.9% | [-46.2%, -35.6%] | 0.0005 | 1159 |
| sasrec vs hybrid_als | -49.0% | [-53.3%, -44.9%] | 0.0005 | 1159 |
| sasrec_shuffled vs sasrec | +23.1% | [+12.6%, +35.4%] | 0.0005 | 1159 |

## Streaming value (first 5 test events of each user used as real-time context)

| Model | Segment | NDCG@10 batch-only | NDCG@10 streaming | Users |
|---|---|---|---|---|
| popularity | warm | 0.1987 | 0.2067 | 1007 |
| popularity | cold | 0.3942 | 0.4278 | 29 |
| popularity | all | 0.2042 | 0.2129 | 1036 |
| implicit_als | warm | 0.2129 | 0.2341 | 1007 |
| implicit_als | cold | 0.3942 | 0.3966 | 29 |
| implicit_als | all | 0.2180 | 0.2387 | 1036 |
| hybrid_als | warm | 0.2190 | 0.2360 | 1007 |
| hybrid_als | cold | 0.3942 | 0.4108 | 29 |
| hybrid_als | all | 0.2239 | 0.2409 | 1036 |
| sasrec | warm | 0.1075 | 0.1759 | 1007 |
| sasrec | cold | 0.3942 | 0.2623 | 29 |
| sasrec | all | 0.1156 | 0.1783 | 1036 |
| sasrec_shuffled | warm | 0.1433 | 0.1606 | 1007 |
| sasrec_shuffled | cold | 0.3942 | 0.2611 | 29 |
| sasrec_shuffled | all | 0.1503 | 0.1634 | 1036 |

## Protocol check: random 80/20 split (leaky)

| Model | NDCG@10 (warm) |
|---|---|
| popularity | 0.1718 |
| implicit_als | 0.2938 |

## Protocol check: leave-one-out (common in papers; leaks the global timeline)

| Model | NDCG@10 | HR@10 | Users |
|---|---|---|---|
| popularity | 0.0193 | 0.0387 | 6040 |
| implicit_als | 0.0359 | 0.0732 | 6040 |
| hybrid_als | 0.0151 | 0.0356 | 6040 |
| sasrec | 0.1758 | 0.3091 | 6040 |
| sasrec_shuffled | 0.0345 | 0.0740 | 6040 |

Hardware: AMD Ryzen 7 7800X3D 8-Core Processor, 16 logical CPUs, 14.7 GB RAM, Linux 6.8.0-138-generic (#138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  7 13:43:15 UTC ).
