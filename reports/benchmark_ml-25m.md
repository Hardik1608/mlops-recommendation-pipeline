# Offline benchmark - ml-25m

Temporal split, fit on train+val, evaluated on test (K=10, relevant = rating >= 4.0).

| Model | RMSE | P@10 | R@10 | NDCG@10 | MAP@10 | NDCG@10 warm | NDCG@10 cold | Coverage | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 0.9773 | 0.1007 | 0.0468 | 0.3357 | 0.0625 | 0.1176 | 0.4513 | 0.006 | 4.61 |
| explicit_als | 0.9736 | 0.0038 | 0.0019 | 0.2963 | 0.0012 | 0.0039 | 0.4513 | 0.056 | 94.93 |
| implicit_als | n/a | 0.1070 | 0.0464 | 0.3374 | 0.0641 | 0.1224 | 0.4513 | 0.052 | 232.81 |
| hybrid_als | n/a | 0.1157 | 0.0530 | 0.3409 | 0.0690 | 0.1325 | 0.4513 | 0.006 | 231.69 |
| sasrec | n/a | 0.0837 | 0.0395 | 0.3279 | 0.0464 | 0.0950 | 0.4513 | 0.059 | 278.51 |
| sasrec_shuffled | n/a | 0.0885 | 0.0371 | 0.3296 | 0.0520 | 0.0998 | 0.4513 | 0.035 | 277.84 |

P/R/MAP and 'warm' columns: 6330 users seen in training; 'cold': 11946 users first seen after the cutoff; NDCG 'all': 18276 users.

## Paired bootstrap over returning test users (NDCG@10)

| Comparison | Lift | 95% CI | p (two-sided) | Users |
|---|---|---|---|---|
| implicit_als vs popularity | +4.0% | [+0.1%, +8.1%] | 0.0480 | 6330 |
| hybrid_als vs popularity | +12.6% | [+10.3%, +15.1%] | 0.0005 | 6330 |
| hybrid_als vs implicit_als | +8.3% | [+5.0%, +11.6%] | 0.0005 | 6330 |
| sasrec vs popularity | -19.3% | [-22.9%, -15.7%] | 0.0005 | 6330 |
| sasrec vs hybrid_als | -28.3% | [-31.1%, -25.4%] | 0.0005 | 6330 |
| sasrec_shuffled vs sasrec | +5.1% | [+0.5%, +9.9%] | 0.0340 | 6330 |

## Streaming value (first 5 test events of each user used as real-time context)

| Model | Segment | NDCG@10 batch-only | NDCG@10 streaming | Users |
|---|---|---|---|---|
| popularity | warm | 0.1104 | 0.1152 | 5348 |
| popularity | cold | 0.2983 | 0.3643 | 11944 |
| popularity | all | 0.2402 | 0.2873 | 17292 |
| implicit_als | warm | 0.1231 | 0.1312 | 5348 |
| implicit_als | cold | 0.2983 | 0.3531 | 11944 |
| implicit_als | all | 0.2441 | 0.2845 | 17292 |
| hybrid_als | warm | 0.1254 | 0.1336 | 5348 |
| hybrid_als | cold | 0.2983 | 0.3608 | 11944 |
| hybrid_als | all | 0.2448 | 0.2906 | 17292 |
| sasrec | warm | 0.0877 | 0.1168 | 5348 |
| sasrec | cold | 0.2983 | 0.3244 | 11944 |
| sasrec | all | 0.2332 | 0.2602 | 17292 |
| sasrec_shuffled | warm | 0.0994 | 0.1050 | 5348 |
| sasrec_shuffled | cold | 0.2983 | 0.3487 | 11944 |
| sasrec_shuffled | all | 0.2368 | 0.2733 | 17292 |

## Protocol check: random 80/20 split (leaky)

| Model | NDCG@10 (warm) |
|---|---|
| popularity | 0.0981 |
| implicit_als | 0.3236 |

## Protocol check: leave-one-out (common in papers; leaks the global timeline)

| Model | NDCG@10 | HR@10 | Users |
|---|---|---|---|
| popularity | 0.0160 | 0.0321 | 162541 |
| implicit_als | 0.0530 | 0.1047 | 162541 |
| hybrid_als | 0.0303 | 0.0547 | 162541 |
| sasrec | 0.0829 | 0.1566 | 162541 |
| sasrec_shuffled | 0.0368 | 0.0750 | 162541 |

Hardware: AMD Ryzen 7 7800X3D 8-Core Processor, 16 logical CPUs, 14.7 GB RAM, Linux 6.8.0-138-generic (#138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  7 13:43:15 UTC ).
