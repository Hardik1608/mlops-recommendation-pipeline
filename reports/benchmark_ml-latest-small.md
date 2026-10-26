# Offline benchmark - ml-latest-small

Temporal split, fit on train+val, evaluated on test (K=10, relevant = rating >= 4.0).

| Model | RMSE | P@10 | R@10 | NDCG@10 | MAP@10 | NDCG@10 warm | NDCG@10 cold | Coverage | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 1.0649 | 0.1560 | 0.0537 | 0.3045 | 0.0814 | 0.1602 | 0.3812 | 0.007 | 0.06 |
| explicit_als | 1.0418 | 0.0640 | 0.0140 | 0.2727 | 0.0251 | 0.0687 | 0.3812 | 0.024 | 23.35 |
| implicit_als | n/a | 0.0960 | 0.0196 | 0.2841 | 0.0636 | 0.1016 | 0.3812 | 0.023 | 5.36 |
| hybrid_als | n/a | 0.1440 | 0.0566 | 0.3041 | 0.0820 | 0.1591 | 0.3812 | 0.007 | 5.28 |

P/R/MAP and 'warm' columns: 25 users seen in training; 'cold': 47 users first seen after the cutoff; NDCG 'all': 72 users.

## Streaming value (first 5 test events of each user used as real-time context)

| Model | Segment | NDCG@10 batch-only | NDCG@10 streaming | Users |
|---|---|---|---|---|
| popularity | warm | 0.1019 | 0.1208 | 25 |
| popularity | cold | 0.2536 | 0.2861 | 47 |
| popularity | all | 0.2010 | 0.2287 | 72 |
| implicit_als | warm | 0.0932 | 0.0965 | 25 |
| implicit_als | cold | 0.2536 | 0.2883 | 47 |
| implicit_als | all | 0.1979 | 0.2217 | 72 |
| hybrid_als | warm | 0.1198 | 0.1350 | 25 |
| hybrid_als | cold | 0.2536 | 0.3000 | 47 |
| hybrid_als | all | 0.2072 | 0.2427 | 72 |

## Protocol check: random 80/20 split (leaky)

| Model | NDCG@10 (warm) |
|---|---|
| popularity | 0.0738 |
| implicit_als | 0.1689 |

Hardware: 13th Gen Intel(R) Core(TM) i7-13620H, 16 logical CPUs, 15.6 GB RAM, Windows 10 (10.0.26200).
