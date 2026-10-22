# A vs B vs C on 10 examinations — result tables

Written 2026-10-19 14:12:54 by `evaluate_autopet3_subset.py`. Descriptive only (n = 10).
**C (autoPET III) was trained on autoPET data that include these 10 examinations; its numbers are not a held-out result.**

Consistency check: 480/480 recomputed A/B values match `per_exam_metrics.csv`. Geometry: all 40 masks on the SUV grid: True.

## Per-examination summary (median [IQR] unless noted)

| Metric | A: Microsoft, 10 epochs | B: testTraining, early-stopped | C: autoPET III (trained on these data) |
|---|---|---|---|
| Dice, mean | 0.354 | 0.333 | 0.924 |
| Dice | 0.261 [0.149, 0.616] | 0.362 [0.163, 0.452] | 0.93 [0.914, 0.949] |
| FPV (ml) | 590 [340, 983] | 184 [106, 280] | 0.137 [0, 1.37] |
| FPV (ml), mean | 647 | 224 | 0.936 |
| FNV (ml) | 0.417 [0.00311, 1.71] | 1.26 [0.0871, 8.3] | 0.106 [0.00311, 0.417] |
| FNV (ml), mean | 3.19 | 9.56 | 0.729 |
| Predicted TMTV (ml) | 888 [664, 1.64e+03] | 428 [257, 688] | 218 [94.9, 499] |
| TMTV abs. error (ml) | 626 [453, 1.01e+03] | 259 [177, 470] | 10.4 [0.6, 16.2] |
| TMTV signed rel. error | 4.55 [0.933, 8.36] | 1.08 [-0.313, 1.95] | 0.0128 [-0.0202, 0.0385] |
| SUVmax abs. error | 42 [6.11, 47.3] | 37 [19.7, 47.7] | 0 [0, 0] |
| Predicted lesions | 71.5 [59.8, 137] | 47 [29, 60] | 9.5 [5.5, 14.2] |
| FP lesions C1, mean | 136 | 27.8 | 3.3 |
| Lesion recall C1 per exam | 0.536 [0.479, 0.958] | 0.505 [0.357, 0.889] | 0.821 [0.703, 0.969] |
| Lesion recall C2 per exam | 0.253 [0.142, 0.377] | 0.132 [0.014, 0.223] | 0.605 [0.478, 0.764] |
| Lesion recall C3 per exam | 0.473 [0.392, 0.701] | 0.428 [0.276, 0.697] | 0.735 [0.635, 0.817] |
| Predicted SUVmax = reference SUVmax (exams) | 1/10 | 1/10 | 8/10 |
| Predicted SUVmax > reference (exams) | 9/10 | 9/10 | 2/10 |
| FNV = 0 (exams) | 3/10 | 3/10 | 3/10 |
| Dice < 0.1 (exams) | 2/10 | 2/10 | 0/10 |
| TMTV Spearman ρ (pred vs ref) | 0.70 | 0.67 | 1.00 |

Reference: 348 lesions (18-connected) in 10 exams.

## Pooled lesion detection

| Model | Criterion | GT found / GT | pred TP | pred FP | recall | precision | F1 |
|---|---|---|---|---|---|---|---|
| A | C1 | 182/348 | 205 | 1355 | 0.52 | 0.13 | 0.21 |
| A | C2 | 82/348 | 82 | 1478 | 0.24 | 0.05 | 0.09 |
| A | C3 | 154/348 | 154 | 1406 | 0.44 | 0.10 | 0.16 |
| B | C1 | 134/348 | 232 | 278 | 0.39 | 0.45 | 0.42 |
| B | C2 | 45/348 | 45 | 465 | 0.13 | 0.09 | 0.10 |
| B | C3 | 125/348 | 125 | 385 | 0.36 | 0.25 | 0.29 |
| C | C1 | 258/348 | 268 | 33 | 0.74 | 0.89 | 0.81 |
| C | C2 | 186/348 | 186 | 115 | 0.53 | 0.62 | 0.57 |
| C | C3 | 235/348 | 235 | 66 | 0.68 | 0.78 | 0.72 |

## Is the 10-case subset representative of A and B on all 145?

| Model | Metric | 145 exams | 10-exam subset |
|---|---|---|---|
| A | Dice mean | 0.301 | 0.354 |
| A | Dice median | 0.267 | 0.261 |
| A | FPV median ml | 471 | 590 |
| A | FNV median ml | 0.485 | 0.417 |
| A | TMTV abs error median ml | 572 | 626 |
| A | Lesion recall C1 median | 0.636 | 0.536 |
| A | Ref TMTV median ml | 163 | 206 |
| A | Ref lesions median | 9 | 8.5 |
| B | Dice mean | 0.277 | 0.333 |
| B | Dice median | 0.259 | 0.362 |
| B | FPV median ml | 291 | 184 |
| B | FNV median ml | 1.8 | 1.26 |
| B | TMTV abs error median ml | 264 | 259 |
| B | Lesion recall C1 median | 0.513 | 0.505 |
| B | Ref TMTV median ml | 163 | 206 |
| B | Ref lesions median | 9 | 8.5 |

## Per examination

| Exam | ref TMTV ml | ref lesions | Dice A | Dice B | Dice C | FPV A | FPV B | FPV C | FNV A | FNV B | FNV C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f47e31ceb5 | 6.7 | 8 | 0.008 | 0.051 | 0.913 | 468.2 | 98.4 | 0.0 | 1.1 | 1.1 | 0.1 |
| 22d07bdc49 | 14.3 | 5 | 0.081 | 0.019 | 0.942 | 141.5 | 466.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| 249dd35d0c | 82.6 | 9 | 0.117 | 0.274 | 0.918 | 1073.0 | 203.2 | 0.0 | 0.3 | 0.3 | 0.1 |
| 62405d117c | 131.3 | 6 | 0.245 | 0.126 | 0.977 | 711.0 | 77.0 | 0.0 | 0.0 | 8.6 | 0.0 |
| c4a686881e | 192.6 | 4 | 0.269 | 0.382 | 0.863 | 713.3 | 268.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| 0b57b247b6 | 219.5 | 4 | 0.390 | 0.359 | 0.849 | 375.5 | 284.5 | 1.3 | 0.0 | 0.0 | 0.0 |
| 1fe6e48293 | 285.0 | 17 | 0.252 | 0.365 | 0.918 | 1376.5 | 537.2 | 0.1 | 0.5 | 7.5 | 0.2 |
| 5ef99b0b6b | 553.1 | 13 | 0.742 | 0.647 | 0.945 | 172.7 | 6.6 | 1.4 | 1.9 | 1.4 | 0.5 |
| 6a3477cd9a | 1554.1 | 157 | 0.692 | 0.628 | 0.950 | 328.2 | 165.2 | 3.7 | 15.0 | 22.4 | 4.2 |
| 30001118d0 | 2481.2 | 125 | 0.748 | 0.476 | 0.964 | 1112.4 | 130.4 | 2.7 | 13.0 | 54.2 | 2.2 |
