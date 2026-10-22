# A vs B: full supervisor-requested evaluation on the 145 out-of-fold predictions

Created 2026-10-18 (see `answers/27-full-evaluation-a-vs-b.md` for the change log and interpretation).

**Experiments**

| | Predictions | Folds |
|---|---|---|
| **A**: Microsoft `lymphoma-segmentation-dnn` (commit 8112924), 10 epochs | `experiments/unet_10ep/work/results/predictions/fold{F}/unet/unet_fold{F}_randcrop192/<case_id>.nii.gz` | seed 42 (`~/data/lymphoma-baseline/cases.json`) |
| **B**: `testTraining` (`scripts/train.py`), budget 500, early-stopped at 112–272 | `/home/eva25/manu-runs/full-training/fold_{F}/predictions/<patient>/<exam>/SEG_pred.nii.gz` | seed 20261006 (`splits.json`) |

Every examination is scored with the prediction of the model that held its patient out. The folds of A and B
differ (audit 26), so **per-fold numbers are not comparable between A and B**. Per-examination numbers are.

Reference masks, PET (SUV) and CT come from `/home/eva25/data/lymphoma-baseline`. Audit 26 showed these
files are byte-identical (SHA-256) to B's source files. No model was retrained. No prediction, checkpoint
or label was modified; all inputs were opened read-only.

## Files

| File | Content |
|---|---|
| `evaluate_a_vs_b.py` | Computes all per-examination metrics (WSL, `~/venvs/lymphoma_seg`) |
| `summarize_a_vs_b.py` | Aggregates, paired tests, plots |
| `per_exam_metrics.csv` | One row per experiment × examination (290 rows), raw metrics |
| `per_exam_paired.csv` | Same, wide format (one row per examination, `<metric>_A`, `<metric>_B`) plus derived errors |
| `aggregate_by_experiment.csv` | Summary statistics per experiment |
| `detection_pooled.csv` | Lesion recall / precision / F1 pooled over all lesions, with patient-bootstrap CIs |
| `paired_tests.csv` | Paired B − A differences, Wilcoxon, Holm-adjusted p, patient-bootstrap CI |
| `results_tables.md` | All tables above in readable form |
| `validation_against_microsoft.csv` | This implementation vs Microsoft's original functions on a sample |
| `compute_manifest.json` | Script hash, Microsoft `metrics.py` hash, input locations, versions |
| `plots/` | `dice_paired_scatter.png`, `metric_distributions.png`, `detection_pooled.png`, `tmtv_agreement.png`, `suvmax_agreement.png`, `dice_by_volume_tertile.png` |

## Definitions (identical for A and B)

All definitions are those of Microsoft's `lymphoma-segmentation-dnn/metrics/metrics.py` and
`segmentation/calculate_test_metrics.py` (unmodified). Arrays are read with Microsoft's
`get_3darray_from_niftipath` (SimpleITK, transposed to x,y,z) on the **native PET grid**. Spacing is
taken from the reference mask. Before scoring, every prediction and reference mask is checked to lie on
the SUV image's grid and to contain only 0 and 1.

| Metric | Definition |
|---|---|
| **Dice** | `2·|P∩G| / (|P|+|G|)`. No examination has an empty reference or an empty prediction, so empty-mask conventions do not arise |
| **Lesion** | Connected component, **18-connectivity** (cc3d), in the reference or predicted mask |
| **False-positive volume (FPV, ml)** | Total volume of predicted lesions that share no voxel with the reference mask |
| **False-negative volume (FNV, ml)** | Total volume of reference lesions that share no voxel with the prediction |
| **Detection, criterion 1** | Predicted lesion TP if it overlaps any reference voxel, else FP. FN = reference lesions with no overlap |
| **Detection, criterion 2** | For each predicted lesion (in label order) take the reference lesion with the highest IoU (first one on ties). TP if IoU ≥ 0.5, and that reference lesion is then removed. Otherwise FP. FN = reference lesions never matched |
| **Detection, criterion 3** | Same best-IoU matching. TP if the predicted lesion contains the SUVmax voxel of its matched reference lesion, which is then removed. Otherwise FP |
| **TMTV (ml)** | Number of mask voxels × voxel volume |
| **SUVmax** | Maximum SUV inside the mask (one value per examination, not per lesion) |
| **SUVmean, TLG** | Mean SUV inside the mask; TLG = Σ lesion MTV × lesion SUVmean (= voxel volume × SUV sum) |
| TMTV / SUVmax error | signed `pred − ref`, absolute, and relative `(pred − ref)/ref` |

Two summaries of detection are reported:
- **Microsoft's per-examination "sensitivity"** `TP/(TP+FN)`, as in `get_prediction_statistics`. For
  criterion 1, TP counts *predicted* components, so several components on one lesion inflate it. It is
  reported for continuity with Microsoft's paper, not as the main number.
- **Lesion recall** = reference lesions found / reference lesions (`(n_ref − FN)/n_ref`), and **lesion
  precision** = TP / (TP + FP) over predicted components. Both are given per examination and pooled over
  all lesions of all examinations, with F1.

Not computed: **Dmax**. Microsoft's implementation builds an N×N×3 distance array over all lesion
voxels, which needs up to hundreds of GiB for these masks (answer 13), and the supervisor did not
request it.

## Implementation and its verification

Microsoft's FPV/FNV and detection functions loop over every (predicted, reference) component pair with a
full-volume array operation each. For A's predictions (median 66 components, up to several hundred) this
takes hours per examination. `evaluate_a_vs_b.py` therefore computes the same quantities from a
component contingency table: intersection counts per (predicted, reference) pair via `bincount`, sizes,
IoU. It reproduces Microsoft's greedy loop exactly:
- predicted lesions are processed in label order;
- the best match uses a strict `>` starting at 0, so the first reference lesion wins ties;
- the SUVmax voxel is the first maximum in the (x,y,z) array's C order, as `np.argmax` picks it;
- Microsoft's failure cases (`list.remove` of an absent lesion; an unmatched predicted lesion covering
  voxel 0; SUVmax ≤ 0) are detected and reported in the `notes` column instead of crashing.

`evaluate_a_vs_b.py --validate N` runs Microsoft's **original** functions next to this implementation on
N evaluations: half from the smallest workloads, half from the lower-quartile workload. Results:
`validation_against_microsoft.csv`, summarized in `answers/27`.

## Statistics

- Per-experiment summaries: mean ± SD, median [IQR], counts.
- Paired comparison per examination (Δ = B − A) for 15 metrics: Wilcoxon signed-rank (normal
  approximation, tie-corrected), Holm adjustment across the 15 metrics, and a 95% bootstrap CI of the mean
  difference that resamples **patients** (10 000 resamples), so the repeated patient's two exams stay
  together.
- Pooled detection CIs: patient bootstrap (2 000 resamples).
- **Limitation:** each configuration was trained once and the folds differ. The tests capture
  case-sampling uncertainty only, not training variance. Treat p-values as descriptive.

## Reproduce

```bash
# inside WSL, from the project root
~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py --validate 8
~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py      # refuses to overwrite per_exam_metrics.csv
~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/summarize_a_vs_b.py
```

Note: the project `.gitignore` ignores `*.md`, so the Markdown files here need `git add -f` if they should
be committed.
