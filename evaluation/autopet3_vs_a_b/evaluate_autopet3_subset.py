"""A vs B vs C on the 10 examinations of experiments/autopet3_subset10/manifest.json.

A = experiments/unet_10ep (Microsoft lymphoma-segmentation-dnn, 10 epochs), out-of-fold prediction
B = /home/eva25/manu-runs/full-training (testTraining, early-stopped), out-of-fold prediction
C = official pretrained autoPET III checkpoint (5-fold ensemble, mirroring TTA), experiments/autopet3_subset10/predictions
    C was trained on autoPET data that INCLUDES these examinations: orientation only, not a held-out result.

Metrics are computed by the unmodified evaluate() of evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py
(same reader, geometry and 0/1 checks, 18-connectivity, IoU 0.5, three detection criteria, FPV, FNV,
TMTV, SUVmax, SUVmean, TLG). Derived columns and aggregates use derive() and aggregate() of
summarize_a_vs_b.py. A and B rows are taken from a_vs_b_full_metrics/per_exam_metrics.csv; all 10 A
and all 10 B examinations are recomputed and must match those rows exactly (consistency check).
Descriptive only: n = 10, no significance tests.

Read-only with respect to all inputs. Refuses to overwrite its outputs.
Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/autopet3_vs_a_b/evaluate_autopet3_subset.py
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AB_DIR = PROJECT / 'evaluation' / 'a_vs_b_full_metrics'
sys.path.insert(0, str(AB_DIR))
import evaluate_a_vs_b as ev  # noqa: E402  (unmodified; also puts Microsoft's metrics on the path)
import summarize_a_vs_b as sa  # noqa: E402

MANIFEST = PROJECT / 'experiments' / 'autopet3_subset10' / 'manifest.json'
C_PRED = PROJECT / 'experiments' / 'autopet3_subset10' / 'predictions'
AB_METRICS = AB_DIR / 'per_exam_metrics.csv'
PLOTS = HERE / 'plots'
ORDER = ['A', 'B', 'C']
COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a'}   # categorical slots 1-3, validated all-pairs
MARKERS = {'A': 'o', 'B': 's', 'C': '^'}                    # secondary encoding (C is below 3:1 contrast)
NAMES = {'A': 'A: Microsoft, 10 epochs', 'B': 'B: testTraining, early-stopped',
         'C': 'C: autoPET III (trained on these data)'}
OUTPUTS = ['per_exam_metrics_10.csv', 'aggregate_10.csv', 'detection_pooled_10.csv', 'subset_vs_full_AB.csv',
           'consistency_check.csv', 'results_tables.md', 'compute_manifest.json']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gzip_complete(path):
    try:
        with gzip.open(path, 'rb') as f:
            while f.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def load_subset():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    ids = [c['case_id'] for c in manifest['cases']]
    if len(ids) != 10 or len(set(ids)) != 10:
        sys.exit('manifest does not list 10 distinct examinations')
    by_id = {c['case_id']: c for c in ev.load_cases()}
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f'manifest cases not among the 145 evaluated examinations: {missing}')
    cases = []
    for i in sorted(ids):
        case = by_id[i]
        case['pred']['C'] = C_PRED / f'{i}.nii.gz'
        case['folds']['C'] = 'ensemble'
        for path in (case['pet'], case['gt'], *case['pred'].values()):
            if not path.is_file():
                sys.exit(f'Missing input: {path}')
        if not gzip_complete(case['pred']['C']):
            sys.exit(f'Incomplete prediction (gzip stream unreadable): {case["pred"]["C"]}')
        cases.append(case)
    return cases


def geometry_report(case):
    """Explicit header comparison of every mask with the SUV image (evaluate() also enforces it)."""
    pet = sitk.ReadImage(str(case['pet']))
    rows = []
    for name, path in [('GT', case['gt'])] + [(e, case['pred'][e]) for e in ORDER]:
        img = sitk.ReadImage(str(path))
        rows.append({'case_id': case['case_id'], 'mask': name, 'size': img.GetSize(),
                     'spacing_max_diff': float(np.max(np.abs(np.subtract(img.GetSpacing(), pet.GetSpacing())))),
                     'origin_max_diff': float(np.max(np.abs(np.subtract(img.GetOrigin(), pet.GetOrigin())))),
                     'direction_max_diff': float(np.max(np.abs(np.subtract(img.GetDirection(), pet.GetDirection())))),
                     'same_size': img.GetSize() == pet.GetSize()})
    return rows


def compare(stored, fresh):
    out = []
    for key, value in fresh.items():
        if key in ('experiment', 'case_id', 'patient', 'notes'):
            continue
        a, b = stored[key], value
        try:
            ok = bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9))
        except (TypeError, ValueError):
            ok = str(a) == str(b)
        out.append({'case_id': fresh['case_id'], 'experiment': fresh['experiment'], 'metric': key,
                    'stored': a, 'recomputed': b, 'match': ok})
    return out


def pooled_detection(df):
    rows = []
    for e in ORDER:
        g = df[df.experiment == e]
        for c in ('C1', 'C2', 'C3'):
            gt, found = int(g.gt_lesions.sum()), int((g.gt_lesions - g[f'FN_{c}']).sum())
            tp, fp = int(g[f'TP_{c}'].sum()), int(g[f'FP_{c}'].sum())
            recall, precision = found / gt, tp / (tp + fp) if tp + fp else np.nan
            rows.append({'experiment': e, 'criterion': c, 'gt_lesions': gt, 'gt_lesions_found': found,
                         'pred_TP': tp, 'pred_FP': fp, 'recall': recall, 'precision': precision,
                         'F1': 2 * recall * precision / (recall + precision) if recall + precision else np.nan,
                         'exams_all_gt_found': int((g[f'FN_{c}'] == 0).sum()),
                         'exams_no_gt_found': int((g[f'FN_{c}'] == g.gt_lesions).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def finish(fig, ax, path):
    """One legend row above the panels (never over data), then save."""
    import matplotlib.pyplot as plt
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8, labelcolor=sa.MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all(df, det):
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    wide = df.pivot(index='case_id', columns='experiment')
    order = wide['GT_TMTV_ml']['A'].sort_values().index            # exams by reference TMTV
    short = [i[6:16] for i in order]
    x = np.arange(len(order))
    offset = {'A': -0.22, 'B': 0.0, 'C': 0.22}

    # 1. Dice per examination
    fig, ax = sa.figure(figsize=(9, 4.2))
    for e in ORDER:
        ax.scatter(x + offset[e], wide['DSC'][e][order], s=46, marker=MARKERS[e], color=COLORS[e],
                   edgecolor=sa.SURFACE, linewidth=1.5, label=NAMES[e], zorder=3)
    sa.style(ax, 'Dice per examination (sorted by reference TMTV, small → large)', None, 'Dice')
    ax.set_xticks(x, short, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    finish(fig, ax, PLOTS / 'dice_per_exam.png')

    # 2. FPV and FNV per examination (two panels, one scale each)
    fig, axes = sa.figure(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], 'FPV_ml', 'False-positive volume (ml, log)'),
                              (axes[1], 'FNV_ml', 'False-negative volume (ml, log)')):
        for e in ORDER:
            ax.scatter(x + offset[e], wide[metric][e][order].clip(lower=0.01), s=40, marker=MARKERS[e],
                       color=COLORS[e], edgecolor=sa.SURFACE, linewidth=1.5, label=NAMES[e], zorder=3)
        sa.style(ax, title)
        ax.set_yscale('log')
        ax.set_xticks(x, short, rotation=35, ha='right', fontsize=8)
    axes[1].text(0.0, -0.32, 'Values of 0 are drawn at 0.01 ml.', transform=axes[1].transAxes, fontsize=8, color=sa.MUTED)
    finish(fig, axes[0], PLOTS / 'fpv_fnv_per_exam.png')

    # 3. Pooled lesion detection: recall and precision per criterion
    fig, axes = sa.figure(1, 2, figsize=(10, 3.8), sharey=True)
    width = 0.26
    for ax, measure in zip(axes, ('recall', 'precision')):
        for k, e in enumerate(ORDER):
            vals = det[det.experiment == e].set_index('criterion')[measure].reindex(['C1', 'C2', 'C3'])
            pos = np.arange(3) + (k - 1) * (width + 0.02)
            ax.bar(pos, vals, width, color=COLORS[e], edgecolor=sa.SURFACE, linewidth=2, label=NAMES[e], zorder=3)
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.015, f'{v:.2f}', ha='center', fontsize=7, color=sa.MUTED)
        sa.style(ax, f'Pooled lesion {measure}')
        ax.set_xticks(range(3), ['C1 any overlap', 'C2 IoU ≥ 0.5', 'C3 SUVmax voxel'], fontsize=8)
        ax.set_ylim(0, 1.05)
    finish(fig, axes[0], PLOTS / 'detection_pooled.png')

    # 4. TMTV and SUVmax agreement (two panels; identity line)
    fig, axes = sa.figure(1, 2, figsize=(10, 4.4))
    for ax, gt, pred, title, log in ((axes[0], 'GT_TMTV_ml', 'Pred_TMTV_ml', 'TMTV (ml, log-log)', True),
                                     (axes[1], 'GT_SUVmax', 'Pred_SUVmax', 'SUVmax', False)):
        values = pd.concat([df[gt], df[pred]])
        lo, hi = max(values[values > 0].min() * 0.7, 1e-2) if log else 0, values.max() * 1.1
        ax.plot([lo, hi], [lo, hi], color=sa.MUTED, linewidth=1, linestyle='--', zorder=1)
        for e in ORDER:
            g = df[df.experiment == e]
            ax.scatter(g[gt], g[pred].clip(lower=lo) if log else g[pred], s=40, marker=MARKERS[e], color=COLORS[e],
                       edgecolor=sa.SURFACE, linewidth=1.5, label=NAMES[e], zorder=3)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        sa.style(ax, title, 'reference', 'predicted')
    finish(fig, axes[0], PLOTS / 'tmtv_suvmax_agreement.png')


# ---------------------------------------------------------------- tables
def results_markdown(df, agg, det, subset_vs_full, consistency, geometry_ok):
    def med_iqr(e, m):
        s = df[df.experiment == e][m].dropna()
        return f'{s.median():.3g} [{s.quantile(.25):.3g}, {s.quantile(.75):.3g}]'

    def mean(e, m):
        return f'{df[df.experiment == e][m].mean():.3g}'

    lines = ['# A vs B vs C on 10 examinations — result tables', '',
             f'Written {time.strftime("%Y-%m-%d %H:%M:%S")} by `evaluate_autopet3_subset.py`. Descriptive only (n = 10).',
             '**C (autoPET III) was trained on autoPET data that include these 10 examinations; its numbers are not a held-out result.**', '',
             f'Consistency check: {int(consistency.match.sum())}/{len(consistency)} recomputed A/B values match `per_exam_metrics.csv`. '
             f'Geometry: all 40 masks on the SUV grid: {geometry_ok}.', '',
             '## Per-examination summary (median [IQR] unless noted)', '',
             '| Metric | ' + ' | '.join(NAMES[e] for e in ORDER) + ' |', '|---|---|---|---|']
    for label, m, kind in [('Dice, mean', 'DSC', 'mean'), ('Dice', 'DSC', 'med'),
                           ('FPV (ml)', 'FPV_ml', 'med'), ('FPV (ml), mean', 'FPV_ml', 'mean'),
                           ('FNV (ml)', 'FNV_ml', 'med'), ('FNV (ml), mean', 'FNV_ml', 'mean'),
                           ('Predicted TMTV (ml)', 'Pred_TMTV_ml', 'med'),
                           ('TMTV abs. error (ml)', 'TMTV_abs_error_ml', 'med'),
                           ('TMTV signed rel. error', 'TMTV_rel_error', 'med'),
                           ('SUVmax abs. error', 'SUVmax_abs_error', 'med'),
                           ('Predicted lesions', 'pred_lesions', 'med'),
                           ('FP lesions C1, mean', 'FP_C1', 'mean'),
                           ('Lesion recall C1 per exam', 'recall_C1', 'med'),
                           ('Lesion recall C2 per exam', 'recall_C2', 'med'),
                           ('Lesion recall C3 per exam', 'recall_C3', 'med')]:
        cells = [mean(e, m) if kind == 'mean' else med_iqr(e, m) for e in ORDER]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    extra = [('Predicted SUVmax = reference SUVmax (exams)', lambda g: f'{int(g.SUVmax_exact.sum())}/10'),
             ('Predicted SUVmax > reference (exams)', lambda g: f'{int((g.Pred_SUVmax > g.GT_SUVmax + 1e-9).sum())}/10'),
             ('FNV = 0 (exams)', lambda g: f'{int((g.FNV_ml == 0).sum())}/10'),
             ('Dice < 0.1 (exams)', lambda g: f'{int((g.DSC < 0.1).sum())}/10'),
             ('TMTV Spearman ρ (pred vs ref)', lambda g: f'{sa.spearman(g.Pred_TMTV_ml, g.GT_TMTV_ml):.2f}')]
    for label, f in extra:
        lines.append(f'| {label} | ' + ' | '.join(f(df[df.experiment == e]) for e in ORDER) + ' |')

    lines += ['', f'Reference: {int(df[df.experiment == "A"].gt_lesions.sum())} lesions (18-connected) in 10 exams.', '',
              '## Pooled lesion detection', '', '| Model | Criterion | GT found / GT | pred TP | pred FP | recall | precision | F1 |',
              '|---|---|---|---|---|---|---|---|']
    for r in det.itertuples():
        lines.append(f'| {r.experiment} | {r.criterion} | {r.gt_lesions_found}/{r.gt_lesions} | {r.pred_TP} | {r.pred_FP} | '
                     f'{r.recall:.2f} | {r.precision:.2f} | {r.F1:.2f} |')
    lines += ['', '## Is the 10-case subset representative of A and B on all 145?', '',
              '| Model | Metric | 145 exams | 10-exam subset |', '|---|---|---|---|']
    for r in subset_vs_full.itertuples():
        lines.append(f'| {r.experiment} | {r.metric} | {r.full_145:.3g} | {r.subset_10:.3g} |')
    lines += ['', '## Per examination', '', '| Exam | ref TMTV ml | ref lesions | Dice A | Dice B | Dice C | FPV A | FPV B | FPV C | FNV A | FNV B | FNV C |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    wide = df.pivot(index='case_id', columns='experiment')
    for i in wide['GT_TMTV_ml']['A'].sort_values().index:
        w = wide.loc[i]
        lines.append(f'| {i[6:16]} | {w["GT_TMTV_ml"]["A"]:.1f} | {int(w["gt_lesions"]["A"])} | '
                     + ' | '.join(f'{w["DSC"][e]:.3f}' for e in ORDER) + ' | '
                     + ' | '.join(f'{w["FPV_ml"][e]:.1f}' for e in ORDER) + ' | '
                     + ' | '.join(f'{w["FNV_ml"][e]:.1f}' for e in ORDER) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    existing = [o for o in OUTPUTS if (HERE / o).exists()]
    if existing:
        sys.exit(f'Outputs exist, refusing to overwrite: {existing}')
    started = time.time()
    cases = load_subset()
    ids = [c['case_id'] for c in cases]

    stored = pd.read_csv(AB_METRICS, keep_default_na=False, na_values=[''])
    stored = stored[stored.case_id.isin(ids)]
    if len(stored) != 20 or set(stored.case_id) != set(ids) or set(stored.experiment) != {'A', 'B'}:
        sys.exit(f'{AB_METRICS} does not hold exactly one A and one B row for each of the 10 examinations')

    geometry, checks, c_rows = [], [], []
    for n, case in enumerate(cases, 1):
        geometry += geometry_report(case)
        for e in ORDER:
            t = time.time()
            row, _ = ev.evaluate(case, e)       # raises on geometry mismatch or non-binary mask
            if e == 'C':
                c_rows.append(row)
            else:
                s = stored[(stored.case_id == case['case_id']) & (stored.experiment == e)].iloc[0]
                checks += compare(s, row)
            print(f'{n:2d}/10 {e} {case["case_id"][:16]} DSC {row["DSC"]:.3f} FPV {row["FPV_ml"]:8.1f} '
                  f'FNV {row["FNV_ml"]:7.1f} lesions gt/pred {row["gt_lesions"]}/{row["pred_lesions"]} '
                  f'{row["notes"]} ({time.time() - t:.1f}s)', flush=True)

    consistency = pd.DataFrame(checks)
    if not consistency.match.all():
        consistency.to_csv(HERE / 'consistency_check.csv', index=False)
        sys.exit(f'Recomputed A/B values differ from {AB_METRICS}: '
                 f'{consistency[~consistency.match].head(10).to_dict("records")}')
    geometry = pd.DataFrame(geometry)
    geometry_ok = bool(geometry.same_size.all() and (geometry.spacing_max_diff <= 1e-3).all() and (geometry.origin_max_diff <= 1e-2).all()
                       and (geometry.direction_max_diff <= 1e-3).all())

    raw = pd.concat([stored, pd.DataFrame(c_rows)], ignore_index=True)
    raw['experiment'] = pd.Categorical(raw.experiment, ORDER)
    raw = raw.sort_values(['case_id', 'experiment']).reset_index(drop=True)
    df = sa.derive(raw)
    df['experiment'] = df.experiment.astype(str)

    agg = sa.aggregate(df)
    det = pooled_detection(df)
    full = sa.derive(pd.read_csv(AB_METRICS, keep_default_na=False, na_values=['']))
    subset_vs_full = pd.DataFrame([
        {'experiment': e, 'metric': label, 'full_145': f(full[full.experiment == e]), 'subset_10': f(df[df.experiment == e])}
        for e in ('A', 'B')
        for label, f in [('Dice mean', lambda g: g.DSC.mean()), ('Dice median', lambda g: g.DSC.median()),
                         ('FPV median ml', lambda g: g.FPV_ml.median()), ('FNV median ml', lambda g: g.FNV_ml.median()),
                         ('TMTV abs error median ml', lambda g: g.TMTV_abs_error_ml.median()),
                         ('Lesion recall C1 median', lambda g: g.recall_C1.median()),
                         ('Ref TMTV median ml', lambda g: g.GT_TMTV_ml.median()),
                         ('Ref lesions median', lambda g: g.gt_lesions.median())]])

    df.to_csv(HERE / 'per_exam_metrics_10.csv', index=False)
    agg.to_csv(HERE / 'aggregate_10.csv', index=False)
    det.to_csv(HERE / 'detection_pooled_10.csv', index=False)
    subset_vs_full.to_csv(HERE / 'subset_vs_full_AB.csv', index=False)
    consistency.to_csv(HERE / 'consistency_check.csv', index=False)
    geometry.to_csv(HERE / 'geometry_check.csv', index=False)
    (HERE / 'results_tables.md').write_text(results_markdown(df, agg, det, subset_vs_full, consistency, geometry_ok),
                                            encoding='utf-8')
    plot_all(df, det)
    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - started),
        'cases': ids, 'rows': len(df),
        'script_sha256': sha256(__file__),
        'evaluate_a_vs_b_sha256': sha256(AB_DIR / 'evaluate_a_vs_b.py'),
        'summarize_a_vs_b_sha256': sha256(AB_DIR / 'summarize_a_vs_b.py'),
        'microsoft_metrics_sha256': sha256(ev.MS_REPO / 'metrics/metrics.py'),
        'ab_per_exam_metrics_sha256': sha256(AB_METRICS),
        'subset_manifest_sha256': sha256(MANIFEST),
        'C_predictions_sha256': {i[:16]: sha256(C_PRED / f'{i}.nii.gz') for i in ids},
        'consistency': f'{int(consistency.match.sum())}/{len(consistency)} match',
        'geometry_ok': geometry_ok,
        'connectivity': ev.CONNECTIVITY, 'iou_threshold': ev.IOU_THRESHOLD,
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
                     'SimpleITK': sitk.Version.VersionString()},
    }
    (HERE / 'compute_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nConsistency: {manifest["consistency"]}; geometry ok: {geometry_ok}; {manifest["seconds"]} s')
    print((HERE / 'results_tables.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
