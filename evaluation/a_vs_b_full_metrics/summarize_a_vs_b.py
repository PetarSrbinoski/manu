"""Aggregate tables, paired A-vs-B comparisons and plots from per_exam_metrics.csv.

Run inside WSL after evaluate_a_vs_b.py:
    ~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/summarize_a_vs_b.py

Writes (next to this file): aggregate_by_experiment.csv, detection_pooled.csv, paired_tests.csv,
per_exam_paired.csv, results_tables.md and plots/*.png. Inputs are only read.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'
COLORS = {'A': '#2a78d6', 'B': '#eb6834'}          # categorical slots 1 and 2, fixed order
NAMES = {'A': 'A: Microsoft, 10 epochs', 'B': 'B: testTraining, early-stopped'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BOOT = 10000
RNG = np.random.default_rng(20261018)


# ---------------------------------------------------------------- per-exam derived columns
def derive(df):
    df = df.copy()
    df['TMTV_error_ml'] = df.Pred_TMTV_ml - df.GT_TMTV_ml
    df['TMTV_abs_error_ml'] = df.TMTV_error_ml.abs()
    df['TMTV_rel_error'] = df.TMTV_error_ml / df.GT_TMTV_ml
    df['SUVmax_error'] = df.Pred_SUVmax - df.GT_SUVmax
    df['SUVmax_abs_error'] = df.SUVmax_error.abs()
    df['SUVmax_rel_error'] = df.SUVmax_error / df.GT_SUVmax
    df['SUVmax_exact'] = np.isclose(df.Pred_SUVmax, df.GT_SUVmax)
    df['TLG_abs_error'] = (df.Pred_TLG - df.GT_TLG).abs()
    df['lesion_count_error'] = df.pred_lesions - df.gt_lesions
    for c in ('C1', 'C2', 'C3'):
        df[f'sens_ms_{c}'] = df[f'TP_{c}'] / (df[f'TP_{c}'] + df[f'FN_{c}'])        # Microsoft's definition
        df[f'recall_{c}'] = (df.gt_lesions - df[f'FN_{c}']) / df.gt_lesions           # share of GT lesions found
        df[f'precision_{c}'] = df[f'TP_{c}'] / (df[f'TP_{c}'] + df[f'FP_{c}']).replace(0, np.nan)
    return df


# ---------------------------------------------------------------- statistics (no scipy available)
def wilcoxon(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float('nan'), float('nan')
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w = ranks[d > 0].sum()
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = (w - n * (n + 1) / 4) / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))


def cluster_bootstrap(values, patients, statistic=np.mean):
    """Resample patients (both exams of the repeated patient stay together)."""
    groups = pd.Series(values).groupby(np.asarray(patients)).apply(list).tolist()
    stats = []
    for _ in range(BOOT):
        pick = RNG.integers(0, len(groups), len(groups))
        stats.append(statistic(np.concatenate([groups[i] for i in pick])))
    return np.percentile(stats, [2.5, 97.5])


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def q(series):
    s = pd.Series(series).dropna()
    return f'{s.median():.3g} [{s.quantile(.25):.3g}, {s.quantile(.75):.3g}]'


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


# ---------------------------------------------------------------- tables
def aggregate(df):
    rows = []
    for e, g in df.groupby('experiment'):
        row = {'experiment': e, 'examinations': len(g), 'patients': g.patient.nunique(),
               'DSC_mean': g.DSC.mean(), 'DSC_sd': g.DSC.std(), 'DSC_median': g.DSC.median(),
               'DSC_q25': g.DSC.quantile(.25), 'DSC_q75': g.DSC.quantile(.75),
               'DSC_zero': int((g.DSC == 0).sum()), 'DSC_below_0.1': int((g.DSC < 0.1).sum())}
        for m in ('FPV_ml', 'FNV_ml', 'TMTV_abs_error_ml', 'TMTV_error_ml', 'TMTV_rel_error',
                  'SUVmax_abs_error', 'SUVmax_error', 'SUVmax_rel_error', 'TLG_abs_error',
                  'pred_lesions', 'lesion_count_error'):
            row[f'{m}_mean'] = g[m].mean()
            row[f'{m}_median'] = g[m].median()
            row[f'{m}_q25'] = g[m].quantile(.25)
            row[f'{m}_q75'] = g[m].quantile(.75)
        row['FNV_zero_exams'] = int((g.FNV_ml == 0).sum())
        row['FPV_zero_exams'] = int((g.FPV_ml == 0).sum())
        row['TMTV_pearson'] = g.Pred_TMTV_ml.corr(g.GT_TMTV_ml)
        row['TMTV_spearman'] = spearman(g.Pred_TMTV_ml, g.GT_TMTV_ml)
        row['SUVmax_pearson'] = g.Pred_SUVmax.corr(g.GT_SUVmax)
        row['SUVmax_spearman'] = spearman(g.Pred_SUVmax, g.GT_SUVmax)
        row['SUVmax_exact_match'] = int(g.SUVmax_exact.sum())
        for c in ('C1', 'C2', 'C3'):
            row[f'sens_ms_{c}_median'] = g[f'sens_ms_{c}'].median()
            row[f'FP_{c}_median'] = g[f'FP_{c}'].median()
            row[f'FP_{c}_mean'] = g[f'FP_{c}'].mean()
            row[f'recall_{c}_median'] = g[f'recall_{c}'].median()
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_detection(df):
    def pooled(g, c):
        gt = g.gt_lesions.sum()
        found = (g.gt_lesions - g[f'FN_{c}']).sum()
        tp, fp = g[f'TP_{c}'].sum(), g[f'FP_{c}'].sum()
        recall, precision = found / gt, tp / (tp + fp)
        return recall, precision, 2 * recall * precision / (recall + precision)

    rows = []
    for e, g in df.groupby('experiment'):
        groups = [x for _, x in g.groupby('patient')]
        for c in ('C1', 'C2', 'C3'):
            recall, precision, f1 = pooled(g, c)
            boot = []
            for _ in range(2000):
                boot.append(pooled(pd.concat([groups[i] for i in RNG.integers(0, len(groups), len(groups))]), c))
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            rows.append({'experiment': e, 'criterion': c, 'gt_lesions': int(g.gt_lesions.sum()),
                         'gt_lesions_found': int((g.gt_lesions - g[f'FN_{c}']).sum()),
                         'pred_TP': int(g[f'TP_{c}'].sum()), 'pred_FP': int(g[f'FP_{c}'].sum()),
                         'recall': recall, 'recall_ci_low': lo[0], 'recall_ci_high': hi[0],
                         'precision': precision, 'precision_ci_low': lo[1], 'precision_ci_high': hi[1],
                         'F1': f1, 'F1_ci_low': lo[2], 'F1_ci_high': hi[2],
                         'exams_all_gt_found': int((g[f'FN_{c}'] == 0).sum()),
                         'exams_no_gt_found': int((g[f'FN_{c}'] == g.gt_lesions).sum())})
    return pd.DataFrame(rows)


PAIRED = [('DSC', 'higher'), ('FPV_ml', 'lower'), ('FNV_ml', 'lower'),
          ('TMTV_abs_error_ml', 'lower'), ('SUVmax_abs_error', 'lower'), ('TLG_abs_error', 'lower'),
          ('recall_C1', 'higher'), ('recall_C2', 'higher'), ('recall_C3', 'higher'),
          ('FP_C1', 'lower'), ('FP_C2', 'lower'), ('FP_C3', 'lower'),
          ('precision_C1', 'higher'), ('precision_C2', 'higher'), ('precision_C3', 'higher')]


def paired(df):
    wide = df.pivot(index='case_id', columns='experiment')
    patients = wide['patient']['A']
    rows = []
    for metric, better in PAIRED:
        a, b = wide[metric]['A'], wide[metric]['B']
        ok = a.notna() & b.notna()
        d = (b - a)[ok]
        z, p = wilcoxon(d)
        lo, hi = cluster_bootstrap(d.to_numpy(), patients[ok].to_numpy())
        rows.append({'metric': metric, 'better_if': better, 'n': int(ok.sum()),
                     'A_mean': a[ok].mean(), 'B_mean': b[ok].mean(), 'A_median': a[ok].median(), 'B_median': b[ok].median(),
                     'mean_diff_B_minus_A': d.mean(), 'boot95_low': lo, 'boot95_high': hi,
                     'median_diff_B_minus_A': d.median(), 'B_better': int(((d > 0) if better == 'higher' else (d < 0)).sum()),
                     'A_better': int(((d < 0) if better == 'higher' else (d > 0)).sum()), 'ties': int((d == 0).sum()),
                     'wilcoxon_z': z, 'wilcoxon_p': p})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm'] = holm(out.wilcoxon_p)
    return out


# ---------------------------------------------------------------- plots
def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_paired_dice(df):
    wide = df.pivot(index='case_id', columns='experiment', values='DSC')
    fig, ax = figure(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle='--')
    ax.scatter(wide.A, wide.B, s=36, color=COLORS['B'], edgecolor=SURFACE, linewidth=1.5, alpha=0.9)
    ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1)
    style(ax, 'Dice per examination (145, paired)', 'A: Microsoft, 10 epochs', 'B: testTraining, early-stopped')
    ax.text(0.97, 0.03, 'below the line: A higher', ha='right', color=MUTED, fontsize=8, transform=ax.transAxes)
    save(fig, 'dice_paired_scatter.png')


def plot_distributions(df):
    metrics = [('DSC', 'Dice', False), ('FPV_ml', 'False-positive volume (ml)', True),
               ('FNV_ml', 'False-negative volume (ml)', True), ('TMTV_abs_error_ml', '|TMTV error| (ml)', True),
               ('SUVmax_abs_error', '|SUVmax error|', True), ('pred_lesions', 'Predicted lesions (18-conn.)', True)]
    fig, axes = figure(2, 3, figsize=(11, 6.4))
    for ax, (m, label, log) in zip(axes.ravel(), metrics):
        data = [df[df.experiment == e][m].to_numpy(float) for e in ('A', 'B')]
        shown = [np.log10(x + 1) if log else x for x in data]
        parts = ax.boxplot(shown, widths=0.5, patch_artist=True, showfliers=False,
                           medianprops={'color': INK, 'linewidth': 2})
        for patch, e in zip(parts['boxes'], ('A', 'B')):
            patch.set_facecolor(COLORS[e]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[e])
        for i, (x, e) in enumerate(zip(shown, ('A', 'B')), 1):
            jitter = RNG.uniform(-0.12, 0.12, len(x))
            ax.scatter(i + jitter, x, s=10, color=COLORS[e], alpha=0.7, linewidth=0)
        ax.set_xticks([1, 2], ['A', 'B'])
        if log:
            ticks = [0, 1, 10, 100, 1000, 10000]
            ticks = [t for t in ticks if t <= max(np.nanmax(x) for x in data) * 1.5]
            ax.set_yticks(np.log10(np.array(ticks) + 1), [str(t) for t in ticks])
        style(ax, label)
    fig.suptitle('Per-examination metrics (145 each; boxes = median and IQR; log scale where values span decades)',
                 color=INK, fontsize=10, x=0.01, ha='left')
    save(fig, 'metric_distributions.png')


def plot_detection(pooled):
    fig, axes = figure(1, 3, figsize=(11, 3.8), sharey=True)
    labels = {'C1': 'Criterion 1: any overlap', 'C2': 'Criterion 2: IoU >= 0.5', 'C3': 'Criterion 3: covers SUVmax voxel'}
    for ax, c in zip(axes, ('C1', 'C2', 'C3')):
        for k, (measure, offset) in enumerate((('recall', -0.18), ('precision', 0.18))):
            for j, e in enumerate(('A', 'B')):
                r = pooled[(pooled.experiment == e) & (pooled.criterion == c)].iloc[0]
                x = k + (j - 0.5) * 0.36
                ax.bar(x, r[measure], width=0.32, color=COLORS[e], label=NAMES[e] if (c == 'C1' and k == 0) else None)
                ax.errorbar(x, r[measure], yerr=[[r[measure] - r[f'{measure}_ci_low']], [r[f'{measure}_ci_high'] - r[measure]]],
                            color=INK, capsize=3, linewidth=1)
                ax.text(x, r[f'{measure}_ci_high'] + 0.02, f'{r[measure]:.2f}', ha='center', color=INK, fontsize=8)
        ax.set_xticks([0, 1], ['lesion recall', 'lesion precision'])
        ax.set_ylim(0, 1.08)
        style(ax, labels[c])
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('Lesion-level detection, pooled over all lesions (95% patient-bootstrap CI)',
                 color=INK, fontsize=10, x=0.01, ha='left')
    save(fig, 'detection_pooled.png')


def plot_agreement(df, gt, pred, title, unit, name, log):
    fig, axes = figure(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    lo = min(df[gt].min(), df[pred].min()); hi = max(df[gt].max(), df[pred].max())
    for ax, e in zip(axes, ('A', 'B')):
        g = df[df.experiment == e]
        ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle='--', linewidth=1)
        ax.scatter(g[gt], g[pred], s=28, color=COLORS[e], edgecolor=SURFACE, linewidth=1.2, alpha=0.9)
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
        style(ax, f'{NAMES[e]}  (Spearman {spearman(g[pred], g[gt]):.2f})', f'reference {unit}', f'predicted {unit}')
    fig.suptitle(f'{title}: prediction vs reference per examination (dashed = perfect agreement)',
                 color=INK, fontsize=10, x=0.01, ha='left')
    save(fig, name)


def plot_dice_by_volume(df):
    wide = df.pivot(index='case_id', columns='experiment', values=['DSC', 'GT_TMTV_ml'])
    gt = wide['GT_TMTV_ml']['A']
    bins = pd.qcut(gt, 3, labels=['small', 'medium', 'large'])
    fig, ax = figure(figsize=(6.4, 4))
    for j, e in enumerate(('A', 'B')):
        means = wide['DSC'][e].groupby(bins, observed=True).mean()
        x = np.arange(3) + (j - 0.5) * 0.36
        ax.bar(x, means.to_numpy(), width=0.32, color=COLORS[e], label=NAMES[e])
        for xi, v in zip(x, means.to_numpy()):
            ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=8, color=INK)
    edges = [f'{gt[bins == b].min():.0f}-{gt[bins == b].max():.0f} ml' for b in ('small', 'medium', 'large')]
    ax.set_xticks(range(3), [f'{b}\n{e}' for b, e in zip(('small', 'medium', 'large'), edges)])
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    style(ax, 'Mean Dice by reference tumour volume (tertiles, ~48 exams each)', None, 'mean Dice')
    save(fig, 'dice_by_volume_tertile.png')


# ---------------------------------------------------------------- report tables
def md_table(frame, floatfmt='{:.3f}'):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v) if abs(v) < 1e5 else f'{v:.3g}'
        return str(v)
    head = '| ' + ' | '.join(frame.columns) + ' |\n|' + '---|' * len(frame.columns) + '\n'
    return head + '\n'.join('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False))


def results_markdown(df, agg, pooled, tests):
    a, b = (agg.set_index('experiment').loc[e] for e in ('A', 'B'))
    lines = ['# A vs B: results tables (generated by summarize_a_vs_b.py)', '',
             'Interpretation and methodology: `README.md` in this folder.', '',
             '## Per-examination metrics (median [IQR] unless stated)', '',
             '| Metric | A: Microsoft, 10 ep | B: testTraining |', '|---|---|---|']
    g = {e: df[df.experiment == e] for e in ('A', 'B')}
    def both(label, f):
        lines.append(f'| {label} | {f(g["A"])} | {f(g["B"])} |')
    both('Examinations / patients', lambda x: f'{len(x)} / {x.patient.nunique()}')
    both('Dice mean ± SD', lambda x: f'{x.DSC.mean():.3f} ± {x.DSC.std():.3f}')
    both('Dice median [IQR]', lambda x: q(x.DSC))
    both('False-positive volume (ml)', lambda x: q(x.FPV_ml))
    both('False-positive volume, mean (ml)', lambda x: f'{x.FPV_ml.mean():.1f}')
    both('False-negative volume (ml)', lambda x: q(x.FNV_ml))
    both('False-negative volume, mean (ml)', lambda x: f'{x.FNV_ml.mean():.1f}')
    both('Exams with FNV = 0', lambda x: f'{(x.FNV_ml == 0).sum()}')
    both('Reference TMTV (ml)', lambda x: q(x.GT_TMTV_ml))
    both('Predicted TMTV (ml)', lambda x: q(x.Pred_TMTV_ml))
    both('TMTV error, signed (ml)', lambda x: q(x.TMTV_error_ml))
    both('TMTV absolute error (ml), mean / median', lambda x: f'{x.TMTV_abs_error_ml.mean():.1f} / {x.TMTV_abs_error_ml.median():.1f}')
    both('TMTV relative error, signed', lambda x: q(x.TMTV_rel_error))
    both('TMTV Spearman ρ (pred vs ref)', lambda x: f'{spearman(x.Pred_TMTV_ml, x.GT_TMTV_ml):.3f}')
    both('SUVmax error, signed', lambda x: q(x.SUVmax_error))
    both('SUVmax absolute error, mean / median', lambda x: f'{x.SUVmax_abs_error.mean():.2f} / {x.SUVmax_abs_error.median():.2f}')
    both('SUVmax relative error, signed', lambda x: q(x.SUVmax_rel_error))
    both('SUVmax identical to reference (exams)', lambda x: f'{int(x.SUVmax_exact.sum())}')
    both('SUVmax Spearman ρ', lambda x: f'{spearman(x.Pred_SUVmax, x.GT_SUVmax):.3f}')
    both('TLG absolute error', lambda x: q(x.TLG_abs_error))
    both('Reference lesions per exam (18-conn.)', lambda x: q(x.gt_lesions))
    both('Predicted lesions per exam', lambda x: q(x.pred_lesions))
    for c, name in (('C1', 'any overlap'), ('C2', 'IoU ≥ 0.5'), ('C3', 'SUVmax voxel')):
        both(f'Sensitivity, Microsoft TP/(TP+FN), {c} ({name})', lambda x, c=c: q(x[f'sens_ms_{c}']))
        both(f'Share of reference lesions found, {c}', lambda x, c=c: q(x[f'recall_{c}']))
        both(f'False-positive lesions per exam, {c}', lambda x, c=c: q(x[f'FP_{c}']))
    lines += ['', '## Lesion detection pooled over all lesions (95% patient-bootstrap CI)', '']
    p = pooled.copy()
    p['recall (CI)'] = p.apply(lambda r: f'{r.recall:.3f} ({r.recall_ci_low:.3f}–{r.recall_ci_high:.3f})', axis=1)
    p['precision (CI)'] = p.apply(lambda r: f'{r.precision:.3f} ({r.precision_ci_low:.3f}–{r.precision_ci_high:.3f})', axis=1)
    p['F1 (CI)'] = p.apply(lambda r: f'{r.F1:.3f} ({r.F1_ci_low:.3f}–{r.F1_ci_high:.3f})', axis=1)
    lines.append(md_table(p[['experiment', 'criterion', 'gt_lesions', 'gt_lesions_found', 'pred_TP', 'pred_FP',
                             'recall (CI)', 'precision (CI)', 'F1 (CI)', 'exams_all_gt_found', 'exams_no_gt_found']]))
    lines += ['', '## Paired A vs B per examination (Δ = B − A)', '',
              'Wilcoxon signed-rank (normal approximation, tie-corrected), Holm adjustment over the 15 metrics. '
              'Bootstrap resamples patients. These tests reflect case sampling only, not training variance '
              '(one training run per configuration).', '']
    t = tests.copy()
    t['mean Δ (95% CI)'] = t.apply(lambda r: f'{r.mean_diff_B_minus_A:.3g} ({r.boot95_low:.3g} to {r.boot95_high:.3g})', axis=1)
    lines.append(md_table(t[['metric', 'better_if', 'n', 'A_mean', 'B_mean', 'A_median', 'B_median', 'mean Δ (95% CI)',
                             'median_diff_B_minus_A', 'B_better', 'A_better', 'ties', 'wilcoxon_p', 'wilcoxon_p_holm']], '{:.4g}'))
    lines += ['', '## Notes from the computation', '']
    notes = df[df.notes.fillna('') != '']
    lines.append('None: no Microsoft edge case (list.remove error, SUVmax ≤ 0) occurred.' if notes.empty else
                 md_table(notes[['experiment', 'case_id', 'notes']]))
    return '\n'.join(lines) + '\n'


def main():
    df = derive(pd.read_csv(HERE / 'per_exam_metrics.csv'))
    if sorted(df.groupby('experiment').size().tolist()) != [145, 145] or df.duplicated(['experiment', 'case_id']).any():
        raise SystemExit('Expected exactly 145 unique examinations per experiment')
    PLOTS.mkdir(exist_ok=True)
    agg, pooled, tests = aggregate(df), pooled_detection(df), paired(df)
    agg.to_csv(HERE / 'aggregate_by_experiment.csv', index=False)
    pooled.to_csv(HERE / 'detection_pooled.csv', index=False)
    tests.to_csv(HERE / 'paired_tests.csv', index=False)
    wide = df.pivot(index='case_id', columns='experiment')
    wide.columns = [f'{m}_{e}' for m, e in wide.columns]
    wide.to_csv(HERE / 'per_exam_paired.csv')
    plot_paired_dice(df)
    plot_distributions(df)
    plot_detection(pooled)
    plot_agreement(df, 'GT_TMTV_ml', 'Pred_TMTV_ml', 'TMTV', '(ml)', 'tmtv_agreement.png', True)
    plot_agreement(df, 'GT_SUVmax', 'Pred_SUVmax', 'SUVmax', 'SUVmax', 'suvmax_agreement.png', True)
    plot_dice_by_volume(df)
    (HERE / 'results_tables.md').write_text(results_markdown(df, agg, pooled, tests), encoding='utf-8')
    print(tests[['metric', 'A_mean', 'B_mean', 'mean_diff_B_minus_A', 'boot95_low', 'boot95_high', 'wilcoxon_p', 'wilcoxon_p_holm']].to_string())
    print(pooled[['experiment', 'criterion', 'recall', 'precision', 'F1']].to_string())


if __name__ == '__main__':
    main()
