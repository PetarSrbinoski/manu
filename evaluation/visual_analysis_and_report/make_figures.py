"""Presentation figures for the visual error analysis (read-only with respect to all inputs).

Figures (figures/, 300 dpi PNG; examinations shown under anonymised labels E1-E7):
  fig01_overview_mip.png        coronal SUV MIP per example, reference and component-level errors of A, B, C
  fig02..fig08_E*.png           per example: CT | SUV | reference | A | B | C on fixed axial and coronal slices
  fig09_fp_anatomy.png          share of all false-positive volume by visually identified structure (A, B)

Error colours, identical in every figure:
  TP aqua   = predicted voxel inside the reference      FP orange = predicted voxel outside the reference
  FN violet = reference voxel not predicted             reference = light fill (reference panel) / outline
Slice panels use voxel-level classes. The MIP overview uses the component-level classes of the evaluation
(FP = predicted lesion touching no reference lesion, FN = reference lesion touched by no prediction), so it
shows exactly what FPV and FNV measure.

Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/visual_analysis_and_report/make_figures.py
"""
import json
import textwrap

import cc3d
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import viz_common as vc  # noqa: E402

FIG = vc.HERE / 'figures'
DPI = 300
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})

# label, exam short id, reason, slices: list of (view, index, purpose)
EXAMPLES = [
    ('E1', '30001118d0', 'High tumour burden; A better than B; B misses many small lesions',
     [('axial', 179, 'largest reference area'), ('axial', 311, 'largest A/B false-positive area'), ('coronal', 178, 'largest reference area')]),
    ('E2', '6a3477cd9a', 'Most lesions (157); all three models miss small lesions',
     [('axial', 144, 'largest reference area'), ('axial', 112, 'false positives at both kidneys'), ('coronal', 180, 'largest reference area')]),
    ('E3', '249dd35d0c', 'B better than A; severe false positives of A',
     [('axial', 255, 'largest reference area'), ('axial', 297, 'largest A false-positive component'), ('coronal', 209, 'largest reference area')]),
    ('E4', '1fe6e48293', 'Largest false-positive volume of A (1376 ml) in several organs',
     [('axial', 93, 'largest reference area'), ('axial', 190, 'false positives of A and B in the heart region'), ('coronal', 184, 'largest reference area')]),
    ('E5', '62405d117c', 'autoPET III best case (Dice 0.977); A better than B; B under-segments the main lesion',
     [('axial', 230, 'largest reference area'), ('axial', 66, 'largest A/B false-positive area'), ('coronal', 236, 'largest reference area')]),
    ('E6', 'f47e31ceb5', 'Smallest tumour burden (6.7 ml): Dice of A and B near 0 because of false positives',
     [('axial', 222, 'largest reference area'), ('axial', 68, 'largest A false-positive component'), ('coronal', 211, 'largest reference area')]),
    ('E7', '22d07bdc49', 'A better than B; B has more false-positive volume',
     [('axial', 65, 'largest reference area'), ('axial', 53, 'largest A/B false-positive area'), ('coronal', 211, 'largest reference area')]),
]
LEGEND = [Patch(color=vc.TP, label='TP: predicted, in reference'), Patch(color=vc.FP, label='FP: predicted, not in reference'),
          Patch(color=vc.FN, label='FN: reference, not predicted'), Patch(facecolor='#f5f5f2', edgecolor=vc.MUTED, label='reference (ground truth)')]
FOOT = '* autoPET III was trained on autoPET data that include these examinations: orientation only, not a held-out result.'


def body_box(ct2d, margin=8):
    rows, cols = np.nonzero(ct2d > -400)
    if rows.size == 0:
        return slice(None), slice(None)
    return (slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
            slice(max(cols.min() - margin, 0), cols.max() + margin + 1))


def overlay(gt2d, pred2d):
    out = np.zeros(gt2d.shape + (4,))
    out[gt2d & pred2d] = to_rgba(vc.TP, 0.9)
    out[pred2d & ~gt2d] = to_rgba(vc.FP, 0.9)
    out[gt2d & ~pred2d] = to_rgba(vc.FN, 0.9)
    return out


def view2d(vol, view, index):
    return vc.axial(vol, index) if view == 'axial' else vc.coronal(vol, index)


def frame(ax, spine=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(spine)


def orientation_marks(ax, view, shape):
    h, w = shape
    top = 'A' if view == 'axial' else 'S'
    for txt, x, y, ha, va in ((top, w / 2, 2, 'center', 'top'), ('R', 2, h / 2, 'left', 'center'), ('L', w - 3, h / 2, 'right', 'center')):
        ax.text(x, y, txt, color='#d8d7d2', fontsize=6, ha=ha, va=va)


def case_figure(label, short, reason, slices, case, image, a, metrics, number):
    sx, sy, sz = image.GetSpacing()
    nz = a['CT'].shape[0]
    cols = ['CT', 'PET (SUV)', 'Reference', vc.MODEL_NAMES['A'], vc.MODEL_NAMES['B'], vc.MODEL_NAMES['C']]
    heights = [1.75 if v == 'axial' else 3.3 for v, _, _ in slices]
    fig, axes = plt.subplots(len(slices), 6, figsize=(12, sum(heights) + 1.3), gridspec_kw={'height_ratios': heights})
    fig.patch.set_facecolor('white')
    for r, (view, index, purpose) in enumerate(slices):
        ct, suv, gt = (view2d(a[k], view, index) for k in ('CT', 'SUV', 'GT'))
        box = body_box(ct)
        aspect = 1.0 if view == 'axial' else sz / sx
        ct_img = vc.ct_window(ct)[box]
        for c in range(6):
            ax = axes[r, c]
            if c == 1:
                ax.imshow(1 - vc.suv_display(suv[box]), cmap='gray', vmin=0, vmax=1, aspect=aspect, interpolation='nearest')
            else:
                ax.imshow(ct_img * (0.75 if c >= 2 else 1), cmap='gray', vmin=0, vmax=1, aspect=aspect, interpolation='nearest')
            if c == 2:
                ref = np.zeros(gt[box].shape + (4,))
                ref[gt[box]] = to_rgba('#f5f5f2', 0.85)
                ax.imshow(ref, aspect=aspect, interpolation='nearest')
            if c >= 3:
                model = 'ABC'[c - 3]
                ax.imshow(overlay(gt[box], view2d(a[model], view, index)[box]), aspect=aspect, interpolation='nearest')
            frame(ax)
            orientation_marks(ax, view, ct_img.shape)
            if r == 0:
                ax.set_title(cols[c], fontsize=8.5, color=vc.INK)
        if view == 'axial':
            where = f'axial {index + 1}/{nz}, {(nz - 1 - index) * sz / 10:.0f} cm below top'
        else:
            where = f'coronal {index + 1}/{a["CT"].shape[1]}'
        axes[r, 0].set_ylabel(where + '\n' + textwrap.fill(purpose, 26), fontsize=6, color=vc.MUTED)
    m = metrics.set_index('experiment')
    line = '   '.join(f"{e}: Dice {m.loc[e, 'DSC']:.3f}, FPV {m.loc[e, 'FPV_ml']:.1f} ml, FNV {m.loc[e, 'FNV_ml']:.1f} ml"
                       for e in 'ABC')
    fig.suptitle(f'{label}  —  {reason}', x=0.01, ha='left', fontsize=10, color=vc.INK)
    fig.text(0.01, 1 - 0.55 / (sum(heights) + 1.3), f"Reference TMTV {m.loc['A', 'GT_TMTV_ml']:.1f} ml, {int(m.loc['A', 'gt_lesions'])} lesions.   {line}",
             fontsize=7, color=vc.MUTED)
    h = sum(heights) + 1.3
    fig.legend(handles=LEGEND, loc='lower left', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.01, 0.2 / h))
    fig.text(0.01, 0.06 / h, f'CT window W400/L40 HU; SUV 0–10 inverted grey. Radiological view (patient right on image left). {FOOT}',
             fontsize=5.5, color=vc.MUTED, ha='left')
    fig.tight_layout(rect=(0, 0.55 / h, 1, 1 - 0.7 / h))
    path = FIG / f'fig{number:02d}_{label}.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def component_classes(gt, pred):
    """Component-level classes as in the evaluation (18-connectivity)."""
    pl = cc3d.connected_components(pred.astype(np.uint8), connectivity=vc.ev.CONNECTIVITY)
    gl = cc3d.connected_components(gt.astype(np.uint8), connectivity=vc.ev.CONNECTIVITY)
    touched_pred = np.unique(pl[gt])
    touched_gt = np.unique(gl[pred])
    fp = (pl > 0) & ~np.isin(pl, touched_pred)
    fn = (gl > 0) & ~np.isin(gl, touched_gt)
    tp = (pl > 0) & ~fp
    return tp, fp, fn


def overview(loaded, metrics):
    rows = ['Reference', vc.MODEL_NAMES['A'], vc.MODEL_NAMES['B'], vc.MODEL_NAMES['C']]
    fig, axes = plt.subplots(4, len(EXAMPLES), figsize=(1.75 * len(EXAMPLES) + 0.6, 12.6))
    fig.patch.set_facecolor('white')
    for c, (label, short, *_rest) in enumerate(EXAMPLES):
        image, a = loaded[short]
        sx, sy, sz = image.GetSpacing()
        suv = 1 - vc.suv_display(vc.mip(a['SUV']))
        m = metrics[metrics.exam == short].set_index('experiment')
        for r in range(4):
            ax = axes[r, c]
            background = 1 - 0.4 * (1 - suv) if r == 0 else suv     # reference row: faded uptake, lesions solid
            ax.imshow(background, cmap='gray', vmin=0, vmax=1, aspect=sz / sx, interpolation='nearest')
            layer = np.zeros(suv.shape + (4,))
            if r == 0:
                layer[vc.mip(a['GT'])] = to_rgba(vc.INK, 1.0)
                sub = f"TMTV {m.loc['A', 'GT_TMTV_ml']:.0f} ml, {int(m.loc['A', 'gt_lesions'])} les."
            else:
                model = 'ABC'[r - 1]
                tp, fp, fn = component_classes(a['GT'], a[model])
                for mask, colour in ((tp, vc.TP), (fn, vc.FN), (fp, vc.FP)):    # later layers on top
                    layer[vc.mip(mask)] = to_rgba(colour, 0.9)
                sub = f"Dice {m.loc[model, 'DSC']:.2f} · FPV {m.loc[model, 'FPV_ml']:.0f} ml"
            ax.imshow(layer, aspect=sz / sx, interpolation='nearest')
            frame(ax)
            ax.set_title(sub, fontsize=6.5, color=vc.MUTED, pad=2)
            if r == 0:
                ax.text(0.5, 1.09, label, transform=ax.transAxes, ha='center', fontsize=10, color=vc.INK)
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=8.5, color=vc.INK)
    legend = [Patch(color=vc.INK, label='reference lesions (uptake faded in this row)'),
              Patch(color=vc.TP, label='predicted lesion touching a reference lesion'),
              Patch(color=vc.FP, label='false-positive lesion (touches no reference lesion)'),
              Patch(color=vc.FN, label='missed reference lesion (no predicted voxel)')]
    fig.legend(handles=legend, loc='lower left', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.01, 0.018))
    fig.suptitle('Coronal maximum-intensity projections of SUV with component-level errors (as used for FPV and FNV)',
                 x=0.01, ha='left', fontsize=10, color=vc.INK)
    fig.text(0.01, 0.004, f'Projection: overlapping classes drawn FP over FN over TP. SUV 0–10 inverted grey. {FOOT}',
             fontsize=5.5, color=vc.MUTED, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    path = FIG / 'fig01_overview_mip.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def anatomy_figure():
    s = pd.read_csv(vc.HERE / 'fp_anatomy_summary.csv')
    groups = ['brain', 'bladder', 'heart', 'kidney', 'unassigned', 'components < 20 ml (not reviewed)']
    names = ['Brain', 'Urinary bladder', 'Heart', 'Kidneys / renal\nregion', 'Unassigned\n(≥ 20 ml)', 'Small components\n< 20 ml (not reviewed)']
    colours = {'A': '#2a78d6', 'B': '#eb6834'}
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(vc.SURFACE)
    y = np.arange(len(groups))
    for k, model in enumerate('AB'):
        sub = s[s.model == model].set_index('group').reindex(groups)
        total = s[(s.model == model) & (s.group == 'all FP components')].volume_ml.iloc[0]
        vals = sub.share_of_all_FP_volume.to_numpy() * 100
        ax.barh(y + (k - 0.5) * 0.38, vals, 0.36, color=colours[model], edgecolor='white', linewidth=2,
                label=f"{'A: Microsoft 10 ep' if model == 'A' else 'B: testTraining'} (total FP volume {total:.0f} ml)")
        for yy, v, ml in zip(y + (k - 0.5) * 0.38, vals, sub.volume_ml):
            ax.text(v + 0.8, yy, f'{v:.1f}%  ({ml:.0f} ml)', va='center', fontsize=6.5, color=vc.MUTED)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 58)
    ax.set_xlabel('share of all false-positive volume in the 10 examinations (%)', fontsize=7.5, color=vc.MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', color=vc.GRID)
    ax.set_axisbelow(True)
    ax.set_title('Where the false-positive volume of A and B lies (visual identification of components ≥ 20 ml)',
                 fontsize=9, loc='left', color=vc.INK)
    ax.legend(loc='lower right', ncol=1, frameon=False, fontsize=7)
    fig.text(0.01, 0.01, 'Structures were identified visually on CT and PET (review/), not by organ segmentation. '
             'autoPET III (C) has 0.9 ml mean FP volume and is not shown.', fontsize=5.8, color=vc.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIG / 'fig09_fp_anatomy.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main():
    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(vc.PROJECT / 'evaluation' / 'autopet3_vs_a_b' / 'per_exam_metrics_10.csv')
    metrics['exam'] = metrics.case_id.str[6:16]
    cases = {c['case_id'][6:16]: c for c in vc.subset_cases()}
    loaded, written = {}, []
    for number, (label, short, reason, slices) in enumerate(EXAMPLES, 2):
        image, a = vc.load(cases[short])          # re-verifies the full geometry of all six images
        loaded[short] = (image, a)
        for view, index, _ in slices:
            limit = a['CT'].shape[0] if view == 'axial' else a['CT'].shape[1]
            if not 0 <= index < limit:
                raise ValueError(f'{label}: {view} index {index} outside 0..{limit - 1}')
        written.append(case_figure(label, short, reason, slices, cases[short], image, a,
                                   metrics[metrics.exam == short], number))
        print('written', written[-1].name, flush=True)
    written.append(overview(loaded, metrics))
    written.append(anatomy_figure())
    mapping = [{'label': l, 'exam_id': cases[s]['case_id'], 'reason': r,
                'slices': [{'view': v, 'index0': i, 'purpose': p} for v, i, p in sl]} for l, s, r, sl in EXAMPLES]
    (vc.HERE / 'selected_examples.json').write_text(json.dumps(mapping, indent=2))
    for p in written:
        print(p.name)


if __name__ == '__main__':
    main()
