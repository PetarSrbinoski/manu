"""Review sheets for the anatomical check of large false-positive components (internal, not for presentation).

For every examination of the subset: coronal PET MIPs with the false-positive components of A and B
(>= 20 ml, numbered), and for each of those components the axial CT, axial PET and coronal CT through its
centroid, with the component outlined. These sheets are what the anatomical assessment in
fp_anatomy_review.csv is based on. Read-only. Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/visual_analysis_and_report/review_fp_sites.py
"""
import cc3d
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import viz_common as vc  # noqa: E402

OUT = vc.HERE / 'review'
MIN_ML = 20


def main():
    OUT.mkdir(exist_ok=True)
    table = pd.read_csv(vc.HERE / 'error_components.csv')
    for case in vc.subset_cases():
        cid = case['case_id']
        sites = table[(table.case_id == cid) & (table.kind == 'FP') & (table.model.isin(['A', 'B']))
                      & (table.volume_ml >= MIN_ML)].sort_values(['model', 'volume_ml'], ascending=[True, False])
        if sites.empty:
            continue
        image, a = vc.load(case)
        sx, sy, sz = image.GetSpacing()
        labels = {m: cc3d.connected_components(a[m].astype(np.uint8), connectivity=vc.ev.CONNECTIVITY) for m in ('A', 'B')}
        rows = 1 + len(sites)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        fig.patch.set_facecolor('white')
        suv_mip = vc.mip(a['SUV'])
        for col, m in enumerate(('A', 'B')):
            ax = axes[0, col]
            ax.imshow(1 - vc.suv_display(suv_mip), cmap='gray', vmin=0, vmax=1, aspect=sz / sx)
            for n, s in enumerate(sites[sites.model == m].itertuples(), 1):
                comp = vc.mip(labels[m] == s.component)
                ax.contour(comp, levels=[0.5], colors=vc.FP, linewidths=1.2)
                ax.text(s.centroid_i, a['SUV'].shape[0] - 1 - s.centroid_k, f'{m}{n}', color=vc.INK, fontsize=9,
                        ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
            ax.contour(vc.mip(a['GT']), levels=[0.5], colors=vc.TP, linewidths=0.8)
            ax.set_title(f'{cid[6:16]} coronal SUV MIP; FP >= {MIN_ML} ml of {m} (orange), reference (aqua)', fontsize=8)
            ax.axis('off')
        ax = axes[0, 2]
        ax.imshow(vc.ct_window(vc.coronal(a['CT'], a['CT'].shape[1] // 2)), cmap='gray', aspect=sz / sx)
        ax.set_title('coronal CT, middle slice (orientation check: head up, R on left)', fontsize=8)
        ax.axis('off')
        counters = {'A': 0, 'B': 0}
        for r, s in enumerate(sites.itertuples(), 1):
            counters[s.model] += 1
            k, j, i = int(round(s.centroid_k)), int(round(s.centroid_j)), int(round(s.centroid_i))
            comp = labels[s.model] == s.component
            if not comp[k].any():                      # centroid slice may miss a curved component
                k = int(np.argmax(comp.sum(axis=(1, 2))))
            if not comp[:, j, :].any():
                j = int(np.argmax(comp.sum(axis=(0, 2))))
            panels = [(vc.ct_window(vc.axial(a['CT'], k)), vc.axial(comp, k), 'gray', 1, f'axial CT k={k}'),
                      (1 - vc.suv_display(vc.axial(a['SUV'], k)), vc.axial(comp, k), 'gray', 1, f'axial SUV k={k}'),
                      (vc.ct_window(vc.coronal(a['CT'], j)), vc.coronal(comp, j), 'gray', sz / sx, f'coronal CT j={j}')]
            for c, (img, mask, cmap, aspect, title) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect=aspect)
                ax.contour(mask, levels=[0.5], colors=vc.FP, linewidths=1.2)
                ax.set_title(f'{s.model}{counters[s.model]}: {s.volume_ml:.0f} ml, SUVmax {s.SUVmax:.1f}, '
                             f'CT {s.CT_mean_HU:.0f} HU, {s.cm_below_top_of_scan:.0f} cm below top — {title}', fontsize=7)
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(OUT / f'{cid[6:16]}.png', dpi=80)
        plt.close(fig)
        print('written', cid[6:16], len(sites), 'sites', flush=True)


if __name__ == '__main__':
    main()
