"""Reproducible single-column DARC figures from audited experiment records.

Run with the bundled Python. No model training or result selection is performed.
All displayed sizes use full-file bytes; scatter positions are never jittered.
"""
from pathlib import Path
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '_runtime_deps/darc_figures_20261103'))
OUT = ROOT / 'output/pdf/darc_single_column_20261103'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pymupdf

font = Path('C:/Windows/Fonts/times.ttf')
font_manager.fontManager.addfont(str(font))
assert font_manager.FontProperties(fname=font).get_name() == 'Times New Roman'
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.weight': 'normal',
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.labelweight': 'normal',
    'axes.titlesize': 9, 'axes.titleweight': 'normal',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': .65,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'axes.unicode_minus': False,
})
INK = '#222222'
GRID = '#E4E7EA'
EXACT = '#426E93'
FINAL = '#C87734'

# Read exact bytes and the reported calibrated MAE from the audited stage table.
stage_source = ROOT / 'pems07_unified_ablation_20261028/README.md'
rows = []
for line in stage_source.read_text(encoding='utf-8').splitlines():
    if not line.startswith('| '):
        continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    if len(cells) != 8 or not re.fullmatch(r'[\d,]+', cells[1]):
        continue
    rows.append({
        'source_name': cells[0], 'parameters': int(cells[1].replace(',', '')),
        'raw_mae': float(cells[2]), 'calibrated_mae': float(cells[3]),
        'checkpoint_bytes': int(cells[4].replace(',', '')),
        'state_bytes': int(cells[5].replace(',', '')),
        'calibrator_bytes': int(cells[6].replace(',', '')),
        'total_bytes': int(cells[7].replace(',', '')),
    })
assert len(rows) == 5
for row in rows:
    assert row['total_bytes'] == sum(row[k] for k in
        ('checkpoint_bytes', 'state_bytes', 'calibrator_bytes'))
assert rows[3]['total_bytes'] - rows[4]['total_bytes'] == 14

codec_source = ROOT / 'contrib3_index_codec_20261020/packages/online/loader_benchmark_20261024_v3/report.json'
report = json.loads(codec_source.read_text(encoding='utf-8'))
datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
kinds = ['legacy_npz', 'generic', 'exact', 'selected']
codec = np.array([[next(r['file_bytes'] for r in report['summary']
                       if r['dataset'] == d and r['kind'] == k)
                   for k in kinds] for d in datasets], dtype=np.int64)
assert codec.sum(axis=0).tolist() == [742194, 349499, 302958, 300314]
normalized = codec / codec[:, [0]] * 100
reduction = (1 - rows[4]['total_bytes'] / rows[2]['total_bytes']) * 100


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#777777')
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)


def export(fig, stem):
    # Fixed 3.5-inch canvas; no tight bounding-box rescaling of physical size.
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}')
    fig.savefig(OUT / f'{stem}.png', dpi=600)
    doc = pymupdf.open(OUT / f'{stem}.pdf')
    assert len(doc) == 1 and abs(doc[0].rect.width - 252) < .01
    doc[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False).save(
        OUT / f'{stem}_pdf_preview.png')
    assert 'TimesNewRoman' in ' '.join(str(f) for f in doc[0].get_fonts())
    print(stem, 'size_pt=', tuple(doc[0].rect), 'fonts=', doc[0].get_fonts())
    doc.close()
    plt.close(fig)


def tradeoff():
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=.17, right=.97, bottom=.19, top=.93)
    style(ax)
    ax.set_xlim(3.8, 16.3)
    ax.set_ylim(18.455, 18.845)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_yticks([18.5, 18.6, 18.7, 18.8])
    ax.grid(axis='y', color=GRID, linewidth=.55)
    ax.set_xlabel('Stored assets (MiB)', labelpad=5)
    ax.set_ylabel('Calibrated test MAE', labelpad=5)
    fig.text(.17, .965, 'PEMS07', ha='left', va='center', fontsize=9)
    x = [r['total_bytes'] / 2**20 for r in rows]
    y = [r['calibrated_mae'] for r in rows]
    ax.scatter(x[0], y[0], s=28, marker='^', color='#969696', zorder=3)
    ax.scatter(x[1], y[1], s=24, marker='o', color=INK, zorder=3)
    ax.scatter(x[2], y[2], s=28, marker='s', color=EXACT, zorder=3)
    ax.scatter(x[3], y[3], s=70, marker='o', facecolors='white',
               edgecolors=EXACT, linewidths=1.05, zorder=4)
    ax.scatter(x[4], y[4], s=19, marker='o', color=FINAL, zorder=5)
    leader = dict(arrowstyle='-', lw=.55, shrinkA=2, shrinkB=4)
    ax.annotate('Student base', xy=(x[1], y[1]), xytext=(4.15, 18.816),
                fontsize=8.5, ha='left', va='center',
                arrowprops={**leader, 'color': INK})
    ax.annotate('Teacher base\n(reference)', xy=(x[0], y[0]),
                xytext=(8.4, 18.784), ha='left', va='center', fontsize=8,
                color='#737373', linespacing=1.1,
                arrowprops={**leader, 'color': '#969696'})
    ax.annotate('Full-node exact', xy=(x[2], y[2]), xytext=(16, 18.702),
                fontsize=8.5, color=EXACT, ha='right', va='center',
                arrowprops={**leader, 'color': EXACT})
    ax.annotate('DARC', xy=(x[4], y[4]), xytext=(7.1, 18.548),
                fontsize=9, color=FINAL, ha='left', va='center',
                arrowprops={**leader, 'color': FINAL})
    ax.annotate('SRM exact', xy=(x[3], y[3]), xytext=(7.1, 18.485),
                fontsize=8.5, color=EXACT, ha='left', va='center',
                arrowprops={**leader, 'color': EXACT})
    ax.annotate('', xy=(x[4], 18.625), xytext=(x[2], 18.625),
                arrowprops=dict(arrowstyle='<->', color='#777777', lw=.65))
    ax.text((x[4]+x[2])/2, 18.641, f'{reduction:.2f}% less storage',
            fontsize=8.5, ha='center', va='bottom')
    export(fig, 'pems07_accuracy_storage')


def encoding():
    fig, ax = plt.subplots(figsize=(3.5, 2.95))
    fig.subplots_adjust(left=.185, right=.975, bottom=.18, top=.80)
    style(ax)
    colors = ['#D9DEE3', '#9DBCCF', EXACT, FINAL]
    names = ['Indexed NPZ', 'Direct compression', 'Fully exact', 'Final state']
    centers = np.arange(4) * 1.28
    bar_h = .19
    for k in range(4):
        ys = centers + (k-1.5) * .225
        ax.barh(ys, normalized[:, k], height=bar_h,
                color=colors[k], label=names[k], linewidth=0, zorder=3)
        if k == 3:
            for i, ypos in enumerate(ys):
                ax.text(normalized[i, k] + 1.8, ypos,
                        f'{normalized[i, k]:.1f}%', va='center',
                        ha='left', fontsize=8, color=INK)
    ax.set_yticks(centers, datasets)
    ax.set_ylim(centers[-1]+.63, -.62)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', length=0, pad=5)
    ax.grid(axis='x', color=GRID, linewidth=.55)
    ax.set_xlabel('State-file size (% of indexed NPZ)', labelpad=5)
    ax.legend(loc='lower left', bbox_to_anchor=(-.18, 1.025), ncol=2,
              frameon=False, borderaxespad=0, handlelength=1.15,
              handletextpad=.45, columnspacing=1.25, labelspacing=.5)
    export(fig, 'retrieval_state_encoding')


if __name__ == '__main__':
    tradeoff()
    encoding()
    (OUT / 'figure_data.json').write_text(json.dumps({
        'font': 'Times New Roman, regular', 'width_inches': 3.5,
        'stage_source': str(stage_source), 'stages': rows,
        'codec_source': str(codec_source), 'datasets': datasets,
        'codec_columns': kinds, 'codec_bytes': codec.tolist(),
        'codec_percent_of_npz': normalized.tolist(),
        'pems07_storage_reduction_percent': reduction,
    }, indent=2), encoding='utf-8')
