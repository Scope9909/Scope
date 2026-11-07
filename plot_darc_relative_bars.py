"""Diverging bars: DARC's relative improvement over the best baseline."""

from pathlib import Path
import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pdf/performance_redrawn'
OUT.mkdir(parents=True, exist_ok=True)

# Read the table data only; do not run the original plotting script.
env = {'np': np}
source = ast.parse((ROOT / 'plot_darc_overall_grouped.py').read_text(encoding='utf-8'))
for node in source.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {'METHODS', 'ROWS'}
        for target in node.targets
    ):
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'results', 'exec'), env)

methods = env['METHODS']
results = np.asarray(env['ROWS'], dtype=float).reshape(len(methods), 4, 3)
assert methods[-2:] == ['DARC (raw)', 'DARC']

# Lower is better for all three metrics. The baseline is chosen independently
# for each dataset and metric, excluding both DARC variants.
reference = np.nanmin(results[:-2], axis=0)
improvement = 100 * (reference - results[-1]) / reference
assert improvement.shape == (4, 3) and np.isfinite(improvement).all()
assert (improvement > 0).sum() == 10

font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 16,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
metrics = ['MAE', 'RMSE', 'MAPE']
palette = ['#547A99', '#6C947D', '#A287A7']
centers = np.array([4.5, 3.0, 1.5, 0.0])
offsets = np.array([.30, 0, -.30])

fig, ax = plt.subplots(figsize=(11.2, 6.0))
fig.subplots_adjust(left=.17, right=.96, top=.84, bottom=.17)
for metric_idx, (metric, color) in enumerate(zip(metrics, palette)):
    yy = centers + offsets[metric_idx]
    xx = improvement[:, metric_idx]
    ax.barh(yy, xx, height=.23, color=color, edgecolor='none', zorder=3)
    ax.scatter(xx, yy, s=28, marker='o', color=color, edgecolor='white',
               linewidth=.5, zorder=4)

ax.set_xlim(-1, 6)
ax.set_ylim(-.63, 5.13)
ax.set_xticks(np.arange(-1, 7, 1))
ax.set_yticks(centers, datasets)
ax.tick_params(axis='x', labelsize=16, length=4, colors='#38454B')
ax.tick_params(axis='y', labelsize=20, length=0, pad=12)
ax.set_xlabel('Improvement over the best baseline (%)', fontsize=19, labelpad=12)
ax.grid(axis='x', color='#DCE3E5', linestyle=(0, (3, 3)), linewidth=.75)
ax.set_axisbelow(True)
ax.axvline(0, color='#34454C', linewidth=1.45, zorder=2)
for y in [3.75, 2.25, .75]:
    ax.axhline(y, color='#E7EBEC', linewidth=.8, zorder=1)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#7B8589')

handles = [Patch(facecolor=color, edgecolor='none', label=name)
           for name, color in zip(metrics, palette)]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .975),
           ncol=3, frameon=False, fontsize=20, handlelength=1.25,
           columnspacing=2.5)

base = OUT / 'darc-relative-improvement-bars'
for suffix in ('pdf', 'svg', 'png'):
    fig.savefig(base.with_suffix('.' + suffix), dpi=240,
                bbox_inches='tight', pad_inches=.08)
plt.close(fig)

print('Improvement (%), dataset rows and metric columns:')
print(np.array2string(improvement, precision=3, suppress_small=True))
