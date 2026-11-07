"""Three PEMS-BAY horizons in one IEEE two-column-width row."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


OUT = Path(__file__).resolve().parent / 'output/pdf/performance_redrawn'
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--bar-style', choices=('plain', 'warm-hatch', 'warm-solid'),
                    default='plain')
args = parser.parse_args()

methods = ['DCRNN', 'GWNet', 'AGCRN', 'STGCN', 'MTGNN',
           'ST-Norm', 'PDFormer', 'STID', 'Scope']
# MAE, RMSE, MAPE at 15, 30, and 60 minutes. The last row's method label
# follows the requested Scope capitalization; its numeric values are unchanged.
values = np.array([
    [1.31, 2.76, 2.73, 1.65, 3.75, 3.71, 1.97, 4.60, 4.68],
    [1.30, 2.73, 2.71, 1.63, 3.73, 3.73, 1.99, 4.60, 4.71],
    [1.35, 2.88, 2.91, 1.67, 3.82, 3.81, 1.94, 4.50, 4.55],
    [1.36, 2.88, 2.86, 1.70, 3.84, 3.79, 2.02, 4.63, 4.72],
    [1.33, 2.80, 2.81, 1.66, 3.77, 3.75, 1.95, 4.50, 4.62],
    [1.33, 2.82, 2.76, 1.65, 3.77, 3.66, 1.92, 4.45, 4.46],
    [1.32, 2.83, 2.78, 1.64, 3.79, 3.71, 1.91, 4.43, 4.51],
    [1.31, 2.79, 2.78, 1.64, 3.73, 3.73, 1.91, 4.42, 4.55],
    [1.30, 2.78, 2.73, 1.62, 3.73, 3.64, 1.88, 4.37, 4.38],
], dtype=float).reshape(9, 3, 3)

font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
emphasize_text = args.bar_style == 'warm-solid'
text_weight = 'bold' if emphasize_text else 'normal'
x_font = 8.8 if emphasize_text else 8
tick_font = 8.3 if emphasize_text else 7.5
axis_title_font = 10.5 if emphasize_text else 9.5
panel_font = 11 if emphasize_text else 10
legend_font = 10 if emphasize_text else 9
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9.3 if emphasize_text else 8.5,
    'font.weight': text_weight,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'hatch.linewidth': .42,
})

if args.bar_style in ('warm-hatch', 'warm-solid'):
    bar_fill = '#FFF8EC'
    bar_edge = '#CFB991'
    scope_fill = '#F4DEB8'
    scope_edge = '#9A7445'
    bar_hatch = '-' if args.bar_style == 'warm-hatch' else ''
else:
    bar_fill = '#E7ECEF'
    bar_edge = '#B7C3C9'
    scope_fill = '#ADC2CD'
    scope_edge = '#577789'
    bar_hatch = ''
# Keep the quantitative lines vivid, as in the supplied reference, while
# muting the large-area bars so that the line trajectories remain legible.
rmse_blue = '#1976B6'
mape_red = '#D94755'


class OutlinedHatchLegend(HandlerBase):
    """Draw the warm hatch and its black outline as separate layers."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        bounds = (xdescent, ydescent, width, height)
        hatch_layer = Rectangle((bounds[0], bounds[1]), bounds[2], bounds[3],
                                facecolor=bar_fill, edgecolor=bar_edge,
                                hatch=bar_hatch, linewidth=.25,
                                transform=trans)
        outline = Rectangle((bounds[0], bounds[1]), bounds[2], bounds[3],
                            facecolor='none', edgecolor='#000000',
                            linewidth=.5, transform=trans)
        return [hatch_layer, outline]


panels = [
    ('15 min', (0.7, 1.5), [0.7, 0.9, 1.1, 1.3, 1.5],
     (2.6, 3.0), [2.6, 2.8, 3.0]),
    ('30 min', (0.7, 1.9), [0.7, 1.0, 1.3, 1.6, 1.9],
     (3.55, 3.95), [3.6, 3.75, 3.9]),
    ('60 min', (0.7, 2.2), [0.7, 1.0, 1.3, 1.6, 1.9, 2.2],
     (4.3, 4.85), [4.3, 4.55, 4.8]),
]
if emphasize_text:
    # Use a clearly labelled, truncated MAE axis in the final warm-solid figure.
    mae_ticks = ([.8, 1.0, 1.2, 1.4],
                 [.8, 1.1, 1.4, 1.7],
                 [.8, 1.2, 1.6, 2.0])
    panels = [(heading, (.8, left_limits[1]), ticks,
               right_limits, right_ticks)
              for (heading, left_limits, _, right_limits, right_ticks), ticks
              in zip(panels, mae_ticks)]
mae_baseline = .8 if emphasize_text else .7

# Wide IEEEtran figure*: axis titles are shared, whereas method/tick labels
# remain on each panel because the panels have different y-axis ranges.
fig, axes = plt.subplots(1, 3,
                         figsize=(7.05, 4.0 if emphasize_text else 3.45))
fig.subplots_adjust(left=.065, right=.93, top=.86,
                    bottom=.34 if emphasize_text else .28, wspace=.30)
x = np.arange(len(methods))

for horizon, (ax, (heading, left_limits, left_ticks,
                   right_limits, right_ticks)) in enumerate(zip(axes, panels)):
    right = ax.twinx()
    right.patch.set_visible(False)
    for i in range(len(methods)):
        is_scope = i == len(methods) - 1
        ax.bar(i, values[i, horizon, 0] - mae_baseline,
               bottom=mae_baseline, width=1.0,
               color=scope_fill if is_scope else bar_fill,
               edgecolor='#000000' if args.bar_style == 'warm-solid'
               else (scope_edge if is_scope else bar_edge),
               hatch=bar_hatch,
               linewidth=(.5 if is_scope else .4)
               if args.bar_style == 'warm-solid'
               else (.65 if is_scope else .4), zorder=2)
        if args.bar_style == 'warm-hatch':
            # Matplotlib uses one edge color for both hatches and borders.
            # Overlay a fine black rectangle to retain ochre hatch strokes.
            ax.add_patch(Rectangle((i - .5, mae_baseline), 1.0,
                                   values[i, horizon, 0] - mae_baseline,
                                   facecolor='none', edgecolor='#000000',
                                   linewidth=.5 if is_scope else .4,
                                   zorder=3))
    right.plot(x, values[:, horizon, 1], color=rmse_blue,
               linewidth=1.55, marker='^', markersize=4.5,
               markeredgewidth=.35, markeredgecolor=rmse_blue,
               zorder=4)
    right.plot(x, values[:, horizon, 2], color=mape_red,
               linewidth=1.55, marker='v', markersize=4.5,
               markeredgewidth=.35, markeredgecolor=mape_red,
               zorder=5)

    ax.set_xlim(-.5, 8.5)
    ax.set_ylim(*left_limits)
    right.set_ylim(*right_limits)
    ax.set_yticks(left_ticks)
    right.set_yticks(right_ticks)
    ax.set_xticks(x, methods, rotation=68, ha='right',
                  rotation_mode='anchor', fontsize=x_font,
                  fontweight=text_weight)
    ax.get_xticklabels()[-1].set_color(scope_edge)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.tick_params(axis='x', length=2.3, width=.65, pad=1)
    ax.tick_params(axis='y', labelsize=tick_font, width=.65, length=2.3,
                   pad=1.5)
    right.tick_params(axis='y', labelsize=tick_font, width=.65, length=2.3,
                      pad=1.5)
    for label in (*ax.get_yticklabels(), *right.get_yticklabels()):
        label.set_fontweight(text_weight)
    ax.grid(axis='y', color='#E4E7E9', linewidth=.3)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_color('#242424')
        side.set_linewidth(.65)
    for side in right.spines.values():
        side.set_color('#242424')
        side.set_linewidth(.65)
    right.spines['left'].set_visible(False)

for ax, panel in zip(axes, panels):
    box = ax.get_position()
    fig.text((box.x0 + box.x1) / 2,
             .13 if emphasize_text else .11, panel[0],
             ha='center', va='center', fontsize=panel_font, fontweight='bold')

# Each axis title appears exactly once for the whole three-panel figure.
fig.text(.017, .55, 'MAE', rotation=90, ha='center', va='center',
         fontsize=axis_title_font, fontweight=text_weight)
fig.text(.985, .55, 'RMSE / MAPE (%)', rotation=90, ha='center',
         va='center', fontsize=axis_title_font, fontweight=text_weight)
fig.text(.5, .025 if emphasize_text else .018,
         'Prediction method', ha='center', va='bottom',
         fontsize=axis_title_font, fontweight=text_weight)
bar_legend = Patch(facecolor=bar_fill,
                   edgecolor='#000000' if args.bar_style == 'warm-solid'
                   else bar_edge,
                   hatch=bar_hatch,
                   label='MAE')
fig.legend(handles=[
    bar_legend,
    Line2D([], [], color=rmse_blue, linewidth=1.55, marker='^',
           markersize=4.5, label='RMSE'),
    Line2D([], [], color=mape_red, linewidth=1.55, marker='v',
           markersize=4.5, label='MAPE (%)'),
], loc='upper center', bbox_to_anchor=(.5, .985), ncol=3,
    frameon=True, facecolor='white', edgecolor='#CECECE',
    prop={'family': 'Times New Roman', 'size': legend_font,
          'weight': text_weight},
    columnspacing=1.1, handlelength=1.7,
    handler_map={bar_legend: OutlinedHatchLegend()}
    if args.bar_style == 'warm-hatch' else None)

basename = 'pemsbay-3horizons-horizontal'
if args.bar_style == 'warm-hatch':
    basename += '-warm-hatch'
elif args.bar_style == 'warm-solid':
    basename += '-warm-solid'
base = OUT / basename
for suffix in ('pdf', 'svg', 'png'):
    fig.savefig(base.with_suffix('.' + suffix), dpi=300,
                bbox_inches='tight', pad_inches=.04)
plt.close(fig)
print('Wrote:', base)
