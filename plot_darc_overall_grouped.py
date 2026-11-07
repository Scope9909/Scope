from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'darc_overall_visualization'
OUT.mkdir(exist_ok=True)
for font_file in ['C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/timesbd.ttf']:
    font_manager.fontManager.addfont(font_file)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'font.size': 16, 'axes.labelsize': 18, 'xtick.labelsize': 13.5,
    'ytick.labelsize': 15, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.linewidth': .9, 'savefig.facecolor': 'white',
})

METHODS = ['HA','GWNet','DCRNN','AGCRN','STGCN','GTS','MTGNN','ST-Norm',
           'GMAN','PDFormer','STID','STAEformer','STWave','DyHSL','MH-MoE',
           'RAST','DARC (raw)','DARC']
# Each row contains MAE, RMSE, MAPE (%) in dataset order 03, 04, 07, 08.
ROWS = [
[32.62,49.89,30.60,42.35,61.66,29.92,49.03,71.18,22.75,36.66,50.45,21.63],
[14.59,25.24,15.52,18.53,29.92,12.89,20.47,33.47,8.61,14.40,23.39,9.21],
[15.54,27.18,15.62,19.63,31.26,13.59,21.16,34.14,9.02,15.22,24.17,10.21],
[15.24,26.65,15.89,19.38,31.25,13.40,20.57,34.40,8.74,15.32,24.41,10.03],
[15.83,27.51,16.13,19.57,31.38,13.44,21.74,35.27,9.24,16.08,25.39,10.60],
[15.41,26.15,15.39,20.96,32.95,14.66,22.15,35.10,9.38,16.49,26.08,10.54],
[14.85,25.23,14.55,19.17,31.70,13.37,20.89,34.06,9.00,15.18,24.24,10.20],
[15.32,25.93,14.37,18.96,30.98,12.69,20.50,34.66,8.75,15.41,24.77,9.76],
[16.87,27.92,18.23,19.14,31.60,13.19,20.97,34.10,9.05,15.31,24.92,10.13],
[14.94,25.39,15.82,18.36,30.03,12.00,19.97,32.95,8.55,13.58,23.41,9.05],
[15.33,27.40,16.40,18.38,29.95,12.04,19.61,32.79,8.30,14.21,23.28,9.27],
[15.35,27.55,15.18,18.22,30.18,11.98,19.14,32.60,8.01,13.46,23.25,8.88],
[14.93,26.50,15.05,18.50,30.39,12.43,19.94,33.88,8.38,13.42,23.40,8.90],
[15.49,27.06,14.38,17.66,29.46,12.42,18.84,31.65,8.11,14.01,22.91,8.60],
[np.nan,np.nan,np.nan,17.88,29.48,11.98,np.nan,np.nan,np.nan,13.33,22.98,8.87],
[15.20,25.81,16.12,18.39,29.93,12.43,19.52,32.73,8.23,14.16,23.33,9.27],
[14.20,24.15,14.44,17.70,29.43,12.08,18.74,32.23,7.86,13.10,22.81,8.63],
[13.97,23.88,14.05,17.55,29.22,12.00,18.51,31.88,7.73,12.88,22.49,8.49],
]
DATA = np.asarray(ROWS, dtype=float).reshape(18,4,3)
COLORS = ['#69849B', '#98A88F', '#AA9CB5']
METRICS = ['MAE', 'RMSE', 'MAPE (%)']
NAMES = ['PEMS03','PEMS04','PEMS07','PEMS08']
LIMITS = [(0,60), (0,70), (0,80), (0,60)]
TICKS = [list(range(0,hi+1,10)) for lo,hi in LIMITS]

assert DATA.shape == (18,4,3)
assert np.isnan(DATA).sum() == 6
assert np.all(np.isfinite(DATA[-2:]))
font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman',weight='bold'),fallback_to_default=False)
print('Bold font:',font_path)

fig = plt.figure(figsize=(23,12.8))
outer = fig.add_gridspec(2,2,left=.05,right=.99,top=.91,bottom=.19,wspace=.11,hspace=.85)
bw = .245
for di,name in enumerate(NAMES):
    bot = fig.add_subplot(outer[di//2,di%2])
    valid = np.all(np.isfinite(DATA[:,di,:]),axis=1)
    names = [m for m,keep in zip(METHODS,valid) if keep]
    values = DATA[valid,di,:]
    x = np.arange(len(names))
    low = LIMITS[di]
    for ax in (bot,):
        ax.set_xlim(-.65,len(names)-.35)
        ax.set_axisbelow(True)
        ax.axvspan(len(names)-2.5,len(names)-.45,color='#EDF3F7',zorder=0)
        ax.grid(axis='y',color='#E0E3E5',linewidth=.7,linestyle=(0,(3,4)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('#5C6268')
        ax.spines['bottom'].set_color('#5C6268')
        ax.tick_params(axis='y',length=3,pad=6,labelsize=15)
        for mi,color in enumerate(COLORS):
            vals=values[:,mi]
            bars=ax.bar(x+(mi-1)*bw,vals,width=bw*.94,color=color,
                        edgecolor='none',linewidth=0,zorder=3)
            for bar in bars[-2:]:
                bar.set_edgecolor('#344A5D')
                bar.set_linewidth(1.0)
    bot.set_ylim(*low)
    bot.set_yticks(TICKS[di])
    bot.tick_params(axis='x',length=0,pad=5,labelsize=15)
    bot.set_xticks(x,names,rotation=50,ha='right',rotation_mode='anchor')
    for txt in bot.get_xticklabels():
        txt.set_fontweight('bold')
        if txt.get_text().startswith('DARC'):
            txt.set_color('#354C63')
            txt.set_fontproperties(font_manager.FontProperties(
                fname='C:/Windows/Fonts/timesbd.ttf',size=txt.get_fontsize(),weight='bold'))
            txt.set_path_effects([path_effects.withStroke(linewidth=.25,foreground='#354C63')])
    assert np.nanmax(values) < low[1]
    bot.set_xlabel('Methods',fontsize=18,labelpad=10,fontweight='bold')
    bot.set_ylabel('Metric value',fontsize=18,labelpad=10,fontweight='bold')
    bot.text(.5,-.51,f'({chr(97+di)}) {name}',transform=bot.transAxes,
             ha='center',va='top',fontsize=24,fontweight='bold')

handles=[Patch(facecolor=c,edgecolor='none',label=m) for c,m in zip(COLORS,METRICS)]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=3,
           frameon=False,fontsize=21,handlelength=1.5,columnspacing=2.8)
svg_path=OUT/'darc-overall-four-datasets-full.svg'
png_path=OUT/'darc-overall-four-datasets-full.png'
pdf_dir=ROOT/'output'/'pdf'
pdf_dir.mkdir(parents=True,exist_ok=True)
fig.savefig(pdf_dir/'darc-overall-four-datasets.pdf')
fig.savefig(svg_path)
fig.savefig(png_path,dpi=220)
fig.savefig(OUT/'darc-overall-four-datasets-full-preview.png',dpi=100)
plt.close(fig)
print(svg_path)
print(png_path)
