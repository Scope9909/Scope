from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'darc_overall_visualization'
PDF = ROOT / 'output' / 'pdf'
OUT.mkdir(exist_ok=True)
PDF.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
plt.rcParams.update({'font.family':'Times New Roman', 'font.weight':'bold',
    'axes.labelweight':'bold', 'font.size':15, 'axes.labelsize':17,
    'xtick.labelsize':15, 'ytick.labelsize':15, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'axes.linewidth':.9})
names = ['DCRNN','GWNet','AGCRN','STGCN','MTGNN','ST-Norm','PDFormer','STID','DARC']
# Rows: methods. Columns: 15, 30, 60 min; each contains MAE, RMSE, MAPE (%).
data = np.array([
    [[1.31,2.76,2.73],[1.65,3.75,3.71],[1.97,4.60,4.68]],
    [[1.30,2.73,2.71],[1.63,3.73,3.73],[1.99,4.60,4.71]],
    [[1.35,2.88,2.91],[1.67,3.82,3.81],[1.94,4.50,4.55]],
    [[1.36,2.88,2.86],[1.70,3.84,3.79],[2.02,4.63,4.72]],
    [[1.33,2.80,2.81],[1.66,3.77,3.75],[1.95,4.50,4.62]],
    [[1.33,2.82,2.76],[1.65,3.77,3.66],[1.92,4.45,4.46]],
    [[1.32,2.83,2.78],[1.64,3.79,3.71],[1.91,4.43,4.51]],
    [[1.31,2.79,2.78],[1.64,3.73,3.73],[1.91,4.42,4.55]],
    [[1.30,2.78,2.73],[1.62,3.73,3.64],[1.88,4.37,4.38]],
])
colors = ['#7599B5','#C6A078','#86A58B','#C98787','#AA98BB',
          '#AA9186','#BE9BB0','#989EA3','#496B60']
markers = ['o','s','D','^','v','P','X','h','o']
h = np.array([15,30,60])
metrics = ['MAE','RMSE','MAPE (%)']
fig, axes = plt.subplots(1,3,figsize=(19,6.8))
fig.subplots_adjust(left=.06,right=.99,top=.80,bottom=.28,wspace=.27)
for j,ax in enumerate(axes):
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#DEE2E5',linestyle=(0,(3,3)),linewidth=.7)
    for i,name in enumerate(names):
        ours = i == 8
        ax.plot(h,data[i,:,j],label=name,color=colors[i],marker=markers[i],
                linewidth=2.8 if ours else 1.45,markersize=8 if ours else 5,
                markeredgecolor='#253E35' if ours else colors[i],
                markeredgewidth=.8,zorder=5 if ours else 2,alpha=1 if ours else .85)
    ax.set_xlim(12,64)
    ax.set_xticks(h)
    ax.set_xlabel('Forecast horizon (min)',labelpad=10)
    ax.set_ylabel(metrics[j],labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#545B61')
    ax.spines['bottom'].set_color('#545B61')
    ax.annotate('',xy=(1.025,0),xytext=(.97,0),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',color='#545B61',lw=.9),
                annotation_clip=False)
    ax.annotate('',xy=(0,1.04),xytext=(0,.97),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',color='#545B61',lw=.9),
                annotation_clip=False)
    ax.text(.5,-.36,f'({chr(97+j)}) {metrics[j]}',transform=ax.transAxes,
            ha='center',va='top',fontsize=21,fontweight='bold')
axes[0].set_ylim(1.23,2.08)
axes[1].set_ylim(2.57,4.80)
axes[2].set_ylim(2.55,4.90)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.97),
           ncol=9,frameon=False,fontsize=15,columnspacing=1.15,handlelength=1.8,
           handletextpad=.5)
fig.savefig(PDF/'darc-pemsbay-horizon-comparison.pdf')
fig.savefig(OUT/'darc-pemsbay-horizon-comparison.svg')
fig.savefig(OUT/'darc-pemsbay-horizon-comparison.png',dpi=220)
plt.close(fig)
print('Saved PEMS-BAY horizon figure: PDF, SVG, PNG.')
