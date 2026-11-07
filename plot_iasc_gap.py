from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent/'output'/'pdf'/'simplified_ablations'
OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/timesbd.ttf')
plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold',
 'axes.labelweight':'bold','font.size':15,'pdf.fonttype':42,'svg.fonttype':'none'})
svd=[18.2097,17.8647,17.8318]
moment=[17.8542,17.8435,17.8325]
gap=[a-b for a,b in zip(svd,moment)]
assert all(abs(a-b)<1e-10 for a,b in zip(gap,[.3555,.0212,-.0007]))
fig,ax=plt.subplots(figsize=(8.6,5.4))
fig.subplots_adjust(left=.32,right=.96,top=.94,bottom=.19)
ax.set_axisbelow(True)
ax.grid(axis='x',color='#DFE4E7',linewidth=.7,linestyle=(0,(3,3)))
ax.axvline(0,color='#58636D',linewidth=1.1)
colors=['#718C79','#718C79','#69849B']
ax.barh(range(3),gap,height=.45,color=colors,zorder=3)
ax.set_xlim(-.025,.40)
ax.set_ylim(2.6,-.6)
ax.set_xticks([0,.1,.2,.3,.4])
ax.set_yticks([0,1,2],['Before training','After training\nwithout KD','After training\nwith KD'])
ax.tick_params(axis='y',length=0,pad=12,labelsize=15)
ax.set_xlabel('MAE difference (SVD - Full Moment)',fontsize=16,labelpad=12)
for side in ['top','right']: ax.spines[side].set_visible(False)
for side in ['left','bottom']: ax.spines[side].set_color('#58636D')
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/('iasc-rank256-horizontal-gap.'+ext),dpi=220)
plt.close(fig)
