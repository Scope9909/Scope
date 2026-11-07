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
GREEN='#718C79'; BLUE='#69849B'; INK='#58636D'
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/(name+'.'+ext),dpi=220)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(8.6,5.7))
fig.subplots_adjust(left=.16,right=.95,top=.94,bottom=.29)
gap=[18.2097-17.8542,17.8647-17.8435,17.8318-17.8325]
ax.set_axisbelow(True)
ax.grid(axis='y',color='#DFE4E7',linewidth=.7,linestyle=(0,(3,3)))
ax.vlines(range(3),0,gap,color=[GREEN,GREEN,BLUE],linewidth=2.4)
ax.scatter(range(3),gap,color=[GREEN,GREEN,BLUE],s=100,zorder=3)
for i,value in enumerate(gap):
    ax.annotate(f'{value:+.4f}',(i,value),xytext=(0,12 if value>=0 else -20),
                textcoords='offset points',ha='center',fontsize=15,
                color=GREEN if value>=0 else BLUE,fontweight='bold')
ax.set_xlim(-.45,2.45); ax.set_ylim(-.04,.41)
ax.set_yticks([0,.1,.2,.3,.4])
ax.set_xticks([0,1,2],['Before training','After training\nwithout KD','After training\nwith KD'])
ax.tick_params(axis='x',length=0,pad=12)
ax.set_ylabel('MAE difference (SVD - Full Moment)',fontsize=16,labelpad=12)
for side in ['top','right','bottom']: ax.spines[side].set_visible(False)
ax.spines['left'].set_color(INK)
ax.annotate('',xy=(2.5,0),xytext=(-.45,0),xycoords='data',
            arrowprops=dict(arrowstyle='-|>',color=INK,lw=1),annotation_clip=False)
ax.annotate('',xy=(0,1.035),xytext=(0,.96),xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>',color=INK,lw=1),annotation_clip=False)
ax.text(.5,-.20,'Initialization advantage',transform=ax.transAxes,
        ha='center',va='top',fontsize=18)
if '--storage-only' not in __import__('sys').argv:
    save(fig,'iasc-gap-labeled')
else:
    plt.close(fig)
if '--iasc-only' in __import__('sys').argv:
    raise SystemExit(0)

fig,ax=plt.subplots(figsize=(8.8,5.7))
fig.subplots_adjust(left=.18,right=.95,top=.79,bottom=.28)
final=[60.34,48.92,22.73,45.54]
ax.set_axisbelow(True)
ax.grid(axis='x',color='#DFE4E7',linewidth=.7,linestyle=(0,(3,3)))
for i,v in enumerate(final): ax.plot([v,100],[i,i],color='#C9D3D9',lw=3)
ax.scatter(final,range(4),s=105,color=GREEN,zorder=3,label='Final state')
ax.scatter([100]*4,range(4),s=105,facecolor='white',edgecolor=BLUE,lw=2,
           zorder=3,label='Original NPZ')
ax.set_yticks(range(4),['PEMS03','PEMS04','PEMS07','PEMS08'])
ax.set_ylim(3.45,-.55); ax.set_xlim(0,108)
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('Relative state-file size (%)',fontsize=16,labelpad=10)
ax.tick_params(axis='y',length=0,pad=10)
for side in ['top','right','left']: ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(INK)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.30),ncol=2,frameon=False,fontsize=16)
ax.text(.5,-.25,'Retrieval-state storage',transform=ax.transAxes,
        ha='center',va='top',fontsize=21)
save(fig,'storage-dumbbell-labeled')
