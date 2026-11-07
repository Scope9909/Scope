"""A simpler, more pictorial SRM validation figure; all numbers from replay."""
import json
from pathlib import Path

from build_pems07_visual_evidence_svg import E, GREEN, GRAY, INK, GRID, L, R, S, T

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'pems07_exact_node_gain_validation.json').read_text(encoding='utf-8'))
assert data['test_evaluated'] is False and data['selected_nodes'] == 121

W, H = 1100, 380
parts = [R(0, 0, W, H, 'white')]
parts += [T(45, 45, 'SRM mask (schematic)', 24, True),
          T(420, 45, 'Retrieval gain on later validation folds', 24, True)]

# Left: a schematic graph deliberately shows a mask, not fabricated road geography.
nodes = [(96, 110), (164, 91), (227, 116), (107, 185), (169, 174),
         (235, 195), (92, 256), (164, 273), (234, 252)]
edges = [(0,1),(1,2),(0,3),(1,4),(2,5),(3,4),(4,5),(3,6),(4,7),(5,8),(6,7),(7,8)]
for i, j in edges:
    x1,y1=nodes[i]; x2,y2=nodes[j]
    parts.append(L(x1,y1,x2,y2,'#b7c6d0',2.8))
selected_demo={0,1,3,4}
for i,(x,y) in enumerate(nodes):
    parts.append(E('circle',{'cx':x,'cy':y,'r':16,'fill':GREEN if i in selected_demo else '#b8c2ca'}))
parts += [E('circle',{'cx':55,'cy':329,'r':7,'fill':GREEN}),T(71,335,'Selected (121): retain history',17),
          E('circle',{'cx':55,'cy':354,'r':7,'fill':'#b8c2ca'}),T(71,360,'Unselected (762): bypass',17),
          L(345,55,345,355,GRID,2)]

# Right: three paired fold comparisons, one shared 0..2.2 MAE-gain scale.
gain_x0, gain_x1 = 545, 980
for q in (0,0.5,1,1.5,2):
    x=gain_x0+q/2.2*(gain_x1-gain_x0)
    parts += [L(x,80,x,305,GRID,1),T(x,331,f'{q:g}',16,anchor='middle')]
parts.append(T((gain_x0+gain_x1)/2,361,'Median node-wise gain (MAE; higher is better)',17,anchor='middle'))

for fold, y in zip(data['folds'],(110,190,270)):
    s=fold['selected']['quantiles_0_5_25_50_75_95_100'][3]
    u=fold['unselected']['quantiles_0_5_25_50_75_95_100'][3]
    assert s>u>=0
    parts += [T(420,y+7,f"Fold {fold['fold']}",19,True),
              R(gain_x0,y-18,(s/2.2)*(gain_x1-gain_x0),16,GREEN),
              E('circle',{'cx':gain_x0+(u/2.2)*(gain_x1-gain_x0),'cy':y+17,'r':6,'fill':GRAY}),
              T(gain_x0+(s/2.2)*(gain_x1-gain_x0)+10,y-4,f'{s:.2f}',17,color=GREEN),
              T(gain_x0+33,y+23,f'{u:.2f}',16,color=GRAY)]

svg=S(W,H,'PEMS07 SRM node selection and held-out gains',
      'Schematic node mask on left. On the right, selected nodes have substantially higher median retrieval gain than unselected nodes in three later validation folds. Data come from the frozen PEMS07 Student, exact full-node state, seed 1; no Test queries used.',parts)
(ROOT/'fig_pems07_srm_simple.svg').write_text(svg,encoding='utf-8')
print('Wrote', ROOT/'fig_pems07_srm_simple.svg')
