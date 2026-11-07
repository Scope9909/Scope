"""Generate editable PEMS07 SVG figures from the frozen evidence ledger."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GREEN, BLUE, GRAY, ORANGE = "#147d66", "#3d70ad", "#75828d", "#cb7d29"
INK, GRID = "#203040", "#dce2e7"

def E(tag, attrs=None, body=""):
    a = "".join(f' {k.replace("_", "-")}="{html.escape(str(v), quote=True)}"' for k,v in (attrs or {}).items())
    return f"<{tag}{a}>{body}</{tag}>"

def T(x,y,s,size=18,bold=False,color=INK,anchor="start",rotate=False):
    a={"x":x,"y":y,"font_size":size,"font_weight":"bold" if bold else "normal","fill":color,"text_anchor":anchor}
    if rotate: a["transform"]=f"rotate(-90 {x} {y})"
    return E("text",a,html.escape(str(s)))

def L(x1,y1,x2,y2,color=GRID,w=1):
    return E("line",{"x1":x1,"y1":y1,"x2":x2,"y2":y2,"stroke":color,"stroke_width":w})

def R(x,y,w,h,color):
    return E("rect",{"x":x,"y":y,"width":w,"height":h,"fill":color})

def P(points,color,w=3.5):
    return E("polyline",{"points":" ".join(f"{x:.2f},{y:.2f}" for x,y in points),"fill":"none","stroke":color,"stroke_width":w})

def S(width,height,title,desc,body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="ct cd">'
            f'<title id="ct">{html.escape(title)}</title><desc id="cd">{html.escape(desc)}</desc>'
            '<g font-family="Arial, Helvetica, sans-serif">'+"".join(body)+'</g></svg>')

def node_gain():
    d=json.loads((ROOT/'pems07_exact_node_gain_validation.json').read_text(encoding='utf-8'))
    assert d['test_evaluated'] is False and d['selection_prefix_samples']>0
    g=d['heldout_gain_by_node']; ids=set(d['selected_node_ids'])
    assert len(g)==883 and len(ids)==121
    groups=[('Selected',[g[i] for i in ids],GREEN),('Unselected',[v for i,v in enumerate(g) if i not in ids],GRAY)]
    a=[R(0,0,1100,430,'white'),T(75,43,'(a) Node-wise retrieval gain',22,True),T(720,43,'(b) Subsequent validation folds',22,True)]
    x0,x1,y0,y1=90,620,100,325
    for q in (0,25,50,75,100):
        y=y1-q/100*(y1-y0); a += [L(x0,y,x1,y),T(x0-13,y+6,q,16,anchor='end')]
    a += [L(x0,y0,x0,y1,INK,1.2),L(x0,y1,x1,y1,INK,1.2)]
    for q in (0,2,4,6,8):
        x=x0+q/8*(x1-x0); a += [L(x,y1,x,y1+6,INK),T(x,y1+28,q,16,anchor='middle')]
    a += [T(355,391,'Gain threshold (MAE; base minus retrieval)',17,anchor='middle'),T(25,215,'Nodes above threshold (%)',17,anchor='middle',rotate=True)]
    for name,v,c in groups:
        pts=[]
        for i in range(161):
            t=i/20; pct=100*sum(z>t for z in v)/len(v)
            pts.append((x0+i/160*(x1-x0),y1-pct/100*(y1-y0)))
        a.append(P(pts,c))
    a += [L(100,72,134,72,GREEN,4),T(143,78,'SRM selected (121)',16),L(345,72,379,72,GRAY,4),T(388,78,'Unselected (762)',16)]
    bx0,bx1,by0,by1=720,1035,100,325
    for q in (0,0.5,1,1.5,2):
        y=by1-q/2.25*(by1-by0); a += [L(bx0,y,bx1,y),T(bx0-13,y+6,f'{q:g}',16,anchor='end')]
    a += [L(bx0,by0,bx0,by1,INK,1.2),L(bx0,by1,bx1,by1,INK,1.2)]
    for name,v,c in groups:
        med=[f[name.lower()]['quantiles_0_5_25_50_75_95_100'][3] for f in d['folds']]
        pts=[(bx0+(i+0.5)*(bx1-bx0)/3,by1-z/2.25*(by1-by0)) for i,z in enumerate(med)]
        a.append(P(pts,c))
        a += [E('circle',{'cx':x,'cy':y,'r':5.5,'fill':c}) for x,y in pts]
    for i in range(3):
        x=bx0+(i+0.5)*(bx1-bx0)/3; a += [L(x,by1,x,by1+6,INK),T(x,by1+28,i+1,16,anchor='middle')]
    a += [T(878,391,'Subsequent validation fold',17,anchor='middle'),T(665,215,'Median node gain (MAE)',17,anchor='middle',rotate=True)]
    out=S(1100,430,'PEMS07 held-out node-wise retrieval gain','Frozen Student and exact full-node state; validation samples after the selection prefix only. Selected nodes have higher gain in each subsequent fold.',a)
    (ROOT/'fig_pems07_srm_node_gain.svg').write_text(out,encoding='utf-8')

def assets():
    rows=[('Full-node exact',2836469,10326391,2156991,18.7459),('SRM exact',2836469,83063,2216230,18.5091),('DARC (SRM + Q8)',2836469,80752,2218527,18.5115)]
    assert [sum(r[1:4]) for r in rows]==[15319851,5135762,5135748]
    a=[R(0,0,1100,385,'white'),T(220,39,'Stored assets',22,True),R(465,23,18,14,BLUE),T(492,37,'Checkpoint',16),R(620,23,18,14,GREEN),T(647,37,'Retrieval state',16),R(828,23,18,14,ORANGE),T(855,37,'Calibrator',16)]
    x0,scale=245,35
    for tick in (0,5,10,15):
        x=x0+tick*scale; a += [L(x,68,x,318),T(x,344,tick,16,anchor='middle')]
    a += [T(x0+7.5*scale,374,'Three stored assets (MiB)',17,anchor='middle'),T(930,66,'Test MAE',17,True)]
    for y,r in zip((107,191,275),rows):
        name,ckpt,state,cal,mae=r; a.append(T(225,y+7,name,18,anchor='end')); left=x0
        for val,c in ((ckpt,BLUE),(state,GREEN),(cal,ORANGE)):
            w=val/2**20*scale; a.append(R(left,y-16,w,32,c)); left+=w
        a += [T(left+9,y+7,f'{sum(r[1:4])/2**20:.2f}',17),T(930,y+7,f'{mae:.4f}',18)]
    out=S(1100,385,'PEMS07 matched staged assets','Matched seed-1 Student rows from the manuscript. SRM reduces three stored assets substantially; Q8 offers negligible additional total reduction.',a)
    (ROOT/'fig_pems07_staged_assets.svg').write_text(out,encoding='utf-8')

if __name__=='__main__':
    node_gain(); assets(); print('Wrote two SVGs to',ROOT)
