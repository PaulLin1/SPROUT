"""
make_ecosystem_html.py
======================
Build a self-contained, dependency-free interactive explorer of the FULL aesthetics ecosystem
(all 1,115 aesthetics, 5,889 links) as a single HTML file. Pan (drag), zoom (wheel), search
(box + autocomplete), hover for details, and toggle labels/edges. Same ForceAtlas2 layout as
Fig 2. Output: ecosystem_explorer.html (open in any browser, works offline).
"""
import csv, json, os, re, sys
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect"))
from aesthetic_map import AESTHETICS

ROOT = os.path.dirname(os.path.abspath(__file__))
AFF = ["unease", "nostalgia", "comfort", "refinement"]
AFF_COLOR = {"unease": "#6FA292", "nostalgia": "#5E7A9B", "comfort": "#C7A35A", "refinement": "#9E7BA6"}
AFF_LABEL = {"unease": "uncanny / unease", "nostalgia": "nostalgia",
             "comfort": "comfort / warmth", "refinement": "refinement / longing"}
W = 2600.0       # viewBox width (arbitrary user units)
PAD = 80.0


def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def main():
    nodes = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_nodes.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_edges.csv"), encoding="utf-8")))
    info = {r["aesthetic"]: r for r in nodes}
    nbn = {norm(r["aesthetic"]): r["aesthetic"] for r in nodes}

    G = nx.Graph(); G.add_nodes_from(info)
    for e in edges:
        a = e["aesthetic"]; b = nbn.get(norm(e["related"]))
        if b and b != a and a in info:
            G.add_edge(a, b)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    comm_nodes = {}
    for n in G:
        comm_nodes.setdefault(int(info[n]["community"]), []).append(n)
    comm_aff = {}
    for c, ns in comm_nodes.items():
        agg = {a: np.mean([float(info[n].get(f"p_{a}", 0) or 0) for n in ns]) for a in AFF}
        comm_aff[c] = max(agg, key=agg.get)

    deg = dict(G.degree())
    seedset = set()
    nn = {norm(n): n for n in G}
    for name in AESTHETICS:
        s = norm(name)
        if s in nn:
            seedset.add(nn[s])
        else:
            for k, v in nn.items():
                if len(s) >= 5 and (k.startswith(s) or s.startswith(k)):
                    seedset.add(v); break

    print(f"layout (ForceAtlas2) on {G.number_of_nodes()} nodes / {G.number_of_edges()} edges ...")
    pos = nx.forceatlas2_layout(G, max_iter=800, scaling_ratio=4.5, gravity=0.55, linlog=True, seed=7)

    nlist = list(G.nodes())
    xs = np.array([pos[n][0] for n in nlist]); ys = np.array([pos[n][1] for n in nlist])
    xr = xs.max() - xs.min(); yr = ys.max() - ys.min()
    sc = (W - 2 * PAD) / xr
    H = yr * sc + 2 * PAD
    px = {n: (PAD + (pos[n][0] - xs.min()) * sc,
              PAD + (ys.max() - pos[n][1]) * sc) for n in nlist}   # flip y for screen

    dmax = max(deg.values())
    rank = {n: i for i, n in enumerate(sorted(G, key=lambda n: deg[n], reverse=True))}
    idx = {n: i for i, n in enumerate(nlist)}

    NODES = []
    for n in nlist:
        x, y = px[n]
        tier = 2 if n in seedset else (1 if rank[n] < 100 else 0)   # 2=focal,1=hub,0=other
        NODES.append({
            "n": n, "x": round(float(x), 1), "y": round(float(y), 1),
            "r": round(float(3 + 2.4 * (deg[n] ** 0.5)), 1),
            "f": round(float(7 + 19 * (deg[n] / dmax) ** 0.5), 1),  # label font (user units)
            "c": AFF_COLOR[comm_aff[int(info[n]["community"])]],
            "d": int(deg[n]), "k": int(info[n]["community"]),
            "a": AFF_LABEL[comm_aff[int(info[n]["community"])]],
            "t": tier,
        })
    EDGES = [[idx[a], idx[b]] for a, b in G.edges()]

    legend = "".join(
        f'<span class="lg"><i style="background:{AFF_COLOR[a]}"></i>{AFF_LABEL[a]}</span>'
        for a in AFF)
    data = json.dumps({"nodes": NODES, "edges": EDGES, "W": int(round(W)), "H": int(round(float(H)))},
                      separators=(",", ":"))

    html = _TEMPLATE.replace("/*LEGEND*/", legend).replace(
        "/*NNODES*/", str(len(NODES))).replace("/*NEDGES*/", str(len(EDGES))).replace(
        "/*DATA*/", data)
    out = os.path.join(ROOT, "ecosystem_explorer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> {out}  ({len(NODES)} nodes, {len(EDGES)} edges, {os.path.getsize(out)//1024} KB)")


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Internet Aesthetics Ecosystem - interactive explorer</title>
<style>
  :root{ --ink:#1f1f1f; --grey:#6b7178; --line:#d7dadd; }
  *{ box-sizing:border-box; }
  html,body{ margin:0; height:100%; font-family:"Times New Roman",Georgia,serif; color:var(--ink); background:#fff; }
  #bar{ position:fixed; top:0; left:0; right:0; z-index:10; background:#ffffffe6; backdrop-filter:blur(3px);
        border-bottom:1px solid var(--line); padding:8px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  #bar h1{ font-size:16px; margin:0 8px 0 0; font-weight:bold; }
  #bar .sub{ font-size:12px; color:var(--grey); margin-right:6px; }
  input,button{ font-family:inherit; font-size:13px; }
  #q{ padding:4px 8px; width:230px; border:1px solid #bbb; border-radius:4px; }
  button{ padding:4px 9px; border:1px solid #bbb; background:#fff; border-radius:4px; cursor:pointer; }
  button:hover{ background:#f2f2f2; }
  button.on{ background:#1f1f1f; color:#fff; border-color:#1f1f1f; }
  .lg{ font-size:12px; color:var(--grey); margin-left:8px; white-space:nowrap; }
  .lg i{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:3px; vertical-align:-1px; }
  #wrap{ position:absolute; top:46px; bottom:0; left:0; right:0; overflow:hidden; cursor:grab; }
  #wrap.drag{ cursor:grabbing; }
  svg{ width:100%; height:100%; display:block; background:#fff; }
  #edges line{ stroke:#9aa0a6; stroke-width:0.5; stroke-opacity:0.13; }
  circle.node{ stroke:#fff; stroke-width:0.6; }
  circle.hit{ fill:#000; opacity:0; }
  text.lab{ fill:#4a4a4a; paint-order:stroke; stroke:#fff; stroke-width:2.4px; stroke-linejoin:round;
            text-anchor:middle; dominant-baseline:central; pointer-events:none; }
  text.lab.seed{ fill:#1f1f1f; font-weight:bold; stroke-width:3px; }
  .hide{ display:none; }
  #tip{ position:fixed; pointer-events:none; z-index:20; background:#1f1f1fee; color:#fff; padding:6px 9px;
        border-radius:5px; font-size:12px; line-height:1.35; max-width:240px; display:none; }
  #tip b{ font-size:13px; }
  #hint{ position:fixed; bottom:8px; right:10px; font-size:11px; color:#aaa; }
</style></head>
<body>
<div id="bar">
  <h1>Internet Aesthetics Ecosystem</h1>
  <span class="sub">/*NNODES*/ aesthetics, /*NEDGES*/ links</span>
  <input id="q" list="names" placeholder="search an aesthetic (e.g. Vaporwave)" autocomplete="off">
  <datalist id="names"></datalist>
  <button id="bLab">Labels: hubs</button>
  <button id="bEdge" class="on">Edges</button>
  <button id="bReset">Reset view</button>
  /*LEGEND*/
</div>
<div id="wrap">
  <svg id="svg" xmlns="http://www.w3.org/2000/svg">
    <g id="vp"><g id="edges"></g><g id="nodes"></g><g id="labels"></g></g>
  </svg>
</div>
<div id="tip"></div>
<div id="hint">drag to pan &middot; scroll to zoom &middot; hover for details</div>
<script>
const DATA = /*DATA*/;
const svg=document.getElementById('svg'), vp=document.getElementById('vp'),
      gE=document.getElementById('edges'), gN=document.getElementById('nodes'), gL=document.getElementById('labels'),
      wrap=document.getElementById('wrap'), tip=document.getElementById('tip');
svg.setAttribute('viewBox', '0 0 '+DATA.W+' '+DATA.H);
const NS='http://www.w3.org/2000/svg';
const N=DATA.nodes;

// ---- edges ----
const ef=document.createDocumentFragment();
for(const [a,b] of DATA.edges){ const l=document.createElementNS(NS,'line');
  l.setAttribute('x1',N[a].x); l.setAttribute('y1',N[a].y); l.setAttribute('x2',N[b].x); l.setAttribute('y2',N[b].y);
  ef.appendChild(l); }
gE.appendChild(ef);

// ---- nodes + labels ----
const labEls=[]; const nf=document.createDocumentFragment(), lf=document.createDocumentFragment();
N.forEach((d,i)=>{
  const c=document.createElementNS(NS,'circle'); c.setAttribute('class','node');
  c.setAttribute('cx',d.x); c.setAttribute('cy',d.y); c.setAttribute('r',d.r); c.setAttribute('fill',d.c);
  c.addEventListener('mouseenter',ev=>showTip(ev,d)); c.addEventListener('mousemove',moveTip);
  c.addEventListener('mouseleave',hideTip); nf.appendChild(c);
  const t=document.createElementNS(NS,'text'); t.setAttribute('class','lab'+(d.t===2?' seed':''));
  t.setAttribute('x',d.x); t.setAttribute('y',d.y - d.r - 2); t.setAttribute('font-size',d.f);
  t.textContent=d.n; t.dataset.tier=d.t; labEls.push(t); lf.appendChild(t);
});
gN.appendChild(nf); gL.appendChild(lf);

// autocomplete
const dl=document.getElementById('names');
N.map(d=>d.n).sort().forEach(n=>{const o=document.createElement('option'); o.value=n; dl.appendChild(o);});

// ---- pan / zoom ----
let k=1,tx=0,ty=0;
function apply(){ vp.setAttribute('transform',`translate(${tx} ${ty}) scale(${k})`); }
function toVB(cx,cy){ const p=svg.createSVGPoint(); p.x=cx; p.y=cy; return p.matrixTransform(svg.getScreenCTM().inverse()); }
wrap.addEventListener('wheel',ev=>{ ev.preventDefault(); const v=toVB(ev.clientX,ev.clientY);
  const f=ev.deltaY<0?1.12:1/1.12; const nk=Math.min(400,Math.max(0.3,k*f));
  tx=v.x-(v.x-tx)*(nk/k); ty=v.y-(v.y-ty)*(nk/k); k=nk; apply(); updateLabels(); },{passive:false});
let drag=false,lx=0,ly=0;
wrap.addEventListener('mousedown',ev=>{drag=true;lx=ev.clientX;ly=ev.clientY;wrap.classList.add('drag');});
window.addEventListener('mouseup',()=>{drag=false;wrap.classList.remove('drag');});
window.addEventListener('mousemove',ev=>{ if(!drag)return; const a=toVB(lx,ly),b=toVB(ev.clientX,ev.clientY);
  tx+=b.x-a.x; ty+=b.y-a.y; lx=ev.clientX; ly=ev.clientY; apply(); });

// ---- tooltip ----
function showTip(ev,d){ tip.innerHTML='<b>'+d.n+'</b><br>community affect: '+d.a+'<br>links: '+d.d+'  &middot;  community '+d.k;
  tip.style.display='block'; moveTip(ev); }
function moveTip(ev){ tip.style.left=(ev.clientX+14)+'px'; tip.style.top=(ev.clientY+14)+'px'; }
function hideTip(){ tip.style.display='none'; }

// ---- labels: size-gated reveal (every visible word stays readable) ----
const modes=['auto','all','none']; let mi=0;
const MINPX=10;                               // ~70%+ of the 13px UI font; words never smaller
function ppu(){ return (svg.clientWidth||window.innerWidth) / DATA.W; }   // screen px per user unit
function updateLabels(){ const m=modes[mi], p=ppu();
  document.getElementById('bLab').textContent='Labels: '+m;
  for(let i=0;i<labEls.length;i++){ const d=N[i]; let show;
    if(m==='none') show=false; else if(m==='all') show=true;
    else show = d.t===2 || (d.f*k*p)>=MINPX;                 // auto: focal always, others when big enough
    labEls[i].classList.toggle('hide',!show); } }
document.getElementById('bLab').addEventListener('click',()=>{ mi=(mi+1)%modes.length; updateLabels(); });
window.addEventListener('resize',updateLabels);
document.getElementById('bEdge').addEventListener('click',e=>{ const on=e.target.classList.toggle('on');
  gE.style.display=on?'':'none'; });

// ---- search ----
function focusNode(name){ const d=N.find(x=>x.n.toLowerCase()===name.toLowerCase())
    || N.find(x=>x.n.toLowerCase().includes(name.toLowerCase())); if(!d)return;
  k=10; tx=DATA.W/2-k*d.x; ty=DATA.H/2-k*d.y; apply(); updateLabels();
  const ring=document.createElementNS(NS,'circle'); ring.setAttribute('cx',d.x); ring.setAttribute('cy',d.y);
  ring.setAttribute('r',d.r+8); ring.setAttribute('fill','none'); ring.setAttribute('stroke','#B4451F');
  ring.setAttribute('stroke-width',3); gN.appendChild(ring);
  setTimeout(()=>ring.remove(),2200); }
document.getElementById('q').addEventListener('change',e=>focusNode(e.target.value));
document.getElementById('q').addEventListener('keydown',e=>{ if(e.key==='Enter')focusNode(e.target.value); });

// ---- reset / initial fit ----
function reset(){ k=1; tx=0; ty=0; apply(); updateLabels(); }
document.getElementById('bReset').addEventListener('click',reset);
reset();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
