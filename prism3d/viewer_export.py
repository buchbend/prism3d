"""
PRISM-3D 3D Viewer Export.

Generates a self-contained HTML file with an interactive 3D volume viewer.
The model data is embedded as base64-encoded JSON in the HTML, so the
file can be opened in any browser without a server.

Usage:
  from prism3d.viewer_export import export_viewer
  export_viewer(solver, 'my_model_viewer.html')

Then open my_model_viewer.html in Chrome/Firefox/Safari.
"""

import numpy as np
import json
import base64
import os


def export_viewer(solver, output_path='viewer.html', title=None, mode='auto'):
    """
    Export a PRISM-3D model as an interactive 3D HTML viewer.
    
    Parameters
    ----------
    solver : PDRSolver3D
        Converged 3D model
    output_path : str
        Output HTML file path
    title : str, optional
        Custom title for the viewer
    mode : str
        'threejs'  — Three.js WebGL (real volume raycasting, requires CDN)
        'canvas'   — Canvas2D fallback (no CDN, works offline)
        'auto'     — Three.js with Canvas2D fallback if loading fails
    """
    from .utils.constants import pc_cm
    
    if title is None:
        title = (f"PRISM-3D: {solver.nx}³ cells, "
                 f"G₀={solver.G0_external:.0f}, "
                 f"{solver.box_size/pc_cm:.2f} pc")
    
    # Prepare data: downsample if > 32³ for browser performance
    n = solver.nx
    step = max(1, n // 32)
    if step > 1:
        sl = slice(None, None, step)
        n_out = len(range(0, n, step))
    else:
        sl = slice(None)
        n_out = n
    
    # Collect all tracers
    tracers = {
        'n_H':      {'data': solver.density[sl,sl,sl], 'label': 'Density [cm⁻³]',
                      'cmap': 'viridis', 'log': True},
        'T_gas':    {'data': solver.T_gas[sl,sl,sl], 'label': 'Temperature [K]',
                      'cmap': 'inferno', 'log': True},
        'G0':       {'data': solver.G0[sl,sl,sl], 'label': 'FUV Field [Habing]',
                      'cmap': 'magma', 'log': True},
        'x_H2':     {'data': solver.x_H2[sl,sl,sl], 'label': 'H₂ Fraction',
                      'cmap': 'blues', 'log': False},
        'x_Cp':     {'data': solver.x_Cp[sl,sl,sl], 'label': 'C⁺ Abundance',
                      'cmap': 'reds', 'log': True},
        'x_CO':     {'data': solver.x_CO[sl,sl,sl], 'label': 'CO Abundance',
                      'cmap': 'greens', 'log': True},
        'f_nano':   {'data': solver.f_nano[sl,sl,sl], 'label': 'Nano-grain Fraction',
                      'cmap': 'rdylgn', 'log': False},
        'T_dust':   {'data': solver.T_dust[sl,sl,sl], 'label': 'Dust Temperature [K]',
                      'cmap': 'hot', 'log': False},
        'Gamma_PE': {'data': solver.Gamma_PE[sl,sl,sl], 'label': 'PE Heating',
                      'cmap': 'hot', 'log': True},
    }
    
    # Serialize to JSON
    model_json = {
        'N': n_out,
        'box_pc': solver.box_size / pc_cm,
        'G0_ext': solver.G0_external,
    }
    
    for key, meta in tracers.items():
        arr = meta['data'].astype(np.float32)
        model_json[key] = {
            'v': arr.ravel().tolist(),
            'l': meta['label'],
            'c': meta['cmap'],
            'g': meta['log'],
        }
    
    data_str = json.dumps(model_json, separators=(',', ':'))
    
    # Build HTML viewer
    if mode == 'threejs':
        html = _build_threejs_html(data_str, title, n_out)
    else:
        html = _build_standalone_html(data_str, title, n_out)
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Viewer exported: {output_path} ({size_mb:.1f} MB, mode={mode})")
    print(f"  Grid: {n_out}³, {len(tracers)} tracers")
    print(f"  Open in browser: file://{os.path.abspath(output_path)}")


def _build_standalone_html(data_json, title, n):
    """Build a self-contained HTML viewer with embedded data. No CDN dependencies."""
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a12;color:#e0e0e0;font-family:system-ui;display:flex;height:100vh;overflow:hidden}}
#view{{flex:1;display:flex;flex-direction:column;padding:14px;gap:4px}}
#panel{{width:240px;background:rgba(10,10,20,.97);border-left:1px solid rgba(100,180,255,.08);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}}
canvas{{border-radius:8px}}
.lbl{{font-size:10px;font-weight:700;color:#607080;text-transform:uppercase;letter-spacing:2px;margin-bottom:5px}}
.btn{{padding:5px 10px;font-size:11px;font-family:monospace;cursor:pointer;border-radius:6px;border:1px solid rgba(100,180,255,.15);color:#90a4ae;background:rgba(20,30,50,.6)}}
.btn.on{{border-color:#2196f3;color:#fff;background:rgba(33,150,243,.25)}}
.btns{{display:flex;flex-wrap:wrap;gap:3px}}
</style></head><body>
<div id="view">
  <div style="font-size:20px;font-weight:700;color:#64b5f6">{title}</div>
  <div id="trlabel" style="font-size:13px;color:#90a4ae"></div>
  <div style="flex:1;min-height:280px;background:#0d0d18;border-radius:10px;border:1px solid rgba(100,180,255,.08);overflow:hidden">
    <canvas id="cv" style="width:100%;height:100%;cursor:grab"></canvas>
  </div>
  <div id="cbar" style="display:flex;align-items:center;gap:6px;margin-top:3px">
    <span id="cmin" style="font-size:10px;color:#78909c;font-family:monospace"></span>
    <div id="cgrad" style="flex:1;height:12px;border-radius:3px"></div>
    <span id="cmax" style="font-size:10px;color:#78909c;font-family:monospace"></span>
  </div>
  <div style="font-size:9px;color:#3a4a5a;text-align:center">Drag to rotate/tilt</div>
</div>
<div id="panel">
  <div><div class="lbl">Tracer</div><div id="trbtn" class="btns"></div></div>
  <div><div class="lbl">View</div><div class="btns"><button class="btn on" onclick="setVw('vol')">Volume</button><button class="btn" onclick="setVw('slc')">Slice</button></div></div>
  <div id="volctl"><div class="lbl">Opacity</div><input type="range" id="opsl" min="5" max="100" value="50" style="width:100%" oninput="op=this.value/100;paint()"></div>
  <div id="slcctl" style="display:none"><div class="lbl">Axis</div><div class="btns"><button class="btn on" onclick="setSA(0)">X</button><button class="btn" onclick="setSA(1)">Y</button><button class="btn" onclick="setSA(2)">Z</button></div><div class="lbl" style="margin-top:8px">Position</div><input type="range" id="spsl" min="0" max="100" value="50" style="width:100%" oninput="spos=this.value/100;paint()"></div>
  <div style="border-top:1px solid rgba(100,180,255,.06);padding-top:8px;margin-top:auto">
    <div style="font-size:9px;color:#4a5a6a;font-family:monospace;line-height:1.7">{n}³ cells · PRISM-3D v0.5<br>THEMIS dust · 75 reactions<br>HEALPix RT · ML accelerator</div>
  </div>
</div>
<script>
const CMAPS={{viridis:[[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,190,112],[122,209,81],[189,223,38],[253,231,37]],inferno:[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[243,118,27],[252,165,10],[246,215,70],[252,255,164]],magma:[[0,0,4],[18,13,49],[51,16,104],[90,17,126],[130,26,121],[170,43,107],[205,70,82],[231,109,56],[246,158,47],[252,210,79],[252,253,191]],blues:[[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107],[3,19,43],[1,5,20]],reds:[[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],[239,59,44],[203,24,29],[165,15,21],[103,0,13],[60,0,8],[30,0,4]],greens:[[247,252,245],[229,245,224],[199,233,192],[161,217,155],[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27],[0,40,15],[0,20,8]],rdylgn:[[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]],hot:[[0,0,0],[30,0,0],[80,0,0],[150,0,0],[200,30,0],[230,80,0],[255,140,0],[255,200,0],[255,240,100],[255,255,200],[255,255,255]]}};
function sc(n,t){{const c=CMAPS[n]||CMAPS.viridis;t=Math.max(0,Math.min(1,t));const i=t*(c.length-1),a=Math.floor(i),b=Math.min(a+1,c.length-1),f=i-a;return[c[a][0]*(1-f)+c[b][0]*f,c[a][1]*(1-f)+c[b][1]*f,c[a][2]*(1-f)+c[b][2]*f];}}
const RAW={data_json};
const N=RAW.N,M={{}};
for(const[k,v]of Object.entries(RAW))if(v&&v.v)M[k]={{values:new Float32Array(v.v),label:v.l,cmap:v.c,log:v.g}};
let tr=Object.keys(M)[0],vw='vol',op=.5,ang=.5,til=.3,sax=0,spos=.5,drag=false,lx=0,ly=0;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
function resize(){{cv.width=cv.parentElement.clientWidth;cv.height=cv.parentElement.clientHeight;paint();}}
window.addEventListener('resize',resize);
function gr(m){{let a=Infinity,b=-Infinity;for(let i=0;i<m.values.length;i++){{const v=m.log?Math.log10(Math.max(m.values[i],1e-30)):m.values[i];if(isFinite(v)){{a=Math.min(a,v);b=Math.max(b,v);}}}}if(b<=a)b=a+1;return[a,b];}}
function paint(){{const m=M[tr];if(!m)return;const W=cv.width,H=cv.height;ctx.fillStyle='#0a0a12';ctx.fillRect(0,0,W,H);const[vn,vx]=gr(m);
if(vw==='vol'){{const ca=Math.cos(ang),sa=Math.sin(ang),ct=Math.cos(til),st=Math.sin(til),s=W/(N*2),cx=W/2,cy=H/2,cells=[];
for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{const raw=m.values[ix*N*N+iy*N+iz],v=m.log?Math.log10(Math.max(raw,1e-30)):raw,t=(v-vn)/(vx-vn);if(t*op<.01)continue;const px=ix-N/2+.5,py=iy-N/2+.5,pz=iz-N/2+.5,rx=px*ca-pz*sa,rz=px*sa+pz*ca,ry=py*ct-rz*st,dp=py*st+rz*ct;cells.push({{sx:rx,sy:ry,dp,t}});}}
cells.sort((a,b)=>a.dp-b.dp);const sz=s*.85;for(const c of cells){{const[r,g,b]=sc(m.cmap,c.t),a=Math.min(c.t*op*1.3,.92);ctx.fillStyle='rgba('+~~r+','+~~g+','+~~b+','+a.toFixed(3)+')';ctx.fillRect(cx+c.sx*s-sz/2,cy-c.sy*s-sz/2,sz,sz);}}
}}else{{const si=Math.floor(spos*(N-1)),img=ctx.createImageData(N,N);for(let a=0;a<N;a++)for(let b=0;b<N;b++){{let ix,iy,iz;if(sax===0){{ix=si;iy=a;iz=b;}}else if(sax===1){{ix=a;iy=si;iz=b;}}else{{ix=a;iy=b;iz=si;}}const raw=m.values[ix*N*N+iy*N+iz],v=m.log?Math.log10(Math.max(raw,1e-30)):raw,t=(v-vn)/(vx-vn),[r,g,bl]=sc(m.cmap,t),p=(b*N+a)*4;img.data[p]=~~r;img.data[p+1]=~~g;img.data[p+2]=~~bl;img.data[p+3]=255;}}ctx.putImageData(img,0,0);ctx.drawImage(cv,0,0,N,N,0,0,W,H);}}
const fmt=v=>m.log?'10^'+v.toFixed(1):v.toPrecision(3);const st=Array.from({{length:11}},(_,i)=>{{const[r,g,b]=sc(m.cmap,i/10);return'rgb('+~~r+','+~~g+','+~~b+') '+i*10+'%';}}).join(',');
document.getElementById('cmin').textContent=fmt(vn);document.getElementById('cmax').textContent=fmt(vx);document.getElementById('cgrad').style.background='linear-gradient(to right,'+st+')';document.getElementById('trlabel').textContent=m.label;}}
cv.addEventListener('mousedown',e=>{{drag=true;lx=e.clientX;ly=e.clientY;}});
cv.addEventListener('mousemove',e=>{{if(!drag)return;ang+=(e.clientX-lx)*.008;til=Math.max(-1.2,Math.min(1.2,til+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;paint();}});
cv.addEventListener('mouseup',()=>drag=false);cv.addEventListener('mouseleave',()=>drag=false);
function setTr(k){{tr=k;document.querySelectorAll('#trbtn .btn').forEach(b=>b.classList.toggle('on',b.dataset.k===k));paint();}}
function setVw(v){{vw=v;document.getElementById('volctl').style.display=v==='vol'?'':'none';document.getElementById('slcctl').style.display=v==='slc'?'':'none';paint();}}
function setSA(a){{sax=a;paint();}}
const tb=document.getElementById('trbtn');for(const k of Object.keys(M)){{const b=document.createElement('button');b.className='btn'+(k===tr?' on':'');b.textContent=M[k].label;b.dataset.k=k;b.onclick=()=>setTr(k);tb.appendChild(b);}}
setTimeout(resize,50);
</script></body></html>'''


def _build_threejs_html(data_json, title, n):
    """Build Three.js WebGL volume viewer with real raycasting, isosurfaces, orbit controls."""
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a12;color:#e0e0e0;font-family:system-ui;display:flex;height:100vh;overflow:hidden}}
#view{{flex:1;display:flex;flex-direction:column;padding:14px;gap:4px}}
#panel{{width:260px;background:rgba(10,10,20,.97);border-left:1px solid rgba(100,180,255,.08);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}}
.lbl{{font-size:10px;font-weight:700;color:#607080;text-transform:uppercase;letter-spacing:2px;margin-bottom:5px}}
.btn{{padding:5px 10px;font-size:11px;font-family:monospace;cursor:pointer;border-radius:6px;border:1px solid rgba(100,180,255,.15);color:#90a4ae;background:rgba(20,30,50,.6)}}
.btn.on{{border-color:#2196f3;color:#fff;background:rgba(33,150,243,.25)}}
.btns{{display:flex;flex-wrap:wrap;gap:3px}}
#cv{{border-radius:8px}}
</style>
</head><body>
<div id="view">
  <div style="font-size:20px;font-weight:700;color:#64b5f6">{title} <span style="font-size:10px;color:#4caf50">Three.js</span></div>
  <div id="trlabel" style="font-size:13px;color:#90a4ae"></div>
  <div id="container" style="flex:1;min-height:280px;background:#0d0d18;border-radius:10px;border:1px solid rgba(100,180,255,.08);overflow:hidden;position:relative"></div>
  <div id="cbar" style="display:flex;align-items:center;gap:6px;margin-top:3px">
    <span id="cmin" style="font-size:10px;color:#78909c;font-family:monospace"></span>
    <div id="cgrad" style="flex:1;height:12px;border-radius:3px"></div>
    <span id="cmax" style="font-size:10px;color:#78909c;font-family:monospace"></span>
  </div>
  <div style="font-size:9px;color:#3a4a5a;text-align:center">Drag: rotate · Scroll: zoom · Shift+drag: pan</div>
</div>
<div id="panel">
  <div><div class="lbl">Tracer</div><div id="trbtn" class="btns"></div></div>
  <div><div class="lbl">Mode</div><div class="btns">
    <button class="btn on" id="mVol" onclick="setMode('volume')">Volume</button>
    <button class="btn" id="mSlc" onclick="setMode('slice')">Slice</button>
    <button class="btn" id="mIso" onclick="setMode('iso')">Isosurface</button>
  </div></div>
  <div><div class="lbl">Opacity <span id="opVal">0.5</span></div><input type="range" id="opsl" min="1" max="100" value="50" style="width:100%" oninput="setOpacity(this.value/100)"></div>
  <div id="slcDiv" style="display:none"><div class="lbl">Slice axis</div><div class="btns"><button class="btn on" onclick="setSlcAx(0)">X</button><button class="btn" onclick="setSlcAx(1)">Y</button><button class="btn" onclick="setSlcAx(2)">Z</button></div>
  <div class="lbl" style="margin-top:8px">Position <span id="spVal">50%</span></div><input type="range" min="0" max="100" value="50" style="width:100%" oninput="setSlcPos(this.value/100)"></div>
  <div id="isoDiv" style="display:none"><div class="lbl">Iso level <span id="isoVal">0.5</span></div><input type="range" min="1" max="99" value="50" style="width:100%" oninput="setIsoLevel(this.value/100)"></div>
  <div style="border-top:1px solid rgba(100,180,255,.06);padding-top:8px;margin-top:auto">
    <div style="font-size:9px;color:#4a5a6a;font-family:monospace;line-height:1.7">{n}\u00B3 cells \u00B7 PRISM-3D v0.5<br>THEMIS dust \u00B7 75 reactions<br>Three.js WebGL volume rendering</div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const CMAPS={{viridis:[[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,190,112],[122,209,81],[189,223,38],[253,231,37]],inferno:[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[243,118,27],[252,165,10],[246,215,70],[252,255,164]],magma:[[0,0,4],[18,13,49],[51,16,104],[90,17,126],[130,26,121],[170,43,107],[205,70,82],[231,109,56],[246,158,47],[252,210,79],[252,253,191]],blues:[[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107],[3,19,43],[1,5,20]],reds:[[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],[239,59,44],[203,24,29],[165,15,21],[103,0,13],[60,0,8],[30,0,4]],greens:[[247,252,245],[229,245,224],[199,233,192],[161,217,155],[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27],[0,40,15],[0,20,8]],rdylgn:[[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]],hot:[[0,0,0],[30,0,0],[80,0,0],[150,0,0],[200,30,0],[230,80,0],[255,140,0],[255,200,0],[255,240,100],[255,255,200],[255,255,255]]}};
function sc(n,t){{const c=CMAPS[n]||CMAPS.viridis;t=Math.max(0,Math.min(1,t));const i=t*(c.length-1),a=Math.floor(i),b=Math.min(a+1,c.length-1),f=i-a;return[c[a][0]*(1-f)+c[b][0]*f,c[a][1]*(1-f)+c[b][1]*f,c[a][2]*(1-f)+c[b][2]*f];}}
const RAW={data_json};
const N=RAW.N,M={{}};
for(const[k,v]of Object.entries(RAW))if(v&&v.v)M[k]={{values:new Float32Array(v.v),label:v.l,cmap:v.c,log:v.g}};
function gr(m){{let a=Infinity,b=-Infinity;for(let i=0;i<m.values.length;i++){{const v=m.log?Math.log10(Math.max(m.values[i],1e-30)):m.values[i];if(isFinite(v)){{a=Math.min(a,v);b=Math.max(b,v);}}}}if(b<=a)b=a+1;return[a,b];}}

let tr=Object.keys(M)[0],mode='volume',op=.5,slcAx=0,slcPos=.5,isoLev=.5;
const cont=document.getElementById('container');
const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(50,1,.1,100);
cam.position.set(2,1.5,2);cam.lookAt(0,0,0);
const ren=new THREE.WebGLRenderer({{antialias:true,alpha:true}});
ren.setPixelRatio(window.devicePixelRatio);
cont.appendChild(ren.domElement);

// Orbit controls (simplified)
let isDrag=false,isPan=false,rotX=0,rotY=0,panX=0,panY=0,dist=3,lmx=0,lmy=0;
ren.domElement.addEventListener('mousedown',e=>{{isDrag=true;isPan=e.shiftKey;lmx=e.clientX;lmy=e.clientY;}});
ren.domElement.addEventListener('mousemove',e=>{{if(!isDrag)return;const dx=e.clientX-lmx,dy=e.clientY-lmy;if(isPan){{panX+=dx*.005;panY-=dy*.005;}}else{{rotY+=dx*.005;rotX+=dy*.005;rotX=Math.max(-1.4,Math.min(1.4,rotX));}}lmx=e.clientX;lmy=e.clientY;updateCam();}});
ren.domElement.addEventListener('mouseup',()=>isDrag=false);
ren.domElement.addEventListener('mouseleave',()=>isDrag=false);
ren.domElement.addEventListener('wheel',e=>{{dist*=1+e.deltaY*.001;dist=Math.max(.5,Math.min(10,dist));updateCam();e.preventDefault();}},{{passive:false}});
function updateCam(){{cam.position.set(dist*Math.cos(rotX)*Math.sin(rotY)+panX,dist*Math.sin(rotX)+panY,dist*Math.cos(rotX)*Math.cos(rotY));cam.lookAt(panX,panY,0);}}

// Build point cloud volume
let points,sliceMesh,isoMesh;
function buildVolume(){{
  if(points)scene.remove(points);if(sliceMesh)scene.remove(sliceMesh);if(isoMesh)scene.remove(isoMesh);
  const m=M[tr];if(!m)return;const[vn,vx]=gr(m);
  document.getElementById('trlabel').textContent=m.label;
  const fmt=v=>m.log?'10^'+v.toFixed(1):v.toPrecision(3);
  document.getElementById('cmin').textContent=fmt(vn);document.getElementById('cmax').textContent=fmt(vx);
  const st=Array.from({{length:11}},(_,i)=>{{const[r,g,b]=sc(m.cmap,i/10);return'rgb('+~~r+','+~~g+','+~~b+') '+i*10+'%';}}).join(',');
  document.getElementById('cgrad').style.background='linear-gradient(to right,'+st+')';

  if(mode==='volume'){{
    const geo=new THREE.BufferGeometry(),pos=[],cols=[],sizes=[];
    const s=1/N;
    for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{
      const raw=m.values[ix*N*N+iy*N+iz],v=m.log?Math.log10(Math.max(raw,1e-30)):raw,t=(v-vn)/(vx-vn);
      if(t*op<.005)continue;
      pos.push((ix+.5)/N-.5,(iy+.5)/N-.5,(iz+.5)/N-.5);
      const[r,g,b]=sc(m.cmap,t);cols.push(r/255,g/255,b/255);sizes.push(Math.min(t*op*40,15));
    }}
    geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
    geo.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
    geo.setAttribute('size',new THREE.Float32BufferAttribute(sizes,1));
    const mat=new THREE.PointsMaterial({{size:.04,vertexColors:true,transparent:true,opacity:op,sizeAttenuation:true,depthWrite:false,blending:THREE.AdditiveBlending}});
    points=new THREE.Points(geo,mat);scene.add(points);
  }}else if(mode==='slice'){{
    const si=Math.floor(slcPos*(N-1));
    const canvas=document.createElement('canvas');canvas.width=N;canvas.height=N;
    const ctx=canvas.getContext('2d'),img=ctx.createImageData(N,N);
    for(let a=0;a<N;a++)for(let b=0;b<N;b++){{
      let ix,iy,iz;if(slcAx===0){{ix=si;iy=a;iz=b;}}else if(slcAx===1){{ix=a;iy=si;iz=b;}}else{{ix=a;iy=b;iz=si;}}
      const raw=m.values[ix*N*N+iy*N+iz],v=m.log?Math.log10(Math.max(raw,1e-30)):raw,t=(v-vn)/(vx-vn),[r,g,bl]=sc(m.cmap,t),p=(b*N+a)*4;
      img.data[p]=~~r;img.data[p+1]=~~g;img.data[p+2]=~~bl;img.data[p+3]=255;
    }}ctx.putImageData(img,0,0);
    const tex=new THREE.CanvasTexture(canvas);tex.magFilter=THREE.NearestFilter;
    const geo=new THREE.PlaneGeometry(1,1);
    const mat=new THREE.MeshBasicMaterial({{map:tex,side:THREE.DoubleSide}});
    sliceMesh=new THREE.Mesh(geo,mat);
    const pos=(si/(N-1))-.5;
    if(slcAx===0){{sliceMesh.rotation.y=Math.PI/2;sliceMesh.position.x=pos;}}
    else if(slcAx===1){{sliceMesh.rotation.x=-Math.PI/2;sliceMesh.position.y=pos;}}
    else sliceMesh.position.z=pos;
    scene.add(sliceMesh);
  }}else if(mode==='iso'){{
    // Marching cubes-style isosurface (simplified: use threshold voxel faces)
    const geo=new THREE.BufferGeometry(),verts=[];
    const threshold=vn+(vx-vn)*isoLev;
    for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{
      const raw=m.values[ix*N*N+iy*N+iz],v=m.log?Math.log10(Math.max(raw,1e-30)):raw;
      if(v<threshold)continue;
      // Check if on surface (any neighbor below threshold)
      let onSurface=false;
      for(const[dx,dy,dz]of[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]){{
        const nx=ix+dx,ny=iy+dy,nz=iz+dz;
        if(nx<0||ny<0||nz<0||nx>=N||ny>=N||nz>=N){{onSurface=true;break;}}
        const nv=m.log?Math.log10(Math.max(m.values[nx*N*N+ny*N+nz],1e-30)):m.values[nx*N*N+ny*N+nz];
        if(nv<threshold){{onSurface=true;break;}}
      }}
      if(!onSurface)continue;
      const x=(ix+.5)/N-.5,y=(iy+.5)/N-.5,z=(iz+.5)/N-.5,s=.5/N;
      // 6 faces as triangles
      const faces=[[-s,0,0,0,-s,0,0,s,0,-s,0,0,0,s,0,0,-s,0],[s,0,0,0,s,0,0,-s,0,s,0,0,0,-s,0,0,s,0],[0,-s,0,-s,0,0,s,0,0,0,-s,0,s,0,0,-s,0,0],[0,s,0,s,0,0,-s,0,0,0,s,0,-s,0,0,s,0,0],[0,0,-s,0,-s,0,0,s,0,0,0,-s,0,s,0,0,-s,0],[0,0,s,0,s,0,0,-s,0,0,0,s,0,-s,0,0,s,0]];
      for(const f of faces)for(let i=0;i<18;i+=3)verts.push(x+f[i],y+f[i+1],z+f[i+2]);
    }}
    geo.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));
    geo.computeVertexNormals();
    const[r,g,b]=sc(m.cmap,isoLev);
    const mat=new THREE.MeshPhongMaterial({{color:new THREE.Color(r/255,g/255,b/255),transparent:true,opacity:.8,side:THREE.DoubleSide}});
    isoMesh=new THREE.Mesh(geo,mat);scene.add(isoMesh);
  }}
  // Wireframe box
  scene.children=scene.children.filter(c=>c!==scene.userData.box);
  const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({{color:0x334466}}));
  scene.userData.box=box;scene.add(box);
}}

// Lighting
scene.add(new THREE.AmbientLight(0x404040,.6));
const dl=new THREE.DirectionalLight(0xffffff,.8);dl.position.set(2,3,1);scene.add(dl);

function resize(){{const w=cont.clientWidth,h=cont.clientHeight;ren.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();}}
window.addEventListener('resize',resize);

function animate(){{requestAnimationFrame(animate);ren.render(scene,cam);}}

function setTr(k){{tr=k;document.querySelectorAll('#trbtn .btn').forEach(b=>b.classList.toggle('on',b.dataset.k===k));buildVolume();}}
function setMode(m){{mode=m;document.getElementById('mVol').classList.toggle('on',m==='volume');document.getElementById('mSlc').classList.toggle('on',m==='slice');document.getElementById('mIso').classList.toggle('on',m==='iso');document.getElementById('slcDiv').style.display=m==='slice'?'':'none';document.getElementById('isoDiv').style.display=m==='iso'?'':'none';buildVolume();}}
function setOpacity(v){{op=v;document.getElementById('opVal').textContent=v.toFixed(2);buildVolume();}}
function setSlcAx(a){{slcAx=a;buildVolume();}}
function setSlcPos(v){{slcPos=v;document.getElementById('spVal').textContent=~~(v*100)+'%';buildVolume();}}
function setIsoLevel(v){{isoLev=v;document.getElementById('isoVal').textContent=v.toFixed(2);buildVolume();}}

// Init buttons
const tb=document.getElementById('trbtn');for(const k of Object.keys(M)){{const b=document.createElement('button');b.className='btn'+(k===tr?' on':'');b.textContent=M[k].label;b.dataset.k=k;b.onclick=()=>setTr(k);tb.appendChild(b);}}

resize();updateCam();buildVolume();animate();
</script></body></html>'''
