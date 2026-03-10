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


def export_viewer(solver, output_path='viewer.html', title=None, mode='auto',
                  distance_pc=414.0):
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
    distance_pc : float
        Source distance [pc] for angular size calculation (default: 414 for Orion Bar).
    """
    from .utils.constants import pc_cm
    
    if title is None:
        title = (f"PRISM-3D: {solver.nx}³ cells, "
                 f"G₀={float(np.max(solver.G0_external)):.0f}, "
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
    # 'rgb' is the flat composite color [R,G,B] used in composite overlay mode
    tracers = {
        'n_H':      {'data': solver.density[sl,sl,sl], 'label': 'Density [cm⁻³]',
                      'cmap': 'viridis', 'log': True, 'rgb': [180, 180, 180]},
        'T_gas':    {'data': solver.T_gas[sl,sl,sl], 'label': 'Temperature [K]',
                      'cmap': 'inferno', 'log': True, 'rgb': [255, 100, 30]},
        'G0':       {'data': solver.G0[sl,sl,sl], 'label': 'FUV Field [Habing]',
                      'cmap': 'magma', 'log': True, 'rgb': [255, 200, 50]},
        'x_H2':     {'data': np.maximum(solver.x_H2[sl,sl,sl], 1e-6), 'label': 'H₂ Fraction',
                      'cmap': 'blues', 'log': True, 'rgb': [50, 120, 255]},
        'x_Cp':     {'data': np.maximum(solver.x_Cp[sl,sl,sl], 1e-10), 'label': 'C⁺ (CII)',
                      'cmap': 'reds', 'log': True, 'rgb': [255, 50, 50]},
        'x_C':      {'data': np.maximum(solver.x_C[sl,sl,sl], 1e-10),
                      'label': 'C (CI)',
                      'cmap': 'oranges', 'log': True, 'rgb': [255, 160, 30]},
        'x_CO':     {'data': np.maximum(solver.x_CO[sl,sl,sl], 1e-8),
                      'label': 'CO Abundance',
                      'cmap': 'greens', 'log': True, 'rgb': [30, 200, 60]},
        'f_nano':   {'data': np.maximum(solver.f_nano[sl,sl,sl], 1e-4), 'label': 'Nano-grain Fraction',
                      'cmap': 'rdylgn', 'log': True, 'rgb': [200, 180, 50]},
        'T_dust':   {'data': solver.T_dust[sl,sl,sl], 'label': 'Dust Temperature [K]',
                      'cmap': 'hot', 'log': False, 'rgb': [255, 140, 0]},
        'Gamma_PE': {'data': solver.Gamma_PE[sl,sl,sl], 'label': 'PE Heating',
                      'cmap': 'hot', 'log': True, 'rgb': [255, 80, 180]},
    }
    
    # Serialize to JSON
    model_json = {
        'N': n_out,
        'box_pc': solver.box_size / pc_cm,
        'G0_ext': float(np.max(solver.G0_external)),
    }
    
    for key, meta in tracers.items():
        arr = meta['data'].astype(np.float32)
        flat = arr.ravel()
        # Compute p2/p98 percentile range for both log and linear
        ranges = {}
        for mode_log, mode_key in [(True, 'log'), (False, 'lin')]:
            if mode_log:
                vals = np.log10(np.maximum(flat, 1e-30))
            else:
                vals = flat.copy()
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))
                if vmax <= vmin:
                    vmax = vmin + 1
            else:
                vmin, vmax = 0.0, 1.0
            ranges[mode_key] = (round(vmin, 4), round(vmax, 4))
        model_json[key] = {
            'v': flat.tolist(),
            'l': meta['label'],
            'c': meta['cmap'],
            'g': meta['log'],
            'r': meta['rgb'],
            'pl': ranges['log'],   # [vmin, vmax] for log scale
            'pp': ranges['lin'],   # [vmin, vmax] for linear scale
        }
    
    # Compute PPV spectra for key lines (with optional velocity field + beam convolution)
    try:
        from .observations.spectra import compute_ppv_cube, convolve_ppv_beam
        from .observations.jwst_pipeline import LINE_BEAMS
        spec_lines = ['CII_158', 'OI_63', 'CI_609', 'CO_1-0', 'CO_2-1', 'CO_3-2']
        spec_colors = ['rgb(255,50,50)', 'rgb(50,170,255)', 'rgb(255,153,0)', 'rgb(50,200,85)', 'rgb(34,170,68)', 'rgb(17,136,51)']
        n_vel_viewer = 32

        # Extract LOS velocity component (axis=2 by default)
        v_los = None
        if solver.velocity_field is not None:
            v_los = solver.velocity_field[2]  # z-component for los_axis=2

        # Auto-adjust velocity range based on velocity field
        vel_range_kms = 8.0
        if v_los is not None:
            v_rms_kms = np.sqrt(np.mean(v_los**2)) / 1e5
            vel_range_kms = max(8.0, 3.0 * v_rms_kms)

        # Pixel angular size
        box_pc = solver.box_size / pc_cm
        pixel_arcsec = (box_pc / n_out) / distance_pc * 206265.0

        spectra_data = {}
        for i, ln in enumerate(spec_lines):
            try:
                ppv = compute_ppv_cube(solver, ln, los_axis=2,
                                       n_vel=n_vel_viewer,
                                       vel_range_kms=vel_range_kms,
                                       velocity_field=v_los)
                # Native cube
                cube = ppv['cube'].astype(np.float32)
                if step > 1:
                    cube = cube[::step, ::step, :]

                entry = {
                    's': cube.reshape(-1, n_vel_viewer).tolist(),
                    'c': spec_colors[i],
                }

                # Beam-convolved version
                beam_fwhm = LINE_BEAMS.get(ln, 0)
                if beam_fwhm > 0 and pixel_arcsec > 0:
                    ppv_conv = convolve_ppv_beam(ppv, beam_fwhm, pixel_arcsec)
                    cube_conv = ppv_conv['cube'].astype(np.float32)
                    if step > 1:
                        cube_conv = cube_conv[::step, ::step, :]
                    entry['sb'] = cube_conv.reshape(-1, n_vel_viewer).tolist()
                    entry['beam'] = round(beam_fwhm, 2)

                spectra_data[ln] = entry
            except Exception:
                pass
        if spectra_data:
            vel_axis = ppv['vel_axis'].tolist()
            model_json['spectra'] = {
                'vel': vel_axis,
                'lines': spectra_data,
                'n_vel': n_vel_viewer,
                'pixel_arcsec': round(pixel_arcsec, 3),
            }
            print(f"  Spectra: {list(spectra_data.keys())}")
            if v_los is not None:
                print(f"  Velocity field: v_rms={np.sqrt(np.mean(v_los**2))/1e5:.2f} km/s")
    except ImportError:
        pass

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
  <div style="font-size:9px;color:#3a4a5a;text-align:center">Drag to rotate/tilt · Click slice to show spectra</div>
  <div id="specPanel" style="display:none;background:#0d0d18;border-radius:10px;border:1px solid rgba(100,180,255,.08);padding:6px 8px;margin-top:4px">
    <div class="lbl" style="margin:0 0 4px 0">Line Profiles <span id="specPos" style="color:#607080;font-weight:400"></span></div>
    <canvas id="specCv" style="width:100%;height:120px;cursor:pointer"></canvas>
  </div>
</div>
<div id="panel">
  <div><div class="lbl">Tracer</div><div id="trbtn" class="btns"></div></div>
  <div><div class="lbl">View</div><div class="btns"><button class="btn on" onclick="setVw('vol')">Volume</button><button class="btn" onclick="setVw('slc')">Slice</button></div></div>
  <div id="obsDiv"><div class="lbl">Observation</div><div class="btns"><button class="btn on" id="obsNat" onclick="setObs(false)">Native</button><button class="btn" id="obsBeam" onclick="setObs(true)">Beam-conv</button></div><div id="beamInfo" style="font-size:9px;color:#607080;margin-top:3px"></div></div>
  <div><div class="lbl">Scale</div><div class="btns"><button class="btn" id="sLog" onclick="setScale('log')">Log</button><button class="btn" id="sLin" onclick="setScale('linear')">Linear</button><button class="btn on" id="sAuto" onclick="setScale('auto')">Auto</button></div></div>
  <div><div class="lbl">Range</div>
    <div style="display:flex;gap:4px;align-items:center"><span style="font-size:10px;width:28px">Min</span><input type="number" id="rMin" step="any" style="flex:1;background:rgba(20,30,50,.6);border:1px solid rgba(100,180,255,.15);border-radius:4px;color:#90a4ae;font-size:10px;padding:3px;font-family:monospace" onchange="setRange()"></div>
    <div style="display:flex;gap:4px;align-items:center;margin-top:3px"><span style="font-size:10px;width:28px">Max</span><input type="number" id="rMax" step="any" style="flex:1;background:rgba(20,30,50,.6);border:1px solid rgba(100,180,255,.15);border-radius:4px;color:#90a4ae;font-size:10px;padding:3px;font-family:monospace" onchange="setRange()"></div>
    <div style="display:flex;gap:3px;margin-top:3px"><button class="btn" style="font-size:9px" onclick="resetRange()">Reset</button><button class="btn" style="font-size:9px" onclick="setFullRange()">Full Range</button></div>
  </div>
  <div id="volctl"><div class="lbl">Opacity</div><input type="range" id="opsl" min="5" max="100" value="50" style="width:100%" oninput="op=this.value/100;paint()"></div>
  <div id="slcctl" style="display:none"><div class="lbl">Axis</div><div class="btns"><button class="btn on" onclick="setSA(0)">X</button><button class="btn" onclick="setSA(1)">Y</button><button class="btn" onclick="setSA(2)">Z</button></div><div class="lbl" style="margin-top:8px">Position</div><input type="range" id="spsl" min="0" max="100" value="50" style="width:100%" oninput="spos=this.value/100;paint()"></div>
  <div style="border-top:1px solid rgba(100,180,255,.06);padding-top:8px;margin-top:auto">
    <div style="font-size:9px;color:#4a5a6a;font-family:monospace;line-height:1.7">{n}³ cells · PRISM-3D v0.5<br>THEMIS dust · 75 reactions<br>HEALPix RT · ML accelerator</div>
  </div>
</div>
<script>
const CMAPS={{viridis:[[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,190,112],[122,209,81],[189,223,38],[253,231,37]],inferno:[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[243,118,27],[252,165,10],[246,215,70],[252,255,164]],magma:[[0,0,4],[18,13,49],[51,16,104],[90,17,126],[130,26,121],[170,43,107],[205,70,82],[231,109,56],[246,158,47],[252,210,79],[252,253,191]],blues:[[1,5,20],[3,19,43],[8,48,107],[8,81,156],[33,113,181],[66,146,198],[107,174,214],[158,202,225],[198,219,239],[222,235,247],[247,251,255]],reds:[[30,0,4],[60,0,8],[103,0,13],[165,15,21],[203,24,29],[239,59,44],[251,106,74],[252,146,114],[252,187,161],[254,224,210],[255,245,240]],greens:[[0,20,8],[0,40,15],[0,68,27],[0,109,44],[35,139,69],[65,171,93],[116,196,118],[161,217,155],[199,233,192],[229,245,224],[247,252,245]],oranges:[[30,10,0],[60,20,0],[90,40,0],[130,60,0],[170,85,0],[200,110,10],[225,140,30],[240,170,60],[250,200,100],[253,225,150],[255,245,200]],rdylgn:[[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]],hot:[[0,0,0],[30,0,0],[80,0,0],[150,0,0],[200,30,0],[230,80,0],[255,140,0],[255,200,0],[255,240,100],[255,255,200],[255,255,255]]}};
function sc(n,t){{const c=CMAPS[n]||CMAPS.viridis;t=Math.max(0,Math.min(1,t));const i=t*(c.length-1),a=Math.floor(i),b=Math.min(a+1,c.length-1),f=i-a;return[c[a][0]*(1-f)+c[b][0]*f,c[a][1]*(1-f)+c[b][1]*f,c[a][2]*(1-f)+c[b][2]*f];}}
const RAW={data_json};
const N=RAW.N,M={{}};
for(const[k,v]of Object.entries(RAW))if(v&&v.v)M[k]={{values:new Float32Array(v.v),label:v.l,cmap:v.c,log:v.g,rgb:v.r,plog:v.pl,plin:v.pp}};
let tr=Object.keys(M)[0],vw='vol',op=.5,ang=.5,til=.3,sax=0,spos=.5,drag=false,lx=0,ly=0;
let scaleMode='auto',userMin=null,userMax=null;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
function resize(){{cv.width=cv.parentElement.clientWidth;cv.height=cv.parentElement.clientHeight;paint();}}
window.addEventListener('resize',resize);
function isLog(m){{if(scaleMode==='log')return true;if(scaleMode==='linear')return false;return m.log;}}
function gr(m){{const p=isLog(m)?m.plog:m.plin;if(p)return[p[0],p[1]];return fullRange(m);}}
function fullRange(m){{const useLog=isLog(m);let a=Infinity,b=-Infinity;for(let i=0;i<m.values.length;i++){{const v=useLog?Math.log10(Math.max(m.values[i],1e-30)):m.values[i];if(isFinite(v)){{a=Math.min(a,v);b=Math.max(b,v);}}}}if(b<=a)b=a+1;return[a,b];}}
function getRange(m){{let[a,b]=gr(m);if(userMin!==null)a=userMin;if(userMax!==null)b=userMax;if(b<=a)b=a+1;return[a,b];}}
function setScale(s){{scaleMode=s;document.getElementById('sLog').classList.toggle('on',s==='log');document.getElementById('sLin').classList.toggle('on',s==='linear');document.getElementById('sAuto').classList.toggle('on',s==='auto');updateRangeInputs();paint();}}
function setRange(){{const mn=document.getElementById('rMin').value,mx=document.getElementById('rMax').value;userMin=mn!==''?parseFloat(mn):null;userMax=mx!==''?parseFloat(mx):null;paint();}}
function resetRange(){{userMin=null;userMax=null;updateRangeInputs();paint();}}
function setFullRange(){{const m=M[tr];if(!m)return;const[a,b]=fullRange(m);userMin=a;userMax=b;document.getElementById('rMin').value=a.toFixed(2);document.getElementById('rMax').value=b.toFixed(2);paint();}}
function updateRangeInputs(){{const m=M[tr];if(!m)return;const[a,b]=gr(m);document.getElementById('rMin').value='';document.getElementById('rMax').value='';document.getElementById('rMin').placeholder=isLog(m)?'10^'+a.toFixed(1):a.toPrecision(3);document.getElementById('rMax').placeholder=isLog(m)?'10^'+b.toFixed(1):b.toPrecision(3);}}
function volCmap(m){{return vw==='vol'?'viridis':m.cmap;}}
function paint(){{const m=M[tr];if(!m)return;const W=cv.width,H=cv.height;ctx.fillStyle='#0a0a12';ctx.fillRect(0,0,W,H);const useLog=isLog(m);const[vn,vx]=getRange(m);const cmap=volCmap(m);
if(vw==='vol'){{const ca=Math.cos(ang),sa=Math.sin(ang),ct=Math.cos(til),st=Math.sin(til),s=W/(N*2),cx=W/2,cy=H/2,cells=[];
for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{const raw=m.values[ix*N*N+iy*N+iz],v=useLog?Math.log10(Math.max(raw,1e-30)):raw,t=Math.max(0,Math.min(1,(v-vn)/(vx-vn)));if(t*op<.01)continue;const px=ix-N/2+.5,py=iy-N/2+.5,pz=iz-N/2+.5,rx=px*ca-pz*sa,rz=px*sa+pz*ca,ry=py*ct-rz*st,dp=py*st+rz*ct;cells.push({{sx:rx,sy:ry,dp,t}});}}
cells.sort((a,b)=>a.dp-b.dp);const sz=s*.85;for(const c of cells){{const[r,g,b]=sc(cmap,c.t),a=Math.min(c.t*op*1.3,.92);ctx.fillStyle='rgba('+~~r+','+~~g+','+~~b+','+a.toFixed(3)+')';ctx.fillRect(cx+c.sx*s-sz/2,cy-c.sy*s-sz/2,sz,sz);}}
}}else{{const si=Math.floor(spos*(N-1)),img=ctx.createImageData(N,N);for(let a=0;a<N;a++)for(let b=0;b<N;b++){{let ix,iy,iz;if(sax===0){{ix=si;iy=a;iz=b;}}else if(sax===1){{ix=a;iy=si;iz=b;}}else{{ix=a;iy=b;iz=si;}}const raw=m.values[ix*N*N+iy*N+iz],v=useLog?Math.log10(Math.max(raw,1e-30)):raw,t=Math.max(0,Math.min(1,(v-vn)/(vx-vn))),[r,g,bl]=sc(cmap,t),p=(b*N+a)*4;img.data[p]=~~r;img.data[p+1]=~~g;img.data[p+2]=~~bl;img.data[p+3]=255;}}ctx.putImageData(img,0,0);ctx.drawImage(cv,0,0,N,N,0,0,W,H);}}
const fmt=v=>useLog?'10^'+v.toFixed(1):v.toPrecision(3);const st=Array.from({{length:11}},(_,i)=>{{const[r,g,b]=sc(cmap,i/10);return'rgb('+~~r+','+~~g+','+~~b+') '+i*10+'%';}}).join(',');
document.getElementById('cmin').textContent=fmt(vn);document.getElementById('cmax').textContent=fmt(vx);document.getElementById('cgrad').style.background='linear-gradient(to right,'+st+')';document.getElementById('trlabel').textContent=m.label;}}
cv.addEventListener('mousedown',e=>{{drag=true;lx=e.clientX;ly=e.clientY;}});
cv.addEventListener('mousemove',e=>{{if(!drag)return;ang+=(e.clientX-lx)*.008;til=Math.max(-1.2,Math.min(1.2,til+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;paint();}});
cv.addEventListener('mouseup',()=>drag=false);cv.addEventListener('mouseleave',()=>drag=false);
function setTr(k){{tr=k;userMin=null;userMax=null;document.querySelectorAll('#trbtn .btn').forEach(b=>b.classList.toggle('on',b.dataset.k===k));updateRangeInputs();paint();}}
function setVw(v){{vw=v;document.getElementById('volctl').style.display=v==='vol'?'':'none';document.getElementById('slcctl').style.display=v==='slc'?'':'none';paint();}}
function setSA(a){{sax=a;paint();}}
const tb=document.getElementById('trbtn');for(const k of Object.keys(M)){{const b=document.createElement('button');b.className='btn'+(k===tr?' on':'');b.textContent=M[k].label;b.dataset.k=k;b.onclick=()=>setTr(k);tb.appendChild(b);}}
// Spectra — individual panels per line with observation mode
const SP=RAW.spectra;
const specCv=document.getElementById('specCv'),specCtx=specCv?specCv.getContext('2d'):null;
let selLine=null,specPx=-1,specPy=-1,obsMode=false;
const lineKeys=SP?Object.keys(SP.lines):[];
const hasBeam=SP&&lineKeys.some(k=>SP.lines[k].sb);
if(!hasBeam)document.getElementById('obsDiv').style.display='none';
function setObs(on){{obsMode=on;document.getElementById('obsNat').classList.toggle('on',!on);document.getElementById('obsBeam').classList.toggle('on',on);const bi=document.getElementById('beamInfo');if(on&&selLine&&SP.lines[selLine]&&SP.lines[selLine].beam)bi.textContent='Beam: '+SP.lines[selLine].beam+'"';else bi.textContent='';if(specPx>=0)drawSpec(specPx,specPy);}}
function fmtI(v){{if(v===0)return'0';const e=Math.floor(Math.log10(Math.abs(v)));if(e>=-1&&e<=2)return v.toPrecision(2);return v.toExponential(1);}}
function specFWHM(s,vel,nv){{let pk=0;for(let i=0;i<nv;i++)if(s[i]>pk)pk=s[i];if(pk<=0)return 0;const hm=pk/2;let i0=-1,i1=-1;for(let i=0;i<nv;i++)if(s[i]>=hm){{if(i0<0)i0=i;i1=i;}};return i1>i0?(vel[i1]-vel[i0]):0;}}
function specInteg(s,dv,nv){{let sum=0;for(let i=0;i<nv;i++)sum+=s[i];return sum*dv;}}
function drawSpec(px,py){{
  if(!SP||!specCtx)return;specPx=px;specPy=py;
  document.getElementById('specPanel').style.display='';
  specCv.width=specCv.parentElement.clientWidth;specCv.height=140;
  const W=specCv.width,H=specCv.height;
  const vel=SP.vel,nv=SP.n_vel,idx=py*N+px,nL=lineKeys.length;
  if(nL===0)return;
  const dv=vel.length>1?vel[1]-vel[0]:1;
  document.getElementById('specPos').textContent='pixel ('+px+', '+py+')';
  const pw=W/nL,pt=16,pb=28,pl=6,pr=2,pH=H-pt-pb;
  specCtx.fillStyle='#0d0d18';specCtx.fillRect(0,0,W,H);
  for(let li=0;li<nL;li++){{
    const ln=lineKeys[li],ld=SP.lines[ln],s=ld.s[idx],sb=ld.sb?ld.sb[idx]:null;
    const x0=li*pw+pl,x1=(li+1)*pw-pr,w=x1-x0;
    // Determine which spectrum to show as primary
    const sPrimary=obsMode&&sb?sb:s;
    const sSecondary=obsMode&&sb?s:null;
    // Per-line max (from both native and convolved for consistent scale)
    let lmax=0;if(s)for(let i=0;i<nv;i++)if(s[i]>lmax)lmax=s[i];
    if(sb)for(let i=0;i<nv;i++)if(sb[i]>lmax)lmax=sb[i];
    if(lmax<=0)lmax=1;
    // Selected highlight
    if(ln===selLine){{specCtx.fillStyle='rgba(33,150,243,.1)';specCtx.fillRect(li*pw,0,pw,H);specCtx.strokeStyle='#2196f3';specCtx.lineWidth=1.5;specCtx.strokeRect(li*pw+.5,.5,pw-1,H-1);}}
    // Label + beam indicator
    specCtx.fillStyle=ld.c;specCtx.font='bold 9px system-ui';specCtx.textAlign='center';
    const beamStr=ld.beam&&obsMode?' ['+ld.beam+'"]':'';
    specCtx.fillText(ln+beamStr,x0+w/2,11);
    // Axes
    specCtx.strokeStyle='#1e2e3e';specCtx.lineWidth=.5;
    specCtx.beginPath();specCtx.moveTo(x0,pt);specCtx.lineTo(x0,pt+pH);specCtx.lineTo(x1,pt+pH);specCtx.stroke();
    // Peak intensity label
    specCtx.fillStyle='#506070';specCtx.font='8px monospace';specCtx.textAlign='left';
    specCtx.fillText(fmtI(lmax),x0+1,pt+8);
    // Velocity labels
    if(li===0||li===nL-1||nL<=3){{
      specCtx.fillStyle='#405060';specCtx.font='7px monospace';specCtx.textAlign='center';
      specCtx.fillText(vel[0].toFixed(0),x0,pt+pH+10);
      specCtx.fillText(vel[nv-1].toFixed(0),x1,pt+pH+10);
    }}
    if(!sPrimary)continue;
    // Draw secondary (dashed) if in obs mode
    if(sSecondary){{specCtx.strokeStyle=ld.c.replace(')',',0.35)').replace('rgb','rgba');specCtx.lineWidth=1;specCtx.setLineDash([3,3]);specCtx.beginPath();for(let i=0;i<nv;i++){{const x=x0+w*i/(nv-1),y=pt+pH*(1-sSecondary[i]/lmax);if(i===0)specCtx.moveTo(x,y);else specCtx.lineTo(x,y);}}specCtx.stroke();specCtx.setLineDash([]);}}
    // Draw primary (solid)
    specCtx.strokeStyle=ld.c;specCtx.lineWidth=1.5;specCtx.beginPath();
    for(let i=0;i<nv;i++){{
      const x=x0+w*i/(nv-1),y=pt+pH*(1-sPrimary[i]/lmax);
      if(i===0)specCtx.moveTo(x,y);else specCtx.lineTo(x,y);
    }}specCtx.stroke();
    specCtx.lineTo(x1,pt+pH);specCtx.lineTo(x0,pt+pH);specCtx.closePath();
    specCtx.fillStyle=ld.c.replace(')',',0.08)').replace('rgb','rgba');
    specCtx.fill();
    // Line parameter annotations
    const pk=Math.max(...sPrimary),fw=specFWHM(sPrimary,vel,nv),intg=specInteg(sPrimary,dv,nv);
    specCtx.fillStyle='#607080';specCtx.font='7px monospace';specCtx.textAlign='center';
    specCtx.fillText('pk:'+fmtI(pk)+' fw:'+fw.toFixed(1)+' I:'+fmtI(intg),x0+w/2,H-2);
  }}
}}
specCv.addEventListener('click',e=>{{
  if(!SP||lineKeys.length===0)return;
  const rect=specCv.getBoundingClientRect();
  const fx=(e.clientX-rect.left)/rect.width;
  const li=Math.floor(fx*lineKeys.length);
  if(li>=0&&li<lineKeys.length){{selLine=lineKeys[li];const ld=SP.lines[selLine];const bi=document.getElementById('beamInfo');if(obsMode&&ld&&ld.beam)bi.textContent='Beam: '+ld.beam+'"';else bi.textContent='';drawSpec(specPx,specPy);}}
}});
cv.addEventListener('click',e=>{{
  if(vw!=='slc')return;
  const rect=cv.getBoundingClientRect();
  const fx=(e.clientX-rect.left)/rect.width,fy=(e.clientY-rect.top)/rect.height;
  const px=Math.floor(fx*N),py=Math.floor(fy*N);
  if(px>=0&&px<N&&py>=0&&py<N)drawSpec(px,py);
}});
updateRangeInputs();setTimeout(resize,50);
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
  <div style="font-size:9px;color:#3a4a5a;text-align:center">Drag: rotate · Scroll: zoom · Shift+drag: pan · Click slice for spectra</div>
  <div id="specPanel" style="display:none;background:#0d0d18;border-radius:10px;border:1px solid rgba(100,180,255,.08);padding:6px 8px;margin-top:4px">
    <div class="lbl" style="margin:0 0 4px 0">Line Profiles <span id="specPos" style="color:#607080;font-weight:400"></span></div>
    <canvas id="specCv" style="width:100%;height:120px;cursor:pointer"></canvas>
  </div>
</div>
<div id="panel">
  <div><div class="lbl">Tracer</div><div id="trbtn" class="btns"></div></div>
  <div><div class="lbl">Mode</div><div class="btns">
    <button class="btn on" id="mVol" onclick="setMode('volume')">Volume</button>
    <button class="btn" id="mCmp" onclick="setMode('composite')">Composite</button>
    <button class="btn" id="mSlc" onclick="setMode('slice')">Slice</button>
    <button class="btn" id="mIso" onclick="setMode('iso')">Isosurface</button>
  </div></div>
  <div id="obsDiv"><div class="lbl">Observation</div><div class="btns"><button class="btn on" id="obsNat" onclick="setObs(false)">Native</button><button class="btn" id="obsBeam" onclick="setObs(true)">Beam-conv</button></div><div id="beamInfo" style="font-size:9px;color:#607080;margin-top:3px"></div></div>
  <div id="cmpDiv" style="display:none"><div class="lbl">Overlay tracers</div><div id="cmpbtns" class="btns"></div></div>
  <div><div class="lbl">Scale</div><div class="btns"><button class="btn" id="sLog" onclick="setScale('log')">Log</button><button class="btn" id="sLin" onclick="setScale('linear')">Linear</button><button class="btn on" id="sAuto" onclick="setScale('auto')">Auto</button></div></div>
  <div><div class="lbl">Range</div>
    <div style="display:flex;gap:4px;align-items:center"><span style="font-size:10px;width:28px">Min</span><input type="number" id="rMin" step="any" style="flex:1;background:rgba(20,30,50,.6);border:1px solid rgba(100,180,255,.15);border-radius:4px;color:#90a4ae;font-size:10px;padding:3px;font-family:monospace" onchange="setRange()"></div>
    <div style="display:flex;gap:4px;align-items:center;margin-top:3px"><span style="font-size:10px;width:28px">Max</span><input type="number" id="rMax" step="any" style="flex:1;background:rgba(20,30,50,.6);border:1px solid rgba(100,180,255,.15);border-radius:4px;color:#90a4ae;font-size:10px;padding:3px;font-family:monospace" onchange="setRange()"></div>
    <div style="display:flex;gap:3px;margin-top:3px"><button class="btn" style="font-size:9px" onclick="resetRange()">Reset</button><button class="btn" style="font-size:9px" onclick="setFullRange()">Full Range</button></div>
  </div>
  <div><div class="lbl">Opacity <span id="opVal">0.5</span></div><input type="range" id="opsl" min="1" max="100" value="50" style="width:100%" oninput="setOpacity(this.value/100)"></div>
  <div id="slcDiv" style="display:none"><div class="lbl">Slice axis</div><div class="btns"><button class="btn on" onclick="setSlcAx(0)">X</button><button class="btn" onclick="setSlcAx(1)">Y</button><button class="btn" onclick="setSlcAx(2)">Z</button></div>
  <div class="lbl" style="margin-top:8px">Position <span id="spVal">50%</span></div><input type="range" min="0" max="100" value="50" style="width:100%" oninput="setSlcPos(this.value/100)"></div>
  <div id="isoDiv" style="display:none"><div class="lbl">Iso level <span id="isoVal">0.5</span></div><input type="range" min="1" max="99" value="50" style="width:100%" oninput="setIsoLevel(this.value/100)"></div>
  <div><div class="lbl">Clip volume</div>
    <div style="display:flex;align-items:center;gap:4px;margin:3px 0"><span style="font-size:10px;width:14px">X</span><input type="range" min="0" max="100" value="0" style="flex:1" oninput="clip[0]=this.value/100;buildVolume()"><input type="range" min="0" max="100" value="100" style="flex:1" oninput="clip[1]=this.value/100;buildVolume()"></div>
    <div style="display:flex;align-items:center;gap:4px;margin:3px 0"><span style="font-size:10px;width:14px">Y</span><input type="range" min="0" max="100" value="0" style="flex:1" oninput="clip[2]=this.value/100;buildVolume()"><input type="range" min="0" max="100" value="100" style="flex:1" oninput="clip[3]=this.value/100;buildVolume()"></div>
    <div style="display:flex;align-items:center;gap:4px;margin:3px 0"><span style="font-size:10px;width:14px">Z</span><input type="range" min="0" max="100" value="0" style="flex:1" oninput="clip[4]=this.value/100;buildVolume()"><input type="range" min="0" max="100" value="100" style="flex:1" oninput="clip[5]=this.value/100;buildVolume()"></div>
  </div>
  <div style="border-top:1px solid rgba(100,180,255,.06);padding-top:8px;margin-top:auto">
    <div style="font-size:9px;color:#4a5a6a;font-family:monospace;line-height:1.7">{n}\u00B3 cells \u00B7 PRISM-3D v0.5<br>THEMIS dust \u00B7 75 reactions<br>Three.js WebGL volume rendering</div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const CMAPS={{viridis:[[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,190,112],[122,209,81],[189,223,38],[253,231,37]],inferno:[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[243,118,27],[252,165,10],[246,215,70],[252,255,164]],magma:[[0,0,4],[18,13,49],[51,16,104],[90,17,126],[130,26,121],[170,43,107],[205,70,82],[231,109,56],[246,158,47],[252,210,79],[252,253,191]],blues:[[1,5,20],[3,19,43],[8,48,107],[8,81,156],[33,113,181],[66,146,198],[107,174,214],[158,202,225],[198,219,239],[222,235,247],[247,251,255]],reds:[[30,0,4],[60,0,8],[103,0,13],[165,15,21],[203,24,29],[239,59,44],[251,106,74],[252,146,114],[252,187,161],[254,224,210],[255,245,240]],greens:[[0,20,8],[0,40,15],[0,68,27],[0,109,44],[35,139,69],[65,171,93],[116,196,118],[161,217,155],[199,233,192],[229,245,224],[247,252,245]],oranges:[[30,10,0],[60,20,0],[90,40,0],[130,60,0],[170,85,0],[200,110,10],[225,140,30],[240,170,60],[250,200,100],[253,225,150],[255,245,200]],rdylgn:[[165,0,38],[215,48,39],[244,109,67],[253,174,97],[254,224,139],[255,255,191],[217,239,139],[166,217,106],[102,189,99],[26,152,80],[0,104,55]],hot:[[0,0,0],[30,0,0],[80,0,0],[150,0,0],[200,30,0],[230,80,0],[255,140,0],[255,200,0],[255,240,100],[255,255,200],[255,255,255]]}};
function sc(n,t){{const c=CMAPS[n]||CMAPS.viridis;t=Math.max(0,Math.min(1,t));const i=t*(c.length-1),a=Math.floor(i),b=Math.min(a+1,c.length-1),f=i-a;return[c[a][0]*(1-f)+c[b][0]*f,c[a][1]*(1-f)+c[b][1]*f,c[a][2]*(1-f)+c[b][2]*f];}}
const RAW={data_json};
const N=RAW.N,M={{}};
for(const[k,v]of Object.entries(RAW))if(v&&v.v)M[k]={{values:new Float32Array(v.v),label:v.l,cmap:v.c,log:v.g,rgb:v.r,plog:v.pl,plin:v.pp}};
let scaleMode='auto',userMin=null,userMax=null;
function isLog(m){{if(scaleMode==='log')return true;if(scaleMode==='linear')return false;return m.log;}}
function gr(m){{const p=isLog(m)?m.plog:m.plin;if(p)return[p[0],p[1]];return fullRange(m);}}
function fullRange(m){{const useLog=isLog(m);let a=Infinity,b=-Infinity;for(let i=0;i<m.values.length;i++){{const v=useLog?Math.log10(Math.max(m.values[i],1e-30)):m.values[i];if(isFinite(v)){{a=Math.min(a,v);b=Math.max(b,v);}}}}if(b<=a)b=a+1;return[a,b];}}
function getRange(m){{let[a,b]=gr(m);if(userMin!==null)a=userMin;if(userMax!==null)b=userMax;if(b<=a)b=a+1;return[a,b];}}
function setScale(s){{scaleMode=s;document.getElementById('sLog').classList.toggle('on',s==='log');document.getElementById('sLin').classList.toggle('on',s==='linear');document.getElementById('sAuto').classList.toggle('on',s==='auto');updateRangeInputs();buildVolume();}}
function setRange(){{const mn=document.getElementById('rMin').value,mx=document.getElementById('rMax').value;userMin=mn!==''?parseFloat(mn):null;userMax=mx!==''?parseFloat(mx):null;buildVolume();}}
function resetRange(){{userMin=null;userMax=null;updateRangeInputs();buildVolume();}}
function setFullRange(){{const m=M[tr];if(!m)return;const[a,b]=fullRange(m);userMin=a;userMax=b;document.getElementById('rMin').value=a.toFixed(2);document.getElementById('rMax').value=b.toFixed(2);buildVolume();}}
function updateRangeInputs(){{const m=M[tr];if(!m)return;const[a,b]=gr(m);document.getElementById('rMin').value='';document.getElementById('rMax').value='';document.getElementById('rMin').placeholder=isLog(m)?'10^'+a.toFixed(1):a.toPrecision(3);document.getElementById('rMax').placeholder=isLog(m)?'10^'+b.toFixed(1):b.toPrecision(3);}}
function volCmap(m){{return(mode==='volume'||mode==='iso')?'viridis':m.cmap;}}

let tr=Object.keys(M)[0],mode='volume',op=.5,slcAx=0,slcPos=.5,isoLev=.5;
const clip=[0,1,0,1,0,1];  // xmin,xmax,ymin,ymax,zmin,zmax as fractions
const cmpSel=new Set(['x_H2','x_CO','n_H']);  // default composite selection
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

// Soft gaussian sprite texture for smooth volume rendering
const spriteCanvas=document.createElement('canvas');spriteCanvas.width=64;spriteCanvas.height=64;
const sctx=spriteCanvas.getContext('2d');
const grad=sctx.createRadialGradient(32,32,0,32,32,32);
grad.addColorStop(0,'rgba(255,255,255,1)');
grad.addColorStop(0.4,'rgba(255,255,255,0.9)');
grad.addColorStop(0.7,'rgba(255,255,255,0.5)');
grad.addColorStop(0.9,'rgba(255,255,255,0.15)');
grad.addColorStop(1,'rgba(255,255,255,0)');
sctx.fillStyle=grad;sctx.fillRect(0,0,64,64);
const sprTex=new THREE.CanvasTexture(spriteCanvas);

// Build point cloud volume
let volObjs=[];
let sliceMesh,isoMesh;
// Cell size scaled so gaussian blobs overlap well.
// The volume cube spans [-0.5, 0.5], cell spacing = 1/N.
// We want ~3x cell spacing so the gaussian tails merge.
const cellSize=5.0/N;
function inClip(ix,iy,iz){{
  const fx=ix/N,fy=iy/N,fz=iz/N;
  return fx>=clip[0]&&fx<=clip[1]&&fy>=clip[2]&&fy<=clip[3]&&fz>=clip[4]&&fz<=clip[5];
}}
function addPoints(m,opMul,flatRGB){{
  const useLog=isLog(m);const[vn,vx]=getRange(m);
  const cmap=volCmap(m);
  const geo=new THREE.BufferGeometry(),pos=[],cols=[];
  for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{
    if(!inClip(ix,iy,iz))continue;
    const raw=m.values[ix*N*N+iy*N+iz],v=useLog?Math.log10(Math.max(raw,1e-30)):raw,t=Math.max(0,Math.min(1,(v-vn)/(vx-vn)));
    if(t<.01)continue;
    pos.push((ix+.5)/N-.5,(iy+.5)/N-.5,(iz+.5)/N-.5);
    const w=t*t;
    if(flatRGB){{cols.push(flatRGB[0]/255*w,flatRGB[1]/255*w,flatRGB[2]/255*w);}}
    else{{const[r,g,b]=sc(cmap,t);cols.push(r/255*w,g/255*w,b/255*w);}}
  }}
  if(pos.length===0)return;
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
  // Opacity scales with slider and inversely with grid depth
  // so 64³ doesn't blow out more than 32³
  const effOp=op*opMul*5.0/N;
  const mat=new THREE.PointsMaterial({{size:cellSize,map:sprTex,vertexColors:true,transparent:true,opacity:effOp,sizeAttenuation:true,depthWrite:false,blending:THREE.AdditiveBlending}});
  const pts=new THREE.Points(geo,mat);scene.add(pts);volObjs.push(pts);
}}
function buildVolume(){{
  for(const o of volObjs)scene.remove(o);volObjs=[];
  if(sliceMesh){{scene.remove(sliceMesh);sliceMesh=null;}}
  if(isoMesh){{scene.remove(isoMesh);isoMesh=null;}}
  const m=M[tr];if(!m)return;const useLog=isLog(m);const[vn,vx]=getRange(m);
  const cmap=volCmap(m);
  document.getElementById('trlabel').textContent=mode==='composite'?'Composite':m.label;
  const fmt=v=>useLog?'10^'+v.toFixed(1):v.toPrecision(3);
  document.getElementById('cmin').textContent=fmt(vn);document.getElementById('cmax').textContent=fmt(vx);
  const st=Array.from({{length:11}},(_,i)=>{{const[r,g,b]=sc(cmap,i/10);return'rgb('+~~r+','+~~g+','+~~b+') '+i*10+'%';}}).join(',');
  document.getElementById('cgrad').style.background='linear-gradient(to right,'+st+')';

  if(mode==='composite'){{
    // Overlay multiple tracers with additive blending — single color per species
    const sel=Array.from(cmpSel).filter(k=>M[k]);
    const opMul=1/Math.max(sel.length,.5);
    for(const k of sel)addPoints(M[k],opMul,M[k].rgb);
    document.getElementById('trlabel').innerHTML=sel.map(k=>{{const c=M[k].rgb;return'<span style="color:rgb('+c[0]+','+c[1]+','+c[2]+')">\u25CF</span> '+M[k].label;}}).join('&ensp;');
  }}else if(mode==='volume'){{
    addPoints(m,1);
  }}else if(mode==='slice'){{
    const si=Math.floor(slcPos*(N-1));
    const canvas=document.createElement('canvas');canvas.width=N;canvas.height=N;
    const ctx=canvas.getContext('2d'),img=ctx.createImageData(N,N);
    for(let a=0;a<N;a++)for(let b=0;b<N;b++){{
      let ix,iy,iz;if(slcAx===0){{ix=si;iy=a;iz=b;}}else if(slcAx===1){{ix=a;iy=si;iz=b;}}else{{ix=a;iy=b;iz=si;}}
      const raw=m.values[ix*N*N+iy*N+iz],v=useLog?Math.log10(Math.max(raw,1e-30)):raw,t=Math.max(0,Math.min(1,(v-vn)/(vx-vn))),[r,g,bl]=sc(cmap,t),p=(b*N+a)*4;
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
    const geo=new THREE.BufferGeometry(),verts=[];
    const threshold=vn+(vx-vn)*isoLev;
    for(let ix=0;ix<N;ix++)for(let iy=0;iy<N;iy++)for(let iz=0;iz<N;iz++){{
      if(!inClip(ix,iy,iz))continue;
      const raw=m.values[ix*N*N+iy*N+iz],v=useLog?Math.log10(Math.max(raw,1e-30)):raw;
      if(v<threshold)continue;
      let onSurface=false;
      for(const[dx,dy,dz]of[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]){{
        const nx=ix+dx,ny=iy+dy,nz=iz+dz;
        if(nx<0||ny<0||nz<0||nx>=N||ny>=N||nz>=N){{onSurface=true;break;}}
        const nv=useLog?Math.log10(Math.max(m.values[nx*N*N+ny*N+nz],1e-30)):m.values[nx*N*N+ny*N+nz];
        if(nv<threshold){{onSurface=true;break;}}
      }}
      if(!onSurface)continue;
      const x=(ix+.5)/N-.5,y=(iy+.5)/N-.5,z=(iz+.5)/N-.5,s=.5/N;
      const faces=[[-s,0,0,0,-s,0,0,s,0,-s,0,0,0,s,0,0,-s,0],[s,0,0,0,s,0,0,-s,0,s,0,0,0,-s,0,0,s,0],[0,-s,0,-s,0,0,s,0,0,0,-s,0,s,0,0,-s,0,0],[0,s,0,s,0,0,-s,0,0,0,s,0,-s,0,0,s,0,0],[0,0,-s,0,-s,0,0,s,0,0,0,-s,0,s,0,0,-s,0],[0,0,s,0,s,0,0,-s,0,0,0,s,0,-s,0,0,s,0]];
      for(const f of faces)for(let i=0;i<18;i+=3)verts.push(x+f[i],y+f[i+1],z+f[i+2]);
    }}
    if(verts.length>0){{
      geo.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));
      geo.computeVertexNormals();
      const[r,g,b]=sc(cmap,isoLev);
      const mat=new THREE.MeshPhongMaterial({{color:new THREE.Color(r/255,g/255,b/255),transparent:true,opacity:.8,side:THREE.DoubleSide}});
      isoMesh=new THREE.Mesh(geo,mat);scene.add(isoMesh);
    }}
  }}
  // Wireframe clip box
  scene.children=scene.children.filter(c=>c!==scene.userData.box);
  const bw=clip[1]-clip[0],bh=clip[3]-clip[2],bd=clip[5]-clip[4];
  const bx=(clip[0]+clip[1])/2-.5,by=(clip[2]+clip[3])/2-.5,bz=(clip[4]+clip[5])/2-.5;
  const boxGeo=new THREE.BoxGeometry(bw,bh,bd);
  const box=new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),new THREE.LineBasicMaterial({{color:0x334466}}));
  box.position.set(bx,by,bz);
  scene.userData.box=box;scene.add(box);
}}

// Lighting
scene.add(new THREE.AmbientLight(0x404040,.6));
const dl=new THREE.DirectionalLight(0xffffff,.8);dl.position.set(2,3,1);scene.add(dl);

function resize(){{const w=cont.clientWidth,h=cont.clientHeight;ren.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();}}
window.addEventListener('resize',resize);

function animate(){{requestAnimationFrame(animate);ren.render(scene,cam);}}

function setTr(k){{tr=k;userMin=null;userMax=null;document.querySelectorAll('#trbtn .btn').forEach(b=>b.classList.toggle('on',b.dataset.k===k));updateRangeInputs();if(mode!=='composite')buildVolume();}}
function setMode(m){{mode=m;
  document.getElementById('mVol').classList.toggle('on',m==='volume');
  document.getElementById('mCmp').classList.toggle('on',m==='composite');
  document.getElementById('mSlc').classList.toggle('on',m==='slice');
  document.getElementById('mIso').classList.toggle('on',m==='iso');
  document.getElementById('cmpDiv').style.display=m==='composite'?'':'none';
  document.getElementById('slcDiv').style.display=m==='slice'?'':'none';
  document.getElementById('isoDiv').style.display=m==='iso'?'':'none';
  buildVolume();
}}
function toggleCmp(k,btn){{if(cmpSel.has(k))cmpSel.delete(k);else cmpSel.add(k);btn.classList.toggle('on');buildVolume();}}
function setOpacity(v){{op=v;document.getElementById('opVal').textContent=v.toFixed(2);buildVolume();}}
function setSlcAx(a){{slcAx=a;buildVolume();}}
function setSlcPos(v){{slcPos=v;document.getElementById('spVal').textContent=~~(v*100)+'%';buildVolume();}}
function setIsoLevel(v){{isoLev=v;document.getElementById('isoVal').textContent=v.toFixed(2);buildVolume();}}

// Init buttons
const tb=document.getElementById('trbtn');for(const k of Object.keys(M)){{const b=document.createElement('button');b.className='btn'+(k===tr?' on':'');b.textContent=M[k].label;b.dataset.k=k;b.onclick=()=>setTr(k);tb.appendChild(b);}}
// Composite toggle buttons
const cb=document.getElementById('cmpbtns');for(const k of Object.keys(M)){{const c=M[k].rgb;const b=document.createElement('button');b.className='btn'+(cmpSel.has(k)?' on':'');b.innerHTML='\u25CF '+M[k].label;b.style.color='rgb('+c[0]+','+c[1]+','+c[2]+')';b.onclick=function(){{toggleCmp(k,this)}};cb.appendChild(b);}}

// Spectra — individual panels per line with observation mode
const SP=RAW.spectra;
const specCv=document.getElementById('specCv'),specCtx=specCv?specCv.getContext('2d'):null;
let selLine=null,specPx=-1,specPy=-1,obsMode=false;
const lineKeys=SP?Object.keys(SP.lines):[];
const hasBeam=SP&&lineKeys.some(k=>SP.lines[k].sb);
if(!hasBeam)document.getElementById('obsDiv').style.display='none';
function setObs(on){{obsMode=on;document.getElementById('obsNat').classList.toggle('on',!on);document.getElementById('obsBeam').classList.toggle('on',on);const bi=document.getElementById('beamInfo');if(on&&selLine&&SP.lines[selLine]&&SP.lines[selLine].beam)bi.textContent='Beam: '+SP.lines[selLine].beam+'"';else bi.textContent='';if(specPx>=0)drawSpec(specPx,specPy);}}
function fmtI(v){{if(v===0)return'0';const e=Math.floor(Math.log10(Math.abs(v)));if(e>=-1&&e<=2)return v.toPrecision(2);return v.toExponential(1);}}
function specFWHM(s,vel,nv){{let pk=0;for(let i=0;i<nv;i++)if(s[i]>pk)pk=s[i];if(pk<=0)return 0;const hm=pk/2;let i0=-1,i1=-1;for(let i=0;i<nv;i++)if(s[i]>=hm){{if(i0<0)i0=i;i1=i;}};return i1>i0?(vel[i1]-vel[i0]):0;}}
function specInteg(s,dv,nv){{let sum=0;for(let i=0;i<nv;i++)sum+=s[i];return sum*dv;}}
function drawSpec(px,py){{
  if(!SP||!specCtx)return;specPx=px;specPy=py;
  document.getElementById('specPanel').style.display='';
  specCv.width=specCv.parentElement.clientWidth;specCv.height=140;
  const W=specCv.width,H=specCv.height;
  const vel=SP.vel,nv=SP.n_vel,idx=py*N+px,nL=lineKeys.length;
  if(nL===0)return;
  const dv=vel.length>1?vel[1]-vel[0]:1;
  document.getElementById('specPos').textContent='pixel ('+px+', '+py+')';
  const pw=W/nL,pt=16,pb=28,pl=6,pr=2,pH=H-pt-pb;
  specCtx.fillStyle='#0d0d18';specCtx.fillRect(0,0,W,H);
  for(let li=0;li<nL;li++){{
    const ln=lineKeys[li],ld=SP.lines[ln],s=ld.s[idx],sb=ld.sb?ld.sb[idx]:null;
    const x0=li*pw+pl,x1=(li+1)*pw-pr,w=x1-x0;
    const sPrimary=obsMode&&sb?sb:s;
    const sSecondary=obsMode&&sb?s:null;
    let lmax=0;if(s)for(let i=0;i<nv;i++)if(s[i]>lmax)lmax=s[i];
    if(sb)for(let i=0;i<nv;i++)if(sb[i]>lmax)lmax=sb[i];
    if(lmax<=0)lmax=1;
    if(ln===selLine){{specCtx.fillStyle='rgba(33,150,243,.1)';specCtx.fillRect(li*pw,0,pw,H);specCtx.strokeStyle='#2196f3';specCtx.lineWidth=1.5;specCtx.strokeRect(li*pw+.5,.5,pw-1,H-1);}}
    specCtx.fillStyle=ld.c;specCtx.font='bold 9px system-ui';specCtx.textAlign='center';
    const beamStr=ld.beam&&obsMode?' ['+ld.beam+'"]':'';
    specCtx.fillText(ln+beamStr,x0+w/2,11);
    specCtx.strokeStyle='#1e2e3e';specCtx.lineWidth=.5;
    specCtx.beginPath();specCtx.moveTo(x0,pt);specCtx.lineTo(x0,pt+pH);specCtx.lineTo(x1,pt+pH);specCtx.stroke();
    specCtx.fillStyle='#506070';specCtx.font='8px monospace';specCtx.textAlign='left';
    specCtx.fillText(fmtI(lmax),x0+1,pt+8);
    if(li===0||li===nL-1||nL<=3){{
      specCtx.fillStyle='#405060';specCtx.font='7px monospace';specCtx.textAlign='center';
      specCtx.fillText(vel[0].toFixed(0),x0,pt+pH+10);
      specCtx.fillText(vel[nv-1].toFixed(0),x1,pt+pH+10);
    }}
    if(!sPrimary)continue;
    if(sSecondary){{specCtx.strokeStyle=ld.c.replace(')',',0.35)').replace('rgb','rgba');specCtx.lineWidth=1;specCtx.setLineDash([3,3]);specCtx.beginPath();for(let i=0;i<nv;i++){{const x=x0+w*i/(nv-1),y=pt+pH*(1-sSecondary[i]/lmax);if(i===0)specCtx.moveTo(x,y);else specCtx.lineTo(x,y);}}specCtx.stroke();specCtx.setLineDash([]);}}
    specCtx.strokeStyle=ld.c;specCtx.lineWidth=1.5;specCtx.beginPath();
    for(let i=0;i<nv;i++){{
      const x=x0+w*i/(nv-1),y=pt+pH*(1-sPrimary[i]/lmax);
      if(i===0)specCtx.moveTo(x,y);else specCtx.lineTo(x,y);
    }}specCtx.stroke();
    specCtx.lineTo(x1,pt+pH);specCtx.lineTo(x0,pt+pH);specCtx.closePath();
    specCtx.fillStyle=ld.c.replace(')',',0.08)').replace('rgb','rgba');
    specCtx.fill();
    const pk=Math.max(...sPrimary),fw=specFWHM(sPrimary,vel,nv),intg=specInteg(sPrimary,dv,nv);
    specCtx.fillStyle='#607080';specCtx.font='7px monospace';specCtx.textAlign='center';
    specCtx.fillText('pk:'+fmtI(pk)+' fw:'+fw.toFixed(1)+' I:'+fmtI(intg),x0+w/2,H-2);
  }}
}}
specCv.addEventListener('click',e=>{{
  if(!SP||lineKeys.length===0)return;
  const rect=specCv.getBoundingClientRect();
  const fx=(e.clientX-rect.left)/rect.width;
  const li=Math.floor(fx*lineKeys.length);
  if(li>=0&&li<lineKeys.length){{selLine=lineKeys[li];const ld=SP.lines[selLine];const bi=document.getElementById('beamInfo');if(obsMode&&ld&&ld.beam)bi.textContent='Beam: '+ld.beam+'"';else bi.textContent='';drawSpec(specPx,specPy);}}
}});
// Raycast click on slice
const raycaster=new THREE.Raycaster();
ren.domElement.addEventListener('click',e=>{{
  if(mode!=='slice'||!sliceMesh||!SP)return;
  const rect=ren.domElement.getBoundingClientRect();
  const mouse=new THREE.Vector2(((e.clientX-rect.left)/rect.width)*2-1,-((e.clientY-rect.top)/rect.height)*2+1);
  raycaster.setFromCamera(mouse,cam);
  const hits=raycaster.intersectObject(sliceMesh);
  if(hits.length===0)return;
  const uv=hits[0].uv;if(!uv)return;
  const px=Math.floor(uv.x*N),py=Math.floor((1-uv.y)*N);
  if(px>=0&&px<N&&py>=0&&py<N)drawSpec(px,py);
}});

resize();updateCam();updateRangeInputs();buildVolume();animate();
</script></body></html>'''
