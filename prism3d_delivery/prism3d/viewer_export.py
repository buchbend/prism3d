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


def export_viewer(solver, output_path='viewer.html', title=None):
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
    
    # Serialize to JSON-compatible format
    model_json = {
        'N': n_out,
        'title': title,
        'box_pc': solver.box_size / pc_cm,
        'G0_external': solver.G0_external,
        'tracers': {}
    }
    
    for key, meta in tracers.items():
        arr = meta['data'].astype(np.float32)
        model_json['tracers'][key] = {
            'values': arr.ravel().tolist(),
            'label': meta['label'],
            'cmap': meta['cmap'],
            'log': meta['log'],
        }
    
    data_json = json.dumps(model_json)
    
    # Read the template HTML
    template_path = os.path.join(os.path.dirname(__file__), 'viewer.html')
    with open(template_path, 'r') as f:
        html = f.read()
    
    # Replace demo data generation with actual model data
    inject_script = f"""
<script>
// --- Actual PRISM-3D model data ---
const MODEL_DATA = {data_json};
const N = MODEL_DATA.N;

function generateDemoData() {{
  const data = {{}};
  for (const [key, meta] of Object.entries(MODEL_DATA.tracers)) {{
    data[key] = {{
      values: new Float32Array(meta.values),
      label: meta.label,
      cmap: meta.cmap,
      log: meta.log,
    }};
  }}
  return data;
}}
</script>
"""
    
    # Insert model data before the main script
    html = html.replace(
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>',
        inject_script + '\n<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    )
    
    # Update title
    html = html.replace('<title>PRISM-3D Interactive Viewer</title>',
                         f'<title>{title}</title>')
    html = html.replace(
        '<div class="panel-subtitle">Interactive Volume Viewer</div>',
        f'<div class="panel-subtitle">{title}</div>'
    )
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Viewer exported: {output_path} ({size_mb:.1f} MB)")
    print(f"  Grid: {n_out}³, {len(tracers)} tracers")
    print(f"  Open in browser: file://{os.path.abspath(output_path)}")
