"""
PRISM-3D Automatic Evaluation & Visualization.

Produces a comprehensive diagnostic report after every model run:
  1. Model summary statistics
  2. Convergence diagnostics
  3. Physics validation checks
  4. Multi-panel midplane slices (3 axes)
  5. Radial/depth profiles
  6. Synthetic observation figures
  7. Dust evolution analysis
  8. HTML summary report

Usage:
  from prism3d.evaluate import evaluate_model
  evaluate_model(solver, output_dir='./results')
"""

import numpy as np
import os
import json
import time


def evaluate_model(solver, output_dir='./results', distance_pc=414, verbose=True):
    """
    Run a complete evaluation suite on a converged PRISM-3D model.
    
    Parameters
    ----------
    solver : PDRSolver3D
        Converged 3D PDR model
    output_dir : str
        Directory for all outputs
    distance_pc : float
        Source distance for synthetic observations
    verbose : bool
    
    Returns
    -------
    report : dict
        Complete evaluation report
    """
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  PRISM-3D Evaluation Suite")
        print(f"{'='*60}")
    
    report = {}
    t0 = time.time()
    
    # 1. Model summary
    report['model'] = _model_summary(solver)
    if verbose:
        _print_summary(report['model'])
    
    # 2. Physics validation
    report['validation'] = _physics_validation(solver)
    if verbose:
        _print_validation(report['validation'])
    
    # 3. Generate synthetic observations
    from .observations.jwst_pipeline import generate_observations
    obs = generate_observations(solver, distance_pc=distance_pc, los_axis=2)
    report['observations'] = {k: float(np.mean(v)) for k, v in obs.items()
                               if isinstance(v, np.ndarray) and v.ndim == 2}
    
    # 4. Generate all figures
    if verbose:
        print(f"\n  Generating figures...")
    
    _plot_midplane_triptych(solver, output_dir)
    _plot_depth_profiles(solver, output_dir)
    _plot_dust_evolution(solver, output_dir)
    _plot_synthetic_observations(solver, obs, output_dir, distance_pc)
    _plot_summary_dashboard(solver, obs, report, output_dir)
    
    # 5. Export interactive 3D viewer
    if verbose:
        print(f"  Exporting 3D viewer...")
    try:
        from .viewer_export import export_viewer
        export_viewer(solver, os.path.join(output_dir, 'viewer_3d.html'), mode='canvas')
        export_viewer(solver, os.path.join(output_dir, 'viewer_3d_webgl.html'), mode='threejs')
    except Exception as e:
        if verbose:
            print(f"  (3D viewer export skipped: {e})")
    
    # 5. Save data
    solver.save(os.path.join(output_dir, 'model.npz'))
    np.savez_compressed(os.path.join(output_dir, 'observations.npz'),
                         **{k: v for k, v in obs.items() if isinstance(v, np.ndarray)})
    
    # 6. Write report
    report['timing'] = {'evaluation_seconds': time.time() - t0}
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 7. HTML report
    _write_html_report(report, output_dir)
    
    if verbose:
        n_figs = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        print(f"\n  Output directory: {output_dir}")
        print(f"  Figures: {n_figs}")
        print(f"  Report: report.json, report.html")
        print(f"  Model:  model.npz")
        print(f"  Evaluation time: {time.time()-t0:.1f}s")
    
    return report


# ============================================================
# Summary Statistics
# ============================================================

def _model_summary(solver):
    """Compute model summary statistics."""
    from .utils.constants import pc_cm
    s = {}
    s['grid'] = f"{solver.nx}x{solver.ny}x{solver.nz}"
    s['n_cells'] = solver.n_cells
    s['box_pc'] = solver.box_size / pc_cm
    s['cell_pc'] = solver.cell_size / pc_cm
    s['G0_external'] = solver.G0_external
    s['zeta_CR_0'] = solver.zeta_CR_0
    
    for name, arr in [
        ('n_H', solver.density), ('T_gas', solver.T_gas),
        ('T_dust', solver.T_dust), ('G0', solver.G0), ('A_V', solver.A_V),
        ('x_H2', solver.x_H2), ('x_CO', solver.x_CO), ('x_Cp', solver.x_Cp),
        ('x_e', solver.x_e), ('f_nano', solver.f_nano), ('E_g', solver.E_g),
    ]:
        s[f'{name}_min'] = float(np.min(arr))
        s[f'{name}_max'] = float(np.max(arr))
        s[f'{name}_mean'] = float(np.mean(arr))
        s[f'{name}_med'] = float(np.median(arr))
    
    # Mass-weighted quantities
    mass = solver.density
    s['T_gas_massweighted'] = float(np.average(solver.T_gas, weights=mass))
    s['x_H2_massweighted'] = float(np.average(solver.x_H2, weights=mass))
    s['x_CO_massweighted'] = float(np.average(solver.x_CO, weights=mass))
    s['f_nano_massweighted'] = float(np.average(solver.f_nano, weights=mass))
    
    # Molecular fraction
    f_mol = 2 * solver.x_H2  # fraction of H in H2
    s['molecular_fraction'] = float(np.average(f_mol, weights=mass))
    
    return s


def _print_summary(s):
    print(f"\n  Model: {s['grid']} ({s['n_cells']} cells), "
          f"{s['box_pc']:.2f} pc, G₀={s['G0_external']}")
    print(f"  n_H:    {s['n_H_min']:.0f} – {s['n_H_max']:.0f} cm⁻³ "
          f"(mean {s['n_H_mean']:.0f})")
    print(f"  T_gas:  {s['T_gas_min']:.0f} – {s['T_gas_max']:.0f} K "
          f"(mass-weighted {s['T_gas_massweighted']:.0f})")
    print(f"  T_dust: {s['T_dust_min']:.0f} – {s['T_dust_max']:.0f} K")
    print(f"  G₀:     {s['G0_min']:.1f} – {s['G0_max']:.1f}")
    print(f"  A_V:    {s['A_V_min']:.2f} – {s['A_V_max']:.2f} mag")
    print(f"  f_mol:  {s['molecular_fraction']:.2f} (mass-weighted)")
    print(f"  f_nano: {s['f_nano_min']:.3f} – {s['f_nano_max']:.3f}")


# ============================================================
# Physics Validation
# ============================================================

def _physics_validation(solver):
    """Run automated physics checks."""
    checks = {}
    
    # H conservation: x_H + 2*x_H2 should = 1
    H_total = solver.x_HI + 2 * solver.x_H2
    checks['H_conservation'] = {
        'mean': float(np.mean(H_total)),
        'std': float(np.std(H_total)),
        'pass': abs(np.mean(H_total) - 1.0) < 0.05
    }
    
    # C conservation
    from .utils.constants import gas_phase_abundances
    x_C_expected = gas_phase_abundances.get('C', 1.4e-4)
    C_total = solver.x_Cp + solver.x_C + solver.x_CO
    checks['C_conservation'] = {
        'mean': float(np.mean(C_total)),
        'expected': x_C_expected,
        'ratio': float(np.mean(C_total) / x_C_expected),
        'pass': abs(np.mean(C_total) / x_C_expected - 1) < 0.3
    }
    
    # Temperature physicality
    checks['T_physical'] = {
        'all_positive': bool(np.all(solver.T_gas > 0)),
        'max_reasonable': bool(np.max(solver.T_gas) < 1e6),
        'pass': bool(np.all(solver.T_gas > 0) and np.max(solver.T_gas) < 1e6)
    }
    
    # Dust evolution occurred
    checks['dust_evolved'] = {
        'f_nano_range': float(np.max(solver.f_nano) - np.min(solver.f_nano)),
        'pass': float(np.max(solver.f_nano) - np.min(solver.f_nano)) > 0.001
    }
    
    # Chemical stratification (C+ should anticorrelate with CO)
    corr = np.corrcoef(solver.x_Cp.ravel(), solver.x_CO.ravel())[0, 1]
    checks['chemical_stratification'] = {
        'Cp_CO_correlation': float(corr),
        'pass': corr < 0  # Should be anti-correlated
    }
    
    n_pass = sum(1 for c in checks.values() if c.get('pass', False))
    checks['summary'] = f"{n_pass}/{len(checks)-1} checks passed"
    
    return checks


def _print_validation(checks):
    print(f"\n  Physics Validation:")
    for name, c in checks.items():
        if name == 'summary':
            continue
        status = "✓" if c.get('pass', False) else "✗"
        if name == 'H_conservation':
            print(f"    {status} H conservation: <H+2H₂> = {c['mean']:.4f} (should be 1.0)")
        elif name == 'C_conservation':
            print(f"    {status} C conservation: ratio = {c['ratio']:.2f} (should be 1.0)")
        elif name == 'T_physical':
            print(f"    {status} Temperature: all positive, max < 10⁶ K")
        elif name == 'dust_evolved':
            print(f"    {status} Dust evolution: Δf_nano = {c['f_nano_range']:.4f}")
        elif name == 'chemical_stratification':
            print(f"    {status} C⁺/CO anticorrelation: r = {c['Cp_CO_correlation']:.2f}")
    print(f"    → {checks['summary']}")


# ============================================================
# Plotting Functions
# ============================================================

def _get_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _plot_midplane_triptych(solver, output_dir):
    """Three midplane slices (xy, xz, yz) for key quantities."""
    plt = _get_matplotlib()
    from .utils.constants import pc_cm
    
    ext = [0, solver.box_size / pc_cm, 0, solver.box_size / pc_cm]
    quantities = [
        ('density', 'n_H [cm⁻³]', 'viridis', True),
        ('T_gas', 'T_gas [K]', 'inferno', True),
        ('G0', 'G₀', 'magma', True),
        ('x_H2', 'x(H₂)', 'RdYlBu_r', False),
        ('x_Cp', 'x(C⁺)', 'YlOrRd', True),
        ('x_CO', 'x(CO)', 'YlGn', True),
        ('f_nano', 'f_nano', 'RdYlGn', False),
        ('Gamma_PE', 'Γ_PE', 'hot', True),
    ]
    
    for axis, axis_label in [(0, 'yz'), (1, 'xz'), (2, 'xy')]:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f'PRISM-3D: {axis_label}-midplane (axis {axis})',
                     fontsize=14, fontweight='bold')
        
        mid_idx = [solver.nx, solver.ny, solver.nz][axis] // 2
        s = [slice(None)] * 3
        s[axis] = mid_idx
        s = tuple(s)
        
        for i, (attr, label, cmap, use_log) in enumerate(quantities):
            ax = axes[i // 4, i % 4]
            data = getattr(solver, attr)[s]
            if use_log and np.any(data > 0):
                vmin = np.percentile(data[data > 0], 2) if np.any(data > 0) else 1e-30
                from matplotlib.colors import LogNorm
                im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap,
                              norm=LogNorm(vmin=max(vmin, 1e-30)))
            else:
                im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('[pc]', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'midplane_{axis_label}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def _plot_depth_profiles(solver, output_dir):
    """1D profiles along the illumination axis (x)."""
    plt = _get_matplotlib()
    from .utils.constants import pc_cm
    
    # Average over y,z
    x_pc = np.linspace(0, solver.box_size/pc_cm, solver.nx)
    
    profiles = {}
    for attr in ['density', 'T_gas', 'T_dust', 'G0', 'A_V',
                  'x_HI', 'x_H2', 'x_Cp', 'x_C', 'x_CO', 'x_e',
                  'f_nano', 'E_g', 'Gamma_PE']:
        arr = getattr(solver, attr)
        profiles[attr] = np.mean(arr, axis=(1, 2))  # Average over y,z
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('PRISM-3D: Depth Profiles (averaged over y,z)',
                 fontsize=14, fontweight='bold')
    
    panels = [
        (0, 0, ['density'], ['n_H'], 'Density [cm⁻³]', True),
        (0, 1, ['T_gas', 'T_dust'], ['T_gas', 'T_dust'], 'Temperature [K]', True),
        (0, 2, ['G0'], ['G₀'], 'FUV Field', True),
        (1, 0, ['x_HI', 'x_H2'], ['H', 'H₂'], 'H Abundances', True),
        (1, 1, ['x_Cp', 'x_C', 'x_CO'], ['C⁺', 'C', 'CO'], 'C Abundances', True),
        (1, 2, ['x_e'], ['e⁻'], 'Electron Fraction', True),
        (2, 0, ['f_nano'], ['f_nano'], 'Nano-grain Fraction', False),
        (2, 1, ['E_g'], ['E_g'], 'Band Gap [eV]', False),
        (2, 2, ['Gamma_PE'], ['Γ_PE'], 'PE Heating', True),
    ]
    
    for r, c, keys, labels, ylabel, log in panels:
        ax = axes[r, c]
        for key, lab in zip(keys, labels):
            ax.plot(x_pc, profiles[key], label=lab, linewidth=2)
        if log and all(np.any(profiles[k] > 0) for k in keys):
            ax.set_yscale('log')
        ax.set_xlabel('x [pc]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_profiles.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_dust_evolution(solver, output_dir):
    """Dust evolution analysis figure."""
    plt = _get_matplotlib()
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('PRISM-3D: THEMIS Dust Evolution Analysis',
                 fontsize=14, fontweight='bold')
    
    G0_flat = solver.G0.ravel()
    fn_flat = solver.f_nano.ravel()
    eg_flat = solver.E_g.ravel()
    pe_flat = solver.Gamma_PE.ravel()
    n_flat = solver.density.ravel()
    
    # f_nano vs G0
    ax = axes[0]
    ax.scatter(G0_flat, fn_flat, c=np.log10(n_flat), cmap='viridis', s=5, alpha=0.5)
    ax.set_xlabel('G₀ [Habing]')
    ax.set_ylabel('f_nano')
    ax.set_xscale('log')
    ax.set_title('Nano-grain fraction vs FUV')
    
    # E_g vs G0
    ax = axes[1]
    ax.scatter(G0_flat, eg_flat, c=np.log10(n_flat), cmap='viridis', s=5, alpha=0.5)
    ax.set_xlabel('G₀ [Habing]')
    ax.set_ylabel('E_g [eV]')
    ax.set_xscale('log')
    ax.set_title('Band gap vs FUV')
    
    # PE heating vs G0*n
    ax = axes[2]
    mask = pe_flat > 0
    ax.scatter(G0_flat[mask] * n_flat[mask], pe_flat[mask],
              c=fn_flat[mask], cmap='RdYlGn', s=5, alpha=0.5)
    ax.set_xlabel('G₀ × n_H')
    ax.set_ylabel('Γ_PE [erg/cm³/s]')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('PE heating vs G₀×n')
    
    # Histogram of f_nano
    ax = axes[3]
    ax.hist(fn_flat, bins=30, color='steelblue', edgecolor='white')
    ax.axvline(1.0, color='red', linestyle='--', label='Diffuse ISM')
    ax.set_xlabel('f_nano')
    ax.set_ylabel('N cells')
    ax.set_title('Nano-grain distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dust_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_synthetic_observations(solver, obs, output_dir, distance_pc):
    """Delegate to jwst_pipeline's plotting."""
    from .observations.jwst_pipeline import plot_observations
    plot_observations(obs, solver,
                      savepath=os.path.join(output_dir, 'synthetic_observations.png'))


def _plot_summary_dashboard(solver, obs, report, output_dir):
    """One-page summary dashboard combining key results."""
    plt = _get_matplotlib()
    from .utils.constants import pc_cm
    
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f'PRISM-3D Model Summary — {report["model"]["grid"]}, '
                 f'G₀={report["model"]["G0_external"]:.0f}',
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
    ext = [0, solver.box_size / pc_cm, 0, solver.box_size / pc_cm]
    mid = solver.get_slice(axis=2)
    
    def imshow(ax, data, title, cmap, log=False):
        if log and np.any(data > 0):
            from matplotlib.colors import LogNorm
            vmin = np.percentile(data[data > 0], 2) if np.any(data > 0) else 1e-30
            im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap,
                          norm=LogNorm(vmin=max(vmin, 1e-30)))
        else:
            im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('[pc]', fontsize=8)
    
    # Row 1: Physical structure
    for i, (key, title, cmap, log) in enumerate([
        ('density', 'n_H [cm⁻³]', 'viridis', True),
        ('T_gas', 'T_gas [K]', 'inferno', True),
        ('G0', 'G₀ [Habing]', 'magma', True),
        ('A_V', 'A_V [mag]', 'bone_r', False),
    ]):
        ax = fig.add_subplot(gs[0, i])
        imshow(ax, mid[key], title, cmap, log)
    
    # Row 2: Chemistry + dust
    for i, (key, title, cmap, log) in enumerate([
        ('x_H2', 'x(H₂)', 'RdYlBu_r', False),
        ('x_CO', 'x(CO)', 'YlGn', True),
        ('f_nano', 'f_nano (THEMIS)', 'RdYlGn', False),
        ('Gamma_PE', 'Γ_PE [erg/cm³/s]', 'hot', True),
    ]):
        ax = fig.add_subplot(gs[1, i])
        imshow(ax, mid[key], title, cmap, log)
    
    # Row 3: Observations + info
    obs_panels = ['line_CII_158', 'dust_PACS160']
    for i, key in enumerate(obs_panels):
        if key in obs and isinstance(obs[key], np.ndarray):
            ax = fig.add_subplot(gs[2, i])
            data = obs[key]
            title = '[CII] 158μm' if 'CII' in key else 'Herschel 160μm'
            imshow(ax, data, title, 'RdYlBu_r' if 'CII' in key else 'hot',
                   log=np.any(data > 0))
    
    # Info panel
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    s = report['model']
    v = report['validation']
    info = (
        f"Model: {s['grid']} cells, {s['box_pc']:.2f} pc box\n"
        f"G₀ = {s['G0_external']:.0f}, ζ_CR = {s['zeta_CR_0']:.1e} s⁻¹\n"
        f"n_H = {s['n_H_min']:.0f} – {s['n_H_max']:.0f} cm⁻³\n"
        f"T_gas = {s['T_gas_min']:.0f} – {s['T_gas_max']:.0f} K "
        f"(<T>_M = {s['T_gas_massweighted']:.0f} K)\n"
        f"f_mol = {s['molecular_fraction']:.2f}\n"
        f"f_nano = {s['f_nano_min']:.3f} – {s['f_nano_max']:.3f}\n"
        f"\nDust: THEMIS 3-component, evolved\n"
        f"Chemistry: 75 reactions, 32 species\n"
        f"RT: HEALPix multi-ray, {12}+ directions\n"
        f"\nValidation: {v['summary']}\n"
        f"  H cons: {v['H_conservation']['mean']:.4f}\n"
        f"  C⁺/CO corr: {v['chemical_stratification']['Cp_CO_correlation']:.2f}"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# HTML Report
# ============================================================

def _write_html_report(report, output_dir):
    """Write an HTML summary report with embedded images."""
    images = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    s = report['model']
    v = report['validation']
    
    html = f"""<!DOCTYPE html>
<html><head>
<title>PRISM-3D Report — {s['grid']}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px;
         margin: auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;
            margin: 20px 0; }}
  .stat {{ background: white; padding: 15px; border-radius: 8px;
           box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
  .stat .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
  .stat .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
  .check {{ padding: 5px 10px; margin: 3px; border-radius: 4px; display: inline-block; }}
  .pass {{ background: #d4edda; color: #155724; }}
  .fail {{ background: #f8d7da; color: #721c24; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.15);
         margin: 10px 0; }}
  .figure {{ background: white; padding: 15px; border-radius: 8px;
             box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: 20px 0; }}
</style>
</head><body>

<h1>🔭 PRISM-3D Model Report</h1>
<p><em>3D PDR with THEMIS Dust Evolution — Generated automatically</em></p>

<div class="stats">
  <div class="stat"><div class="value">{s['grid']}</div><div class="label">Grid</div></div>
  <div class="stat"><div class="value">{s['G0_external']:.0f}</div><div class="label">G₀ [Habing]</div></div>
  <div class="stat"><div class="value">{s['molecular_fraction']:.0%}</div><div class="label">Molecular fraction</div></div>
  <div class="stat"><div class="value">{s['f_nano_min']:.2f}–{s['f_nano_max']:.2f}</div><div class="label">f_nano range</div></div>
  <div class="stat"><div class="value">{s['T_gas_massweighted']:.0f} K</div><div class="label">⟨T_gas⟩_M</div></div>
  <div class="stat"><div class="value">{s['n_H_min']:.0f}–{s['n_H_max']:.0f}</div><div class="label">n_H range [cm⁻³]</div></div>
  <div class="stat"><div class="value">{s['A_V_min']:.1f}–{s['A_V_max']:.1f}</div><div class="label">A_V range [mag]</div></div>
  <div class="stat"><div class="value">{s['box_pc']:.2f} pc</div><div class="label">Box size</div></div>
</div>

<h2>Physics Validation</h2>
"""
    
    for name, c in v.items():
        if name == 'summary':
            continue
        cls = 'pass' if c.get('pass', False) else 'fail'
        icon = '✓' if c.get('pass') else '✗'
        html += f'<span class="check {cls}">{icon} {name}</span>\n'
    html += f'<p><strong>{v["summary"]}</strong></p>\n'
    
    html += '<h2>Figures</h2>\n'
    for img in images:
        label = img.replace('.png', '').replace('_', ' ').title()
        html += f'''<div class="figure">
  <h3>{label}</h3>
  <img src="{img}" alt="{label}">
</div>\n'''
    
    html += """
<hr>
<p><em>Generated by PRISM-3D v0.5 — PDR Radiative transfer with Integrated Structure and Microphysics</em></p>
</body></html>"""
    
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html)
