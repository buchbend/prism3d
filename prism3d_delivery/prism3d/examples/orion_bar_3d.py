"""
Orion Bar PDR Model for PRISM-3D.

Sets up a 3D model of the Orion Bar based on the PDRs4All JWST
observations (Berné et al. 2022; Habart et al. 2024; Peeters et al. 2024).

The Orion Bar is the prototypical bright PDR:
  - Distance: 414 pc (Menten+ 2007)
  - Illumination: θ¹ Ori C (O7V), G₀ ≈ 2-7 × 10⁴ Habing
  - Geometry: nearly edge-on ridge, inclined ~4° from face-on
  - Structure: IF → atomic layer → DF1/DF2/DF3 → molecular zone
  - Density: n_H ~ 10⁴-10⁵ cm⁻³ at dissociation fronts
  - Size: ~0.4 pc × 0.1 pc visible ridge

The model uses a density structure motivated by:
1. An exponential ramp from HII region to molecular zone
2. Embedded clumps at the dissociation fronts
3. Turbulent substructure matching the observed filaments

References:
  Berné et al. 2022, PASP 134, 054301 (PDRs4All design)
  Habart et al. 2024, A&A 685, A73 (NIR/MIR imaging)
  Peeters et al. 2024, A&A 685, A74 (NIRSpec spectroscopy)
  Elyajouri et al. 2024, A&A 685, A76 (Dust evolution)
  Goicoechea et al. 2016, Nature 537, 207 (Density structure)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def orion_bar_density(n_cells, box_size_cm):
    """
    Generate an Orion Bar-like density structure.
    
    The density increases exponentially from the HII region side
    (x=0, low density) into the molecular cloud (x=box, high density),
    with turbulent fluctuations and embedded dense clumps.
    
    Parameters
    ----------
    n_cells : int
        Cells per dimension
    box_size_cm : float
        Physical box size [cm]
    
    Returns
    -------
    density : ndarray (n_cells, n_cells, n_cells)
    cell_size : float
    """
    cs = box_size_cm / n_cells
    x = np.linspace(0, 1, n_cells)  # Normalized position
    
    # Base density profile: exponential ramp
    # x=0: HII/PDR boundary, n ~ 5000 cm⁻³
    # x=0.3: Dissociation front, n ~ 50000 cm⁻³
    # x=1: Deep molecular zone, n ~ 100000 cm⁻³
    n_surface = 5e3
    n_deep = 1e5
    n_profile = n_surface * np.exp(np.log(n_deep / n_surface) * x)
    
    # Make 3D: density varies primarily along x (illumination axis)
    density = np.zeros((n_cells, n_cells, n_cells))
    for i in range(n_cells):
        density[i, :, :] = n_profile[i]
    
    # Add turbulent fluctuations (moderate Mach ~ 2)
    rng = np.random.RandomState(42)
    kx = np.fft.fftfreq(n_cells, d=cs)
    ky = np.fft.fftfreq(n_cells, d=cs)
    kz = np.fft.fftfreq(n_cells, d=cs)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0
    
    amplitude = K**(-3.7 / 2.0)
    amplitude[0, 0, 0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, (n_cells, n_cells, n_cells))
    gaussian = np.real(np.fft.ifftn(amplitude * np.exp(1j * phases)))
    gaussian *= 0.5 / max(np.std(gaussian), 1e-10)  # sigma(ln n) ~ 0.5 (Mach~2)
    
    density *= np.exp(gaussian - 0.5**2 / 2)
    
    # Add dense clumps at the dissociation front (x ~ 0.2-0.4)
    # These represent the "terraced" DF1/DF2/DF3 structure
    n_clumps = 5
    clump_centers = np.array([
        [0.2, 0.3, 0.5],
        [0.25, 0.7, 0.3],
        [0.3, 0.5, 0.7],
        [0.35, 0.2, 0.6],
        [0.28, 0.6, 0.4],
    ]) * box_size_cm
    
    xx = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    yy = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    zz = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
    
    for ic in range(n_clumps):
        cx, cy, cz = clump_centers[ic]
        r_clump = 0.03 * box_size_cm * (0.8 + 0.4 * rng.rand())
        dist = np.sqrt((XX - cx)**2 + (YY - cy)**2 + (ZZ - cz)**2)
        clump = 2e5 * np.exp(-0.5 * (dist / r_clump)**2)
        density = np.maximum(density, clump)
    
    density = np.maximum(density, 100.0)  # Floor
    return density, cs


def run_orion_bar(n_cells=16, n_iter=5):
    """
    Run a 3D Orion Bar model.
    
    Parameters
    ----------
    n_cells : int
        Cells per dimension
    n_iter : int
        Number of solver iterations
    """
    from prism3d.core.solver_3d import PDRSolver3D
    from prism3d.observations.jwst_pipeline import generate_observations, plot_observations
    from prism3d.utils.constants import pc_cm
    
    print("="*60)
    print("  PRISM-3D: Orion Bar 3D Model")
    print("  Based on PDRs4All JWST observations")
    print("="*60)
    
    # Physical setup following Habart+ 2024
    box_pc = 0.4   # 0.4 pc across the bar
    box_cm = box_pc * pc_cm
    G0 = 2.6e4     # Habing, from θ¹ Ori C at the IF
    zeta_CR = 1e-16  # Somewhat attenuated by surrounding cloud
    
    print(f"\n  Distance: 414 pc")
    print(f"  Box:      {box_pc} pc ({box_pc*1e3:.0f} mpc)")
    print(f"  G₀:       {G0:.0e} Habing (θ¹ Ori C)")
    print(f"  Grid:     {n_cells}³ = {n_cells**3} cells")
    print(f"  Cell:     {box_pc/n_cells*1e3:.1f} mpc = {box_pc/n_cells*414*206265:.0f} AU")
    
    # Generate Orion Bar density structure
    density, cs = orion_bar_density(n_cells, box_cm)
    print(f"\n  Density: {np.min(density):.0f} – {np.max(density):.0f} cm⁻³")
    print(f"  Mean:    {np.mean(density):.0f} cm⁻³")
    
    # Run solver (illumination from x=0 face)
    solver = PDRSolver3D(
        density, box_cm,
        G0_external=G0,
        zeta_CR_0=zeta_CR,
        nside_rt=1
    )
    
    t0 = time.time()
    solver.run(max_iterations=n_iter, convergence_tol=0.1, verbose=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")
    
    # Generate observations at 414 pc
    print("\nGenerating synthetic JWST observations...")
    obs = generate_observations(solver, distance_pc=414, los_axis=1)
    
    pix_arcsec = obs['pixel_size_arcsec']
    print(f"  Pixel: {pix_arcsec:.1f}″ ({pix_arcsec/0.1:.0f}× JWST NIRCam resolution)")
    print(f"  FOV:   {n_cells * pix_arcsec:.0f}″ × {n_cells * pix_arcsec:.0f}″")
    
    # Save
    outdir = './orion_bar_output'
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, f'orion_bar_{n_cells}cube.npz'))
    plot_observations(obs, solver,
                      savepath=os.path.join(outdir, f'orion_bar_{n_cells}cube.png'))
    
    # Also make a custom Orion Bar figure showing the layered structure
    plot_orion_bar_structure(solver, obs, outdir, n_cells)
    
    return solver, obs


def plot_orion_bar_structure(solver, obs, outdir, n_cells):
    """Custom figure showing the PDR layering across the bar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from prism3d.utils.constants import pc_cm
    
    # Get x-z slice through the middle (y = mid)
    mid = solver.get_slice(axis=1, index=n_cells // 2)
    ext_pc = [0, solver.box_size / pc_cm, 0, solver.box_size / pc_cm]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('PRISM-3D: Orion Bar — Cross-Section through the PDR\n'
                 r'G₀ = 2.6×10⁴, $\theta^1$ Ori C illumination from left',
                 fontsize=14, fontweight='bold')
    
    # Top row: physical structure
    panels_top = [
        ('density', r'$n_H$ [cm⁻³]', 'viridis', True),
        ('T_gas', r'$T_{gas}$ [K]', 'inferno', True),
        ('G0', r'$G_0$ [Habing]', 'magma', True),
        ('f_nano', r'$f_{nano}$ (THEMIS)', 'RdYlGn', False),
        ('A_V', r'$A_V$ [mag]', 'bone_r', False),
    ]
    
    for i, (key, label, cmap, log) in enumerate(panels_top):
        ax = axes[0, i]
        data = mid[key]
        if log and np.any(data > 0):
            im = ax.imshow(np.log10(np.maximum(data, 1e-30)).T, origin='lower',
                          extent=ext_pc, cmap=cmap, aspect='equal')
            cb = plt.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label(f'log₁₀({label})', fontsize=9)
        else:
            im = ax.imshow(data.T, origin='lower', extent=ext_pc, cmap=cmap, aspect='equal')
            cb = plt.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label(label, fontsize=9)
        ax.set_xlabel('x [pc] (→ θ¹ Ori C)', fontsize=9)
        ax.set_ylabel('z [pc]', fontsize=9)
        ax.set_title(label, fontsize=10)
    
    # Bottom row: chemistry  
    panels_bot = [
        ('x_H2', r'$x$(H₂)', 'RdYlBu_r'),
        ('x_Cp', r'$x$(C⁺)', 'YlOrRd'),
        ('x_CO', r'$x$(CO)', 'YlGn'),
        ('x_e', r'$x$(e⁻)', 'PuBu'),
        ('Gamma_PE', r'$\Gamma_{PE}$', 'hot'),
    ]
    
    for i, (key, label, cmap) in enumerate(panels_bot):
        ax = axes[1, i]
        data = mid[key]
        if np.any(data > 0):
            vmin = np.percentile(data[data > 0], 2) if np.any(data > 0) else 1e-30
            from matplotlib.colors import LogNorm
            im = ax.imshow(data.T, origin='lower', extent=ext_pc, cmap=cmap,
                          norm=LogNorm(vmin=max(vmin, 1e-30)), aspect='equal')
        else:
            im = ax.imshow(data.T + 1e-30, origin='lower', extent=ext_pc, cmap=cmap,
                          aspect='equal')
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(label, fontsize=9)
        ax.set_xlabel('x [pc] (→ θ¹ Ori C)', fontsize=9)
        ax.set_ylabel('z [pc]', fontsize=9)
        ax.set_title(label, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, f'orion_bar_structure_{n_cells}cube.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--iter', type=int, default=5)
    args = p.parse_args()
    run_orion_bar(n_cells=args.n, n_iter=args.iter)
