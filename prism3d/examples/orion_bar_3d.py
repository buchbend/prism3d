"""
Orion Bar PDR Model for PRISM-3D.

Sets up a 3D model of the Orion Bar based on the PDRs4All JWST
observations (Berné et al. 2022; Habart et al. 2024; Peeters et al. 2024).

The Orion Bar is the prototypical bright PDR:
  - Distance: 414 pc (Menten+ 2007)
  - Illumination: θ¹ Ori C (O7V), G₀ ≈ 2-7 × 10⁴ Habing
  - Geometry: nearly edge-on ridge, inclined ~4° from edge-on
  - Structure: IF → atomic layer → DF1/DF2/DF3 → molecular zone
  - Density: n_H ~ 10⁴-10⁵ cm⁻³ at dissociation fronts
  - Size: ~0.4 pc × 0.1 pc visible ridge
  - Thermal pressure: P/k ≈ 10⁸ cm⁻³ K at the compressed layer

The model uses a density structure motivated by:
1. An isobaric pressure profile: n(x) = P₀/(k·T(x))
2. Embedded clumps at the dissociation fronts (DF1/DF2/DF3)
3. Turbulent substructure matching the observed filaments
4. Directional illumination from θ¹ Ori C (not isotropic)
5. Nearly edge-on viewing geometry (~4° inclination)

References:
  Berné et al. 2022, PASP 134, 054301 (PDRs4All design)
  Habart et al. 2024, A&A 685, A73 (NIR/MIR imaging)
  Peeters et al. 2024, A&A 685, A74 (NIRSpec spectroscopy)
  Elyajouri et al. 2024, A&A 685, A76 (Dust evolution)
  Goicoechea et al. 2016, Nature 537, 207 (Density structure, P/k~10⁸)
  Andree-Labsch et al. 2017, A&A 598, A2 (3D clumpy Orion Bar model)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Orion Bar physical constants ──────────────────────────────────

DISTANCE_PC = 414           # Menten+ 2007
G0_ORION = 2.6e4            # Habing units, θ¹ Ori C at the IF
ZETA_CR = 1e-16             # Attenuated by surrounding Orion A cloud
P_THERMAL = 1e8             # P/k [cm⁻³ K], Goicoechea+ 2016
INCLINATION_DEG = 4.0       # Viewing angle from edge-on (Hogerheijde+ 1995)

# Source direction: θ¹ Ori C illuminates from the -x side
SOURCE_DIRECTION = np.array([-1.0, 0.0, 0.0])


def orion_bar_density(n_cells, box_size_cm, P_over_k=P_THERMAL):
    """
    Generate an Orion Bar-like density structure using an isobaric
    pressure profile.

    The gas is approximately in pressure equilibrium at the compressed
    PDR layer (Goicoechea+ 2016, Joblin+ 2018).  As the gas temperature
    drops from ~500 K at the UV-exposed surface to ~50 K in the
    molecular zone, the density rises as n = P/(k T).

    A turbulent component and embedded dense clumps at the dissociation
    fronts (DF1/DF2/DF3) are overlaid on this base profile.

    Parameters
    ----------
    n_cells : int
        Cells per dimension.
    box_size_cm : float
        Physical box size [cm].
    P_over_k : float
        Thermal pressure P/k [cm⁻³ K].  Default 10⁸ (Orion Bar).

    Returns
    -------
    density : ndarray (n_cells, n_cells, n_cells)
    cell_size : float
    """
    cs = box_size_cm / n_cells
    x = np.linspace(0, 1, n_cells)  # Normalised position along illumination axis

    # ── Density profile motivated by observations ────────────────
    # The Orion Bar is NOT uniformly isobaric.  The thermal pressure
    # peaks at the compressed DF layer (P/k ~ 10⁸, Goicoechea+ 2016)
    # and is lower in the atomic and deep molecular zones.
    #
    # Observed structure (Habart+ 2024, Goicoechea+ 2016):
    #   x=0:    IF boundary, warm atomic gas,   n ~ 5×10³, T ~ 3000 K
    #   x~0.15: Atomic PDR layer,               n ~ 1×10⁴, T ~ 500 K
    #   x~0.30: Compressed DF (P/k ~ 10⁸),     n ~ 1-2×10⁵, T ~ 500 K
    #   x~0.50: Molecular zone behind DFs,      n ~ 5×10⁴, T ~ 50 K
    #   x=1:    Deep molecular cloud,           n ~ 1×10⁵, T ~ 20-30 K
    #
    # Use a smooth profile with a density peak at the DF.
    n_IF = 5e3        # near ionisation front
    n_DF = 2e5        # compressed dissociation front (P/k ~ 10⁸)
    n_deep = 1e5      # deep molecular zone

    # Two-component profile: ramp up to DF, then settle to n_deep
    # Gaussian bump at x=0.30 for the compressed DF layer
    ramp = n_IF * np.exp(np.log(n_deep / n_IF) * x)
    df_bump = (n_DF - ramp) * np.exp(-0.5 * ((x - 0.30) / 0.08)**2)
    n_profile = ramp + np.maximum(df_bump, 0.0)

    # Make 3D
    density = np.zeros((n_cells, n_cells, n_cells))
    for i in range(n_cells):
        density[i, :, :] = n_profile[i]

    # ── Turbulent fluctuations (moderate, Mach ~ 2) ─────────────
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
    sigma_turb = 0.5  # σ(ln n) ~ 0.5 → Mach ~ 2
    gaussian *= sigma_turb / max(np.std(gaussian), 1e-10)

    density *= np.exp(gaussian - sigma_turb**2 / 2)

    # ── Dense clumps at dissociation fronts (DF1/DF2/DF3) ───────
    # 5 clumps at x ∈ [0.2, 0.4], representing the terraced structure
    # seen in JWST H₂ maps (Habart+ 2024)
    clump_centers = np.array([
        [0.22, 0.30, 0.50],
        [0.28, 0.70, 0.30],
        [0.32, 0.50, 0.70],
        [0.38, 0.20, 0.60],
        [0.30, 0.60, 0.40],
    ]) * box_size_cm

    xx = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    yy = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    zz = np.linspace(0, box_size_cm, n_cells) + 0.5 * cs
    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')

    for ic in range(len(clump_centers)):
        cx, cy, cz = clump_centers[ic]
        r_clump = 0.03 * box_size_cm * (0.8 + 0.4 * rng.rand())
        dist = np.sqrt((XX - cx)**2 + (YY - cy)**2 + (ZZ - cz)**2)
        clump = 2e5 * np.exp(-0.5 * (dist / r_clump)**2)
        density = np.maximum(density, clump)

    density = np.maximum(density, 100.0)  # Floor
    return density, cs


def run_orion_bar(n_cells=16, n_iter=5):
    """
    Run a 3D Orion Bar model with directional illumination.

    Parameters
    ----------
    n_cells : int
        Cells per dimension.
    n_iter : int
        Number of solver iterations.
    """
    from prism3d.solver_3d import PDRSolver3D
    from prism3d.observations.jwst_pipeline import generate_observations, plot_observations
    from prism3d.radiative_transfer.fuv_rt_3d import directional_G0
    from prism3d.utils.constants import pc_cm

    print("=" * 60)
    print("  PRISM-3D: Orion Bar 3D Model")
    print("  Isobaric profile + directional illumination")
    print("  Based on PDRs4All JWST observations")
    print("=" * 60)

    # Physical setup
    box_pc = 0.4
    box_cm = box_pc * pc_cm

    print(f"\n  Distance:     {DISTANCE_PC} pc")
    print(f"  Box:          {box_pc} pc ({box_pc*1e3:.0f} mpc)")
    print(f"  G₀:           {G0_ORION:.0e} Habing (θ¹ Ori C, directional)")
    print(f"  P/k:          {P_THERMAL:.0e} cm⁻³ K (isobaric)")
    print(f"  Inclination:  {INCLINATION_DEG}° from edge-on")
    print(f"  Grid:         {n_cells}³ = {n_cells**3} cells")
    print(f"  Cell:         {box_pc/n_cells*1e3:.1f} mpc = "
          f"{box_pc/n_cells*DISTANCE_PC*206265:.0f} AU")

    # Generate density structure
    density, cs = orion_bar_density(n_cells, box_cm)
    print(f"\n  Density: {np.min(density):.0f} – {np.max(density):.0f} cm⁻³")
    print(f"  Mean:    {np.mean(density):.0f} cm⁻³")

    # Directional G₀: illumination from -x side (θ¹ Ori C)
    nside_rt = 2 if n_cells >= 32 else 1
    G0_rays = directional_G0(G0_ORION, SOURCE_DIRECTION, nside=nside_rt)
    print(f"  RT rays:  {12 * nside_rt**2} (nside={nside_rt})")
    print(f"  G₀ rays:  min={np.min(G0_rays):.0f}, max={np.max(G0_rays):.0f}")

    # Create solver with per-ray G₀
    solver = PDRSolver3D(
        density, box_cm,
        G0_external=G0_rays,
        zeta_CR_0=ZETA_CR,
        nside_rt=nside_rt,
    )

    t0 = time.time()
    solver.run(max_iterations=n_iter, convergence_tol=0.1, verbose=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")

    # Generate observations with inclination
    print("\nGenerating synthetic JWST observations...")
    obs = generate_observations(
        solver, distance_pc=DISTANCE_PC, los_axis=1,
        inclination_deg=INCLINATION_DEG,
    )

    pix_arcsec = obs['pixel_size_arcsec']
    print(f"  Pixel: {pix_arcsec:.1f}″ ({pix_arcsec/0.1:.0f}× JWST NIRCam resolution)")
    print(f"  FOV:   {n_cells * pix_arcsec:.0f}″ × {n_cells * pix_arcsec:.0f}″")

    # Save
    outdir = './orion_bar_output'
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, f'orion_bar_{n_cells}cube.npz'))
    plot_observations(obs, solver,
                      savepath=os.path.join(outdir, f'orion_bar_{n_cells}cube.png'))

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
                 r'G₀ = 2.6×10⁴ (directional), P/k = 10⁸ cm⁻³ K, '
                 r'$\theta^1$ Ori C illumination from left',
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
