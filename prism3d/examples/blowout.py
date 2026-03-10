"""
Blowout Model for PRISM-3D — Young O Star Breaking Out of Its Natal Cloud.

A young O star (O5V–O7V) embedded in its natal molecular cloud has
ionised and evacuated a cavity.  The cavity is asymmetric: the cloud
is thinner along one direction and the star is about to blow out,
creating a champagne flow.

Physical setup:
  - O5V star: L_bol ~ 2×10⁵ L_sun, L_FUV ~ 6×10⁴ L_sun
  - Natal cloud: n ~ 3000 cm⁻³, box ~ 3 pc
  - Cavity: R ~ 0.5 pc, n ~ 10 cm⁻³ (hot ionised gas)
  - Swept-up shell: n ~ 3×10⁴ cm⁻³, thinned on blowout side
  - Pillars of dense gas protruding into the cavity

The FUV field is computed via point-source RT: each cell sees
G₀ = L_FUV / (4π r²) attenuated by the column from star to cell.

References:
  Tenorio-Tagle 1979, A&A 71, 59 (champagne model)
  Arthur & Hoare 2006, ApJS 165, 283 (3D simulations)
  Deharveng+ 2010, A&A 523, A6 (triggered star formation)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism3d.utils.constants import pc_cm, Lsun_erg


# ── Stellar parameters ───────────────────────────────────────────

# O5V star (Martins+ 2005)
L_BOL = 2e5 * Lsun_erg           # Bolometric luminosity [erg/s]
L_FUV = 0.3 * L_BOL              # FUV (6–13.6 eV): ~30% of L_bol for O stars
T_EFF = 41000                     # Effective temperature [K]

# Cloud parameters
BOX_PC = 3.0                      # Box size [pc]
N_CLOUD = 3000                    # Ambient cloud density [cm⁻³]
DISTANCE_PC = 1500                # Typical galactic HII region distance


def run_blowout(n_cells=32, n_iter=10, dust_steps=5, accelerator=None):
    """
    Run a 3D blowout model: O star breaking out of its natal cloud.

    Parameters
    ----------
    n_cells : int
        Cells per dimension.
    n_iter : int
        Max inner iterations per dust step.
    dust_steps : int
        Outer dust evolution steps.
    accelerator : str, optional
        Path to ML accelerator .pkl file.
    """
    from prism3d.density_fields import embedded_star_cloud
    from prism3d.solver_3d import PDRSolver3D
    from prism3d.observations.jwst_pipeline import generate_observations, plot_observations

    print("=" * 60)
    print("  PRISM-3D: Blowout — O Star Breaking Out of Natal Cloud")
    print("=" * 60)

    box_cm = BOX_PC * pc_cm

    # Generate density field with cavity
    density, cs, star_pos = embedded_star_cloud(
        n_cells, box_cm,
        n_cloud=N_CLOUD,
        n_cavity=10,
        R_cavity_frac=0.15,
        n_shell_factor=10,
        shell_width_frac=0.02,
        sigma_ln=1.2,
        star_offset=(0.0, 0.0, -0.05),  # Slightly off-centre toward blowout
        blowout_axis=2,                   # Blowing out along +z
        blowout_thin=0.5,                 # Shell 50% thinner on blowout side
        seed=42,
    )

    R_cav_pc = 0.15 * BOX_PC
    star_r_pc = np.linalg.norm(star_pos - 0.5 * box_cm) / pc_cm

    # G₀ at the cavity wall (for diagnostics)
    G0_wall = L_FUV / (4 * np.pi * (R_cav_pc * pc_cm)**2 * 1.6e-3)

    print(f"\n  Star: O5V, L_FUV = {L_FUV/Lsun_erg:.0e} L_sun")
    print(f"  Cloud: n = {N_CLOUD} cm⁻³, box = {BOX_PC} pc")
    print(f"  Cavity: R ~ {R_cav_pc:.2f} pc")
    print(f"  G₀ at cavity wall: {G0_wall:.0f} Habing")
    print(f"  Grid: {n_cells}³ = {n_cells**3:,} cells")
    print(f"  Cell: {BOX_PC/n_cells*1e3:.1f} mpc")
    print(f"  Density: {np.min(density):.0f} – {np.max(density):.0f} cm⁻³ "
          f"(mean {np.mean(density):.0f})")

    # Create solver with point-source RT
    star = {'position': star_pos, 'L_FUV': L_FUV}
    solver = PDRSolver3D(
        density, box_cm,
        G0_external=1.0,      # Weak background ISRF
        zeta_CR_0=2e-16,
        nside_rt=1,
        star=star,
    )

    if accelerator:
        from prism3d.chemistry.accelerator import ChemistryAccelerator
        solver.accelerator = ChemistryAccelerator.load(accelerator)
        print(f"  Accelerator: {accelerator}")

    print(f"\n  Running solver (point-source RT)...")
    t0 = time.time()
    solver.run(max_iterations=n_iter, convergence_tol=0.1,
               dust_steps=dust_steps, verbose=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")

    # Generate observations
    print("\nGenerating synthetic observations...")
    obs = generate_observations(solver, distance_pc=DISTANCE_PC, los_axis=2)

    pix_arcsec = obs['pixel_size_arcsec']
    print(f"  Pixel: {pix_arcsec:.2f}″")
    print(f"  FOV:   {n_cells * pix_arcsec:.0f}″ × {n_cells * pix_arcsec:.0f}″")

    # Save
    outdir = './blowout_output'
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, f'blowout_{n_cells}cube.npz'))
    plot_observations(obs, solver,
                      savepath=os.path.join(outdir, f'blowout_{n_cells}cube.png'))

    # Custom blowout figure
    plot_blowout_structure(solver, outdir, n_cells, star_pos)

    return solver, obs


def plot_blowout_structure(solver, outdir, n_cells, star_pos):
    """Multi-panel figure showing the cavity and PDR shell structure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # x-z slice through the star (y = star_y)
    js = int(star_pos[1] / solver.cell_size)
    js = min(max(js, 0), n_cells - 1)
    mid = solver.get_slice(axis=1, index=js)
    ext_pc = [0, solver.box_size / pc_cm, 0, solver.box_size / pc_cm]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('PRISM-3D: O Star Blowout — Cross-Section through Cavity\n'
                 f'L_FUV = {L_FUV/Lsun_erg:.0e} L☉, '
                 f'n_cloud = {N_CLOUD} cm⁻³, '
                 f'box = {BOX_PC} pc',
                 fontsize=14, fontweight='bold')

    # Star marker position in pc
    sx_pc = star_pos[0] / pc_cm
    sz_pc = star_pos[2] / pc_cm

    def add_star(ax):
        ax.plot(sx_pc, sz_pc, '*', color='yellow', markersize=15,
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)

    panels = [
        ('density',  r'$n_H$ [cm⁻³]',  'viridis',  True),
        ('T_gas',    r'$T_{gas}$ [K]',  'inferno',  True),
        ('G0',       r'$G_0$ [Habing]', 'magma',    True),
        ('A_V',      r'$A_V$ [mag]',    'bone_r',   False),
        ('x_H2',    r'$x$(H₂)',         'RdYlBu_r', True),
        ('x_Cp',    r'$x$(C⁺)',         'YlOrRd',   True),
        ('x_CO',    r'$x$(CO)',         'YlGn',     True),
        ('f_nano',  r'$f_{nano}$',      'RdYlGn',   False),
    ]

    for idx, (key, label, cmap, log) in enumerate(panels):
        ax = axes[idx // 4, idx % 4]
        data = mid[key]
        if log and np.any(data > 0):
            vmin = np.percentile(data[data > 0], 2)
            im = ax.imshow(data.T, origin='lower', extent=ext_pc, cmap=cmap,
                          norm=LogNorm(vmin=max(vmin, 1e-30)), aspect='equal')
        else:
            im = ax.imshow(data.T, origin='lower', extent=ext_pc, cmap=cmap,
                          aspect='equal')
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(label, fontsize=9)
        ax.set_xlabel('x [pc]', fontsize=9)
        ax.set_ylabel('z [pc]', fontsize=9)
        ax.set_title(label, fontsize=10)
        add_star(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, f'blowout_structure_{n_cells}cube.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='PRISM-3D: O star blowout model')
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--iter', type=int, default=10)
    p.add_argument('--dust-steps', type=int, default=5)
    p.add_argument('--accelerator', type=str, default=None)
    args = p.parse_args()
    run_blowout(n_cells=args.n, n_iter=args.iter,
                dust_steps=args.dust_steps, accelerator=args.accelerator)
