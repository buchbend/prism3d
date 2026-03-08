"""
3D PDR Model: Turbulent molecular cloud illuminated by FUV radiation.

This is the flagship demonstration of PRISM-3D's 3D capability.
Sets up a turbulent density field and runs the full 3D PDR calculation.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_3d_pdr(n_cells=8, n_iter=4):
    """
    Run a 3D PDR model of a turbulent cloud.
    
    Parameters
    ----------
    n_cells : int
        Cells per dimension. Total cells = n_cells³.
        Use 8 for quick test, 16-32 for science.
    """
    from prism3d.core.density_fields import fractal_turbulent, density_field_stats
    from prism3d.core.solver_3d import PDRSolver3D
    from prism3d.utils.constants import pc_cm, AV_per_NH, NH_per_AV
    
    print("="*60)
    print("  PRISM-3D: 3D Turbulent Cloud PDR Model")
    print("="*60)
    
    # Physical setup: 2 pc box, mean n ~ 300 cm⁻³, Mach 5 turbulence
    box_size = 2.0 * pc_cm  # 2 pc
    n_mean = 300.0           # cm⁻³
    G0 = 10.0                # Habing (moderate FUV, like near OB association)
    zeta_CR = 2e-16          # s⁻¹
    
    print(f"\nPhysical setup:")
    print(f"  Box:     {box_size/pc_cm:.1f} pc")
    print(f"  <n_H>:   {n_mean:.0f} cm⁻³")
    print(f"  G₀:      {G0:.0f} Habing (isotropic)")
    print(f"  ζ_CR:    {zeta_CR:.1e} s⁻¹")
    print(f"  Grid:    {n_cells}³ = {n_cells**3} cells")
    print(f"  Cell:    {box_size/n_cells/pc_cm:.4f} pc")
    
    # Generate turbulent density field
    print(f"\nGenerating turbulent density field...")
    density, cell_size = fractal_turbulent(
        n_cells, box_size, n_mean=n_mean,
        sigma_ln=1.5,  # Mach ~ 5
        spectral_index=-3.7,
        seed=42
    )
    density_field_stats(density)
    
    # Create 3D solver
    print(f"\nInitializing 3D solver...")
    solver = PDRSolver3D(
        density, box_size,
        G0_external=G0,
        zeta_CR_0=zeta_CR,
        nside_rt=1,  # 12 rays (fast)
        fixed_T=None  # Solve thermal balance
    )
    
    # Run
    print(f"\nRunning 3D PDR model...")
    t0 = time.time()
    converged = solver.run(max_iterations=n_iter, convergence_tol=0.1, verbose=True)
    total_time = time.time() - t0
    
    print(f"\nTotal wall time: {total_time:.1f}s")
    print(f"Time per cell per iteration: {total_time/n_cells**3/n_iter*1000:.1f} ms")
    
    # Generate diagnostic plots
    plot_3d_results(solver, n_cells)
    
    return solver


def plot_3d_results(solver, n_cells):
    """Generate diagnostic slice plots of the 3D model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from prism3d.utils.constants import pc_cm
    
    # Get midplane slice
    mid = solver.get_slice(axis=2, index=n_cells // 2)
    
    extent_pc = solver.box_size / pc_cm
    ext = [0, extent_pc, 0, extent_pc]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'PRISM-3D: 3D Turbulent Cloud PDR ({n_cells}³ cells, z-midplane)',
                 fontsize=14, fontweight='bold')
    
    # Row 1: Physical conditions
    plots = [
        ('density', 'n_H [cm⁻³]', True),
        ('T_gas', 'T_gas [K]', True),
        ('G0', 'G₀ [Habing]', True),
        ('A_V', 'A_V [mag]', False),
    ]
    
    for i, (key, label, use_log) in enumerate(plots):
        ax = axes[0, i]
        data = mid[key]
        if use_log and np.any(data > 0):
            im = ax.imshow(np.log10(np.maximum(data, 1e-30)).T, origin='lower',
                          extent=ext, cmap='viridis')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f'log₁₀({label})')
        else:
            im = ax.imshow(data.T, origin='lower', extent=ext, cmap='viridis')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(label)
        ax.set_xlabel('x [pc]')
        ax.set_ylabel('y [pc]')
        ax.set_title(label)
    
    # Row 2: Chemical abundances
    chem_plots = [
        ('x_H2', 'x(H₂)', 'RdYlBu_r'),
        ('x_Cp', 'x(C⁺)', 'YlOrRd'),
        ('x_CO', 'x(CO)', 'YlGn'),
        ('x_e', 'x(e⁻)', 'PuBu'),
    ]
    
    for i, (key, label, cmap) in enumerate(chem_plots):
        ax = axes[1, i]
        data = mid[key]
        vmin = max(np.min(data[data > 0]), 1e-15) if np.any(data > 0) else 1e-15
        im = ax.imshow(np.log10(np.maximum(data, vmin)).T, origin='lower',
                      extent=ext, cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'log₁₀({label})')
        ax.set_xlabel('x [pc]')
        ax.set_ylabel('y [pc]')
        ax.set_title(label)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/home/claude/prism3d_3d_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: prism3d_3d_result.png")


if __name__ == '__main__':
    solver = run_3d_pdr(n_cells=8, n_iter=3)
