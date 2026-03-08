"""
Example: 1D PDR Slab Model — Orion Bar Configuration

This example sets up a standard 1D plane-parallel PDR model
similar to the Orion Bar, for benchmarking against the Meudon
PDR code and the Röllig+ (2007) benchmark results.

Parameters:
- n_H = 1e4 cm⁻³ (constant density)
- G0 = 1e4 (Habing units, strong FUV field from θ¹ Ori C)
- ζ_CR = 2e-16 s⁻¹
- Total depth: A_V = 10 mag
- Solar metallicity

This produces the classic PDR stratification:
- Surface: H, C⁺, high T (~500-1000 K)
- A_V ~ 1-2: H→H₂ transition, [CII] emission peak
- A_V ~ 3-4: C⁺→C→CO transition, [CI] emission peak
- Deep: H₂, CO, low T (~20-50 K)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prism3d.core.grid import OctreeGrid
from prism3d.core.solver import PDRSolver
from prism3d.utils.constants import pc_cm, AV_per_NH, NH_per_AV


def setup_orion_bar(n_cells_1d=32):
    """
    Set up a 1D slab model for the Orion Bar PDR.

    Uses a 3D grid but with uniform conditions in y and z,
    and variation only along x (illumination direction).
    """
    # Physical parameters
    n_H = 1e4             # cm⁻³
    G0 = 1e4              # Habing units
    zeta_CR = 2e-16       # s⁻¹

    # Box size: depth to reach A_V = 10
    # N_H = A_V / AV_per_NH
    N_H_total = 10.0 / AV_per_NH  # cm⁻²
    depth = N_H_total / n_H       # cm
    box_size = depth

    print(f"Orion Bar PDR model:")
    print(f"  n_H = {n_H:.0e} cm⁻³")
    print(f"  G0 = {G0:.0e}")
    print(f"  ζ_CR = {zeta_CR:.2e} s⁻¹")
    print(f"  Depth = {depth:.2e} cm = {depth/pc_cm:.4f} pc")
    print(f"  A_V,max = 10 mag")
    print(f"  N_cells = {n_cells_1d}")

    # Create grid (use n_base=1 in y,z for effectively 1D)
    # For true 1D, we use a thin slab (n_base x 1 x 1)
    grid = OctreeGrid(box_size, n_base=n_cells_1d, max_level=3)

    # Initialize as 1D slab
    grid.setup_1d_slab(n_H=n_H, G0_surface=G0)

    return grid, G0, zeta_CR


def run_model(n_cells=8, max_iter=15):
    """
    Run the Orion Bar model and generate diagnostic plots.

    Parameters
    ----------
    n_cells : int
        Number of cells along the illumination direction.
        Keep small (4-8) for quick testing.
    max_iter : int
        Maximum solver iterations.
    """
    print("=" * 60)
    print("PRISM-3D: Orion Bar 1D Benchmark")
    print("=" * 60)

    # Setup
    grid, G0, zeta_CR = setup_orion_bar(n_cells_1d=n_cells)

    # Create solver with 1D RT (fast)
    solver = PDRSolver(
        grid,
        G0_external=G0,
        zeta_CR_0=zeta_CR,
        use_1d_rt=True,
        cr_model='M'
    )

    # Run
    converged = solver.run(
        max_iterations=max_iter,
        convergence_tol=0.05,
        verbose=True
    )

    # Extract 1D profile
    profile = solver.get_1d_profile(axis=0)

    # Generate plots
    plot_pdr_structure(profile, n_cells)

    return profile, solver


def plot_pdr_structure(profile, n_cells):
    """Generate diagnostic plots of the PDR structure."""

    A_V = profile['A_V']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'PRISM-3D: Orion Bar PDR (n_H=10⁴, G₀=10⁴, {n_cells} cells)',
                 fontsize=14, fontweight='bold')

    # 1. Temperature profile
    ax = axes[0, 0]
    ax.semilogy(A_V, profile['T_gas'], 'r-o', label='T_gas', markersize=4)
    ax.semilogy(A_V, profile['T_dust'], 'b--o', label='T_dust', markersize=4)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Temperature Structure')
    ax.legend()
    ax.set_xlim(0, max(A_V))
    ax.grid(True, alpha=0.3)

    # 2. H/H2 transition
    ax = axes[0, 1]
    ax.semilogy(A_V, profile['x_HI'], 'b-o', label='H', markersize=4)
    ax.semilogy(A_V, 2*profile['x_H2'], 'r-o', label='2×H₂', markersize=4)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('Abundance (relative to n_H)')
    ax.set_title('H / H₂ Transition')
    ax.legend()
    ax.set_ylim(1e-6, 2)
    ax.grid(True, alpha=0.3)

    # 3. C+/C/CO transition
    ax = axes[0, 2]
    ax.semilogy(A_V, profile['x_Cp'], 'b-o', label='C⁺', markersize=4)
    ax.semilogy(A_V, profile['x_C'], 'g-o', label='C', markersize=4)
    ax.semilogy(A_V, profile['x_CO'], 'r-o', label='CO', markersize=4)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('Abundance (relative to n_H)')
    ax.set_title('C⁺ / C / CO Transition')
    ax.legend()
    ax.set_ylim(1e-10, 1e-3)
    ax.grid(True, alpha=0.3)

    # 4. FUV field
    ax = axes[1, 0]
    ax.semilogy(A_V, profile['G0'], 'orange', marker='o', markersize=4)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('G₀ [Habing]')
    ax.set_title('FUV Field Attenuation')
    ax.grid(True, alpha=0.3)

    # 5. Electron fraction and CR rate
    ax = axes[1, 1]
    ax.semilogy(A_V, profile['x_e'], 'purple', marker='o', label='x_e', markersize=4)
    ax2 = ax.twinx()
    ax2.semilogy(A_V, profile['zeta_CR'], 'gray', marker='s', label='ζ_CR', markersize=4, alpha=0.7)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('x_e', color='purple')
    ax2.set_ylabel('ζ_CR [s⁻¹]', color='gray')
    ax.set_title('Ionization & Cosmic Rays')
    ax.grid(True, alpha=0.3)

    # 6. Heating/cooling balance
    ax = axes[1, 2]
    ax.semilogy(A_V, profile['Gamma'], 'r-o', label='Heating', markersize=4)
    ax.semilogy(A_V, profile['Lambda'], 'b--o', label='Cooling', markersize=4)
    ax.set_xlabel('A_V [mag]')
    ax.set_ylabel('Rate [erg/cm³/s]')
    ax.set_title('Heating / Cooling Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/prism3d_orion_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: prism3d_orion_bar.png")


if __name__ == '__main__':
    profile, solver = run_model(n_cells=4, max_iter=10)
