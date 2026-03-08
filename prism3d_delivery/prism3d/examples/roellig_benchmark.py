"""
Röllig+ (2007) PDR Benchmark Suite for PRISM-3D.

Implements the 8 standard benchmark models from:
  "A photon dominated region code comparison study"
  Röllig, Abel, Bell, Bensch, Black, et al.
  A&A 467, 187-206 (2007)

Benchmark models:
  F1-F4: Fixed (isothermal) temperature, tests chemistry + RT only
  V1-V4: Variable temperature, tests full thermal balance

  Model   n [cm⁻³]   χ [Draine]   T_gas [K]   Notes
  ─────   ────────   ──────────   ─────────   ─────
  F1      10³        10           50          Low n, low χ
  F2      10³        10⁵          50          Low n, high χ
  F3      10⁵·⁵      10           50          High n, low χ
  F4      10⁵·⁵      10⁵          50          High n, high χ
  V1      10³        10           solved      Low n, low χ
  V2      10³        10⁵          solved      Low n, high χ
  V3      10⁵·⁵      10           solved      High n, low χ
  V4      10⁵·⁵      10⁵          solved      High n, high χ

Standard benchmark conventions (Röllig+ 2007, Sect. 4):
  - χ is in Draine (1978) field units: χ=1 → F_FUV = 2.7e-3 erg/cm²/s
  - G₀ (Habing) = χ / 1.71 (conversion factor)
  - Plane-parallel semi-infinite slab, illuminated from one side
  - Constant density (no density gradient)
  - Total depth: A_V = 10 mag
  - Cosmic ray ionization rate: ζ = 5e-17 s⁻¹
  - Solar elemental abundances (see below)
  - Standardized chemical network: H, He, C, O, S, Si, Fe + electrons
  - Results plotted vs A_V,eff (effective visual extinction from surface)

Standardized abundances (gas-phase, relative to n_H):
  He: 0.1,  C: 1.4e-4,  O: 3.0e-4,  S: 2.8e-5,
  Si: 1.7e-6, Fe: 1.7e-7, N: 7.5e-5
  (Note: these are the BENCHMARK abundances, not solar or depleted)

Key comparables:
  - T(A_V) profiles (V models)
  - n(H), n(H2), n(C+), n(C), n(CO) vs A_V
  - Surface [CII] 158 μm, [OI] 63 μm intensities
  - CO column densities
"""

import numpy as np
import sys
import os
import time

# Ensure our package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Röllig benchmark abundances
# ============================================================
ROELLIG_ABUNDANCES = {
    'He': 0.1,
    'C':  1.4e-4,
    'O':  3.0e-4,
    'N':  7.5e-5,
    'S':  2.8e-5,
    'Si': 1.7e-6,
    'Fe': 1.7e-7,
}

# Draine → Habing conversion
CHI_TO_G0 = 1.0 / 1.71   # G0 = chi / 1.71

# Benchmark CR ionization rate
ZETA_CR_BENCHMARK = 5e-17  # s⁻¹

# ============================================================
# Benchmark model definitions
# ============================================================
BENCHMARK_MODELS = {
    'F1': {'n_H': 1e3,      'chi': 10,    'T_fixed': 50.0,  'variable_T': False},
    'F2': {'n_H': 1e3,      'chi': 1e5,   'T_fixed': 50.0,  'variable_T': False},
    'F3': {'n_H': 10**5.5,  'chi': 10,    'T_fixed': 50.0,  'variable_T': False},
    'F4': {'n_H': 10**5.5,  'chi': 1e5,   'T_fixed': 50.0,  'variable_T': False},
    'V1': {'n_H': 1e3,      'chi': 10,    'T_fixed': None,   'variable_T': True},
    'V2': {'n_H': 1e3,      'chi': 1e5,   'T_fixed': None,   'variable_T': True},
    'V3': {'n_H': 10**5.5,  'chi': 10,    'T_fixed': None,   'variable_T': True},
    'V4': {'n_H': 10**5.5,  'chi': 1e5,   'T_fixed': None,   'variable_T': True},
}


def setup_benchmark_model(model_name, n_cells=32, use_3d_rt=False, nside_rt=2):
    """
    Set up a Röllig benchmark model.

    Parameters
    ----------
    model_name : str
        One of: F1, F2, F3, F4, V1, V2, V3, V4
    n_cells : int
        Number of cells along the slab depth.
    use_3d_rt : bool
        If True, use full 3D HEALPix ray tracing.
        If False, use fast 1D column density integration.
    nside_rt : int
        HEALPix resolution (only used if use_3d_rt=True).

    Returns
    -------
    solver : PDRSolver
        Configured solver ready to run.
    model_params : dict
        Model parameters for reference.
    """
    from prism3d.core.grid import OctreeGrid, OctreeNode, CellData
    from prism3d.core.solver import PDRSolver
    from .benchmark_config import (
        apply_benchmark_overrides, build_benchmark_network,
        AV_PER_NH_BENCHMARK, NH_PER_AV_BENCHMARK
    )
    
    # Apply benchmark-specific constants FIRST
    apply_benchmark_overrides()
    AV_per_NH = AV_PER_NH_BENCHMARK
    NH_per_AV = NH_PER_AV_BENCHMARK

    if model_name not in BENCHMARK_MODELS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: {list(BENCHMARK_MODELS.keys())}")

    params = BENCHMARK_MODELS[model_name].copy()
    n_H = params['n_H']
    chi = params['chi']
    G0 = chi * CHI_TO_G0
    T_fixed = params['T_fixed']
    variable_T = params['variable_T']

    # Slab depth: A_V = 10 mag
    AV_max = 10.0
    N_H_total = AV_max * NH_per_AV
    depth = N_H_total / n_H  # cm

    print(f"═══════════════════════════════════════════════════")
    print(f"  PRISM-3D Röllig Benchmark: Model {model_name}")
    print(f"═══════════════════════════════════════════════════")
    print(f"  n_H  = {n_H:.2e} cm⁻³")
    print(f"  χ    = {chi:.0e} (Draine) → G₀ = {G0:.1f} (Habing)")
    print(f"  ζ_CR = {ZETA_CR_BENCHMARK:.1e} s⁻¹")
    print(f"  A_V  = 0 → {AV_max} mag")
    print(f"  Depth = {depth:.2e} cm = {depth/3.086e18:.3f} pc")
    if T_fixed is not None:
        print(f"  T_gas = {T_fixed} K (FIXED)")
    else:
        print(f"  T_gas = solved self-consistently")
    print(f"  Cells = {n_cells}")
    print(f"  3D RT = {use_3d_rt} (nside={nside_rt})")
    print(f"═══════════════════════════════════════════════════")

    # ── Build 1D grid ──
    cell_size = depth / n_cells
    grid = OctreeGrid(box_size=depth, n_base=1, max_level=0)
    grid.root_nodes = []
    half_depth = depth / 2.0

    for i in range(n_cells):
        center = np.array([-half_depth + (i + 0.5) * cell_size, 0.0, 0.0])
        node = OctreeNode(center, cell_size, level=0)
        node.data = CellData()
        node.data.n_H = n_H
        grid.root_nodes.append(node)

    grid.n_base = 1
    grid.base_cell_size = depth
    grid.invalidate_cache()

    # ── Initialize with benchmark abundances ──
    x_C = ROELLIG_ABUNDANCES['C']
    x_O = ROELLIG_ABUNDANCES['O']

    for i, node in enumerate(grid.root_nodes):
        x = (i + 0.5) * cell_size
        N_H_col = n_H * x
        node.data.A_V = N_H_col * AV_per_NH
        node.data.G0 = G0 * np.exp(-1.8 * node.data.A_V)
        node.data.N_H_total = N_H_col
        A = node.data.A_V

        # Initial conditions based on depth
        if A < 0.5:
            node.data.x_HI = 1.0; node.data.x_H2 = 1e-6
            node.data.x_Cp = x_C; node.data.x_C = 1e-10; node.data.x_CO = 1e-15
            node.data.x_O = x_O
            node.data.T_gas = T_fixed if T_fixed else 500.0
        elif A < 2.0:
            f = (A - 0.5) / 1.5
            node.data.x_HI = 1.0 - f; node.data.x_H2 = f / 2.0
            node.data.x_Cp = x_C * (1 - 0.5 * f)
            node.data.x_C = x_C * 0.3 * f
            node.data.x_CO = x_C * 0.2 * f
            node.data.x_O = x_O - node.data.x_CO
            node.data.T_gas = T_fixed if T_fixed else 200.0
        elif A < 5.0:
            fCO = (A - 2.0) / 3.0
            node.data.x_HI = 0.01; node.data.x_H2 = 0.495
            node.data.x_Cp = max(x_C * (1 - 3 * fCO), 1e-9)
            node.data.x_C = x_C * (1 - fCO) * 0.3
            node.data.x_CO = x_C * fCO * 0.9
            node.data.x_O = x_O - node.data.x_CO
            node.data.T_gas = T_fixed if T_fixed else 60.0
        else:
            node.data.x_HI = 0.005; node.data.x_H2 = 0.4975
            node.data.x_Cp = 1e-9
            node.data.x_C = x_C * 0.05
            node.data.x_CO = x_C * 0.95
            node.data.x_O = x_O - node.data.x_CO
            node.data.T_gas = T_fixed if T_fixed else 15.0

        # No metals in benchmark — set to zero
        node.data.x_Sp = 0.0; node.data.x_S = 0.0
        node.data.x_Sip = 0.0; node.data.x_Si = 0.0
        node.data.x_Fep = 0.0; node.data.x_Fe = 0.0
        # Electrons from C+ only (plus CR floor)
        node.data.x_e = node.data.x_Cp + 1e-8

    # ── Create solver with BENCHMARK network ──
    benchmark_net = build_benchmark_network()
    
    solver = PDRSolver(
        grid,
        G0_external=G0,
        zeta_CR_0=ZETA_CR_BENCHMARK,
        use_1d_rt=(not use_3d_rt),
        nside_rt=nside_rt,
        cr_model='M'
    )
    
    # Inject benchmark chemistry into the solver
    from prism3d.chemistry.solver import ChemistrySolver
    solver.chem_solver = ChemistrySolver(network=benchmark_net)

    # ── Override RT with simple 1D for the 1D benchmark ──
    if not use_3d_rt:
        solver.rt = _Simple1DRT(grid, G0)

    # ── For fixed-T models, disable thermal solver ──
    if T_fixed is not None:
        solver._fixed_T = T_fixed
    else:
        solver._fixed_T = None

    return solver, params


class _Simple1DRT:
    """Fast 1D column density integrator for benchmarking."""
    def __init__(self, grid, G0):
        self.grid = grid
        self.G0_external = G0

    def compute_1d_column_densities(self, axis=0, direction=1):
        from prism3d.utils.constants import AV_per_NH  # Will be overridden to benchmark value
        leaves = sorted(self.grid.get_leaves(), key=lambda l: l.center[0])
        N_H = N_H2 = N_CO = 0.0
        for leaf in leaves:
            ds = leaf.size
            leaf.data.N_H_total = N_H + 0.5 * leaf.data.n_H * ds
            leaf.data.N_H2 = N_H2 + 0.5 * leaf.data.n_H * leaf.data.x_H2 * 2.0 * ds
            leaf.data.N_CO = N_CO + 0.5 * leaf.data.n_H * leaf.data.x_CO * ds
            leaf.data.A_V = leaf.data.N_H_total * AV_per_NH
            leaf.data.G0 = self.G0_external * np.exp(-1.8 * leaf.data.A_V)
            N_H += leaf.data.n_H * ds
            N_H2 += leaf.data.n_H * leaf.data.x_H2 * 2.0 * ds
            N_CO += leaf.data.n_H * leaf.data.x_CO * ds

    def compute_fuv_field(self):
        self.compute_1d_column_densities()


def run_benchmark(model_name, n_cells=32, max_iter=10, use_3d_rt=False,
                  nside_rt=2, convergence_tol=0.05):
    """
    Run a single Röllig benchmark model and return results.

    Returns
    -------
    profile : dict
        Arrays of A_V, T_gas, T_dust, n_H, n(H), n(H2),
        n(C+), n(C), n(CO), n(O), n(e) vs depth.
    """
    solver, params = setup_benchmark_model(
        model_name, n_cells=n_cells,
        use_3d_rt=use_3d_rt, nside_rt=nside_rt
    )

    t0 = time.time()
    converged = solver.run(
        max_iterations=max_iter,
        convergence_tol=convergence_tol,
        verbose=True
    )
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Extract profile
    leaves = sorted(solver.grid.get_leaves(), key=lambda l: l.center[0])
    n_H = params['n_H']

    profile = {
        'model': model_name,
        'n_H': n_H,
        'chi': params['chi'],
        'converged': converged,
        'wall_time': elapsed,
        'A_V': np.array([l.data.A_V for l in leaves]),
        'T_gas': np.array([l.data.T_gas for l in leaves]),
        'T_dust': np.array([l.data.T_dust for l in leaves]),
        'G0': np.array([l.data.G0 for l in leaves]),
        # Densities (not fractions) for Röllig comparison
        'n_HI': np.array([l.data.x_HI * n_H for l in leaves]),
        'n_H2': np.array([l.data.x_H2 * n_H for l in leaves]),
        'n_Cp': np.array([l.data.x_Cp * n_H for l in leaves]),
        'n_C':  np.array([l.data.x_C * n_H for l in leaves]),
        'n_CO': np.array([l.data.x_CO * n_H for l in leaves]),
        'n_O':  np.array([l.data.x_O * n_H for l in leaves]),
        'n_e':  np.array([l.data.x_e * n_H for l in leaves]),
        # Fractions too
        'x_HI': np.array([l.data.x_HI for l in leaves]),
        'x_H2': np.array([l.data.x_H2 for l in leaves]),
        'x_Cp': np.array([l.data.x_Cp for l in leaves]),
        'x_C':  np.array([l.data.x_C for l in leaves]),
        'x_CO': np.array([l.data.x_CO for l in leaves]),
        'x_e':  np.array([l.data.x_e for l in leaves]),
    }

    return profile


def plot_benchmark_comparison(profiles, save_path=None):
    """
    Plot benchmark results in the Röllig+ (2007) format.

    Generates a 2×3 panel figure for each model:
    - n(H), n(H2) vs A_V
    - n(C+), n(C), n(CO) vs A_V
    - T(A_V) for V models
    - n(e) vs A_V
    - G0 vs A_V
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_models = len(profiles)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {'F1': '#1f77b4', 'F2': '#ff7f0e', 'F3': '#2ca02c', 'F4': '#d62728',
              'V1': '#1f77b4', 'V2': '#ff7f0e', 'V3': '#2ca02c', 'V4': '#d62728'}
    styles = {'F1': '-', 'F2': '-', 'F3': '-', 'F4': '-',
              'V1': '--', 'V2': '--', 'V3': '--', 'V4': '--'}

    for p in profiles:
        name = p['model']
        A = p['A_V']
        c = colors.get(name, 'black')
        ls = styles.get(name, '-')
        lbl = f"{name} (n={p['n_H']:.0e}, χ={p['chi']:.0e})"

        # Panel 1: H/H2
        ax = axes[0, 0]
        ax.semilogy(A, p['n_HI'], color=c, ls=ls, lw=1.5, label=f'{name} H')
        ax.semilogy(A, p['n_H2'], color=c, ls=':', lw=1.5)

        # Panel 2: C+/C/CO
        ax = axes[0, 1]
        ax.semilogy(A, p['n_Cp'], color=c, ls=ls, lw=1.5, label=f'{name} C+')
        ax.semilogy(A, p['n_C'], color=c, ls='--', lw=1.0)
        ax.semilogy(A, p['n_CO'], color=c, ls=':', lw=1.5)

        # Panel 3: Temperature
        ax = axes[0, 2]
        ax.semilogy(A, p['T_gas'], color=c, ls=ls, lw=1.5, label=name)

        # Panel 4: Electron density
        ax = axes[1, 0]
        ax.semilogy(A, p['n_e'], color=c, ls=ls, lw=1.5, label=name)

        # Panel 5: FUV field
        ax = axes[1, 1]
        ax.semilogy(A, np.maximum(p['G0'], 1e-3), color=c, ls=ls, lw=1.5, label=name)

    # Labels
    axes[0, 0].set_xlabel('A_V [mag]'); axes[0, 0].set_ylabel('n [cm⁻³]')
    axes[0, 0].set_title('H / H₂ densities'); axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_xlabel('A_V [mag]'); axes[0, 1].set_ylabel('n [cm⁻³]')
    axes[0, 1].set_title('C⁺ (solid) / C (dash) / CO (dot)'); axes[0, 1].legend(fontsize=7)
    axes[0, 2].set_xlabel('A_V [mag]'); axes[0, 2].set_ylabel('T [K]')
    axes[0, 2].set_title('Gas Temperature'); axes[0, 2].legend(fontsize=7)
    axes[1, 0].set_xlabel('A_V [mag]'); axes[1, 0].set_ylabel('n_e [cm⁻³]')
    axes[1, 0].set_title('Electron density'); axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_xlabel('A_V [mag]'); axes[1, 1].set_ylabel('G₀ [Habing]')
    axes[1, 1].set_title('FUV Field'); axes[1, 1].legend(fontsize=7)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    rows = []
    for p in profiles:
        rows.append([
            p['model'],
            f"{p['n_H']:.0e}",
            f"{p['chi']:.0e}",
            'Y' if p['converged'] else 'N',
            f"{p['wall_time']:.0f}s",
        ])
    table = ax.table(cellText=rows,
                     colLabels=['Model', 'n_H', 'χ', 'Conv', 'Time'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Benchmark Summary')

    fig.suptitle('PRISM-3D: Röllig+ (2007) PDR Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.close()


def run_all_benchmarks(models=None, n_cells=24, max_iter=8, save_dir=None):
    """
    Run all (or selected) Röllig benchmark models.

    Parameters
    ----------
    models : list, optional
        List of model names. Default: all 8.
    """
    if models is None:
        models = ['F1', 'F2', 'F3', 'F4', 'V1', 'V2', 'V3', 'V4']

    profiles = []
    for name in models:
        print(f"\n{'='*60}")
        print(f"  Running benchmark model {name}")
        print(f"{'='*60}\n")
        try:
            p = run_benchmark(name, n_cells=n_cells, max_iter=max_iter)
            profiles.append(p)
        except Exception as e:
            print(f"  ERROR in model {name}: {e}")

    if save_dir:
        plot_benchmark_comparison(
            profiles,
            save_path=os.path.join(save_dir, 'roellig_benchmark.png')
        )

    return profiles


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Röllig PDR benchmark')
    parser.add_argument('--models', nargs='+', default=['V1', 'V2'],
                       help='Models to run (e.g., F1 F2 V1 V2)')
    parser.add_argument('--cells', type=int, default=24)
    parser.add_argument('--iter', type=int, default=8)
    parser.add_argument('--output', default='/home/claude')
    args = parser.parse_args()

    profiles = run_all_benchmarks(
        models=args.models,
        n_cells=args.cells,
        max_iter=args.iter,
        save_dir=args.output
    )
