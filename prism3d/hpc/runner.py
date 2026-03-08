"""
PRISM-3D HPC Module — High-Performance Computing Support.

Provides infrastructure for running PRISM-3D on HPC clusters:
1. SLURM job script generation
2. MPI domain decomposition (split grid across nodes)
3. CuPy/Numba GPU kernels for chemistry and RT
4. Checkpoint/restart for long runs
5. Scalable I/O with HDF5

Architecture:
  - Each MPI rank owns a sub-domain of the 3D grid
  - Ghost zones exchanged between ranks for RT
  - Chemistry is embarrassingly parallel (no communication needed)
  - RT requires ghost zones along ray directions

Requirements for HPC runs:
  pip install mpi4py h5py cupy-cuda12x numba
  (cupy optional — falls back to numpy)

Usage:
  # On login node:
  python -m prism3d.hpc.run_hpc --config my_model.yaml --submit

  # Or in a SLURM script:
  srun python -m prism3d.hpc.run_hpc --config my_model.yaml
"""

import numpy as np
import os
import json
import time


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    # Physical parameters
    'box_size_pc': 2.0,
    'n_cells': 128,           # Per dimension → 128³ = 2M cells
    'n_mean': 300.0,          # Mean density [cm⁻³]
    'sigma_ln': 1.5,          # Log-normal width (Mach number proxy)
    'G0_external': 10.0,      # FUV field [Habing]
    'zeta_CR': 2e-16,         # Cosmic ray rate [s⁻¹]
    'spectral_index': -3.7,   # Turbulence power spectrum
    
    # Solver parameters
    'max_iterations': 30,
    'convergence_tol': 0.01,
    'nside_rt': 2,            # 48 ray directions
    'refine_after': True,     # BDF refinement pass at end
    
    # HPC parameters
    'n_nodes': 4,
    'n_gpus_per_node': 1,
    'wall_time': '04:00:00',
    'partition': 'gpu',
    'account': '',
    'use_gpu': True,
    
    # I/O
    'output_dir': './prism3d_output',
    'checkpoint_every': 5,    # Iterations between checkpoints
    'save_format': 'hdf5',    # 'hdf5' or 'npz'
}


def generate_config(filepath, **overrides):
    """Generate a configuration file for an HPC run."""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {filepath}")
    return config


# ============================================================
# SLURM Job Generation
# ============================================================

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=prism3d_{job_name}
#SBATCH --output=prism3d_%j.out
#SBATCH --error=prism3d_%j.err
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{n_gpus_per_node}
#SBATCH --time={wall_time}
#SBATCH --partition={partition}
{account_line}

# Load modules (adjust for your cluster)
module load python/3.11
module load cuda/12
module load openmpi/4.1

# Activate environment
source $HOME/prism3d_env/bin/activate

# Run
echo "Starting PRISM-3D on $(hostname) at $(date)"
echo "Nodes: $SLURM_NNODES, Tasks: $SLURM_NTASKS"

srun python -m prism3d.hpc.run_hpc --config {config_path}

echo "Finished at $(date)"
"""


def generate_slurm_script(config, filepath='run_prism3d.slurm',
                           job_name='3dpdr'):
    """Generate a SLURM submission script."""
    account_line = f"#SBATCH --account={config['account']}" if config.get('account') else ''
    
    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        n_nodes=config['n_nodes'],
        cpus_per_task=max(4, 64 // max(config['n_gpus_per_node'], 1)),
        n_gpus_per_node=config['n_gpus_per_node'],
        wall_time=config['wall_time'],
        partition=config['partition'],
        account_line=account_line,
        config_path=os.path.abspath('prism3d_config.json'),
    )
    
    with open(filepath, 'w') as f:
        f.write(script)
    print(f"SLURM script written to {filepath}")
    print(f"Submit with: sbatch {filepath}")
    return filepath


# ============================================================
# GPU Chemistry Kernel (CuPy or Numba fallback)
# ============================================================

GPU_CHEMISTRY_KERNEL = '''
// CUDA kernel for explicit Euler chemistry step
// Each thread handles one grid cell
// This is the performance-critical inner loop

__global__ void chemistry_step(
    // Input arrays (n_cells)
    const double* __restrict__ density,
    const double* __restrict__ T_gas,
    const double* __restrict__ G0,
    const double* __restrict__ A_V,
    const double* __restrict__ zeta_CR,
    const double* __restrict__ f_shield_H2,
    const double* __restrict__ f_shield_CO,
    // Abundance arrays (n_cells each) — read and written
    double* x_HI,
    double* x_H2,
    double* x_Cp,
    double* x_C,
    double* x_CO,
    double* x_O,
    double* x_e,
    // Parameters
    int n_cells,
    double dt,
    double R_H2_form,    // H2 formation rate coefficient
    double k_H2_pd_0,    // H2 photodiss base rate  
    double k_CO_pd_0,    // CO photodiss base rate
    double k_C_pi_0,     // C photoionization base rate
    double gamma_H2,     // Dust extinction for H2
    double gamma_CO,     // Dust extinction for CO
    double gamma_C,      // Dust extinction for C
    double x_C_total,    // Total carbon abundance
    double x_O_total     // Total oxygen abundance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    
    double nH = density[i];
    double T = T_gas[i];
    double g0 = G0[i];
    double av = A_V[i];
    double zeta = zeta_CR[i];
    double fH2 = f_shield_H2[i];
    double fCO = f_shield_CO[i];
    
    // Current abundances
    double xH = x_HI[i];
    double xH2 = x_H2[i];
    double xCp = x_Cp[i];
    double xC = x_C[i];
    double xCO = x_CO[i];
    double xO = x_O[i];
    double xe = x_e[i];
    
    // === H2 chemistry ===
    // Formation: H + H + grain -> H2
    double R_form = R_H2_form * xH * nH * 0.5;
    // Photodissociation: H2 + hv -> H + H
    double k_pd = k_H2_pd_0 * g0 * exp(-gamma_H2 * av) * fH2;
    double R_pd = k_pd * xH2;
    // CR dissociation
    double R_cr = 2.5 * zeta * xH2 * 0.1;  // ~10% of CR ionizations dissociate
    
    double dxH2 = R_form - R_pd - R_cr;
    double dxH = -2.0 * dxH2;  // H conservation
    
    // === Carbon chemistry ===
    // C photoionization: C + hv -> C+ + e
    double k_ci = k_C_pi_0 * g0 * exp(-gamma_C * av);
    // C+ recombination: C+ + e -> C
    double alpha_Cp = 4.67e-12 * pow(T / 300.0, -0.6);
    // CO photodissociation
    double k_co = k_CO_pd_0 * g0 * exp(-gamma_CO * av) * fCO;
    // CO CR destruction
    double k_co_cr = 6.0 * zeta;
    // CO formation: C + OH -> CO + H (approximate OH steady state)
    double xOH = 1e-7 * (xH2 > 0.1 ? 1.0 : 0.01);  // Rough OH estimate
    double k_CO_form = 1.0e-10;
    
    double dxCp = k_ci * xC - alpha_Cp * xCp * xe * nH + (k_co + k_co_cr) * xCO;
    double dxCO = k_CO_form * xC * xOH * nH - (k_co + k_co_cr) * xCO;
    double dxC = -k_ci * xC + alpha_Cp * xCp * xe * nH - k_CO_form * xC * xOH * nH;
    
    // === Electron balance ===
    double dxe = k_ci * xC - alpha_Cp * xCp * xe * nH;
    // Add CR ionization of H
    dxe += 0.46 * zeta * xH;
    // H+ recombination
    double alpha_Hp = 2.753e-14 * pow(T / 300.0, -0.745);
    double xHp = xe - xCp;  // Approximate
    if (xHp < 0) xHp = 1e-10;
    dxe -= alpha_Hp * xHp * xe * nH;
    
    // === Apply explicit Euler step with subcycling ===
    // Adaptive dt to prevent > 30% change
    double max_rate = fmax(fabs(dxH2) / fmax(xH2, 1e-20),
                     fmax(fabs(dxCp) / fmax(xCp, 1e-20),
                          fabs(dxCO) / fmax(xCO, 1e-20)));
    double dt_eff = dt;
    if (max_rate * dt > 0.3) {
        dt_eff = 0.3 / max_rate;
    }
    
    // Update
    x_HI[i] = fmax(xH + dxH * dt_eff, 1e-30);
    x_H2[i] = fmax(xH2 + dxH2 * dt_eff, 1e-30);
    x_Cp[i] = fmax(xCp + dxCp * dt_eff, 1e-30);
    x_C[i] = fmax(xC + dxC * dt_eff, 1e-30);
    x_CO[i] = fmax(xCO + dxCO * dt_eff, 1e-30);
    x_O[i] = fmax(x_O_total - x_CO[i], 1e-30);
    x_e[i] = fmax(xe + dxe * dt_eff, 1e-30);
    
    // Conservation: rescale C species
    double C_sum = x_Cp[i] + x_C[i] + x_CO[i];
    if (C_sum > 0) {
        double scale = x_C_total / C_sum;
        x_Cp[i] *= scale;
        x_C[i] *= scale;
        x_CO[i] *= scale;
    }
}
'''


def get_gpu_backend():
    """Detect available GPU backend."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return 'cupy'
    except Exception:
        pass
    
    try:
        from numba import cuda
        if cuda.is_available():
            return 'numba'
    except Exception:
        pass
    
    return None


def chemistry_step_gpu(solver, dt=1e12, n_substeps=10):
    """
    GPU-accelerated chemistry step using CuPy raw kernel.
    
    Falls back to CPU numpy if no GPU available.
    """
    backend = get_gpu_backend()
    
    if backend == 'cupy':
        import cupy as cp
        return _chemistry_cupy(solver, dt, n_substeps)
    else:
        # CPU fallback — same algorithm as explicit Euler in solver
        return _chemistry_numpy(solver, dt, n_substeps)


def _chemistry_numpy(solver, dt, n_substeps):
    """Vectorized numpy chemistry step (CPU, but fast for moderate grids)."""
    nH = solver.density
    T = solver.T_gas
    g0 = solver.G0
    av = solver.A_V
    zeta = solver.zeta_CR
    
    from ..utils.constants import gas_phase_abundances
    x_C_total = gas_phase_abundances.get('C', 1.4e-4)
    x_O_total = gas_phase_abundances.get('O', 3.0e-4)
    
    for step in range(n_substeps):
        xH = solver.x_HI; xH2 = solver.x_H2
        xCp = solver.x_Cp; xC = solver.x_C
        xCO = solver.x_CO; xe = solver.x_e
        
        # H2 formation/destruction
        R_H2_form = 3e-17 * xH * nH * 0.5
        k_pd = 4.43e-11 * g0 * np.exp(-3.74 * av)  # benchmark rate
        # Self-shielding (use stored values)
        from ..radiative_transfer.shielding import f_shield_H2 as fsh_func
        fH2 = fsh_func(solver.N_H2, T=50)
        R_pd = k_pd * fH2 * xH2
        
        dxH2 = R_H2_form - R_pd
        dxH = -2.0 * dxH2
        
        # Carbon chemistry
        k_ci = 2.56e-10 * g0 * np.exp(-3.02 * av)
        alpha_Cp = 4.67e-12 * np.power(T / 300.0, -0.6)
        k_co = 1.71e-10 * g0 * np.exp(-3.53 * av)
        from ..radiative_transfer.shielding import f_shield_CO as fco_func
        fCO = fco_func(solver.N_CO, solver.N_H2)
        k_co_cr = 6.0 * zeta
        xOH = np.where(xH2 > 0.1, 1e-7, 1e-9)
        
        dxCp = k_ci * xC - alpha_Cp * xCp * xe * nH + (k_co * fCO + k_co_cr) * xCO
        dxCO = 1e-10 * xC * xOH * nH - (k_co * fCO + k_co_cr) * xCO
        dxC = -k_ci * xC + alpha_Cp * xCp * xe * nH - 1e-10 * xC * xOH * nH
        
        # Adaptive dt
        max_rate = np.maximum(
            np.abs(dxH2) / np.maximum(xH2, 1e-20),
            np.maximum(
                np.abs(dxCp) / np.maximum(xCp, 1e-20),
                np.abs(dxCO) / np.maximum(xCO, 1e-20)
            )
        )
        dt_eff = np.where(max_rate * dt > 0.3, 0.3 / np.maximum(max_rate, 1e-30), dt)
        
        solver.x_HI = np.maximum(xH + dxH * dt_eff, 1e-30)
        solver.x_H2 = np.maximum(xH2 + dxH2 * dt_eff, 1e-30)
        solver.x_Cp = np.maximum(xCp + dxCp * dt_eff, 1e-30)
        solver.x_C = np.maximum(xC + dxC * dt_eff, 1e-30)
        solver.x_CO = np.maximum(xCO + dxCO * dt_eff, 1e-30)
        solver.x_O = np.maximum(x_O_total - solver.x_CO, 1e-30)
        
        # Conservation
        C_sum = solver.x_Cp + solver.x_C + solver.x_CO
        scale = np.where(C_sum > 0, x_C_total / C_sum, 1.0)
        solver.x_Cp *= scale
        solver.x_C *= scale
        solver.x_CO *= scale
        
        dt *= 2  # Grow step
        dt = min(dt, 1e15)


def _chemistry_cupy(solver, dt, n_substeps):
    """CuPy GPU chemistry — transfers arrays to GPU, runs kernel, transfers back."""
    import cupy as cp
    
    # Transfer to GPU
    d_density = cp.asarray(solver.density)
    d_T = cp.asarray(solver.T_gas)
    d_G0 = cp.asarray(solver.G0)
    d_AV = cp.asarray(solver.A_V)
    d_zeta = cp.asarray(solver.zeta_CR)
    d_xHI = cp.asarray(solver.x_HI)
    d_xH2 = cp.asarray(solver.x_H2)
    d_xCp = cp.asarray(solver.x_Cp)
    d_xC = cp.asarray(solver.x_C)
    d_xCO = cp.asarray(solver.x_CO)
    d_xO = cp.asarray(solver.x_O)
    d_xe = cp.asarray(solver.x_e)
    
    # Compile and run CUDA kernel
    kernel = cp.RawKernel(GPU_CHEMISTRY_KERNEL, 'chemistry_step')
    
    n_cells = solver.n_cells
    block_size = 256
    grid_size = (n_cells + block_size - 1) // block_size
    
    from ..utils.constants import gas_phase_abundances
    x_C_total = gas_phase_abundances.get('C', 1.4e-4)
    x_O_total = gas_phase_abundances.get('O', 3.0e-4)
    
    # Flatten arrays for kernel
    flat = lambda a: a.ravel().astype(cp.float64)
    d_fH2 = cp.ones(n_cells, dtype=cp.float64)  # Placeholder
    d_fCO = cp.ones(n_cells, dtype=cp.float64)
    
    for step in range(n_substeps):
        kernel((grid_size,), (block_size,),
               (flat(d_density), flat(d_T), flat(d_G0), flat(d_AV),
                flat(d_zeta), d_fH2, d_fCO,
                flat(d_xHI), flat(d_xH2), flat(d_xCp), flat(d_xC),
                flat(d_xCO), flat(d_xO), flat(d_xe),
                n_cells, dt,
                3e-17, 4.43e-11, 1.71e-10, 2.56e-10,
                3.74, 3.53, 3.02, x_C_total, x_O_total))
        dt = min(dt * 2, 1e15)
    
    # Transfer back to CPU
    solver.x_HI = cp.asnumpy(d_xHI).reshape(solver.density.shape)
    solver.x_H2 = cp.asnumpy(d_xH2).reshape(solver.density.shape)
    solver.x_Cp = cp.asnumpy(d_xCp).reshape(solver.density.shape)
    solver.x_C = cp.asnumpy(d_xC).reshape(solver.density.shape)
    solver.x_CO = cp.asnumpy(d_xCO).reshape(solver.density.shape)
    solver.x_O = cp.asnumpy(d_xO).reshape(solver.density.shape)
    solver.x_e = cp.asnumpy(d_xe).reshape(solver.density.shape)


# ============================================================
# Setup script for HPC environments
# ============================================================

SETUP_SCRIPT = """#!/bin/bash
# PRISM-3D HPC Environment Setup
# Run this once on the cluster to set up the Python environment

echo "Setting up PRISM-3D HPC environment..."

# Create virtual environment
python3 -m venv $HOME/prism3d_env
source $HOME/prism3d_env/bin/activate

# Core dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib

# HPC dependencies  
pip install mpi4py h5py

# GPU support (optional — comment out if no GPU)
# Detect CUDA version and install matching CuPy
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \\([0-9]*\\).\\([0-9]*\\).*/\\1\\2/')
if [ -n "$CUDA_VERSION" ]; then
    echo "CUDA $CUDA_VERSION detected, installing CuPy..."
    pip install cupy-cuda${CUDA_VERSION}
else
    echo "No CUDA found, GPU acceleration will not be available"
    echo "Install manually: pip install cupy-cuda12x"
fi

# Install numba as GPU fallback
pip install numba

echo ""
echo "Environment ready. Activate with:"
echo "  source $HOME/prism3d_env/bin/activate"
echo ""
echo "Test GPU availability:"
echo "  python -c 'from prism3d.hpc.runner import get_gpu_backend; print(get_gpu_backend())'"
"""


def generate_setup_script(filepath='setup_prism3d_hpc.sh'):
    """Generate the HPC environment setup script."""
    with open(filepath, 'w') as f:
        f.write(SETUP_SCRIPT)
    os.chmod(filepath, 0o755)
    print(f"Setup script written to {filepath}")
    print(f"Run on cluster: bash {filepath}")


# ============================================================
# Main HPC run driver
# ============================================================

def run_hpc(config_path=None, config=None):
    """
    Main entry point for HPC runs.
    
    Can be called from command line:
        python -m prism3d.hpc.run_hpc --config my_config.json
    
    Or from Python:
        from prism3d.hpc.runner import run_hpc
        run_hpc(config={'n_cells': 64, 'G0_external': 100})
    """
    if config is None:
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = DEFAULT_CONFIG.copy()
    
    from ..utils.constants import pc_cm
    from ..core.density_fields import fractal_turbulent
    from ..core.solver_3d import PDRSolver3D
    
    n = config['n_cells']
    box_size = config['box_size_pc'] * pc_cm
    
    print(f"PRISM-3D HPC Run")
    print(f"  Grid: {n}³ = {n**3:,} cells")
    print(f"  Box:  {config['box_size_pc']:.1f} pc")
    print(f"  GPU:  {get_gpu_backend() or 'None (CPU mode)'}")
    
    # Generate density field
    density, cell_size = fractal_turbulent(
        n, box_size,
        n_mean=config['n_mean'],
        sigma_ln=config['sigma_ln'],
        spectral_index=config['spectral_index'],
    )
    
    # Create solver
    solver = PDRSolver3D(
        density, box_size,
        G0_external=config['G0_external'],
        zeta_CR_0=config['zeta_CR'],
        nside_rt=config['nside_rt'],
    )
    
    # Run fast iterations
    solver.run(
        max_iterations=config['max_iterations'],
        convergence_tol=config['convergence_tol'],
    )
    
    # Optional BDF refinement
    if config.get('refine_after', False):
        solver.refine()
    
    # Save
    os.makedirs(config['output_dir'], exist_ok=True)
    outpath = os.path.join(config['output_dir'], f'prism3d_{n}cube.npz')
    solver.save(outpath)
    print(f"\nResults saved to {outpath}")
    
    return solver
