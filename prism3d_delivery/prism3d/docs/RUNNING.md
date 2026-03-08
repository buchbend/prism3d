# PRISM-3D: Running Guide

## Quick Start

```bash
# Clone and install
git clone <repo_url>
cd prism3d
pip install -e .

# Quick test (8³ grid, ~20 seconds)
python -m prism3d.run --n 8 --G0 100

# Higher resolution (16³, ~4 minutes)
python -m prism3d.run --n 16 --G0 500 --output ./my_run
```

## Command-Line Interface

PRISM-3D provides a single CLI entry point: `python -m prism3d.run`

### Basic Usage

```bash
# Turbulent cloud with custom parameters
python -m prism3d.run \
    --n 32 \           # 32³ cells
    --G0 1000 \        # FUV field [Habing]
    --box 2.0 \        # Box size [pc]
    --n-mean 300 \     # Mean density [cm⁻³]
    --output ./run1

# With BDF chemistry refinement (slower but more accurate)
python -m prism3d.run --n 16 --G0 100 --refine
```

### Presets for Well-Known PDRs

```bash
# Orion Bar (G0=2×10⁴, n=5×10⁴, 0.4 pc box)
python -m prism3d.run --orion-bar --n 32

# Horsehead Nebula (G0=100, n=2000, 0.5 pc box)
python -m prism3d.run --horsehead --n 32
```

### From Configuration File

```bash
python -m prism3d.run --config my_model.json
```

Example `my_model.json`:
```json
{
    "n": 64,
    "G0": 1000,
    "box": 5.0,
    "n_mean": 300,
    "sigma_ln": 1.5,
    "zeta_cr": 2e-16,
    "nside": 2,
    "max_iter": 20,
    "tol": 0.01,
    "refine": true,
    "output": "./science_run"
}
```

### All Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n` | 8 | Cells per dimension (total = n³) |
| `--box` | 2.0 | Box size [pc] |
| `--n-mean` | 500 | Mean H density [cm⁻³] |
| `--sigma-ln` | 1.5 | Log-normal width (~Mach number) |
| `--G0` | 100 | External FUV field [Habing] |
| `--zeta-cr` | 2e-16 | Cosmic ray ionization rate [s⁻¹] |
| `--nside` | 1 | HEALPix nside (12×nside² rays) |
| `--max-iter` | 10 | Maximum solver iterations |
| `--tol` | 0.05 | Convergence tolerance |
| `--refine` | false | BDF chemistry refinement pass |
| `--output` | ./prism3d_output | Output directory |
| `--seed` | 42 | Random seed for density field |

## Output Files

Each run produces in the output directory:
- `model_NNcube.npz` — Full model state (loadable for analysis)
- `observations_NNcube.npz` — Synthetic observation arrays
- `observations_NNcube.png` — 20-panel diagnostic figure

### Loading Results in Python

```python
import numpy as np
from prism3d.core.solver_3d import PDRSolver3D

# Load a saved model
solver = PDRSolver3D.load('output/model_16cube.npz')

# Access any quantity
print(solver.T_gas.shape)      # (16, 16, 16)
print(solver.f_nano.mean())    # Nano-grain fraction
print(solver.x_CO.max())       # Peak CO abundance

# Get a 2D slice
mid = solver.get_slice(axis=2)  # z-midplane
# Returns dict: density, T_gas, T_dust, G0, A_V, x_H2, x_CO, ...

# Generate observations at a different distance
from prism3d.observations.jwst_pipeline import generate_observations
obs = generate_observations(solver, distance_pc=1000, los_axis=0)
```

## Performance Guide

| Grid | Cells | Time (laptop) | Time (A100 GPU) | Memory |
|---|---|---|---|---|
| 8³ | 512 | 20s | <1s | <100 MB |
| 16³ | 4,096 | 4 min | ~5s | ~200 MB |
| 32³ | 32,768 | 30 min | ~30s | ~1 GB |
| 64³ | 262,144 | ~4 hr | ~4 min | ~8 GB |
| 128³ | 2,097,152 | HPC only | ~30 min | ~60 GB |

Scaling is linear in cell count. RT is negligible; chemistry and dust 
dominate equally at ~45% each.

For runs above 32³, use the HPC deployment (see below).


---

# HPC Deployment

## Setup on Cluster (once)

```bash
# Create Python environment
python3 -m venv $HOME/prism3d_env
source $HOME/prism3d_env/bin/activate
pip install numpy scipy matplotlib

# Optional: GPU support
pip install cupy-cuda12x  # Adjust for your CUDA version

# Install PRISM-3D
cd /path/to/prism3d
pip install -e .
```

## Generate and Submit SLURM Jobs

```bash
# Generate a SLURM script
python -m prism3d.run --gen-slurm \
    --n 128 --G0 1000 --box 5.0 \
    --nodes 4 --partition gpu-a100 --wall-time 08:00:00

# Submit
sbatch run_prism3d_128cube.slurm
```

## Batch Parameter Surveys

```python
# Generate a grid of models varying G0 and density
import json, os

for G0 in [10, 100, 1000, 10000]:
    for n_mean in [100, 1000, 10000]:
        config = {
            "n": 64, "G0": G0, "n_mean": n_mean,
            "box": 2.0, "output": f"./survey/G0_{G0}_n_{n_mean}"
        }
        fname = f"config_G0_{G0}_n_{n_mean}.json"
        with open(fname, 'w') as f:
            json.dump(config, f)
        os.system(f"sbatch run_prism3d.slurm --config {fname}")
```


---

# Benchmark Validation

## Röllig et al. (2007) Benchmark

```python
from prism3d.examples.roellig_benchmark import run_benchmark

# Run model F1 (n=1e3, χ=10, T=50K fixed)
profiles = run_benchmark('F1', n_cells=24, max_iter=8)

# Run all 8 benchmark models
from prism3d.examples.roellig_benchmark import run_all_benchmarks
all_profiles = run_all_benchmarks()
```

## Expected Results

| Quantity | Röllig range | PRISM-3D |
|---|---|---|
| H→H₂ transition | AV ≈ 1.5–2.5 | 1.5 ✓ |
| C⁺→CO crossover | AV ≈ 2.5–4.0 | 2.5–3.5 ✓ |
| n(CO) at depth | 0.14 cm⁻³ | 0.133 ✓ |
| n(H₂) at depth | 500 cm⁻³ | 498 ✓ |


---

# THEMIS Dust Model

PRISM-3D includes the first 3D implementation of the THEMIS dust framework
(Jones et al. 2013, 2017). Each cell tracks:

- `f_nano` — nano-grain abundance relative to diffuse ISM (1.0 = standard)
- `E_g` — band gap [eV] (0 = aromatic/graphitic, 2.5 = aliphatic)

Dust evolves based on local FUV field:
- UV photo-destruction depletes nano-grains (τ ~ 10⁴ yr at G₀=10⁴)
- UV aromatization lowers E_g (τ ~ 10³ yr at G₀=10⁴)
- Replenishment from large-grain fragmentation (τ ~ 10⁶ yr)

This produces spatially varying PE heating, H₂ formation rates, and
dust emission — the key physics seen by JWST in the Orion Bar
(Elyajouri et al. 2024: 15× nano-grain depletion).

```python
# Access dust state from a model
solver = PDRSolver3D.load('model.npz')
print(solver.f_nano.min(), solver.f_nano.max())  # Nano-grain range
print(solver.E_g.min(), solver.E_g.max())         # Band gap range
```
