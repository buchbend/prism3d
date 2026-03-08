# PRISM-3D

**3D PDR code with THEMIS dust evolution, ML-accelerated chemistry, and synthetic observation pipeline.**

PRISM-3D solves the 3D photodissociation region problem with self-consistent
dust evolution. It couples multi-ray FUV radiative transfer with a
three-component THEMIS grain model, 75-reaction gas-phase chemistry, and
15 heating/cooling processes. A machine-learning accelerator provides
10,000× speedup for production 3D runs. The integrated observation pipeline
produces JWST, Herschel, and ALMA maps with proper beam handling.

---

## Quick Start

```bash
pip install -e .
python -m prism3d.run --n 8 --G0 100           # 20 seconds
python -m prism3d.run --orion-bar --n 32        # Orion Bar model
```

Or with Make:

```bash
make install
make test
make run N=32 G0=500
```

Every run automatically produces:

```
output/
├── model.npz                    3D model state (reload with PDRSolver3D.load)
├── viewer_3d.html               Interactive 3D browser viewer (no CDN needed)
├── summary_dashboard.png        One-page visual overview
├── depth_profiles.png           1D profiles through the PDR
├── dust_evolution.png           Nano-grain depletion analysis
├── synthetic_observations.png   20 JWST/Herschel/ALMA maps
├── midplane_{xy,xz,yz}.png     Midplane slices (3 axes × 8 quantities)
├── report.html                  Visual HTML report
└── report.json                  Machine-readable statistics
```

---

## Installation

### Laptop / Workstation

```bash
pip install -e .           # numpy, scipy, matplotlib, scikit-learn
pip install astropy h5py   # Optional: FITS I/O, HDF5 support
```

### HPC Cluster (pip)

```bash
python3 -m venv $HOME/prism3d_env
source $HOME/prism3d_env/bin/activate
pip install numpy scipy matplotlib scikit-learn astropy h5py
pip install -e /path/to/prism3d
```

### HPC Cluster (Apptainer — recommended)

```bash
make container              # Builds prism3d.sif (~800 MB)
apptainer run prism3d.sif --n 64 --G0 1000 --output ./results
```

See "HPC Deployment" below.

---

## Running Models

### Command Line

```bash
# Custom parameters
python -m prism3d.run \
    --n 64 --G0 1000 --box 2.0 --n-mean 300 \
    --nside 2 --max-iter 20 --refine \
    --output ./my_run

# From JSON config
python -m prism3d.run --config model.json

# Presets
python -m prism3d.run --orion-bar --n 64       # G₀=2.6e4, n=5e4, 0.4 pc
python -m prism3d.run --horsehead --n 64       # G₀=100, n=2e3, 0.5 pc
```

### From Python

```python
from prism3d.core.density_fields import fractal_turbulent
from prism3d.core.solver_3d import PDRSolver3D
from prism3d.evaluate import evaluate_model
from prism3d.utils.constants import pc_cm

density, _ = fractal_turbulent(64, 2.0 * pc_cm, n_mean=1000)
solver = PDRSolver3D(density, 2.0 * pc_cm, G0_external=500)
solver.run(max_iterations=10)
solver.refine()                    # BDF accuracy pass
evaluate_model(solver, './results')
```

### Parameters

| Parameter     | Default | Description                        |
|---------------|--------:|-------------------------------------|
| `--n`         |       8 | Cells per axis (total = n³)        |
| `--box`       |     2.0 | Box size [pc]                      |
| `--n-mean`    |     500 | Mean H density [cm⁻³]             |
| `--sigma-ln`  |     1.5 | Log-normal width                   |
| `--G0`        |     100 | External FUV field [Habing]        |
| `--zeta-cr`   |  2e-16  | Cosmic ray ionization rate [s⁻¹]  |
| `--nside`     |       1 | HEALPix nside (12×nside² rays)    |
| `--max-iter`  |      10 | Maximum solver iterations          |
| `--tol`       |    0.05 | Convergence tolerance              |
| `--refine`    |   false | BDF chemistry refinement pass      |

---

## HPC Deployment

### 1. Build Container

```bash
# On a build node with fakeroot or root access:
make container
# → prism3d.sif (~800 MB Apptainer image)
```

Copy `prism3d.sif` to your cluster. It contains Python 3.11, all
dependencies, and PRISM-3D installed.

### 2. Generate SLURM Scripts

```bash
make slurm N=128 G0=1000
# → run_prism3d_128cube.slurm

# Or for ML training:
make slurm-train TRAIN_SAMPLES=50000
# → train_accelerator_50000.slurm
```

### 3. Submit Jobs

```bash
sbatch run_prism3d_128cube.slurm
```

The generated SLURM script uses Apptainer if `prism3d.sif` exists,
otherwise calls Python directly. Example SLURM content:

```bash
#!/bin/bash
#SBATCH --job-name=prism3d_128
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=prism3d_%j.log

if [ -f prism3d.sif ]; then
    apptainer run prism3d.sif --n 128 --G0 1000 --output ./run_${SLURM_JOB_ID}
else
    python -m prism3d.run --n 128 --G0 1000 --output ./run_${SLURM_JOB_ID}
fi
```

### 4. Train ML Accelerator

The ML accelerator must be trained once using the BDF solver, then
provides 10⁴× speedup for all subsequent runs.

```bash
# On HPC (SLURM):
make slurm-train TRAIN_SAMPLES=50000
sbatch train_accelerator_50000.slurm

# Or interactively:
make train TRAIN_SAMPLES=50000
```

| Samples  | Time (10 cores) | MAE       |
|---------:|----------------:|-----------|
|    5,000 |          25 min | ~0.05 dex |
|   50,000 |           4 hr  | ~0.01 dex |
|  200,000 |          16 hr  | ~0.005 dex|

### 5. Performance

| Grid  |      Cells | Laptop  | 1× H100    |
|------:|-----------:|---------|------------|
|    8³ |        512 | 20s     | < 1s       |
|   16³ |      4,096 | 4 min   | ~5s        |
|   32³ |     32,768 | 30 min  | ~30s       |
|   64³ |    262,144 | ~4 hr   | ~4 min     |
|  128³ |  2,097,152 | —       | ~30 min    |
|  250³ | 15,625,000 | —       | ~26 min    |

---

## Observation Comparison Pipeline

### Download Orion Bar Data (PDRs4All)

```bash
make data                       # Template spectra (~10 MB)
make data-full                  # Full JWST + Herschel (~10 GB)
make data-prepare N=32          # Reproject to model grid
```

### Hybrid Beam Matching

Observations and model are compared at the resolution of the coarser partner:

- Observation finer than model → observation convolved to model beam
- Observation coarser than model → model convolved to observation beam

This means a 32³ model (cell ~ 3″) gives you valid comparisons against
all JWST lines (convolved from 0.1–0.8″ to 3″), ALMA CO (from 0.5″),
*and* Herschel [CII]/[OI] (model convolved to 11″).

### Compare

```python
import numpy as np
from prism3d.core.solver_3d import PDRSolver3D
from prism3d.observations.from_observations import full_comparison

solver = PDRSolver3D.load('results/model.npz')
obs = np.load('prepared/orion_bar_prepared.npz', allow_pickle=True)

results = full_comparison(solver, {
    'CII_158': obs['CII_158_matched'],
    'PACS160': obs['PACS160_matched'],
    'CO_3-2':  obs['CO_3-2_matched'],
})
for k, s in results.items():
    if not k.startswith('_'):
        print(f"{k}: chi2={s['chi2_reduced']:.1f}, r={s['correlation']:.2f}")
```

---

## Spectral Line Cubes

```python
from prism3d.observations.spectra import compute_ppv_cube, extract_spectrum

ppv = compute_ppv_cube(solver, 'CII_158', v_turb_kms=2.0, n_vel=64)
spec = extract_spectrum(ppv, aperture_pix=5)
print(f"[CII] FWHM = {spec['fwhm_kms']:.1f} km/s, tau = {ppv['peak_tau']:.1f}")
```

The spectral cubes use proper radiative transfer with optical depth:
[OI] 63μm is self-absorbed (τ > 4), [CII] is moderately thick (τ ~ 1),
[CI] 609μm is optically thin.

---

## FLASH / AREPO Post-Processing

```python
from prism3d.observations.from_observations import read_flash_hdf5, read_arepo_hdf5

# FLASH AMR
density, cell_size = read_flash_hdf5('flash_chk_0100', field='dens')

# AREPO Voronoi → Cartesian
density = read_arepo_hdf5('arepo_snap_050.hdf5', n_grid=128)

# Run PRISM-3D on the hydro output
solver = PDRSolver3D(density, box, G0_external=100)
solver.run(max_iterations=10)
```

---

## Code Structure

```
prism3d/
├── run.py                  CLI runner with presets
├── evaluate.py             Auto-evaluation (7 figure types + HTML report)
├── viewer_export.py        3D browser viewer (pure JS, no CDN)
├── core/
│   ├── solver_3d.py        3D solver (main engine)
│   ├── density_fields.py   Turbulent density generators
│   └── grid.py             Octree AMR grid
├── chemistry/
│   ├── network.py          75 reactions, 32 species
│   ├── solver.py           BDF + explicit Euler
│   ├── accelerator.py      ML chemistry (GBRT, 10⁴× speedup)
│   └── train_hpc.py        HPC training pipeline
├── radiative_transfer/
│   ├── fuv_rt_3d.py        3D multi-ray FUV RT (HEALPix)
│   └── shielding.py        H₂/CO self-shielding
├── grains/
│   └── themis.py           THEMIS 3-component dust model
├── thermal/
│   ├── heating.py          7 heating processes
│   ├── cooling.py          8 cooling processes
│   └── balance.py          Thermal equilibrium
├── observations/
│   ├── jwst_pipeline.py    Synthetic maps + beam convolution
│   ├── spectra.py          PPV cubes with optical depth RT
│   └── from_observations.py FITS ingest + beam-matched comparison
├── data/
│   ├── download_data.py    PDRs4All data downloader
│   └── prepare_data.py     Reproject + beam-match pipeline
├── examples/
│   ├── orion_bar_3d.py     Orion Bar geometry
│   └── roellig_benchmark.py Röllig+ 2007 validation
├── Makefile                Build/run/deploy targets
├── apptainer.def           Container definition
└── pyproject.toml          Package metadata
```

12,800+ lines across 47 files.
