# CLAUDE.md — PRISM-3D Project Briefing

This document is for Claude (or any developer) starting a new session on this project.
Read this first, then Architecture.md for implementation details.

## What Is PRISM-3D?

PRISM-3D (**PDR Radiative transfer with Integrated Structure and Microphysics**) is a
3D photodissociation region (PDR) code written in Python. It models the transition zone
between ionized/atomic gas (exposed to UV from a nearby star) and cold molecular gas
(shielded from UV), solving simultaneously for:

- **FUV radiative transfer** through a 3D density field (multi-ray HEALPix)
- **Gas-phase chemistry** (75 reactions, 32 species) with an ML accelerator (10⁴× speedup)
- **Thermal balance** (7 heating + 8 cooling processes)
- **THEMIS dust model** (3-component grain populations with UV-driven evolution)
- **Synthetic observations** (JWST, Herschel, ALMA maps with correct beam sizes)
- **Spectral line cubes** (PPV cubes with proper optical depth radiative transfer)

The unique selling point vs. existing codes (Meudon PDR, KOSMA-τ, 3D-PDR):
**PRISM-3D is the first 3D PDR code with self-consistent dust evolution.**
The THEMIS grain model tracks nano-grain destruction/formation per cell, which directly
feeds back into photoelectric heating — the dominant heating mechanism in PDRs.

## Project Status

**v0.5** — 12,998 lines across 50 files. Fully functional for production runs.

### What Works
- 3D solver converges on turbulent density fields (Röllig+ 2007 benchmark passed)
- THEMIS dust evolution (nano-grain depletion in UV-exposed gas)
- ML chemistry accelerator (GBRT, 10⁴× faster than BDF, 0.01 dex accuracy)
- Synthetic JWST/Herschel/ALMA observations with 30 beam sizes
- PPV spectral cubes with optical depth RT ([OI] self-absorbed, [CII] thick, [CI] thin)
- Hybrid beam matching for model-observation comparison
- Interactive 3D viewer (Canvas2D offline + Three.js WebGL)
- Automated evaluation pipeline producing 7 figure types + HTML report
- Apptainer container + Makefile + SLURM generation for HPC

### Known Issues / Next Steps
1. **ML accuracy at intermediate AV**: CO is underestimated at AV 2-4 because the accelerator
   was trained on fast-Euler data. Fix: retrain with BDF data on HPC (50k samples, ~4 hr).
2. **No density dependence in ML chemistry**: x(H₂) barely varies with n_H at fixed AV.
   Same fix as above — BDF training data captures density dependence.
3. **Thermal solver is Python loops**: now the bottleneck (~19% of time). Vectorizing
   Brent's method or moving to a Newton solver would help.
4. **H₂ rovibrational levels**: needed for JWST NIR line comparison (1-0 S(1) etc.).
5. **Time-dependent chemistry mode**: needed for over-pressurized gas dynamics.
6. **AMR wiring**: octree grid exists but isn't connected to the 3D solver yet.
   Needed for JWST-resolution zoom-ins (250³ on 0.02 pc).

## Physics Background

### PDR Structure
A PDR is illuminated from one side by UV photons (G₀ in Habing units, 1.6×10⁻³ erg/cm²/s).
Moving into the cloud:

```
Star → [HII region] → [Atomic PDR] → [Dissociation front] → [Molecular cloud]
                        C⁺, O, H         H→H₂, C⁺→CO          H₂, CO, dust
                        T ~ 500 K         AV ~ 1.5-4            T ~ 20-50 K
```

Key transitions:
- **H → H₂** at AV ~ 1.5–2.5 (depends on G₀/n ratio)
- **C⁺ → C → CO** at AV ~ 2.5–4 (CO requires both H₂ shielding and dust shielding)
- **Photoelectric heating** dominates at AV < 3, cosmic ray heating dominates deeper

### THEMIS Dust Model
Three grain populations (Jones+ 2013, 2017):
1. **a-C(:H) nano-particles** (0.4–20 nm): PAH-like, dominate PE heating, destroyed by UV
2. **Large a-C(:H) grains** (20 nm–0.2 μm): carbon mantles on silicate cores
3. **a-Sil grains** (20 nm–0.3 μm): amorphous silicates with iron inclusions

Per-cell state: `f_nano` (nano-grain mass fraction) and `E_g` (band gap eV).
UV photo-processing destroys nano-grains (τ ~ 10⁴ yr at G₀ = 10⁴), while
accretion in dense gas replenishes them. This creates a spatial gradient in
PE heating efficiency that 1D models cannot capture in 3D turbulent clouds.

### Röllig+ (2007) Benchmark
The standard PDR code comparison. Our results on the V1 benchmark (n=10³, G₀=10):

| Metric              | Röllig range | PRISM-3D | Status |
|---------------------|-------------|----------|--------|
| H→H₂ transition AV  | 1.5–2.5     | 1.5      | ✅     |
| C⁺→CO crossover AV  | 2.5–4.0     | 2.5–3.5  | ✅     |
| n(CO) at depth       | ~0.14       | 0.133    | ✅     |
| n(H₂) at depth       | ~500        | 498      | ✅     |
| T surface            | 50–120 K    | 47 K     | ✅     |

## Key Files to Know

| File | Lines | What It Does |
|------|------:|--------------|
| `core/solver_3d.py` | 477 | **The main engine.** 3D iterative solver: RT → chemistry → dust → thermal, per-cell THEMIS state |
| `grains/themis.py` | 799 | THEMIS dust: 3 populations, PE heating, H₂ formation, stochastic heating, SED, UV evolution |
| `chemistry/network.py` | 1004 | 75 reactions, 32 species. Rate coefficients, ODE RHS |
| `chemistry/accelerator.py` | 566 | ML GBRT ensemble: predict abundances from (n,T,G₀,AV,ζ,f_H2,f_CO) |
| `observations/jwst_pipeline.py` | 657 | Synthetic maps: dust continuum, line emission with escape-probability RT, 30 beam sizes |
| `observations/spectra.py` | 500 | PPV cubes with formal solution RT (optical depth per channel) |
| `observations/from_observations.py` | 630 | FITS ingest, SED fitting, 3D inversion, hybrid beam matching, χ² comparison |
| `evaluate.py` | 568 | Auto-evaluation: 7 figure types, 5 physics checks, HTML report, viewer export |
| `data/prepare_data.py` | 682 | Reproject JWST/Herschel to model grid, hybrid beam matching |
| `viewer_export.py` | 340 | Two backends: Canvas2D (offline) + Three.js (WebGL isosurfaces) |

## Where Things Live

```
prism3d/
├── core/           Physics engine (solver, density fields, grid)
├── chemistry/      Network, ODE solver, ML accelerator, HPC training
├── radiative_transfer/  FUV RT (1D and 3D), H₂/CO shielding
├── grains/         THEMIS dust model, legacy MRN
├── thermal/        Heating (PE, CR, H₂ form/pump, ...), cooling ([CII], [OI], CO, ...)
├── observations/   Synthetic obs pipeline, spectra, FITS comparison
├── data/           PDRs4All downloader + preparation
├── examples/       Orion Bar, Horsehead, Röllig benchmark
├── hpc/            SLURM generation, GPU CUDA kernel
├── utils/          Physical constants
├── Makefile        Build/run/deploy
├── apptainer.def   Container definition
└── pyproject.toml  Package metadata
```

## Test Data: PDRs4All Orion Bar

The primary benchmark dataset is the JWST PDRs4All ERS program (Program 1288):
- **Fully public** (zero proprietary period)
- NIRSpec IFU (0.97–5.27 μm), MIRI MRS (4.9–27.9 μm), NIRCam, MIRI imaging
- Template spectra for 5 PDR zones: HII, Atomic, DF1, DF2, DF3
- Complementary: ALMA HCO⁺/CO, Herschel PACS/SPIRE
- Physical parameters: G₀ = 2.6×10⁴, n = 5×10⁴ cm⁻³, d = 414 pc

Download with: `make data` (templates) or `make data-full` (everything).

## Running

```bash
make install && make test                    # Quick validation
make run N=32 G0=500                         # Production run
make container && make container-run N=64    # Apptainer on HPC
```

Every run produces a model.npz, viewer HTML, figures, and report in the output directory.

## The A&A Letter

A draft letter (v2) is in `paper_v2/prism3d_letter_v2.tex`. It covers:
architecture, Röllig benchmark, 3D turbulent cloud results, Orion Bar model,
ML accelerator performance, and the observation-to-model workflow. 6 pages,
14 references, 4 figures.

## Session History

This project was built in a single extended session. Three transcript files
record the full development history:
1. `2026-03-08-14-03-12`: Foundation — 1D solver, chemistry, thermal, RT, Röllig benchmark
2. `2026-03-08-18-19-05`: 3D extension — turbulent clouds, THEMIS, ML accelerator, A&A paper
3. `2026-03-08-21-49-03`: Pipeline — synthetic obs, data download, beam matching, spectra, viewer

Transcripts are in `/mnt/transcripts/` if available.
