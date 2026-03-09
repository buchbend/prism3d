# Architecture.md — PRISM-3D Implementation Details

Deep technical reference for each subsystem. Read CLAUDE.md first for overview.

---

## 1. Solver Architecture (`core/solver_3d.py`, 477 lines)

### Two-Level Iteration

The solver uses a two-level iteration that respects physical timescales:

- **Outer loop** (dust evolution, slow: 10⁴–10⁶ yr timescale):
  advances THEMIS grain state (f_nano, E_g) by one dust timestep (10 kyr),
  then re-converges the inner loop. Default: 5 dust steps.

- **Inner loop** (chemistry + thermal equilibrium, fast: 10²–10⁴ yr):
  iterates RT → shielding → CR → PE heating → chemistry → thermal balance
  to convergence with **fixed** dust properties.

```
Outer (dust step d = 0..N_dust):
│  Evolve dust: f_nano, E_g  (time-dependent, _evolve_dust_state)
│
│  Inner (iteration i = 0..N_iter):
│  ├─ 1. FUV Radiative Transfer  → G₀(x,y,z), A_V(x,y,z)
│  ├─ 2. H₂/CO Shielding         → f_shield_H2, f_shield_CO per cell
│  ├─ 3. CR Attenuation           → ζ_CR(x,y,z)
│  ├─ 4. PE Heating + T_dust      → Γ_PE, T_dust  (instantaneous, _update_dust_heating)
│  ├─ 5. Chemistry                → abundances (ML or Euler)
│  ├─ 6. Damping                  → mode-dependent (see below)
│  ├─ 7. Thermal Balance          → T_gas per cell (adaptive per-cell damping)
│  └─ 8. Convergence Check        → p99(ΔT/T, Δx_H2, Δx_CO) < tol
```

This separation is physically motivated: gas reaches chemical and thermal
equilibrium quickly given a fixed grain population, while dust evolution
responds slowly to the ambient radiation field. Mixing both in a single
loop prevented convergence (dust kept changing PE heating rates).

On the very first inner iteration (d=0, i=0), `_set_analytical_ic()` seeds
position-dependent abundances from the A_V/G₀ field using tanh-profile
approximations for the H/H₂ and C⁺/CO transitions.

### Damping Strategy

#### The physical problem: operator-split bistability

PRISM-3D uses operator splitting: chemistry is solved at fixed T, then thermal
balance is solved at fixed abundances. This is the standard approach in all PDR
codes (Meudon, KOSMA-τ, 3D-PDR, Cloudy) and is mathematically a **fixed-point
iteration** on the coupled system (T, abundances).

The complication arises at the **H/H₂ dissociation front** (A_V ≈ 1.5–2.5).
In these cells the thermal balance has two self-consistent solutions:

1. **Atomic state**: x_H₂ ≈ 0, weak H₂ cooling, T ≈ 200–500 K → at this T,
   photodissociation dominates → x_H₂ stays low ✓
2. **Molecular state**: x_H₂ ≈ 0.5, strong H₂/CO cooling, T ≈ 30–80 K → at
   this T, formation on grains dominates → x_H₂ stays high ✓

Both are self-consistent, but operator splitting bounces between them:
small ΔT → chemistry flips x_H₂ → cooling rate changes → T flips back.
This is not a numerical bug — it is a genuine **physical bistability** of the
H/H₂ transition under operator splitting. The true solution lies on the
separatrix between the two states.

In a 32³ grid with G₀=500, roughly 1500–3000 cells (~5–9%) sit at this
transition zone. The rest of the grid converges without issue.

#### Under-relaxation (damping)

The standard remedy for oscillatory fixed-point iterations is
**under-relaxation**: instead of `x_{n+1} = F(x_n)`, use
`x_{n+1} = w × x_n + (1−w) × F(x_n)`, where `w ∈ [0, 1)`.
The converged solution x* satisfies x* = F(x*) regardless of w —
damping only affects the path to convergence, not the endpoint.

Damping differs by chemistry mode because the ML and Euler paths have
fundamentally different iteration behavior:

- **ML path**: The accelerator predicts equilibrium abundances for a given T
  in one shot. Damping these globally would create an inconsistent state (not
  equilibrium for *any* T), causing thermal balance to overcorrect. Therefore:
  **no global abundance damping; selective per-cell abundance damping only for
  identified oscillating cells; adaptive per-cell T damping for all cells.**

- **Euler path**: Chemistry is evolved incrementally via subcycled timesteps,
  which is inherently partially damped by construction.
  **50/50 abundance damping + 50/50 T damping** (uniform across all cells).

#### Adaptive per-cell T damping

A single damping factor cannot work for the whole grid: well-behaved cells
converge best with light damping (w ≈ 0.5), while bistable cells need heavy
damping (w → 1). The solver tracks a per-cell damping weight array `w_damp`:

- Each cell starts at `w = 0.5` (50/50 blend)
- After each thermal solve:
  - `|ΔT/T| > 0.5` → `w += 0.15` (fast ramp: cell is oscillating)
  - `|ΔT/T| < 0.1` → `w -= 0.02` (slow decay: cell is settling)
- Clamped to `w ∈ [0.5, 0.98]`
- Applied as: `T_new = w × T_old + (1−w) × T_solved`

At `w = 0.98`, a persistent oscillator moves only 2% toward the new
solution each step. Over many iterations, this converges to the
**time-averaged state** between the two bistable solutions — which is the
physically correct answer for the dissociation front location.

The fast ramp (+0.15 per hit) ensures problematic cells are identified
within 3 iterations. The slow decay (−0.02) keeps well-behaved cells
responsive. This is standard practice in CFD/FEM nonlinear solvers.

#### Selective abundance damping in ML mode

The `w_damp` array also controls abundance damping in ML mode.
Most cells should NOT have their ML-predicted abundances damped (the ML
gives the correct equilibrium for the current T). But for cells already
identified as oscillating (`w_damp > 0.55`, i.e. flagged for ≥1 iteration
with `|ΔT/T| > 0.5`), abundances are blended with the same weight:

```
x_H₂_damped = w × x_H₂_old + (1−w) × x_H₂_ML
```

Without this, the T↔H₂ feedback loop persists: damping only T is
insufficient because the ML predicts sharply different x_H₂ for T values
on either side of the bistable transition (e.g. x_H₂ ≈ 0 at 300 K vs.
x_H₂ ≈ 0.45 at 60 K). Damping T slows the T oscillation but does not
break the chemistry jump. Damping both T and abundances together for the
same cells freezes the coupled oscillation.

Well-behaved cells (~92–95% of the grid) remain completely undamped.

### Convergence Metric

The solver reports three quantities per iteration:
- `dT`: relative temperature change (95th percentile)
- `dH2`: relative H₂ change (99th percentile, floor 0.01 in denominator)
- `dCO`: relative CO change (99th percentile, floor 1e-6 in denominator)

Convergence is declared when `max(dT, dH2, dCO) < tol` (default tol = 0.05).

**Why p95 for T but p99 for chemistry**: The ~5% of cells at the dissociation
front are genuinely bistable under operator splitting. Their T oscillation is
a consequence of the numerical method (operator splitting), not a failure to
find the right answer — the spatially-averaged position of the front IS
well-determined. Using p99 for T would require these cells to fully settle,
which demands very heavy damping (w > 0.99) and many iterations for marginal
benefit. At p95, convergence means 95% of the grid has ΔT/T < 5%, which is
sufficient for all downstream quantities (line emission, dust evolution).

Chemistry (dH2, dCO) uses p99 because the abundance damping in ML mode
effectively freezes the oscillating cells — their dH2 contribution is
suppressed by the per-cell damping, so the p99 reflects real convergence.

**Denominator floors**: The relative change |Δx|/x diverges when x is near
zero (e.g. x_H₂ = 10⁻⁶ in fully atomic gas). Floors of 0.01 for H₂ and
10⁻⁶ for CO prevent these irrelevant cells from dominating the metric. A cell
going from x_H₂ = 0.001 → 0.002 registers as |Δ|/0.01 = 0.1, not |Δ|/0.001 = 1.

The log additionally reports:
- `dT(p99)`: the 99th percentile of ΔT/T (for monitoring the front cells)
- `max(dT)`: absolute worst cell
- `n_osc`: count of cells with |ΔT/T| > tol (should plateau, not grow)

Inner convergence typically in 5–15 iterations. The solver stores per-cell arrays:
- `density`, `T_gas`, `T_dust` — shape (nx, ny, nz)
- `x_H2`, `x_HI`, `x_Cp`, `x_C`, `x_CO`, `x_O`, `x_e`, `x_OH`, `x_HCOp` — abundances
- `G0`, `A_V` — FUV field and visual extinction
- `f_nano`, `E_g`, `Gamma_PE` — THEMIS dust state

### Chemistry Modes
Three modes, selected by `--accelerator` flag and `--refine`:
1. **ML Accelerator** (`--accelerator path.pkl`): GBRT prediction, 0.046 ms/cell batch.
   Predicts equilibrium abundances in one shot from (n, T, G₀, A_V, ζ, f_sh).
   Conservation enforced in `predict_batch()`. Requires a trained `.pkl` file
   (train with `make container-train`).
2. **Explicit Euler** (default, no accelerator): 200 subcycled substeps per inner
   iteration. Adaptive per-cell timestep with 10% rate limiter. Enforces H and C
   conservation after each substep. Convergence check every 20 substeps for early exit.
3. **BDF** (`--refine`): `scipy.integrate.solve_ivp` with BDF, 500 ms/cell.
   Post-convergence refinement pass for maximum accuracy.

With ML, the thermal solver (vectorized bisection) becomes the bottleneck.

### Save/Load
`solver.save('model.npz')` and `PDRSolver3D.load('model.npz')` serialize all
state arrays including dust state. The npz includes metadata (box size, G₀, etc.).

---

## 2. THEMIS Dust Model (`grains/themis.py`, 799 lines)

### Three Populations

| Population | Size Range | Composition | Role |
|------------|-----------|-------------|------|
| a-C(:H) nano | 0.4–20 nm | Hydrogenated amorphous carbon | PE heating, PAH emission |
| Large a-C(:H) | 20 nm–0.2 μm | Carbon mantles | H₂ formation, FUV extinction |
| a-Sil | 20 nm–0.3 μm | Mg-rich silicates + Fe inclusions | FUV extinction, FIR emission |

### Per-Cell State
- `f_nano`: nano-grain mass fraction relative to ISM standard (0 = fully depleted, 1 = ISM)
- `E_g`: optical band gap [eV] (0.1 = aromatic/graphitic, 2.67 = aliphatic)

### Key Methods
```python
dust = THEMISDust(f_nano=0.8, E_g=0.5)
Gamma_PE = dust.photoelectric_heating(n_H, T, G0, x_e)     # erg/s/cm³
R_H2     = dust.h2_formation_rate(n_H, T, G0)              # cm⁻³/s
sigma_d  = dust.fuv_cross_section()                         # cm² per H
P_T      = dust.stochastic_temperature_distribution(G0)     # for nano-grains
SED      = dust.dust_sed(T_dust, G0)                        # per wavelength
```

### Photoelectric Heating
Uses WD01-style formalism averaged over grain charge distribution P(Z):
1. Compute charging parameter γ = G₀ T^0.5 / (n_e × f_nano)
2. Sum over grain sizes: each size contributes PE yield × UV absorption × P(Z)
3. Recombination cooling: electrons recombining on grains remove energy

The WD01 fit: ε_PE = 0.049 / (1 + (γ/1925)^0.73) + 0.037 × (T/10⁴)^0.7 / (1 + (γ/5000))
Multiplied by f_nano for the nano-grain contribution.

### UV Evolution
Dust state evolves in the **outer loop** only (`_evolve_dust_state`):
- **Photo-destruction**: df/dt = -f / τ_destr, τ = 10⁴ yr × (10⁴/G₀)
- **Aromatization**: E_g decreases (→ 0.1 eV) under UV exposure
- **Replenishment**: accretion in dense gas, df/dt = (1-f) × n_H / n_crit

Each outer step advances by dt = 10 kyr. Between dust steps, the inner
loop converges chemistry and thermal balance with fixed f_nano/E_g.

PE heating and T_dust are **instantaneous** quantities recomputed every
inner iteration (`_update_dust_heating`) — they depend on the current
electron density and temperature, not on dust evolution timescales.

---

## 3. Chemistry (`chemistry/`, 1958 lines total)

### Network (`network.py`, 1004 lines)
75 reactions in 8 categories:
1. **Photodissociation**: H₂ + hν → 2H (with self-shielding), CO + hν → C + O
2. **Photoionization**: C + hν → C⁺ + e⁻
3. **Cosmic ray**: H₂ + CR → H₂⁺ + e⁻ → H + H⁺ + e⁻
4. **Ion-molecule**: C⁺ + OH → CO⁺ + H, HCO⁺ + e⁻ → CO + H
5. **Neutral-neutral**: O + H₂ → OH + H, C + H₂ → CH + H
6. **Radiative association**: C⁺ + H₂ → CH₂⁺ + hν
7. **Grain-surface**: H + H + grain → H₂ + grain (THEMIS-dependent rate)
8. **Recombination**: C⁺ + e⁻ + grain → C + grain

Species: H, H⁺, H₂, H₂⁺, He, He⁺, C, C⁺, CO, O, O⁺, OH, H₂O, CH, CH⁺,
CH₂, CH₂⁺, HCO⁺, e⁻, plus simplified intermediates.

### ODE Solver (`solver.py`, 388 lines)
Two modes:
- **Explicit Euler**: Fixed timestep (10⁶ s), 100 sub-steps. Fast but inaccurate
  for stiff systems. Initializes at approximate equilibrium.
- **BDF**: `scipy.integrate.solve_ivp` with `method='BDF'`, Jacobian from `network.py`.
  Accurate to 10⁻⁶ relative tolerance. 500 ms/cell.

Conservation is enforced as post-processing:
- H conservation: n(H) + 2n(H₂) = n_H
- C conservation: n(C⁺) + n(C) + n(CO) = 1.4×10⁻⁴ × n_H
- Charge: n(e⁻) = Σ n(ions)

### ML Accelerator (`accelerator.py`, 566 lines)
Gradient Boosted Regression Trees (GBRT from scikit-learn):
- **Input features** (7): log(n_H), log(T), log(G₀), A_V, log(ζ_CR), log(f_shield_H₂), log(f_shield_CO)
- **Output** (9): log abundances of H, H₂, C⁺, C, CO, O, e⁻, OH, HCO⁺
- **One model per species** (9 GBRT ensembles, 200 trees each, depth 6)
- **Batch prediction**: vectorized across all cells, 0.046 ms/cell
- **Training**: `train_hpc.py` generates parameter-space samples, solves with BDF,
  trains GBRT. 50,000 samples recommended (4 hr on 10 cores).

**Current limitation**: trained on Euler data (not BDF), so CO is systematically
underestimated at AV 2-4. Retraining with BDF data is the #1 priority.

---

## 4. Radiative Transfer (`radiative_transfer/`, 325 lines)

### 3D FUV RT (`fuv_rt_3d.py`, 198 lines)
Multi-ray RT using HEALPix angular discretization:
1. Generate `12 × nside²` ray directions from HEALPix
2. For each ray, march through the 3D grid accumulating optical depth:
   `τ += n_H × σ_dust × ds` where σ_dust comes from THEMIS
3. Attenuate: `G₀(cell) = G₀_external × Σ_rays (1/N_rays) × exp(-τ_ray)`
4. Compute A_V = τ / 1.086 for each cell

The illumination comes from the x=0 face (external UV source).
Internal sources (e.g. embedded stars) are not yet implemented.

### Shielding (`shielding.py`)
- **H₂ self-shielding**: Draine & Bertoldi (1996) fitting function, depends on N(H₂) column
- **CO self-shielding**: Visser+ (2009) table interpolation, depends on N(CO) and N(H₂)
- Applied per-ray during the RT sweep

---

## 5. Thermal Balance (`thermal/`, 822 lines)

### Heating Processes (`heating.py`, 296 lines)
1. **Photoelectric** (PE): dominant at AV < 3. From THEMIS `Gamma_PE` (includes grain charge distribution)
2. **Cosmic ray**: Γ_CR = ζ × n_H × 20 eV per ionization
3. **H₂ formation**: 4.48 eV per H₂ formed on grain surfaces
4. **H₂ UV pumping**: fluorescent cascade deposits ~2 eV per pumping event
5. **C photoionization**: excess energy from C → C⁺ + e⁻
6. **Gas-grain collisional**: warm gas heats cold dust (or vice versa)
7. **Turbulent dissipation**: Γ_turb ∝ ρ × v_turb³ / L (small)

### Cooling Processes (`cooling.py`, 363 lines)
1. **[CII] 158 μm**: ²P₃/₂ → ²P₁/₂, A = 2.3×10⁻⁶ s⁻¹. Dominant coolant in atomic PDR.
2. **[OI] 63 μm**: ³P₁ → ³P₂, A = 8.9×10⁻⁵ s⁻¹. Important at high T.
3. **[OI] 145 μm**: ³P₀ → ³P₁
4. **CO rotational**: J-ladder from J=1-0 to J=10-9 with escape probability β(τ)
5. **[CI] 609 μm**: ³P₁ → ³P₀
6. **H₂ rovibrational**: quadrupole lines, important above ~500 K
7. **Gas-grain**: Λ_gd ∝ n_H² × T^0.5 × (T - T_dust)
8. **Lyα**: hydrogen recombination (minor in PDRs)

### Solver (`balance.py`, 163 lines)
Brent's method root-finding: solve Γ(T) = Λ(T) for each cell independently.
Bracket: [10 K, 10⁵ K]. Tolerance: 0.01 K.

**Performance note**: this is currently Python loops over cells (not vectorized).
It's ~19% of total runtime with ML chemistry. Vectorizing Brent or switching to
a Newton method with analytical Jacobian would help.

---

## 6. Synthetic Observations (`observations/jwst_pipeline.py`, 657 lines)

### Line Emission Maps
For each emission line:
1. Compute emissivity j = n_upper × A_ul × hν / (4π) per cell
2. Compute absorption κ from lower/upper level populations (LTE)
3. Integrate along LOS with escape probability: I += j × ds × β(τ)
4. β(τ) = (1 - exp(-τ)) / τ smoothly transitions thin → thick

### PPV Spectral Cubes (`spectra.py`, 500 lines)
Full formal solution per velocity channel:
1. Line profile φ(v) = Gaussian(σ_thermal + σ_turbulent) per cell
2. At each LOS cell: `I_v = I_v × exp(-dτ) + S_v × (1 - exp(-dτ))`
3. Source function S_v = j × φ_v / (κ × φ_Hz) (in velocity units)
4. Produces flat-topped profiles for optically thick lines ([OI], low-J CO)

**Consistency**: map (escape probability) and PPV (formal solution) agree to
0.8–1.0 for all lines. Remaining ~20% for thick lines is the real difference
between the two standard RT methods.

### Beam Catalog
30 beam sizes for all instruments (in arcsec FWHM):

| Instrument | Beam | Examples |
|------------|------|---------|
| JWST NIRSpec | 0.10–0.13″ | H₂ 1-0 S(1), Brα |
| JWST MIRI MRS | 0.20–0.55″ | H₂ 0-0 lines, PAH bands |
| ALMA Band 6/7 | 0.3–2.0″ | CO, HCO⁺ |
| Herschel PACS | 4.5–11.4″ | [OI] 63, [CII] 158, 70/160 μm |
| Herschel SPIRE | 18–36″ | 250/350/500 μm |

### Dust Continuum
Modified blackbody with THEMIS opacity:
`I_ν = n_H × m_H × (D/G) × κ_ν × B_ν(T_dust) × ds`
β = 1.8, κ₀ = 10 cm²/g at 250 μm.

---

## 7. Observation Comparison (`observations/from_observations.py`, 630 lines)

### Hybrid Beam Matching
At comparison time, whichever is finer gets convolved:
```
obs finer than model → convolve OBSERVATION to model beam
obs coarser than model → convolve MODEL to observation beam
```

Kernel: σ²_kernel = σ²_target - σ²_native (quadrature difference).

### Metrics
- χ² reduced (10% calibration uncertainty assumed)
- Log-space Pearson correlation
- Median scale factor (systematic offset)
- Resolution status: 'obs_convolved_to_model' or 'model_convolved_to_obs'

### Data Preparation (`data/prepare_data.py`, 682 lines)
Reads FITS cubes from MAST (JWST) and ESASky (Herschel), reprojects to
the model grid via WCS → pixel coordinate mapping, extracts emission lines
from spectral cubes, derives T_dust/N_H from continuum SED fitting.

---

## 8. Evaluation Pipeline (`evaluate.py`, 568 lines)

Automatically produces 7 figure types:
1. **Summary dashboard**: 8-panel overview (density, T, G₀, chemistry, dust)
2. **Midplane slices** (×3): XY, XZ, YZ at midplane, 8 quantities each
3. **Depth profiles**: 1D cuts along illumination axis (Röllig-style)
4. **Dust evolution**: f_nano vs G₀, E_g distribution, PE heating efficiency
5. **Synthetic observations**: 20-panel JWST/Herschel/ALMA maps
6. **Viewer** (×2): Canvas2D (offline) + Three.js (WebGL with isosurfaces)
7. **HTML report**: all figures + validation statistics

5 automatic physics checks:
- H conservation (error < 5%)
- C conservation (error < 25% — known offset, see CLAUDE.md)
- Temperature physicality (10 K < T < 10⁵ K)
- Dust evolution occurred (f_nano range > 0.01)
- C⁺/CO anticorrelation (Pearson r < -0.3)

---

## 9. Viewer (`viewer_export.py`, 340 lines)

Two backends:
- **Canvas2D** (`mode='canvas'`): Sorts cells back-to-front, draws colored rectangles
  with alpha blending. No dependencies. Works offline. Drag to rotate/tilt.
- **Three.js** (`mode='threejs'`): WebGL point cloud with additive blending, orbit controls,
  zoom, pan. Isosurface mode with threshold-based surface extraction and Phong shading.
  Requires CDN (three.min.js r128).

Both embed model data as JSON in the HTML. Typical size: 580 KB for 16³.

---

## 10. HPC Infrastructure

### Apptainer (`apptainer.def`)
Python 3.11 slim base, installs numpy/scipy/matplotlib/scikit-learn/astropy/h5py.
Entry point: `python -m prism3d.run "$@"`.

### SLURM Generation (`hpc/runner.py` + `run.py --gen-slurm`)
Generates job scripts with correct resource requests for the grid size.
Auto-detects GPU availability and uses CUDA kernel if present.

### Makefile
15 targets covering the full workflow:
`install → test → run → train → data → container → slurm → clean`

---

## Numerical Choices and Why

| Choice | Reason |
|--------|--------|
| Two-level iteration (dust outer, chem+thermal inner) | Respects physical timescales; single-loop mixing prevented convergence |
| Analytical IC from A_V/G₀ | Seeds near-equilibrium abundances; uniform IC left chemistry flat |
| H + C conservation enforcement per substep | `np.maximum` clipping breaks conservation; explicit rescaling required |
| Adaptive per-cell damping w ∈ [0.5, 0.98] | Bistable front cells need heavy damping; well-behaved cells need light |
| Selective ML abundance damping (w > 0.55 only) | Undamped ML abundances are correct; only oscillating cells need blending |
| p95 for T convergence, p99 for chemistry | ~5% of cells at dissociation front are genuinely bistable under operator splitting |
| Denominator floors (H₂: 0.01, CO: 1e-6) | Prevents near-zero cells from inflating relative change metric |
| Explicit Euler for chemistry default | 10× faster than BDF, acceptable with good IC and 200 substeps |
| ML accelerator as primary | 10⁴× faster, makes 3D feasible on laptops |
| Escape probability for line maps | Standard in PDR codes, O(N) per cell, handles optically thick lines |
| Formal solution for PPV cubes | Exact per-channel, captures self-absorption profile shapes |
| HEALPix for ray directions | Uniform angular sampling, easy to increase resolution |
| GBRT over neural nets | Faster training, interpretable, stable extrapolation |
| Brent for thermal balance | Guaranteed convergence, no Jacobian needed |
| JSON+HTML for viewer | No server, works offline, browser-native |
