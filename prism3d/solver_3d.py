"""
3D PDR Solver for PRISM-3D.

Operates on regular 3D grids (numpy arrays) for maximum performance.
This is the main entry point for 3D PDR calculations.

Workflow:
  1. Set up 3D density field
  2. Multi-ray FUV RT → G0(x,y,z), AV, column densities
  3. Self-shielding factors
  4. CR attenuation
  5. Chemistry at every cell (vectorized where possible)
  6. Thermal balance at every cell
  7. Iterate to convergence

All physics steps are vectorized over the full 3D grid using numpy.
Per-cell Python loops are only used for the BDF refinement pass.
"""

import numpy as np
import time
from ._rust_backend import has_rust, get_rust
from .radiative_transfer.fuv_rt_3d import compute_fuv_field_3d
from .radiative_transfer.shielding import f_shield_H2, f_shield_CO
from .chemistry.solver import ChemistrySolver
from .thermal.balance import ThermalSolver
from .cosmic_rays.attenuation import cr_ionization_rate
from .grains.themis import THEMISDust
from .grains.themis_tables import THEMISTables
from .utils.constants import (k_boltz, eV_to_erg, m_H, h_planck, c_light,
                               fine_structure_lines, gas_phase_abundances)


class PDRSolver3D:
    """
    3D PDR solver on a regular Cartesian grid.

    Parameters
    ----------
    density : ndarray (nx, ny, nz)
        Hydrogen number density [cm^-3].
    box_size : float
        Physical box size [cm].
    G0_external : float or ndarray (n_rays,)
        External FUV field [Habing units].  If scalar, isotropic.
        If array, per-ray directional illumination (use ``directional_G0``).
    zeta_CR_0 : float
        Surface cosmic ray ionization rate [s^-1].
    nside_rt : int
        HEALPix resolution for FUV ray tracing.
    fixed_T : float, optional
        If set, use fixed gas temperature (for isothermal benchmarks).
    """

    def __init__(self, density, box_size, G0_external=1.0,
                 zeta_CR_0=2e-16, nside_rt=1, fixed_T=None,
                 chem_network=None, tile_size=64, velocity_field=None,
                 star=None):
        self.density = np.asarray(density, dtype=np.float64)
        self.nx, self.ny, self.nz = self.density.shape
        self.n_cells = self.nx * self.ny * self.nz
        self.box_size = box_size
        self.cell_size = box_size / self.nx  # Assumes cubic cells
        self.G0_external = G0_external
        self.zeta_CR_0 = zeta_CR_0
        self.nside_rt = nside_rt
        self.fixed_T = fixed_T
        self.tile_size = tile_size  # For tiled thermal solver
        self.velocity_field = np.asarray(velocity_field, dtype=np.float64) if velocity_field is not None else None
        # Point source: dict with 'position' [cm] and 'L_FUV' [erg/s]
        self.star = star

        # Initialize state arrays
        self._init_state()

        # THEMIS lookup tables (precompute once, ~0.5 s)
        self._themis_ref = THEMISDust(f_nano=1.0, E_g=0.1)
        self.themis_tables = THEMISTables(self._themis_ref)

        # Log backend
        if has_rust():
            core = get_rust()
            print(f"  Rust core v{core.version()}: thermal + chemistry + PE (rayon parallel)")
        else:
            print("  Backend: Python/NumPy (Rust core not available)")

        # Solvers (kept for BDF refinement pass)
        self.chem_solver = ChemistrySolver(network=chem_network)
        self.thermal_solver = ThermalSolver()

        # Diagnostics
        self.timing = {}
        self.convergence_history = []

    def _init_state(self):
        """Initialize all 3D state arrays.

        Uses float32 for state arrays to halve memory at 1024³
        (4.3 GB vs 8.6 GB per array).  density stays float64 for RT.
        Chemistry conservation uses float64 intermediates internally.
        """
        shape = (self.nx, self.ny, self.nz)
        # Use float32 for large grids (> 128³), float64 for small
        self._use_f32 = self.n_cells > 128**3
        dt = np.float32 if self._use_f32 else np.float64

        # Radiation field (scalar estimate for initialisation; overwritten by RT)
        G0_init = float(np.max(self.G0_external)) if not np.isscalar(self.G0_external) else self.G0_external
        self.G0 = np.full(shape, G0_init, dtype=dt)
        self.A_V = np.zeros(shape, dtype=dt)
        self.N_H = np.zeros(shape, dtype=dt)
        self.N_H2 = np.zeros(shape, dtype=dt)
        self.N_CO = np.zeros(shape, dtype=dt)
        self.zeta_CR = np.full(shape, self.zeta_CR_0, dtype=dt)

        # Temperature
        if self.fixed_T is not None:
            self.T_gas = np.full(shape, self.fixed_T, dtype=dt)
        else:
            self.T_gas = np.full(shape, 50.0, dtype=dt)
        self.T_dust = np.full(shape, 20.0, dtype=dt)

        # Abundances (relative to n_H)
        x_C = gas_phase_abundances.get('C', 1.4e-4)
        x_O = gas_phase_abundances.get('O', 3.0e-4)

        self.x_HI = np.full(shape, 0.5, dtype=dt)
        self.x_H2 = np.full(shape, 0.25, dtype=dt)
        self.x_Cp = np.full(shape, x_C * 0.5, dtype=dt)
        self.x_C = np.full(shape, x_C * 0.1, dtype=dt)
        self.x_CO = np.full(shape, x_C * 0.4, dtype=dt)
        self.x_O = np.full(shape, x_O - x_C * 0.4, dtype=dt)
        self.x_e = np.full(shape, x_C * 0.5, dtype=dt)
        self.x_OH = np.full(shape, 1e-8, dtype=dt)
        self.x_H2O = np.full(shape, 1e-8, dtype=dt)
        self.x_HCOp = np.full(shape, 1e-10, dtype=dt)

        # Heating/cooling diagnostics
        self.Gamma = np.zeros(shape, dtype=dt)
        self.Lambda = np.zeros(shape, dtype=dt)

        # ML accelerator (set externally before calling run())
        self.accelerator = None

        # THEMIS dust state (per cell)
        self.f_nano = np.ones(shape, dtype=dt)
        self.E_g = np.full(shape, 0.1, dtype=dt)
        self.Gamma_PE = np.zeros(shape, dtype=dt)
        self._R_H2 = np.full(shape, 3e-17, dtype=dt)

    # ================================================================
    # AV/G0-dependent initial conditions
    # ================================================================

    def _set_analytical_ic(self):
        """Set position-dependent initial abundances from local A_V and G0.

        Called once after the first RT step so that A_V and G0 fields
        are available.  Uses tanh-profile approximations for the H/H2
        and C+/C/CO transitions, giving the iterative solver a head
        start close to equilibrium.
        """
        x_C = gas_phase_abundances.get('C', 1.4e-4)
        x_O = gas_phase_abundances.get('O', 3.0e-4)
        av = self.A_V
        g0 = self.G0

        # H/H2 transition: shifts to higher AV for stronger radiation
        av_H2 = 1.5 + 0.3 * np.log10(np.maximum(g0, 0.1))
        f_H2 = 0.5 * (1.0 + np.tanh((av - av_H2) / 0.8))
        self.x_H2 = np.clip(f_H2 * 0.5, 1e-6, 0.5)
        self.x_HI = np.clip(1.0 - 2.0 * self.x_H2, 1e-6, 1.0)

        # C+/C/CO transition
        av_CO = 3.0 + 0.2 * np.log10(np.maximum(g0, 0.1))
        f_CO = 0.5 * (1.0 + np.tanh((av - av_CO) / 1.0))
        self.x_CO = np.clip(x_C * f_CO * 0.95, 1e-10, x_C)
        self.x_C = np.clip(x_C * f_CO * 0.05, 1e-10, x_C)
        self.x_Cp = np.clip(x_C * (1.0 - f_CO), 1e-10, x_C)

        # Carbon conservation
        C_sum = self.x_Cp + self.x_C + self.x_CO
        scale = np.where(C_sum > 0, x_C / C_sum, 1.0)
        self.x_Cp *= scale
        self.x_C *= scale
        self.x_CO *= scale

        # Derived species
        self.x_O = np.maximum(x_O - self.x_CO, 1e-10)
        self.x_e = np.maximum(self.x_Cp + self.x_HCOp, 1e-10)
        self.x_OH = 1e-9 + 9e-8 * np.clip(f_H2, 0, 1)

    # ================================================================
    # Main solver loop
    # ================================================================

    def run(self, max_iterations=20, convergence_tol=0.05,
            dust_steps=5, verbose=True):
        """
        Run the 3D PDR model to convergence.

        Two-level iteration:
          Outer: dust evolution steps (slow timescale, 10⁴–10⁶ yr)
          Inner: RT → shielding → CR → chemistry → thermal (fast,
                 iterates to steady state for fixed dust properties)

        Parameters
        ----------
        max_iterations : int
            Max inner iterations per dust step.
        convergence_tol : float
            Relative convergence threshold for inner loop.
        dust_steps : int
            Number of outer dust evolution steps.
        """
        if verbose:
            print(f"PRISM-3D 3D Solver")
            print(f"Grid: {self.nx}x{self.ny}x{self.nz} = {self.n_cells} cells")
            print(f"G0 = {self.G0_external}, zeta_CR = {self.zeta_CR_0:.1e}")
            print(f"RT: nside={self.nside_rt} ({12*self.nside_rt**2} rays)")
            if self.fixed_T:
                print(f"T = {self.fixed_T} K (fixed)")
            if self.accelerator is not None:
                print(f"Chemistry: ML accelerator")
            else:
                print(f"Chemistry: explicit Euler")
            print(f"Dust steps: {dust_steps}, "
                  f"max inner iterations: {max_iterations}")
            if self._use_f32:
                gb = self.n_cells * 4 / 1e9
                print(f"State dtype: float32 ({gb:.1f} GB/array, "
                      f"~{gb*24:.0f} GB total)")
            print(f"{'='*60}")

        t_start = time.time()
        global_iter = 0

        for dust_step in range(dust_steps):
            # ── Outer: evolve dust ──────────────────────────
            if dust_step > 0:
                t0 = time.time()
                old_f_nano = self.f_nano.copy()
                self._evolve_dust_state()
                df_nano = np.max(np.abs(self.f_nano - old_f_nano) /
                                 np.maximum(old_f_nano, 1e-10))
                self.timing['dust_evolve'] = time.time() - t0
                if verbose:
                    print(f"\n── Dust step {dust_step}/{dust_steps} "
                          f"| f_nano={np.mean(self.f_nano):.3f} "
                          f"| df_nano={df_nano:.2e} ──")
            else:
                if verbose:
                    print(f"\n── Dust step 0/{dust_steps} "
                          f"| f_nano={np.mean(self.f_nano):.3f} "
                          f"(initial) ──")

            # ── Inner: converge RT + chemistry + thermal ────
            converged = False
            # Per-cell damping weight: increases for oscillating cells
            w_damp = np.full(self.density.shape, 0.5)

            for iteration in range(max_iterations):
                t_iter = time.time()

                old_T = self.T_gas.copy()
                old_x_H2 = self.x_H2.copy()
                old_x_CO = self.x_CO.copy()

                # Step 1: FUV Radiative Transfer
                t0 = time.time()
                if self.star is not None:
                    from .radiative_transfer.fuv_rt_3d import compute_fuv_field_point_source
                    G0, A_V, N_H, N_H2, N_CO = compute_fuv_field_point_source(
                        self.density, self.star['position'],
                        self.star['L_FUV'], self.cell_size,
                        x_H2=self.x_H2, x_CO=self.x_CO,
                    )
                else:
                    G0, A_V, N_H, N_H2, N_CO = compute_fuv_field_3d(
                        self.density, self.G0_external, self.cell_size,
                        x_H2=self.x_H2, x_CO=self.x_CO,
                        nside=self.nside_rt
                    )
                # Cast RT outputs to state dtype (float32 for large grids)
                if self._use_f32:
                    self.G0 = G0.astype(np.float32)
                    self.A_V = A_V.astype(np.float32)
                    self.N_H = N_H.astype(np.float32)
                    self.N_H2 = N_H2.astype(np.float32)
                    self.N_CO = N_CO.astype(np.float32)
                else:
                    self.G0, self.A_V = G0, A_V
                    self.N_H, self.N_H2, self.N_CO = N_H, N_H2, N_CO
                self.timing['RT'] = time.time() - t0

                # On very first iteration, seed from analytical profiles
                if dust_step == 0 and iteration == 0:
                    self._set_analytical_ic()

                # Step 2: Self-shielding
                t0 = time.time()
                f_sh_H2 = f_shield_H2(self.N_H2, T=np.mean(self.T_gas))
                f_sh_CO = f_shield_CO(self.N_CO, self.N_H2)
                self.timing['shielding'] = time.time() - t0

                # Step 3: CR attenuation
                t0 = time.time()
                self.zeta_CR = cr_ionization_rate(self.N_H, model='M',
                                                   zeta_0=self.zeta_CR_0)
                self.timing['CR'] = time.time() - t0

                # Update PE heating and T_dust for current dust state
                t0 = time.time()
                self._update_dust_heating()
                self.timing['dust'] = time.time() - t0

                # Step 4: Chemistry
                t0 = time.time()
                if self.accelerator is not None:
                    from .chemistry.accelerator import solve_chemistry_ml
                    n_chem_conv = solve_chemistry_ml(
                        self, self.accelerator, f_sh_H2, f_sh_CO)
                else:
                    n_chem_conv = self._solve_chemistry_vec(f_sh_H2, f_sh_CO)
                self.timing['chemistry'] = time.time() - t0

                # Damping: Euler damps all abundances uniformly;
                # ML damps only oscillating cells (using per-cell w_damp)
                if iteration > 0:
                    if self.accelerator is not None:
                        # ML mode: selectively damp oscillating cells
                        osc = w_damp > 0.55
                        if np.any(osc):
                            w = w_damp[osc]
                            self.x_H2[osc] = (w * old_x_H2[osc] +
                                               (1 - w) * self.x_H2[osc])
                            self.x_HI[osc] = np.maximum(
                                1.0 - 2.0 * self.x_H2[osc], 1e-30)
                            self.x_CO[osc] = (w * old_x_CO[osc] +
                                              (1 - w) * self.x_CO[osc])
                            x_C_total = gas_phase_abundances.get(
                                'C', 1.4e-4)
                            self.x_Cp[osc] = np.maximum(
                                x_C_total - self.x_CO[osc] -
                                self.x_C[osc], 1e-30)
                    else:
                        # Euler mode: damp all abundances uniformly
                        self.x_H2 = 0.5 * old_x_H2 + 0.5 * self.x_H2
                        self.x_HI = np.maximum(
                            1.0 - 2.0 * self.x_H2, 1e-30)
                        self.x_CO = 0.5 * old_x_CO + 0.5 * self.x_CO
                        x_C_total = gas_phase_abundances.get('C', 1.4e-4)
                        self.x_Cp = np.maximum(
                            x_C_total - self.x_CO - self.x_C, 1e-30)

                # Step 5: Thermal balance with adaptive per-cell damping
                t0 = time.time()
                if self.fixed_T is None:
                    T_new = self.T_gas.copy()
                    self._solve_thermal_vec(f_sh_H2)
                    if iteration > 0:
                        # Adaptive damping: cells with large relative T
                        # change get damped harder (up to 0.95)
                        rel_dT = np.abs(self.T_gas - old_T) / \
                            np.maximum(old_T, 10.0)
                        # Increase damping for oscillating cells
                        # Ramp fast (+0.15) so persistent oscillators
                        # are frozen within 3 iterations
                        w_damp = np.clip(
                            np.where(rel_dT > 0.5, w_damp + 0.15,
                                     np.where(rel_dT < 0.1,
                                              w_damp - 0.02, w_damp)),
                            0.5, 0.98)
                        self.T_gas = w_damp * old_T + \
                            (1.0 - w_damp) * self.T_gas
                self.timing['thermal'] = time.time() - t0

                # Convergence: use 99th percentile, not max
                # (a handful of bistable transition cells shouldn't
                # block convergence of the other 32,000+ cells)
                mask_T = old_T > 10
                mask_H2 = old_x_H2 > 1e-6
                mask_CO = old_x_CO > 1e-8

                rel_T = (np.abs(self.T_gas[mask_T] - old_T[mask_T])
                         / old_T[mask_T]) if np.any(mask_T) \
                    else np.array([0.0])
                # H2 floor=0.01: cells below x_H2=0.01 don't
                # inflate rel change (x_H2 ranges 0–0.5)
                rel_H2 = (np.abs(self.x_H2[mask_H2] - old_x_H2[mask_H2])
                          / np.maximum(old_x_H2[mask_H2], 0.01)) \
                    if np.any(mask_H2) else np.array([0.0])
                # CO floor=1e-6: avoids blow-up at low CO
                rel_CO = (np.abs(self.x_CO[mask_CO] - old_x_CO[mask_CO])
                          / np.maximum(old_x_CO[mask_CO], 1e-6)) \
                    if np.any(mask_CO) else np.array([0.0])

                # p99 for logging, p95 for convergence check.
                # The ~5% of cells at the dissociation front are
                # genuinely bistable under operator splitting —
                # they shouldn't block convergence of the other 95%.
                dT_p99 = float(np.percentile(rel_T, 99))
                dT = float(np.percentile(rel_T, 95))
                dH2 = float(np.percentile(rel_H2, 99))
                dCO = float(np.percentile(rel_CO, 99))
                delta_max = max(dT, dH2, dCO)

                # Also track max for diagnostics
                dT_max = float(np.max(rel_T))
                dH2_max = float(np.max(rel_H2))
                n_hot = int(np.sum(rel_T > convergence_tol))

                dt_iter = time.time() - t_iter

                if verbose:
                    print(f"  {dust_step}.{iteration:2d} | "
                          f"dT={dT:.2e}(p99={dT_p99:.2e}) "
                          f"dH2={dH2:.2e} dCO={dCO:.2e} | "
                          f"max(dT)={dT_max:.1e} n_osc={n_hot} | "
                          f"t={dt_iter:.1f}s "
                          f"[RT:{self.timing.get('RT',0):.1f} "
                          f"chem:{self.timing.get('chemistry',0):.1f} "
                          f"therm:{self.timing.get('thermal',0):.1f}]")

                self.convergence_history.append({
                    'dust_step': dust_step, 'iteration': iteration,
                    'dT': dT, 'dT_p99': dT_p99,
                    'dH2': dH2, 'dCO': dCO,
                    'dT_max': dT_max, 'dH2_max': dH2_max,
                    'n_oscillating': n_hot,
                    'delta_max': delta_max,
                    'f_nano_mean': float(np.mean(self.f_nano)),
                })
                global_iter += 1

                if delta_max < convergence_tol and iteration > 2:
                    converged = True
                    if verbose:
                        print(f"  → Converged after {iteration+1} "
                              f"iterations ({n_hot} cells still "
                              f"oscillating)")
                    break

            if not converged and verbose:
                print(f"  → Not converged after {max_iterations} "
                      f"iterations (p99={delta_max:.2e}, "
                      f"max={dT_max:.2e}, {n_hot} cells)")

        total = time.time() - t_start
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Finished: {dust_steps} dust steps, "
                  f"{global_iter} total iterations, {total:.1f}s")
            print(f"  f_nano: {np.mean(self.f_nano):.3f} "
                  f"(range {np.min(self.f_nano):.3f}–"
                  f"{np.max(self.f_nano):.3f})")
            self._print_timing()

        return converged

    # ================================================================
    # Vectorized dust evolution
    # ================================================================

    def _evolve_dust_state(self):
        """Time-evolve THEMIS grain state (f_nano, E_g).

        This is the slow process (10⁴–10⁶ yr timescale).  Called once
        per outer dust step, NOT every inner iteration.
        """
        dt_dust = 1e4 * 3.15e7  # 10 kyr per step

        G0 = self.G0
        nH = self.density
        f_n = self.f_nano
        eg = self.E_g

        # Photo-destruction timescale
        tau_dest = 1e4 * 3.15e7 / np.maximum(G0 / 1e4, 0.01)
        f_dest = np.exp(-dt_dust / tau_dest)

        # Replenishment in shielded gas
        f_replenish = np.minimum(dt_dust / (1e6 * 3.15e7), 0.1)

        # Accretion in dense, shielded gas
        f_accrete = np.where(
            (G0 < 1) & (nH > 1e4),
            np.minimum(dt_dust / (1e5 * 3.15e7 * 1e4 / nH), 0.1),
            0.0
        )

        self.f_nano = np.clip(
            f_n * f_dest + (1.0 - f_n) * f_replenish + f_accrete,
            0.01, 2.0
        )

        # Band gap evolution
        tau_aromat = 1e3 * 3.15e7 / np.maximum(G0, 0.01)
        Eg_eq = np.where(
            G0 > 10,
            0.01,
            np.minimum(0.1 + 0.5 * np.log10(np.maximum(nH, 1.0)), 2.5)
        )
        self.E_g = np.clip(
            eg + (Eg_eq - eg) * np.minimum(dt_dust / tau_aromat, 1.0),
            0.01, 2.5
        )

    def _update_dust_heating(self):
        """Compute PE heating rate and T_dust for current grain state.

        Uses THEMIS lookup tables for proper charge-distribution PE
        heating (replaces BT94 approximation).  Instantaneous — no
        time evolution.  Called every inner iteration.
        """
        dt = self.T_gas.dtype
        self.Gamma_PE = self.themis_tables.pe_heating_vec(
            self.G0, self.T_gas, self.x_e, self.density,
            self.f_nano, self.E_g
        ).astype(dt)

        # THEMIS-calibrated dust temperature
        self.T_dust = self.themis_tables.dust_temperature_vec(
            self.G0, self.A_V, nH=self.density, T_gas=self.T_gas
        ).astype(dt)

        # Precompute THEMIS H2 formation rate for use in chemistry + thermal
        self._R_H2 = self.themis_tables.h2_formation_rate_vec(
            self.T_gas, self.T_dust, self.f_nano
        ).astype(dt)

    # ================================================================
    # Vectorized chemistry (explicit Euler on full grid)
    # ================================================================

    def _solve_chemistry_vec(self, f_sh_H2, f_sh_CO, n_substeps=200,
                              conv_tol=1e-4):
        """Vectorized explicit Euler chemistry on the full 3D grid.

        Uses Rust backend when available for parallel per-cell integration.
        """
        nH = self.density
        T = self.T_gas
        zeta = self.zeta_CR

        x_C_total = gas_phase_abundances.get('C', 1.4e-4)
        x_O_total = gas_phase_abundances.get('O', 3.0e-4)

        # Precompute T-independent rate coefficients (constant over substeps)
        k_pd_base = 4.43e-11 * self.G0 * np.exp(-3.74 * self.A_V) * f_sh_H2
        k_ci = 2.56e-10 * self.G0 * np.exp(-3.02 * self.A_V)
        alpha_Cp = 4.67e-12 * np.power(T / 300.0, -0.6)
        k_co = 1.71e-10 * self.G0 * np.exp(-3.53 * self.A_V) * f_sh_CO
        k_co_cr = 6.0 * zeta
        R_cr_base = 2.5 * zeta * 0.1
        k_co_total = k_co + k_co_cr
        R_H2_half_nH = self._R_H2 * nH * 0.5

        # ── Rust fast path ──
        if has_rust() and self.accelerator is None:
            core = get_rust()
            to_f64_flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()
            shape = self.T_gas.shape
            dt_state = self.T_gas.dtype
            # Prepare state as f64 flat arrays
            x_hi = to_f64_flat(self.x_HI)
            x_h2 = to_f64_flat(self.x_H2)
            x_cp = to_f64_flat(self.x_Cp)
            x_c = to_f64_flat(self.x_C)
            x_co = to_f64_flat(self.x_CO)
            x_o = to_f64_flat(self.x_O)
            x_e = to_f64_flat(self.x_e)
            core.solve_chemistry_euler(
                to_f64_flat(nH),
                to_f64_flat(k_pd_base),
                to_f64_flat(k_ci),
                to_f64_flat(alpha_Cp),
                to_f64_flat(k_co_total),
                to_f64_flat(R_H2_half_nH),
                to_f64_flat(R_cr_base),
                x_C_total,
                x_hi, x_h2, x_cp, x_c, x_co, x_o, x_e,
                to_f64_flat(self.x_HCOp),
                n_substeps, conv_tol,
            )
            self.x_HI = x_hi.reshape(shape).astype(dt_state)
            self.x_H2 = x_h2.reshape(shape).astype(dt_state)
            self.x_Cp = x_cp.reshape(shape).astype(dt_state)
            self.x_C = x_c.reshape(shape).astype(dt_state)
            self.x_CO = x_co.reshape(shape).astype(dt_state)
            self.x_O = x_o.reshape(shape).astype(dt_state)
            self.x_e = x_e.reshape(shape).astype(dt_state)
            return self.n_cells

        # ── Python fallback ──
        dt = 1e10  # Initial timestep [s]

        for step in range(n_substeps):
            xH = self.x_HI
            xH2 = self.x_H2
            xCp = self.x_Cp
            xC = self.x_C
            xCO = self.x_CO
            xe = self.x_e

            # H2 formation/destruction (THEMIS grain-surface rate)
            R_H2_form = R_H2_half_nH * xH
            R_pd = k_pd_base * xH2
            R_cr = R_cr_base * xH2

            dxH2 = R_H2_form - R_pd - R_cr
            dxH = -2.0 * dxH2

            # Carbon chemistry
            alpha_ne = alpha_Cp * xe * nH
            xOH = 1e-9 + 9e-8 * np.clip(xH2 * 10.0, 0, 1)

            dxCp = k_ci * xC - alpha_ne * xCp + k_co_total * xCO
            dxCO = 1e-10 * xC * xOH * nH - k_co_total * xCO
            dxC = -k_ci * xC + alpha_ne * xCp - 1e-10 * xC * xOH * nH

            # Adaptive dt per cell
            max_rate = np.maximum(
                np.abs(dxH2) / np.maximum(xH2, 1e-20),
                np.maximum(
                    np.abs(dxCp) / np.maximum(xCp, 1e-20),
                    np.abs(dxCO) / np.maximum(xCO, 1e-20)
                )
            )
            dt_eff = np.where(max_rate * dt > 0.1,
                              0.1 / np.maximum(max_rate, 1e-30), dt)

            self.x_HI = np.maximum(xH + dxH * dt_eff, 1e-30)
            self.x_H2 = np.maximum(xH2 + dxH2 * dt_eff, 1e-30)
            self.x_Cp = np.maximum(xCp + dxCp * dt_eff, 1e-30)
            self.x_C = np.maximum(xC + dxC * dt_eff, 1e-30)
            self.x_CO = np.maximum(xCO + dxCO * dt_eff, 1e-30)
            self.x_O = np.maximum(x_O_total - self.x_CO, 1e-30)

            # H conservation: x_H + 2*x_H2 = 1
            H_sum = self.x_HI + 2.0 * self.x_H2
            self.x_HI /= H_sum
            self.x_H2 /= H_sum

            # C conservation: rescale C species (upcast sum to float64
            # only for float32 grids where x_CO << x_Cp would lose bits)
            if self._use_f32:
                C_sum = (self.x_Cp.astype(np.float64)
                         + self.x_C.astype(np.float64)
                         + self.x_CO.astype(np.float64))
                scale = np.where(C_sum > 0, x_C_total / C_sum, 1.0)
                scale = scale.astype(self.x_Cp.dtype)
            else:
                C_sum = self.x_Cp + self.x_C + self.x_CO
                scale = np.where(C_sum > 0, x_C_total / C_sum, 1.0)
            self.x_Cp *= scale
            self.x_C *= scale
            self.x_CO *= scale

            # Electron balance
            self.x_e = np.maximum(self.x_Cp + self.x_HCOp, 1e-30)

            # Convergence check every 20 steps
            if step > 0 and step % 20 == 0:
                rel_H2 = np.max(np.abs(self.x_H2 - xH2) /
                                np.maximum(self.x_H2, 1e-20))
                rel_CO = np.max(np.abs(self.x_CO - xCO) /
                                np.maximum(self.x_CO, 1e-20))
                if max(rel_H2, rel_CO) < conv_tol:
                    break

            dt = min(dt * 3, 1e16)

        return self.n_cells

    # ================================================================
    # Vectorized thermal balance (bisection on full grid)
    # ================================================================

    def _solve_thermal_vec(self, f_sh_H2):
        """Vectorized hybrid bisection-Newton thermal equilibrium solver.

        When Rust backend is available, uses rayon for parallel per-cell
        solving (no tiling needed). Otherwise, for large grids (> 256³),
        uses tiled processing to keep the working set in L3 cache.
        """
        # Rust handles all grid sizes efficiently via rayon work-stealing
        if has_rust() or self.n_cells <= 256**3:
            self._solve_thermal_full(f_sh_H2)
        else:
            self._solve_thermal_tiled(f_sh_H2)

    def _prepare_thermal_ctx(self, f_sh_H2):
        """Precompute all T-independent quantities for thermal solver.

        During thermal equilibrium, only T varies. All abundances,
        density, G0, AV are fixed. Precomputing saves ~6 heating terms
        and density products from being recomputed 16× per solve.
        """
        nH = self.density

        # --- T-independent heating sum ---
        # All heating except gas-grain is T-independent
        uv_atten = self.G0 * np.exp(-3.74 * self.A_V) * f_sh_H2
        Gamma_fixed = self.Gamma_PE.copy()
        Gamma_fixed += 3.3e-11 * uv_atten * self.x_H2 * nH * (0.4 * eV_to_erg)
        Gamma_fixed += self._R_H2 * self.x_HI * nH**2 * (1.5 * eV_to_erg)
        Gamma_fixed += 3.0e-10 * uv_atten * self.x_H2 * nH * (2.0 * eV_to_erg) * (nH / (nH + 3e4))
        Gamma_fixed += self.zeta_CR * (self.x_HI + 2.0 * self.x_H2) * nH * (20.0 * eV_to_erg)
        Gamma_fixed += 3.0e-10 * self.G0 * np.exp(-3.0 * self.A_V) * self.x_C * nH * eV_to_erg

        # Gas-grain coefficient: Gamma_gg(T) = gg_coeff * sqrt(T) * (T_dust - T)
        mu = np.where(self.x_H2 > 0.5, 2.0, 1.0)
        gg_coeff = 2.0 * 0.35 * 1e-21 * nH**2 * np.sqrt(
            8.0 * k_boltz / (np.pi * mu * m_H)) * k_boltz

        # --- Cooling density products (T-independent) ---
        n_e = self.x_e * nH
        n_HI = self.x_HI * nH
        n_H2 = self.x_H2 * nH
        n_CO = self.x_CO * nH
        nCp_nH = self.x_Cp * nH
        nO_nH = self.x_O * nH
        nC_nH = self.x_C * nH

        # --- Fine-structure line constants (avoid dict lookup per eval) ---
        _, CII_f, CII_A, CII_T, CII_gu, CII_gl = fine_structure_lines['CII_158']
        _, OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl = fine_structure_lines['OI_63']
        _, OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl = fine_structure_lines['OI_145']
        CI_consts = []
        for name in ('CI_609', 'CI_370'):
            _, f, A, Tul, gu, gl = fine_structure_lines[name]
            CI_consts.append((f, A, Tul, gu, gl))

        # --- CO J-level constants (precompute scalars once) ---
        B_rot = 57.635968e9
        mu_d = 0.1098e-18
        co_J = []
        for J_u in range(1, 41):
            freq = 2.0 * B_rot * J_u
            T_u = h_planck * B_rot * J_u * (J_u + 1) / k_boltz
            A_co = (64.0 * np.pi**4 * freq**3 * mu_d**2 * J_u) / \
                   (3.0 * h_planck * c_light**3 * (2 * J_u + 1))
            g_ratio = (2 * J_u + 1) / (2 * (J_u - 1) + 1)
            J_factor = 1.0 + 0.1 * J_u
            Ahv = A_co * h_planck * freq
            co_J.append((T_u, A_co, g_ratio, J_factor, Ahv))

        return (Gamma_fixed, gg_coeff, self.T_dust,
                n_e, n_HI, n_H2, n_CO, nCp_nH, nO_nH, nC_nH, mu, nH,
                CII_f, CII_A, CII_T, CII_gu, CII_gl,
                OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl,
                OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl,
                CI_consts, co_J)

    def _solve_thermal_full(self, f_sh_H2, n_bisect=10, n_newton=3):
        """Hybrid bisection-Newton solver on full grid.

        Precomputes all T-independent quantities once, then runs
        10 bisection + 3 Newton steps using only T-dependent ops.
        Uses Rust backend when available for ~10-50x speedup.
        """
        ctx = self._prepare_thermal_ctx(f_sh_H2)

        # ── Rust fast path ──
        if has_rust():
            (Gamma_fixed, gg_coeff, T_dust_ctx,
             n_e, n_HI, n_H2, n_CO, nCp_nH, nO_nH, nC_nH, mu, nH,
             CII_f, CII_A, CII_T, CII_gu, CII_gl,
             OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl,
             OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl,
             CI_consts, co_J) = ctx
            core = get_rust()
            # Flatten to f64 1D arrays
            to_f64_flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()
            t_out = np.ascontiguousarray(self.T_gas, dtype=np.float64).ravel()
            # Pack line constants
            cii_p = np.array([CII_f, CII_A, CII_T, CII_gu, CII_gl], dtype=np.float64)
            oi63_p = np.array([OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl], dtype=np.float64)
            oi145_p = np.array([OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl], dtype=np.float64)
            ci_p = np.array([v for tup in CI_consts for v in tup], dtype=np.float64)
            core.solve_thermal_vec(
                to_f64_flat(Gamma_fixed),
                to_f64_flat(gg_coeff),
                to_f64_flat(T_dust_ctx),
                to_f64_flat(n_e),
                to_f64_flat(n_HI),
                to_f64_flat(n_H2),
                to_f64_flat(n_CO),
                to_f64_flat(nCp_nH),
                to_f64_flat(nO_nH),
                to_f64_flat(nC_nH),
                cii_p, oi63_p, oi145_p, ci_p,
                n_bisect, n_newton,
                t_out,
            )
            self.T_gas = t_out.reshape(self.T_gas.shape).astype(self.T_gas.dtype)
            self.Gamma = self._total_heating_vec(self.T_gas, f_sh_H2)
            self.Lambda = self._total_cooling_vec(self.T_gas)
            return

        # ── Python fallback ──
        T_lo = np.full_like(self.T_gas, 10.0)
        T_hi = np.full_like(self.T_gas, 1e5)

        # Phase 1: bisection to narrow bracket
        for _ in range(n_bisect):
            T_mid = np.sqrt(T_lo * T_hi)
            net = _net_heating_ctx(T_mid, ctx)
            pos = net > 0
            T_lo = np.where(pos, T_mid, T_lo)
            T_hi = np.where(~pos, T_mid, T_hi)

        # Phase 2: Newton-Raphson with clamping to bracket
        T = np.sqrt(T_lo * T_hi)
        for _ in range(n_newton):
            f = _net_heating_ctx(T, ctx)
            pos = f > 0
            T_lo = np.where(pos, np.maximum(T, T_lo), T_lo)
            T_hi = np.where(~pos, np.minimum(T, T_hi), T_hi)
            eps = T * 0.01
            fp = _net_heating_ctx(T + eps, ctx)
            dfdT = (fp - f) / eps
            dT = -f / np.where(np.abs(dfdT) > 1e-50, dfdT, -1e-50)
            dT = np.clip(dT, -(T - T_lo), T_hi - T)
            dT = np.clip(dT, -0.5 * T, 0.5 * T)
            T = T + dT

        self.T_gas = T

        # Compute final rates for diagnostics
        self.Gamma = self._total_heating_vec(T, f_sh_H2)
        self.Lambda = self._total_cooling_vec(T)

    def _solve_thermal_tiled(self, f_sh_H2, n_bisect=10, n_newton=3):
        """Tiled hybrid bisection-Newton for large grids.

        Processes the grid in tile_size³ chunks so that the working
        set fits in L3 cache.  Precomputes T-independent context
        per tile before the bisection loop.
        """
        # Precompute full-grid context, then slice per tile
        ctx_full = self._prepare_thermal_ctx(f_sh_H2)

        ts = self.tile_size
        nx, ny, nz = self.nx, self.ny, self.nz

        for ix0 in range(0, nx, ts):
            ix1 = min(ix0 + ts, nx)
            for iy0 in range(0, ny, ts):
                iy1 = min(iy0 + ts, ny)
                for iz0 in range(0, nz, ts):
                    iz1 = min(iz0 + ts, nz)
                    sl = (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1))

                    # Slice the precomputed context for this tile
                    ctx = _slice_ctx(ctx_full, sl)

                    # Phase 1: bisection to narrow bracket
                    T_lo = np.full(self.density[sl].shape, 10.0,
                                   dtype=self.T_gas.dtype)
                    T_hi = np.full(self.density[sl].shape, 1e5,
                                   dtype=self.T_gas.dtype)

                    for _ in range(n_bisect):
                        T_mid = np.sqrt(T_lo * T_hi)
                        net = _net_heating_ctx(T_mid, ctx)
                        pos = net > 0
                        T_lo = np.where(pos, T_mid, T_lo)
                        T_hi = np.where(~pos, T_mid, T_hi)

                    # Phase 2: Newton-Raphson with clamping
                    T = np.sqrt(T_lo * T_hi)
                    for _ in range(n_newton):
                        f = _net_heating_ctx(T, ctx)
                        pos = f > 0
                        T_lo = np.where(pos, np.maximum(T, T_lo), T_lo)
                        T_hi = np.where(~pos, np.minimum(T, T_hi), T_hi)
                        eps = T * 0.01
                        fp = _net_heating_ctx(T + eps, ctx)
                        dfdT = (fp - f) / eps
                        dT = -f / np.where(np.abs(dfdT) > 1e-50,
                                           dfdT, -1e-50)
                        dT = np.clip(dT, -(T - T_lo), T_hi - T)
                        dT = np.clip(dT, -0.5 * T, 0.5 * T)
                        T = T + dT

                    self.T_gas[sl] = T

        # Diagnostics (full-grid, computed once after tiled solve)
        self.Gamma = self._total_heating_vec(self.T_gas, f_sh_H2)
        self.Lambda = self._total_cooling_vec(self.T_gas)

    def _net_heating_vec(self, T, f_sh_H2):
        """Net heating rate (Gamma - Lambda) at temperature T for all cells."""
        return self._total_heating_vec(T, f_sh_H2) - self._total_cooling_vec(T)

    def _total_heating_vec(self, T, f_sh_H2):
        """Vectorized total heating rate at temperature T [erg/cm3/s]."""
        nH = self.density
        G0 = self.G0
        AV = self.A_V
        zeta = self.zeta_CR
        xe = self.x_e
        xHI = self.x_HI
        xH2 = self.x_H2
        xC = self.x_C
        T_dust = self.T_dust

        # PE heating (THEMIS charge-distribution, precomputed)
        Gamma = self.Gamma_PE.copy()

        # H2 photodissociation
        k_pd = 3.3e-11 * G0 * np.exp(-3.74 * AV) * f_sh_H2
        Gamma += k_pd * xH2 * nH * 0.4 * eV_to_erg

        # H2 formation on grains (THEMIS surface rate)
        Gamma += self._R_H2 * xHI * nH * nH * 1.5 * eV_to_erg

        # H2 UV pumping
        k_pump = 3.0e-10 * G0 * np.exp(-3.74 * AV) * f_sh_H2
        f_heat = nH / (nH + 3e4)
        Gamma += k_pump * xH2 * nH * 2.0 * eV_to_erg * f_heat

        # Cosmic ray heating
        Gamma += zeta * (xHI * nH + 2.0 * xH2 * nH) * 20.0 * eV_to_erg

        # Gas-grain collisional (positive when T_dust > T, negative otherwise)
        mu = np.where(xH2 > 0.5, 2.0, 1.0)
        v_th = np.sqrt(8.0 * k_boltz * T / (np.pi * mu * m_H))
        Gamma += 2.0 * 0.35 * 1e-21 * nH**2 * v_th * k_boltz * (T_dust - T)

        # C photoionization
        Gamma += 3.0e-10 * G0 * np.exp(-3.0 * AV) * xC * nH * 1.0 * eV_to_erg

        return Gamma

    def _total_cooling_vec(self, T):
        """Vectorized total cooling rate at temperature T [erg/cm3/s]."""
        nH = self.density
        xe = self.x_e
        xHI = self.x_HI
        xH2 = self.x_H2
        xCp = self.x_Cp
        xC = self.x_C
        xO = self.x_O
        xCO = self.x_CO
        T_dust = self.T_dust

        n_e = xe * nH
        n_HI = xHI * nH
        n_H2 = xH2 * nH

        Lambda = np.zeros_like(T)

        # [CII] 158 um
        _, freq, A_ul, T_ul, g_u, g_l = fine_structure_lines['CII_158']
        gamma_e = 8.63e-6 / (g_l * np.sqrt(T)) * 2.15 * np.exp(-T_ul / T)
        gamma_H = 8.0e-10 * (T / 100.0)**0.07
        gamma_H2 = 4.9e-10 * (T / 100.0)**0.12
        q_coll = gamma_e * n_e + gamma_H * n_HI + gamma_H2 * n_H2
        R_ul = A_ul + q_coll
        R_lu = (g_u / g_l) * q_coll * np.exp(-T_ul / T)
        n_u = xCp * nH * R_lu / (R_lu + R_ul)
        Lambda += n_u * A_ul * h_planck * freq

        # [OI] 63 um
        _, freq_63, A_63, T_63, g_u_63, g_l_63 = fine_structure_lines['OI_63']
        gamma_H_63 = 9.2e-12 * T**0.67
        gamma_H2_63 = 4.5e-12 * T**0.64
        gamma_e_63 = 1.4e-8 * (T / 100.0)**0.39
        q_63 = gamma_H_63 * n_HI + gamma_H2_63 * n_H2 + gamma_e_63 * n_e
        R_lu_63 = (g_u_63 / g_l_63) * q_63 * np.exp(-T_63 / T)
        R_ul_63 = A_63 + q_63
        n_u_63 = xO * nH * R_lu_63 / (R_lu_63 + R_ul_63)
        Lambda += n_u_63 * A_63 * h_planck * freq_63

        # [OI] 145 um
        _, freq_145, A_145, T_145, g_u_145, g_l_145 = fine_structure_lines['OI_145']
        gamma_H_145 = 4.0e-12 * T**0.60
        gamma_H2_145 = 2.0e-12 * T**0.58
        gamma_e_145 = 5.0e-9 * (T / 100.0)**0.39
        q_145 = gamma_H_145 * n_HI + gamma_H2_145 * n_H2 + gamma_e_145 * n_e
        R_lu_145 = (g_u_145 / g_l_145) * q_145 * np.exp(-T_145 / T)
        R_ul_145 = A_145 + q_145
        n_u_145 = xO * nH * R_lu_145 / (R_lu_145 + R_ul_145)
        Lambda += n_u_145 * A_145 * h_planck * freq_145

        # [CI] 609 + 370 um
        for line_name in ('CI_609', 'CI_370'):
            _, freq_ci, A_ci, T_ci, g_u_ci, g_l_ci = fine_structure_lines[line_name]
            gamma_H_ci = 1.0e-10 * (T / 100.0)**0.3
            gamma_H2_ci = 5.0e-11 * (T / 100.0)**0.3
            gamma_e_ci = 2.0e-7 * (T / 100.0)**(-0.5)
            q_ci = gamma_H_ci * n_HI + gamma_H2_ci * n_H2 + gamma_e_ci * n_e
            R_lu_ci = (g_u_ci / g_l_ci) * q_ci * np.exp(-T_ci / T)
            R_ul_ci = A_ci + q_ci
            n_u_ci = xC * nH * R_lu_ci / (R_lu_ci + R_ul_ci)
            Lambda += n_u_ci * A_ci * h_planck * freq_ci

        # CO rotational (loop over J, vectorized over cells)
        n_CO = xCO * nH
        B_rot = 57.635968e9
        mu_d = 0.1098e-18
        for J_u in range(1, 41):
            freq_co = 2.0 * B_rot * J_u
            T_u = h_planck * B_rot * J_u * (J_u + 1) / k_boltz
            mask = T_u < 5.0 * T
            if not np.any(mask):
                break
            A_co = (64.0 * np.pi**4 * freq_co**3 * mu_d**2 * J_u) / \
                   (3.0 * h_planck * c_light**3 * (2 * J_u + 1))
            g_u_co = 2 * J_u + 1
            g_l_co = 2 * (J_u - 1) + 1
            gamma_H2_co = 3.3e-11 * (T / 100.0)**0.5 * (1.0 + 0.1 * J_u)
            R_lu_co = (g_u_co / g_l_co) * gamma_H2_co * n_H2 * np.exp(-T_u / T)
            R_ul_co = A_co + gamma_H2_co * n_H2
            n_u_co = n_CO * R_lu_co / (R_lu_co + R_ul_co)
            Lambda += np.where(mask, n_u_co * A_co * h_planck * freq_co, 0.0)

        # H2 rovibrational (Glover & Abel 2008)
        logT = np.clip(np.log10(np.maximum(T, 100.0)), 2.0, 4.5)
        log_LH = (-24.311 + 3.5692 * logT - 11.332 * logT**2
                   + 15.738 * logT**3 - 10.581 * logT**4
                   + 3.5803 * logT**5 - 0.48365 * logT**6)
        log_LH2 = (-24.311 + 4.6585 * logT - 14.272 * logT**2
                    + 19.779 * logT**3 - 13.255 * logT**4
                    + 4.4840 * logT**5 - 0.60504 * logT**6)
        L_H = 10**log_LH
        L_H2_cool = 10**log_LH2
        L_LTE = 10**(-19.703 + 0.5 * logT)
        L0 = L_H * n_HI + L_H2_cool * n_H2
        Lambda_H2 = n_H2 * L0 * L_LTE / np.maximum(L0 + L_LTE, 1e-50)
        Lambda += np.where(T >= 100, Lambda_H2, 0.0)

        # Lyman-alpha
        q_lu = 2.41e-6 / np.sqrt(T) * 0.486 * np.exp(-1.18e5 / T)
        Lambda += np.where(T >= 3000, n_HI * n_e * q_lu * 10.2 * eV_to_erg, 0.0)

        # Gas-grain cooling (positive when T_gas > T_dust)
        mu = np.where(xH2 > 0.5, 2.0, 1.0)
        v_th = np.sqrt(8.0 * k_boltz * T / (np.pi * mu * m_H))
        gg_cool = 2.0 * 0.35 * 1e-21 * nH**2 * v_th * k_boltz * (T - T_dust)
        Lambda += np.maximum(gg_cool, 0.0)

        # Recombination cooling
        alpha_Cp = 4.67e-12 * (T / 300.0)**(-0.6)
        Lambda += xCp * nH * n_e * alpha_Cp * k_boltz * T

        return Lambda

    # ================================================================
    # BDF refinement (serial, high-accuracy)
    # ================================================================

    def refine(self, verbose=True):
        """
        Run a BDF refinement pass on the converged explicit solution.

        This replaces the fast explicit chemistry with a full BDF ODE
        solve at each cell, using the current state as initial conditions.
        Much slower but much more accurate.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BDF REFINEMENT PASS ({self.n_cells} cells)")
            print(f"{'='*60}")

        # Get current shielding
        f_sh_H2 = f_shield_H2(self.N_H2, T=np.mean(self.T_gas))
        f_sh_CO = f_shield_CO(self.N_CO, self.N_H2)

        t0 = time.time()
        n_conv = self._solve_chemistry_serial(f_sh_H2, f_sh_CO, fast=False)
        dt = time.time() - t0

        if verbose:
            print(f"  Chemistry: {n_conv}/{self.n_cells} converged in {dt:.1f}s")
            print(f"  ({dt/self.n_cells*1000:.0f} ms/cell)")

        # Re-solve thermal balance with refined chemistry
        if self.fixed_T is None:
            t0 = time.time()
            self._solve_thermal_vec(f_sh_H2)
            if verbose:
                print(f"  Thermal: {time.time()-t0:.1f}s")

        if verbose:
            print(f"  Refinement complete.")

    # ================================================================
    # Serial per-cell solvers (for BDF refinement)
    # ================================================================

    def _solve_chemistry_serial(self, f_sh_H2, f_sh_CO, fast=False):
        """Solve chemistry at every cell (serial, for BDF accuracy)."""
        n_conv = 0

        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    n_H = self.density[ix, iy, iz]
                    T = self.T_gas[ix, iy, iz]
                    G0 = self.G0[ix, iy, iz]
                    A_V = self.A_V[ix, iy, iz]
                    zeta = self.zeta_CR[ix, iy, iz]

                    fH2 = float(f_sh_H2[ix, iy, iz]) if hasattr(f_sh_H2, '__getitem__') else float(f_sh_H2)
                    fCO = float(f_sh_CO[ix, iy, iz]) if hasattr(f_sh_CO, '__getitem__') else float(f_sh_CO)

                    x_init = {
                        'H': self.x_HI[ix, iy, iz],
                        'H2': self.x_H2[ix, iy, iz],
                        'C+': self.x_Cp[ix, iy, iz],
                        'C': self.x_C[ix, iy, iz],
                        'CO': self.x_CO[ix, iy, iz],
                        'O': self.x_O[ix, iy, iz],
                        'e-': self.x_e[ix, iy, iz],
                        'OH': self.x_OH[ix, iy, iz],
                        'H2O': self.x_H2O[ix, iy, iz],
                        'HCO+': self.x_HCOp[ix, iy, iz],
                    }

                    result, conv = self.chem_solver.solve_steady_state(
                        n_H=n_H, T=T, G0=G0, A_V=A_V, zeta_CR=zeta,
                        x_init=x_init, f_shield_H2=fH2, f_shield_CO=fCO,
                        fast=fast
                    )

                    self.x_HI[ix, iy, iz] = result.get('H', self.x_HI[ix, iy, iz])
                    self.x_H2[ix, iy, iz] = result.get('H2', self.x_H2[ix, iy, iz])
                    self.x_Cp[ix, iy, iz] = result.get('C+', self.x_Cp[ix, iy, iz])
                    self.x_C[ix, iy, iz] = result.get('C', self.x_C[ix, iy, iz])
                    self.x_CO[ix, iy, iz] = result.get('CO', self.x_CO[ix, iy, iz])
                    self.x_O[ix, iy, iz] = result.get('O', self.x_O[ix, iy, iz])
                    self.x_e[ix, iy, iz] = result.get('e-', self.x_e[ix, iy, iz])
                    self.x_OH[ix, iy, iz] = result.get('OH', self.x_OH[ix, iy, iz])
                    self.x_H2O[ix, iy, iz] = result.get('H2O', self.x_H2O[ix, iy, iz])
                    self.x_HCOp[ix, iy, iz] = result.get('HCO+', self.x_HCOp[ix, iy, iz])

                    if conv:
                        n_conv += 1

        return n_conv

    # ================================================================
    # I/O and utilities
    # ================================================================

    def save(self, filepath):
        """Save model state to numpy archive for HPC workflows."""
        save_dict = dict(
            density=self.density,
            T_gas=self.T_gas, T_dust=self.T_dust,
            G0=self.G0, A_V=self.A_V,
            N_H=self.N_H, N_H2=self.N_H2, N_CO=self.N_CO,
            zeta_CR=self.zeta_CR,
            x_HI=self.x_HI, x_H2=self.x_H2,
            x_Cp=self.x_Cp, x_C=self.x_C, x_CO=self.x_CO,
            x_O=self.x_O, x_e=self.x_e,
            x_OH=self.x_OH, x_H2O=self.x_H2O, x_HCOp=self.x_HCOp,
            Gamma=self.Gamma, Lambda=self.Lambda,
            Gamma_PE=self.Gamma_PE,
            f_nano=self.f_nano, E_g=self.E_g,
            box_size=self.box_size, G0_external=self.G0_external,
            zeta_CR_0=self.zeta_CR_0,
        )
        if self.velocity_field is not None:
            save_dict['velocity_field'] = self.velocity_field
        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, filepath):
        """Load model state from numpy archive."""
        data = np.load(filepath)
        vel = data['velocity_field'] if 'velocity_field' in data else None
        solver = cls(
            data['density'], float(data['box_size']),
            G0_external=float(data['G0_external']),
            zeta_CR_0=float(data['zeta_CR_0']),
            velocity_field=vel,
        )
        dt = solver.T_gas.dtype  # float32 or float64 based on grid size
        for key in ['T_gas', 'T_dust', 'G0', 'A_V', 'N_H', 'N_H2', 'N_CO',
                     'zeta_CR', 'x_HI', 'x_H2', 'x_Cp', 'x_C', 'x_CO',
                     'x_O', 'x_e', 'x_OH', 'x_H2O', 'x_HCOp',
                     'Gamma', 'Lambda', 'Gamma_PE', 'f_nano', 'E_g']:
            if key in data:
                setattr(solver, key, data[key].astype(dt))
        return solver

    def _print_timing(self):
        total = sum(self.timing.values())
        if total > 0:
            print(f"\nTiming breakdown:")
            for k, v in sorted(self.timing.items(), key=lambda x: -x[1]):
                print(f"  {k:15s}: {v:6.2f}s ({100*v/total:5.1f}%)")

    def get_slice(self, axis=0, index=None):
        """
        Extract a 2D slice through the 3D model.

        Parameters
        ----------
        axis : int
            Slice axis (0=x, 1=y, 2=z).
        index : int, optional
            Slice index. Default: middle.

        Returns
        -------
        data : dict
            Dictionary of 2D arrays for all quantities.
        """
        if index is None:
            index = [self.nx, self.ny, self.nz][axis] // 2

        s = [slice(None)] * 3
        s[axis] = index
        s = tuple(s)

        return {
            'density': self.density[s],
            'T_gas': self.T_gas[s],
            'T_dust': self.T_dust[s],
            'G0': self.G0[s],
            'A_V': self.A_V[s],
            'x_HI': self.x_HI[s],
            'x_H2': self.x_H2[s],
            'x_Cp': self.x_Cp[s],
            'x_C': self.x_C[s],
            'x_CO': self.x_CO[s],
            'x_O': self.x_O[s],
            'x_e': self.x_e[s],
            'f_nano': self.f_nano[s],
            'E_g': self.E_g[s],
            'Gamma_PE': self.Gamma_PE[s],
        }


# ====================================================================
# Context-based thermal evaluation (free functions)
# ====================================================================

def _slice_ctx(ctx, sl):
    """Slice a precomputed thermal context for a tile."""
    # ctx is a tuple: (Gamma_fixed, gg_coeff, T_dust,
    #   n_e, n_HI, n_H2, n_CO, nCp_nH, nO_nH, nC_nH, mu, nH,
    #   CII_f, CII_A, CII_T, CII_gu, CII_gl,
    #   OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl,
    #   OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl,
    #   CI_consts, co_J)
    # Slice only the array elements (indices 0-11), keep scalars as-is
    return (ctx[0][sl], ctx[1][sl], ctx[2][sl],
            ctx[3][sl], ctx[4][sl], ctx[5][sl], ctx[6][sl],
            ctx[7][sl], ctx[8][sl], ctx[9][sl], ctx[10][sl], ctx[11][sl],
            *ctx[12:])


def _net_heating_ctx(T, ctx):
    """Merged net heating rate using precomputed T-independent context.

    Heating: only gas-grain depends on T (everything else precomputed).
    Cooling: all T-dependent, but density products and line constants cached.
    """
    (Gamma_fixed, gg_coeff, T_dust,
     n_e, n_HI, n_H2, n_CO, nCp_nH, nO_nH, nC_nH, mu, nH,
     CII_f, CII_A, CII_T, CII_gu, CII_gl,
     OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl,
     OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl,
     CI_consts, co_J) = ctx

    # --- Heating: precomputed sum + gas-grain (only T-dependent term) ---
    sqrtT = np.sqrt(T)
    net = Gamma_fixed + gg_coeff * sqrtT * (T_dust - T)

    # --- Cooling (subtract from net) ---
    inv_T = 1.0 / T

    # [CII] 158 um
    gamma_e = 8.63e-6 / (CII_gl * sqrtT) * 2.15 * np.exp(-CII_T * inv_T)
    gamma_H = 8.0e-10 * (T * 0.01)**0.07
    gamma_H2 = 4.9e-10 * (T * 0.01)**0.12
    q_coll = gamma_e * n_e + gamma_H * n_HI + gamma_H2 * n_H2
    R_ul = CII_A + q_coll
    R_lu = (CII_gu / CII_gl) * q_coll * np.exp(-CII_T * inv_T)
    net -= nCp_nH * R_lu / (R_lu + R_ul) * CII_A * h_planck * CII_f

    # [OI] 63 um
    gamma_H_63 = 9.2e-12 * T**0.67
    gamma_H2_63 = 4.5e-12 * T**0.64
    gamma_e_63 = 1.4e-8 * (T * 0.01)**0.39
    q_63 = gamma_H_63 * n_HI + gamma_H2_63 * n_H2 + gamma_e_63 * n_e
    R_lu_63 = (OI63_gu / OI63_gl) * q_63 * np.exp(-OI63_T * inv_T)
    R_ul_63 = OI63_A + q_63
    net -= nO_nH * R_lu_63 / (R_lu_63 + R_ul_63) * OI63_A * h_planck * OI63_f

    # [OI] 145 um
    gamma_H_145 = 4.0e-12 * T**0.60
    gamma_H2_145 = 2.0e-12 * T**0.58
    gamma_e_145 = 5.0e-9 * (T * 0.01)**0.39
    q_145 = gamma_H_145 * n_HI + gamma_H2_145 * n_H2 + gamma_e_145 * n_e
    R_lu_145 = (OI145_gu / OI145_gl) * q_145 * np.exp(-OI145_T * inv_T)
    R_ul_145 = OI145_A + q_145
    net -= nO_nH * R_lu_145 / (R_lu_145 + R_ul_145) * OI145_A * h_planck * OI145_f

    # [CI] 609 + 370 um
    T_01 = T * 0.01
    T_01_03 = T_01**0.3
    T_01_m05 = T_01**(-0.5)
    for ci_f, ci_A, ci_T, ci_gu, ci_gl in CI_consts:
        q_ci = 1.0e-10 * T_01_03 * n_HI + 5.0e-11 * T_01_03 * n_H2 + 2.0e-7 * T_01_m05 * n_e
        R_lu_ci = (ci_gu / ci_gl) * q_ci * np.exp(-ci_T * inv_T)
        R_ul_ci = ci_A + q_ci
        net -= nC_nH * R_lu_ci / (R_lu_ci + R_ul_ci) * ci_A * h_planck * ci_f

    # CO rotational (precomputed J-level constants)
    sqrtT_01 = np.sqrt(T_01)
    for T_u, A_co, g_ratio, J_factor, Ahv in co_J:
        mask = T_u < 5.0 * T
        if not np.any(mask):
            break
        gamma_H2_co = 3.3e-11 * sqrtT_01 * J_factor
        R_lu_co = g_ratio * gamma_H2_co * n_H2 * np.exp(-T_u * inv_T)
        R_ul_co = A_co + gamma_H2_co * n_H2
        net -= np.where(mask, n_CO * R_lu_co / (R_lu_co + R_ul_co) * Ahv, 0.0)

    # H2 rovibrational (Glover & Abel 2008)
    logT = np.clip(np.log10(np.maximum(T, 100.0)), 2.0, 4.5)
    log_LH = (-24.311 + 3.5692 * logT - 11.332 * logT**2
               + 15.738 * logT**3 - 10.581 * logT**4
               + 3.5803 * logT**5 - 0.48365 * logT**6)
    log_LH2 = (-24.311 + 4.6585 * logT - 14.272 * logT**2
                + 19.779 * logT**3 - 13.255 * logT**4
                + 4.4840 * logT**5 - 0.60504 * logT**6)
    L_H = 10**log_LH
    L_H2_cool = 10**log_LH2
    L_LTE = 10**(-19.703 + 0.5 * logT)
    L0 = L_H * n_HI + L_H2_cool * n_H2
    Lambda_H2 = n_H2 * L0 * L_LTE / np.maximum(L0 + L_LTE, 1e-50)
    net -= np.where(T >= 100, Lambda_H2, 0.0)

    # Lyman-alpha
    q_lu = 2.41e-6 / sqrtT * 0.486 * np.exp(-1.18e5 * inv_T)
    net -= np.where(T >= 3000, n_HI * n_e * q_lu * 10.2 * eV_to_erg, 0.0)

    # Gas-grain cooling (heating part already in Gamma_fixed via gg_coeff)
    gg_cool = gg_coeff * sqrtT * (T - T_dust)
    net -= np.maximum(gg_cool, 0.0)

    # Recombination cooling
    alpha_Cp = 4.67e-12 * (T / 300.0)**(-0.6)
    net -= nCp_nH * n_e * alpha_Cp * k_boltz * T

    return net
