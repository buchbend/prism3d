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
"""

import numpy as np
import time
from ..radiative_transfer.fuv_rt_3d import compute_fuv_field_3d
from ..radiative_transfer.shielding import f_shield_H2, f_shield_CO
from ..chemistry.solver import ChemistrySolver
from ..thermal.balance import ThermalSolver
from ..cosmic_rays.attenuation import cr_ionization_rate
from ..grains.themis import THEMISDust


class PDRSolver3D:
    """
    3D PDR solver on a regular Cartesian grid.
    
    Parameters
    ----------
    density : ndarray (nx, ny, nz)
        Hydrogen number density [cm⁻³].
    box_size : float
        Physical box size [cm].
    G0_external : float
        External FUV field [Habing units].
    zeta_CR_0 : float
        Surface cosmic ray ionization rate [s⁻¹].
    nside_rt : int
        HEALPix resolution for FUV ray tracing.
    fixed_T : float, optional
        If set, use fixed gas temperature (for isothermal benchmarks).
    """
    
    def __init__(self, density, box_size, G0_external=1.0,
                 zeta_CR_0=2e-16, nside_rt=1, fixed_T=None,
                 chem_network=None):
        self.density = np.asarray(density, dtype=np.float64)
        self.nx, self.ny, self.nz = self.density.shape
        self.n_cells = self.nx * self.ny * self.nz
        self.box_size = box_size
        self.cell_size = box_size / self.nx  # Assumes cubic cells
        self.G0_external = G0_external
        self.zeta_CR_0 = zeta_CR_0
        self.nside_rt = nside_rt
        self.fixed_T = fixed_T
        
        # Initialize state arrays
        self._init_state()
        
        # Solvers
        self.chem_solver = ChemistrySolver(network=chem_network)
        self.thermal_solver = ThermalSolver()
        
        # Diagnostics
        self.timing = {}
        self.convergence_history = []
    
    def _init_state(self):
        """Initialize all 3D state arrays."""
        shape = (self.nx, self.ny, self.nz)
        
        # Radiation field
        self.G0 = np.full(shape, self.G0_external)
        self.A_V = np.zeros(shape)
        self.N_H = np.zeros(shape)
        self.N_H2 = np.zeros(shape)
        self.N_CO = np.zeros(shape)
        self.zeta_CR = np.full(shape, self.zeta_CR_0)
        
        # Temperature
        if self.fixed_T is not None:
            self.T_gas = np.full(shape, self.fixed_T)
        else:
            self.T_gas = np.full(shape, 50.0)
        self.T_dust = np.full(shape, 20.0)
        
        # Abundances (relative to n_H)
        from ..utils.constants import gas_phase_abundances
        x_C = gas_phase_abundances.get('C', 1.4e-4)
        x_O = gas_phase_abundances.get('O', 3.0e-4)
        
        self.x_HI = np.full(shape, 0.5)
        self.x_H2 = np.full(shape, 0.25)
        self.x_Cp = np.full(shape, x_C * 0.5)
        self.x_C = np.full(shape, x_C * 0.1)
        self.x_CO = np.full(shape, x_C * 0.4)
        self.x_O = np.full(shape, x_O - x_C * 0.4)
        self.x_e = np.full(shape, x_C * 0.5)
        self.x_OH = np.full(shape, 1e-8)
        self.x_H2O = np.full(shape, 1e-8)
        self.x_HCOp = np.full(shape, 1e-10)
        
        # Heating/cooling diagnostics
        self.Gamma = np.zeros(shape)
        self.Lambda = np.zeros(shape)
        
        # THEMIS dust state (per cell)
        self.f_nano = np.ones(shape)       # Nano-grain fraction (1.0 = diffuse ISM)
        self.E_g = np.full(shape, 0.1)     # Band gap [eV] (0.1 = diffuse ISM)
        self.Gamma_PE = np.zeros(shape)    # PE heating rate per cell
    
    def run(self, max_iterations=20, convergence_tol=0.05, verbose=True):
        """
        Run the 3D PDR model to convergence.
        """
        if verbose:
            print(f"PRISM-3D 3D Solver")
            print(f"Grid: {self.nx}×{self.ny}×{self.nz} = {self.n_cells} cells")
            print(f"G0 = {self.G0_external}, ζ_CR = {self.zeta_CR_0:.1e}")
            print(f"RT: nside={self.nside_rt} ({12*self.nside_rt**2} rays)")
            if self.fixed_T:
                print(f"T = {self.fixed_T} K (fixed)")
            print(f"{'='*60}")
        
        t_start = time.time()
        converged = False
        
        for iteration in range(max_iterations):
            t_iter = time.time()
            
            # Save old values for convergence
            old_T = self.T_gas.copy()
            old_x_H2 = self.x_H2.copy()
            old_x_CO = self.x_CO.copy()
            
            # Step 1: FUV Radiative Transfer (vectorized)
            t0 = time.time()
            self.G0, self.A_V, self.N_H, self.N_H2, self.N_CO = \
                compute_fuv_field_3d(
                    self.density, self.G0_external, self.cell_size,
                    x_H2=self.x_H2, x_CO=self.x_CO,
                    nside=self.nside_rt
                )
            self.timing['RT'] = time.time() - t0
            
            # Step 2: Self-shielding (vectorized)
            t0 = time.time()
            f_sh_H2 = f_shield_H2(self.N_H2, T=np.mean(self.T_gas))
            f_sh_CO = f_shield_CO(self.N_CO, self.N_H2)
            self.timing['shielding'] = time.time() - t0
            
            # Step 3: CR attenuation (vectorized)
            t0 = time.time()
            self.zeta_CR = cr_ionization_rate(self.N_H, model='M',
                                               zeta_0=self.zeta_CR_0)
            self.timing['CR'] = time.time() - t0
            
            # Step 3b: THEMIS dust evolution + PE heating (vectorized)
            t0 = time.time()
            dt_dust = 1e4 * 3.15e7  # 10 kyr per iteration
            
            # Create ONE dust object and reuse — avoid per-cell construction
            # Group cells by similar f_nano (within 5%) to reuse THEMIS objects
            dust_cache = {}
            
            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz in range(self.nz):
                        G0_loc = self.G0[ix, iy, iz]
                        n_H_loc = self.density[ix, iy, iz]
                        f_n = self.f_nano[ix, iy, iz]
                        eg = self.E_g[ix, iy, iz]
                        
                        # Evolve dust (cheap — no object creation needed)
                        # Photo-destruction timescale
                        tau_dest = 1e4 * 3.15e7 / max(G0_loc / 1e4, 0.01)
                        f_dest = np.exp(-dt_dust / tau_dest)
                        tau_replenish = 1e6 * 3.15e7
                        f_replenish = min(dt_dust / tau_replenish, 0.1)
                        f_accrete = 0.0
                        if G0_loc < 1 and n_H_loc > 1e4:
                            f_accrete = min(dt_dust / (1e5 * 3.15e7 * 1e4 / n_H_loc), 0.1)
                        
                        f_new = f_n * f_dest + (1.0 - f_n) * f_replenish + f_accrete
                        f_new = np.clip(f_new, 0.01, 2.0)
                        self.f_nano[ix, iy, iz] = f_new
                        
                        # Band gap evolution
                        tau_aromat = 1e3 * 3.15e7 / max(G0_loc, 0.01)
                        Eg_eq = 0.01 if G0_loc > 10 else min(0.1 + 0.5 * np.log10(max(n_H_loc, 1)), 2.5)
                        eg_new = eg + (Eg_eq - eg) * min(dt_dust / tau_aromat, 1.0)
                        self.E_g[ix, iy, iz] = np.clip(eg_new, 0.01, 2.5)
                        
                        # Cached THEMIS PE heating
                        # Round f_nano to nearest 0.05 for cache efficiency
                        f_key = round(f_new * 20) / 20
                        eg_key = round(eg_new * 10) / 10
                        cache_key = (f_key, eg_key)
                        
                        if cache_key not in dust_cache:
                            dust_cache[cache_key] = THEMISDust(f_nano=f_key, E_g=eg_key)
                        
                        dust_obj = dust_cache[cache_key]
                        self.Gamma_PE[ix, iy, iz] = dust_obj.photoelectric_heating(
                            n_H_loc, self.T_gas[ix, iy, iz],
                            G0_loc, self.x_e[ix, iy, iz]
                        )
                        
                        # Dust temperature (simple: use silicate equilibrium)
                        F_abs = 1.6e-3 * G0_loc * np.exp(-1.8 * self.A_V[ix, iy, iz])
                        self.T_dust[ix, iy, iz] = max(5, min(200,
                            16.4 * (F_abs / 1.6e-3)**0.2))  # Approx T_dust ~ 16*(G0_eff)^0.2
            
            self.timing['dust'] = time.time() - t0
            
            # Step 4: Chemistry (cell by cell — the bottleneck)
            t0 = time.time()
            n_chem_conv = self._solve_chemistry_all(f_sh_H2, f_sh_CO,
                                                      fast=True)  # Always fast in 3D
            self.timing['chemistry'] = time.time() - t0
            
            # Step 5: Dust temperature (handled by THEMIS above)
            self.timing['dust_T'] = 0.0
            
            # Step 6: Thermal balance
            t0 = time.time()
            if self.fixed_T is None:
                self._solve_thermal_all(f_sh_H2)
                # Damping
                if iteration > 0:
                    self.T_gas = 0.5 * old_T + 0.5 * self.T_gas
            self.timing['thermal'] = time.time() - t0
            
            # Convergence check
            mask_T = old_T > 10
            mask_H2 = old_x_H2 > 1e-6
            mask_CO = old_x_CO > 1e-8
            
            dT = np.max(np.abs(self.T_gas[mask_T] - old_T[mask_T]) / old_T[mask_T]) if np.any(mask_T) else 0
            dH2 = np.max(np.abs(self.x_H2[mask_H2] - old_x_H2[mask_H2]) / old_x_H2[mask_H2]) if np.any(mask_H2) else 0
            dCO = np.max(np.abs(self.x_CO[mask_CO] - old_x_CO[mask_CO]) / old_x_CO[mask_CO]) if np.any(mask_CO) else 0
            delta_max = max(dT, dH2, dCO)
            
            dt_iter = time.time() - t_iter
            
            if verbose:
                f_nano_mean = np.mean(self.f_nano)
                print(f"Iter {iteration:3d} | "
                      f"ΔT={dT:.2e} ΔH2={dH2:.2e} ΔCO={dCO:.2e} | "
                      f"f_nano={f_nano_mean:.3f} | "
                      f"chem:{n_chem_conv}/{self.n_cells} | "
                      f"t={dt_iter:.1f}s "
                      f"[RT:{self.timing.get('RT',0):.1f} "
                      f"dust:{self.timing.get('dust',0):.1f} "
                      f"chem:{self.timing.get('chemistry',0):.1f} "
                      f"therm:{self.timing.get('thermal',0):.1f}]")
            
            self.convergence_history.append({
                'iteration': iteration, 'dT': dT, 'dH2': dH2,
                'dCO': dCO, 'delta_max': delta_max
            })
            
            if delta_max < convergence_tol and iteration > 2:
                converged = True
                if verbose:
                    total = time.time() - t_start
                    print(f"\nCONVERGED after {iteration+1} iterations ({total:.1f}s)")
                    self._print_timing()
                break
        
        if not converged and verbose:
            print(f"\nWARNING: Not converged after {max_iterations} iterations")
        
        return converged
    
    def refine(self, verbose=True):
        """
        Run a BDF refinement pass on the converged explicit solution.
        
        This replaces the fast explicit chemistry with a full BDF ODE
        solve at each cell, using the current state as initial conditions.
        Much slower but much more accurate — run after the fast iterations
        converge to get publication-quality abundances.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BDF REFINEMENT PASS ({self.n_cells} cells)")
            print(f"{'='*60}")
        
        # Get current shielding
        f_sh_H2 = f_shield_H2(self.N_H2, T=np.mean(self.T_gas))
        f_sh_CO = f_shield_CO(self.N_CO, self.N_H2)
        
        t0 = time.time()
        n_conv = self._solve_chemistry_all(f_sh_H2, f_sh_CO, fast=False)
        dt = time.time() - t0
        
        if verbose:
            print(f"  Chemistry: {n_conv}/{self.n_cells} converged in {dt:.1f}s")
            print(f"  ({dt/self.n_cells*1000:.0f} ms/cell)")
        
        # Re-solve thermal balance with refined chemistry
        if self.fixed_T is None:
            t0 = time.time()
            self._solve_thermal_all(f_sh_H2)
            if verbose:
                print(f"  Thermal: {time.time()-t0:.1f}s")
        
        if verbose:
            print(f"  Refinement complete.")
    
    def save(self, filepath):
        """Save model state to numpy archive for HPC workflows."""
        np.savez_compressed(filepath,
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
    
    @classmethod
    def load(cls, filepath):
        """Load model state from numpy archive."""
        data = np.load(filepath)
        solver = cls(
            data['density'], float(data['box_size']),
            G0_external=float(data['G0_external']),
            zeta_CR_0=float(data['zeta_CR_0']),
        )
        for key in ['T_gas', 'T_dust', 'G0', 'A_V', 'N_H', 'N_H2', 'N_CO',
                     'zeta_CR', 'x_HI', 'x_H2', 'x_Cp', 'x_C', 'x_CO',
                     'x_O', 'x_e', 'x_OH', 'x_H2O', 'x_HCOp', 'Gamma', 'Lambda']:
            if key in data:
                setattr(solver, key, data[key])
        return solver
    
    def _solve_chemistry_all(self, f_sh_H2, f_sh_CO, fast=False):
        """Solve chemistry at every cell."""
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
                    
                    # Build initial abundances from current state
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
                    
                    # Write back
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
    
    def _solve_thermal_all(self, f_sh_H2):
        """Solve thermal balance at every cell."""
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    fH2 = float(f_sh_H2[ix, iy, iz]) if hasattr(f_sh_H2, '__getitem__') else float(f_sh_H2)
                    
                    T_eq, G, L, _, _ = self.thermal_solver.solve(
                        n_H=self.density[ix, iy, iz],
                        T_dust=self.T_dust[ix, iy, iz],
                        G0=self.G0[ix, iy, iz],
                        A_V=self.A_V[ix, iy, iz],
                        zeta_CR=self.zeta_CR[ix, iy, iz],
                        x_e=self.x_e[ix, iy, iz],
                        x_HI=self.x_HI[ix, iy, iz],
                        x_H2=self.x_H2[ix, iy, iz],
                        x_Cp=self.x_Cp[ix, iy, iz],
                        x_C=self.x_C[ix, iy, iz],
                        x_O=self.x_O[ix, iy, iz],
                        x_CO=self.x_CO[ix, iy, iz],
                        f_shield_H2=fH2,
                        N_CO=self.N_CO[ix, iy, iz]
                    )
                    
                    self.T_gas[ix, iy, iz] = T_eq
                    self.Gamma[ix, iy, iz] = G
                    self.Lambda[ix, iy, iz] = L
    
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
