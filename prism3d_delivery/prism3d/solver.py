"""
Main PDR Solver for PRISM-3D.

Orchestrates the iterative solution of the coupled system:
  FUV RT → Chemistry → Thermal balance → (repeat)

Supports:
- Steady-state iteration to convergence
- Time-dependent evolution with operator splitting
- Adaptive mesh refinement during iteration
- Comprehensive logging and diagnostics

The basic iteration loop:
1. Compute FUV radiation field (ray tracing)
2. Compute self-shielding factors (H2, CO)
3. Compute CR ionization rate (column-density dependent)
4. Solve chemistry (ODE integration or steady state)
5. Compute dust temperature
6. Solve thermal balance (heating = cooling)
7. Check convergence → if not converged, go to 1
8. Optionally refine mesh and repeat
"""

import numpy as np
import time
from ..radiative_transfer.fuv_rt import FUVRadiativeTransfer
from ..radiative_transfer.shielding import f_shield_H2, f_shield_CO
from ..chemistry.solver import ChemistrySolver
from ..thermal.balance import ThermalSolver
from ..cosmic_rays.attenuation import cr_ionization_rate


class PDRSolver:
    """
    Main solver for PRISM-3D PDR models.

    Parameters
    ----------
    grid : OctreeGrid
        The computational grid with initial conditions set.
    G0_external : float or callable
        External FUV field [Habing units].
    zeta_CR_0 : float
        Surface cosmic ray ionization rate [s⁻¹].
    nside_rt : int
        HEALPix resolution for FUV ray tracing.
    cr_model : str
        Cosmic ray attenuation model ('L', 'M', 'H').
    use_1d_rt : bool
        If True, use fast 1D column density calculation.
        Set True for benchmarking against 1D codes.
    metallicity : float
        Metallicity relative to solar (scales abundances and dust).
    """

    def __init__(self, grid, G0_external=1.0, zeta_CR_0=2e-16,
                 nside_rt=2, cr_model='M', use_1d_rt=False,
                 metallicity=1.0):
        self.grid = grid
        self.G0_external = G0_external
        self.zeta_CR_0 = zeta_CR_0
        self.cr_model = cr_model
        self.use_1d_rt = use_1d_rt
        self.metallicity = metallicity

        # Initialize sub-solvers
        self.rt = FUVRadiativeTransfer(grid, G0_external, nside=nside_rt)
        self.chem_solver = ChemistrySolver()
        self.thermal_solver = ThermalSolver()

        # Convergence tracking
        self.iteration = 0
        self.convergence_history = []
        self.timing = {}

    def run(self, max_iterations=100, convergence_tol=1e-2,
            refine_every=0, verbose=True):
        """
        Run the PDR model to convergence.

        Parameters
        ----------
        max_iterations : int
            Maximum number of global iterations.
        convergence_tol : float
            Convergence criterion: max relative change in T and
            abundances between iterations.
        refine_every : int
            If > 0, perform AMR every this many iterations.
        verbose : bool
            Print progress information.

        Returns
        -------
        converged : bool
            Whether the model converged within max_iterations.
        """
        if verbose:
            print(f"PRISM-3D Solver v0.1")
            print(f"Grid: {self.grid}")
            print(f"G0 = {self.G0_external}, ζ_CR = {self.zeta_CR_0:.2e}")
            print(f"{'='*60}")

        converged = False
        t_start = time.time()

        for iteration in range(max_iterations):
            self.iteration = iteration
            t_iter_start = time.time()

            # Store old values for convergence check
            leaves = self.grid.get_leaves()
            old_T = np.array([l.data.T_gas for l in leaves])
            old_x_H2 = np.array([l.data.x_H2 for l in leaves])
            old_x_CO = np.array([l.data.x_CO for l in leaves])

            # ========================================
            # Step 1: Radiative Transfer
            # ========================================
            t0 = time.time()
            if self.use_1d_rt:
                self.rt.compute_1d_column_densities(axis=0, direction=1)
            else:
                self.rt.compute_fuv_field()
            self.timing['RT'] = time.time() - t0

            # ========================================
            # Step 2: Self-shielding
            # ========================================
            t0 = time.time()
            for leaf in leaves:
                leaf.data._f_shield_H2 = float(
                    f_shield_H2(leaf.data.N_H2, T=leaf.data.T_gas)
                )
                leaf.data._f_shield_CO = float(
                    f_shield_CO(leaf.data.N_CO, leaf.data.N_H2)
                )
            self.timing['shielding'] = time.time() - t0

            # ========================================
            # Step 3: Cosmic ray attenuation
            # ========================================
            t0 = time.time()
            for leaf in leaves:
                leaf.data.zeta_CR = float(cr_ionization_rate(
                    leaf.data.N_H_total,
                    model=self.cr_model,
                    zeta_0=self.zeta_CR_0
                ))
            self.timing['CR'] = time.time() - t0

            # ========================================
            # Step 4: Chemistry
            # ========================================
            t0 = time.time()
            n_chem_converged = 0
            for leaf in leaves:
                d = leaf.data

                # Build initial abundance dict from cell data
                x_init = self._cell_to_abundances(d)

                # Get shielding factors
                fsh_H2 = getattr(d, '_f_shield_H2', 1.0)
                fsh_CO = getattr(d, '_f_shield_CO', 1.0)

                # Solve chemistry
                abundances, chem_conv = self.chem_solver.solve_steady_state(
                    n_H=d.n_H, T=d.T_gas, G0=d.G0, A_V=d.A_V,
                    zeta_CR=d.zeta_CR, x_init=x_init,
                    f_shield_H2=fsh_H2, f_shield_CO=fsh_CO,
                    t_max=1e14, rtol=1e-3
                )

                # Write back to cell
                self._abundances_to_cell(abundances, d)

                if chem_conv:
                    n_chem_converged += 1

            self.timing['chemistry'] = time.time() - t0

            # ========================================
            # Step 5: Dust temperature
            # ========================================
            t0 = time.time()
            for leaf in leaves:
                d = leaf.data
                d.T_dust = self.thermal_solver.compute_dust_temperature(
                    d.G0, d.A_V, n_H=d.n_H, T_gas=d.T_gas
                )
            self.timing['dust_T'] = time.time() - t0

            # ========================================
            # Step 6: Thermal balance
            # ========================================
            t0 = time.time()
            # Check if fixed temperature mode (for Röllig F models)
            fixed_T = getattr(self, '_fixed_T', None)

            if fixed_T is not None:
                # Fixed temperature: skip thermal balance solve
                for leaf in leaves:
                    leaf.data.T_gas = fixed_T
            else:
                for leaf in leaves:
                    d = leaf.data
                    fsh_H2 = getattr(d, '_f_shield_H2', 1.0)

                    T_eq, Gamma, Lambda, _, _ = self.thermal_solver.solve(
                        n_H=d.n_H, T_dust=d.T_dust,
                        G0=d.G0, A_V=d.A_V, zeta_CR=d.zeta_CR,
                        x_e=d.x_e, x_HI=d.x_HI, x_H2=d.x_H2,
                        x_Cp=d.x_Cp, x_C=d.x_C, x_O=d.x_O, x_CO=d.x_CO,
                        f_shield_H2=fsh_H2, N_CO=d.N_CO
                    )

                    d.T_gas = T_eq
                    d.Gamma_total = Gamma
                    d.Lambda_total = Lambda
            self.timing['thermal'] = time.time() - t0

            # ========================================
            # Step 6b: Temperature damping
            # ========================================
            if fixed_T is None and iteration > 0:
                damping = 0.5
                for ii, leaf in enumerate(leaves):
                    T_new = leaf.data.T_gas
                    T_old = old_T[ii]
                    leaf.data.T_gas = T_old * damping + T_new * (1.0 - damping)

            # ========================================
            # Step 7: Convergence check
            # ========================================
            new_T = np.array([l.data.T_gas for l in leaves])
            new_x_H2 = np.array([l.data.x_H2 for l in leaves])
            new_x_CO = np.array([l.data.x_CO for l in leaves])

            # Relative changes
            mask_T = old_T > 10
            mask_H2 = old_x_H2 > 1e-6
            mask_CO = old_x_CO > 1e-8

            dT_max = np.max(np.abs(new_T[mask_T] - old_T[mask_T]) / old_T[mask_T]) if np.any(mask_T) else 0
            dH2_max = np.max(np.abs(new_x_H2[mask_H2] - old_x_H2[mask_H2]) / old_x_H2[mask_H2]) if np.any(mask_H2) else 0
            dCO_max = np.max(np.abs(new_x_CO[mask_CO] - old_x_CO[mask_CO]) / old_x_CO[mask_CO]) if np.any(mask_CO) else 0

            delta_max = max(dT_max, dH2_max, dCO_max)

            self.convergence_history.append({
                'iteration': iteration,
                'delta_T': dT_max,
                'delta_H2': dH2_max,
                'delta_CO': dCO_max,
                'delta_max': delta_max,
                'n_chem_converged': n_chem_converged,
                'n_cells': len(leaves),
            })

            t_iter = time.time() - t_iter_start

            if verbose:
                print(f"Iter {iteration:3d} | "
                      f"ΔT={dT_max:.2e} ΔH2={dH2_max:.2e} ΔCO={dCO_max:.2e} | "
                      f"chem:{n_chem_converged}/{len(leaves)} | "
                      f"t={t_iter:.1f}s")

            if delta_max < convergence_tol and iteration > 2:
                converged = True
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"CONVERGED after {iteration+1} iterations")
                    print(f"Total time: {time.time()-t_start:.1f}s")
                    self._print_timing()
                break

            # ========================================
            # Step 8: Optional AMR refinement
            # ========================================
            if refine_every > 0 and iteration > 0 and iteration % refine_every == 0:
                n_refined = self.grid.refine_by_gradient('x_H2', threshold=0.05)
                if verbose and n_refined > 0:
                    print(f"  AMR: refined {n_refined} cells → {self.grid.n_cells} total")

        if not converged and verbose:
            print(f"\nWARNING: Did not converge after {max_iterations} iterations")
            print(f"Max change: {delta_max:.2e} (target: {convergence_tol:.2e})")

        return converged

    def _cell_to_abundances(self, d):
        """Convert CellData to abundance dict for chemistry solver."""
        from ..utils.constants import solar_abundances

        return {
            'H': d.x_HI,
            'H2': d.x_H2,
            'H+': d.x_Hp,
            'H-': 1e-12,
            'H2+': 1e-12,
            'H3+': 1e-10,
            'C+': d.x_Cp,
            'C': d.x_C,
            'CO': d.x_CO,
            'O': d.x_O,
            'O+': 1e-12,
            'OH': d.x_OH,
            'OH+': 1e-12,
            'H2O': d.x_H2O,
            'H2O+': 1e-12,
            'H3O+': 1e-12,
            'O2': getattr(d, 'x_O2', 1e-10),
            'CH': d.x_CHx,
            'CH+': 1e-12,
            'CH2+': 1e-14,
            'CO+': 1e-14,
            'HCO+': d.x_HCOp,
            'S': d.x_S,
            'S+': d.x_Sp,
            'Si': d.x_Si,
            'Si+': d.x_Sip,
            'Fe': d.x_Fe,
            'Fe+': d.x_Fep,
            'e-': d.x_e,
            'He': solar_abundances['He'],
            'He+': d.x_Hep,
        }

    def _abundances_to_cell(self, abundances, d):
        """Write abundance dict back to CellData."""
        d.x_HI = abundances.get('H', d.x_HI)
        d.x_H2 = abundances.get('H2', d.x_H2)
        d.x_Hp = abundances.get('H+', d.x_Hp)
        d.x_Cp = abundances.get('C+', d.x_Cp)
        d.x_C = abundances.get('C', d.x_C)
        d.x_CO = abundances.get('CO', d.x_CO)
        d.x_O = abundances.get('O', d.x_O)
        d.x_OH = abundances.get('OH', d.x_OH)
        d.x_H2O = abundances.get('H2O', d.x_H2O)
        d.x_CHx = abundances.get('CH', d.x_CHx)
        d.x_HCOp = abundances.get('HCO+', d.x_HCOp)
        d.x_S = abundances.get('S', d.x_S)
        d.x_Sp = abundances.get('S+', d.x_Sp)
        d.x_Si = abundances.get('Si', d.x_Si)
        d.x_Sip = abundances.get('Si+', d.x_Sip)
        d.x_Fe = abundances.get('Fe', d.x_Fe)
        d.x_Fep = abundances.get('Fe+', d.x_Fep)
        d.x_Hep = abundances.get('He+', d.x_Hep)

        # Electron fraction from charge balance
        d.x_e = (d.x_Cp + d.x_Sp + d.x_Sip + d.x_Fep + d.x_Hp
                + abundances.get('HCO+', 0) + abundances.get('H3+', 0))

    def _print_timing(self):
        """Print timing breakdown."""
        total = sum(self.timing.values())
        print(f"\nTiming breakdown:")
        for key, val in sorted(self.timing.items(), key=lambda x: -x[1]):
            print(f"  {key:15s}: {val:6.2f}s ({100*val/total:5.1f}%)")

    def get_1d_profile(self, axis=0):
        """
        Extract 1D profiles along an axis (for comparison with 1D codes).

        Returns arrays sorted by position along the axis.
        """
        leaves = self.grid.get_leaves()

        # Sort by position
        positions = np.array([l.center[axis] for l in leaves])
        order = np.argsort(positions)

        profile = {
            'position': positions[order],
            'A_V': np.array([leaves[i].data.A_V for i in order]),
            'T_gas': np.array([leaves[i].data.T_gas for i in order]),
            'T_dust': np.array([leaves[i].data.T_dust for i in order]),
            'n_H': np.array([leaves[i].data.n_H for i in order]),
            'G0': np.array([leaves[i].data.G0 for i in order]),
            'x_HI': np.array([leaves[i].data.x_HI for i in order]),
            'x_H2': np.array([leaves[i].data.x_H2 for i in order]),
            'x_Cp': np.array([leaves[i].data.x_Cp for i in order]),
            'x_C': np.array([leaves[i].data.x_C for i in order]),
            'x_CO': np.array([leaves[i].data.x_CO for i in order]),
            'x_O': np.array([leaves[i].data.x_O for i in order]),
            'x_e': np.array([leaves[i].data.x_e for i in order]),
            'x_OH': np.array([leaves[i].data.x_OH for i in order]),
            'zeta_CR': np.array([leaves[i].data.zeta_CR for i in order]),
            'Gamma': np.array([leaves[i].data.Gamma_total for i in order]),
            'Lambda': np.array([leaves[i].data.Lambda_total for i in order]),
        }

        return profile
