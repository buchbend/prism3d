"""
Precomputed lookup tables for vectorized THEMIS dust physics.

Replaces per-cell Python loops in THEMISDust with table interpolation
that can be evaluated over entire 3D grids in one vectorized call.

Two tables:
  1. PE heating: PE_energy(psi, E_g) per grain bin — encodes the charge-
     distribution physics.  At runtime, 32 np.interp calls replace
     3 × 12 × 15 Python loop iterations per cell.
  2. H2 formation: precomputed surface area sums per population so the
     rate is a simple vectorized function of (T_gas, T_dust, f_nano).

Build once at solver init (< 1 s), use every iteration.
"""

import numpy as np
from ..utils.constants import (
    k_boltz, eV_to_erg, e_charge, m_H, m_electron, sigma_sb, h_planck, c_light
)

# THEMIS constants (duplicated from themis.py for self-containment)
F_HABING = 1.6e-3        # erg cm⁻² s⁻¹
E_FUV_MEAN = 10.0 * eV_to_erg  # Mean FUV photon energy [erg]


class THEMISTables:
    """
    Precomputed tables for vectorized THEMIS PE heating and H2 formation.

    Parameters
    ----------
    themis : THEMISDust instance
        Used to extract grain populations and their properties.
    n_psi : int
        Number of grid points in charging parameter psi.
    n_Eg : int
        Number of grid points in band gap E_g.
    """

    def __init__(self, themis, n_psi=200, n_Eg=20):
        self.n_psi = n_psi
        self.n_Eg = n_Eg

        # Grids
        self.log_psi_grid = np.linspace(-1, 7, n_psi)  # log10(psi) from 0.1 to 10^7
        self.Eg_grid = np.linspace(0.01, 2.5, n_Eg)

        # Build tables from THEMIS populations
        self._build_pe_tables(themis)
        self._build_h2_tables(themis)

    # ----------------------------------------------------------------
    # PE heating tables
    # ----------------------------------------------------------------

    def _build_pe_tables(self, themis):
        """
        Build PE lookup tables: charge-averaged Y*E_pe per grain bin.

        For each size bin, tabulate the charge-averaged product
        Y(Z) * E_pe(Z) as a function of (psi, E_g).  At runtime,
        Gamma_PE = sum_bins [ n_gr * sigma_abs * phi_UV * table(psi, E_g) ]

        The charge loop is vectorized over all (psi, E_g) grid points
        simultaneously for each bin, keeping build time < 1 s.
        """
        all_pops = themis.populations
        bins = []

        for pop in all_pops:
            is_carbon = 'aC' in pop.name
            is_nano = pop.name == 'aC_nano'

            for i in range(pop.n_bins):
                bins.append({
                    'a': pop.a[i],
                    'sigma_abs': np.pi * pop.a[i]**2 * pop.Q_abs_FUV[i],
                    'dn_da_da': pop.dn_da[i] * pop.da[i],
                    'is_carbon': is_carbon,
                    'is_nano': is_nano,
                    'W_eV_base': 4.0 if is_carbon else 8.0,
                    'Y_0': 0.14 if is_carbon else 0.05,
                })

        self.n_bins = len(bins)
        self.bin_sigma_abs = np.array([b['sigma_abs'] for b in bins])
        self.bin_dn_da_da = np.array([b['dn_da_da'] for b in bins])
        self.bin_is_nano = np.array([b['is_nano'] for b in bins])

        # Table: shape (n_bins, n_psi, n_Eg)
        self.pe_table = np.zeros((self.n_bins, self.n_psi, self.n_Eg))

        # Grids as 2D arrays for vectorized charge loop
        psi_2d = 10.0 ** self.log_psi_grid[:, None]  # (n_psi, 1)
        Eg_2d = self.Eg_grid[None, :]                 # (1, n_Eg)

        for ib, b in enumerate(bins):
            a = b['a']
            Y_0 = b['Y_0']
            is_carbon = b['is_carbon']

            if is_carbon:
                W_eV_2d = 4.0 + Eg_2d  # (1, n_Eg)
            else:
                W_eV_2d = np.full((1, 1), 8.0)

            # Max charge: shape broadcasts to (1, n_Eg) or (1, 1)
            Z_max_2d = np.maximum(
                (10.0 - W_eV_2d) * a * eV_to_erg / e_charge**2, 0.5
            )

            # Determine the largest Z_max across all E_g values
            Z_max_int = int(np.max(Z_max_2d)) + 2
            Z_range = np.arange(-1, Z_max_int)  # (n_Z,)

            # Mean charge: shape (n_psi, 1)
            # Clip to Z_max per E_g: shape (n_psi, n_Eg)
            Z_mean_raw = psi_2d * a / 5e-7  # (n_psi, 1)
            Z_mean = np.minimum(Z_mean_raw, Z_max_2d)  # (n_psi, n_Eg)

            # Gaussian charge distribution: (n_psi, n_Eg, n_Z)
            sigma_Z = np.maximum(1.0, np.sqrt(Z_mean + 1))  # (n_psi, n_Eg)
            Z_3d = Z_range[None, None, :]  # (1, 1, n_Z)
            P_Z = np.exp(-0.5 * ((Z_3d - Z_mean[:, :, None])
                                  / sigma_Z[:, :, None])**2)
            P_Z /= np.sum(P_Z, axis=2, keepdims=True)

            # Yield per charge: Y_Z = Y_0 * max(1 - max(Z,0)/(Z_max+1), 0.01)
            Z_pos = np.maximum(Z_3d, 0)
            Y_Z = Y_0 * np.maximum(
                1.0 - Z_pos / (Z_max_2d[:, :, None] + 1), 0.01
            )
            # Small grain quantum yield enhancement
            if a < 1e-7:
                Y_Z = Y_Z * min(3.0, 1e-7 / a)

            # Ionization potential per charge
            IP = (W_eV_2d[:, :, None] * eV_to_erg
                  + Z_pos * e_charge**2 / max(a, 1e-8))
            E_pe = np.maximum(E_FUV_MEAN - IP, 0.0)

            # Charge-averaged Y * E_pe: sum over Z
            # Mask negligible P_Z
            YE = np.sum(P_Z * Y_Z * E_pe, axis=2)  # (n_psi, n_Eg)

            self.pe_table[ib] = YE

            # For silicates, E_g doesn't matter — replicate column 0
            if not is_carbon:
                self.pe_table[ib, :, :] = self.pe_table[ib, :, 0:1]

    def pe_heating_vec(self, G0, T, x_e, nH, f_nano, E_g):
        """
        Vectorized THEMIS PE heating rate [erg cm⁻³ s⁻¹].

        All inputs are arrays of the same shape (or broadcastable).
        Uses Rust backend when available for parallel per-cell evaluation.

        Parameters
        ----------
        G0, T, x_e, nH : ndarray
            Radiation field, temperature, electron fraction, density.
        f_nano : ndarray
            Nano-grain scaling factor per cell.
        E_g : ndarray
            Band gap per cell [eV].

        Returns
        -------
        Gamma_PE : ndarray
            PE heating rate per cell.
        """
        from .._rust_backend import has_rust, get_rust

        shape = nH.shape

        if has_rust():
            core = get_rust()
            to_f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()

            # Grid parameters (uniform spacing)
            log_psi_min = float(self.log_psi_grid[0])
            log_psi_step = float(self.log_psi_grid[1] - self.log_psi_grid[0])
            eg_min = float(self.Eg_grid[0])
            eg_step = float(self.Eg_grid[1] - self.Eg_grid[0])

            # Output array
            gamma_pe_out = np.zeros(nH.size, dtype=np.float64)

            core.pe_heating_vec(
                np.ascontiguousarray(self.pe_table, dtype=np.float64),
                log_psi_min, log_psi_step, self.n_psi,
                eg_min, eg_step, self.n_Eg,
                np.ascontiguousarray(self.bin_sigma_abs, dtype=np.float64),
                np.ascontiguousarray(self.bin_dn_da_da, dtype=np.float64),
                np.ascontiguousarray(self.bin_is_nano.astype(np.float64)),
                to_f64(G0), to_f64(T), to_f64(x_e), to_f64(nH),
                to_f64(f_nano), to_f64(E_g),
                gamma_pe_out,
            )
            return gamma_pe_out.reshape(shape)

        # ── Python fallback ──
        n_e = np.maximum(x_e * nH, 1e-10 * nH)
        psi = G0 * np.sqrt(T) / n_e
        log_psi = np.log10(np.maximum(psi, 0.1))

        phi_UV = F_HABING * G0 / E_FUV_MEAN  # photon flux [cm⁻² s⁻¹]

        flat_log_psi = log_psi.ravel()
        flat_Eg = E_g.ravel()
        n_cells = flat_log_psi.size

        # Precompute interpolation indices and weights for psi
        psi_idx, psi_w = _interp_weights(flat_log_psi, self.log_psi_grid)
        # Precompute interpolation indices and weights for E_g
        Eg_idx, Eg_w = _interp_weights(flat_Eg, self.Eg_grid)

        Gamma_PE = np.zeros(n_cells)

        for ib in range(self.n_bins):
            # Bilinear interpolation from table
            t = self.pe_table[ib]  # shape (n_psi, n_Eg)

            # 4 corners
            v00 = t[psi_idx, Eg_idx]
            v10 = t[psi_idx + 1, Eg_idx]
            v01 = t[psi_idx, Eg_idx + 1]
            v11 = t[psi_idx + 1, Eg_idx + 1]

            # Bilinear
            YE = (v00 * (1 - psi_w) * (1 - Eg_w) +
                  v10 * psi_w * (1 - Eg_w) +
                  v01 * (1 - psi_w) * Eg_w +
                  v11 * psi_w * Eg_w)

            # Scale nano-carbon bins by f_nano
            n_gr = self.bin_dn_da_da[ib]
            if self.bin_is_nano[ib]:
                Gamma_PE += (f_nano.ravel() * n_gr * nH.ravel()
                             * self.bin_sigma_abs[ib]
                             * phi_UV.ravel() * YE)
            else:
                Gamma_PE += (n_gr * nH.ravel()
                             * self.bin_sigma_abs[ib]
                             * phi_UV.ravel() * YE)

        return Gamma_PE.reshape(shape)

    # ----------------------------------------------------------------
    # H2 formation rate tables
    # ----------------------------------------------------------------

    def _build_h2_tables(self, themis):
        """
        Precompute grain surface area sums per population category.

        The H2 formation rate becomes:
          R = 0.5 * v_H * S_H * (eps_nano * Sigma_nano * f_nano
                                 + eps_largeC * Sigma_largeC
                                 + eps_Sil * Sigma_Sil)
        where Sigma_* are the precomputed total cross sections per H.
        """
        self.sigma_nano = 0.0
        self.sigma_largeC = 0.0
        self.sigma_Sil = 0.0

        for pop in themis.populations:
            area_sum = np.sum(np.pi * pop.a**2 * pop.dn_da * pop.da)
            if pop.name == 'aC_nano':
                self.sigma_nano = area_sum
            elif pop.name == 'aC_large':
                self.sigma_largeC = area_sum
            elif pop.name == 'aSil':
                self.sigma_Sil = area_sum

    def h2_formation_rate_vec(self, T_gas, T_dust, f_nano):
        """
        Vectorized H2 formation rate coefficient [cm³/s].

        Accounts for:
        - T-dependent sticking coefficient (Hollenbach & McKee 1979)
        - T_dust-dependent recombination efficiency per material
        - Nano-grain abundance scaling

        Rate = R * n(H) * n_H gives formation rate [cm⁻³ s⁻¹].
        """
        v_H = np.sqrt(8.0 * k_boltz * T_gas / (np.pi * m_H))

        # Sticking coefficient
        S_H = 1.0 / (1.0 + 0.04 * np.sqrt(T_gas + T_dust)
                      + 0.002 * T_gas + 8e-6 * T_gas**2)

        # Silicate recombination efficiency: physisorption-dominated
        eps_Sil = np.where(
            T_dust < 14, 1.0,
            np.where(T_dust < 25, np.exp(-(T_dust - 14) / 5.0), 0.01)
        )

        # Carbonaceous recombination efficiency: chemisorption up to ~400 K
        eps_C = np.maximum(
            np.where(
                T_dust < 50, 0.8,
                np.where(T_dust < 400,
                         0.8 * np.exp(-(T_dust - 50) / 200.0),
                         0.01)
            ),
            0.005  # Eley-Rideal floor
        )

        # Total rate coefficient
        R = 0.5 * v_H * S_H * (
            eps_C * self.sigma_nano * f_nano +
            eps_C * self.sigma_largeC +
            eps_Sil * self.sigma_Sil
        )

        return R

    # ----------------------------------------------------------------
    # Dust equilibrium temperature (vectorized)
    # ----------------------------------------------------------------

    def dust_temperature_vec(self, G0, A_V, nH=None, T_gas=None):
        """
        Vectorized dust equilibrium temperature.

        Uses THEMIS-calibrated scaling (accounts for grain composition).
        More accurate than the simple BT94 T_dust formula.
        """
        # Absorbed FUV + optical flux
        F_abs = F_HABING * G0 * np.exp(-1.8 * A_V)
        F_abs += F_HABING * 0.5 * np.exp(-A_V)  # optical ISRF
        F_abs = np.maximum(F_abs, 1e-30)

        # Silicate-dominated emission: beta=2
        # T = (F_abs * Q_ratio / (16 pi^2 sigma_SB a^2 Q_eff))^(1/6)
        # Simplified to a calibrated power law
        T_dust = 16.4 * (F_abs / F_HABING) ** 0.167  # ~ (1/6)

        T_dust = np.clip(T_dust, 5.0, 200.0)

        # Gas-grain coupling at high density
        if nH is not None and T_gas is not None:
            f_couple = np.clip(nH / 1e7, 0.0, 1.0)
            high_n = nH > 1e5
            T_dust = np.where(
                high_n,
                T_dust * (1 - f_couple) + T_gas * f_couple,
                T_dust
            )

        return T_dust


def _interp_weights(x, grid):
    """
    Compute interpolation indices and weights for 1D linear interpolation.

    Returns (idx, weight) such that:
      f(x) ≈ table[idx] * (1 - weight) + table[idx+1] * weight

    Clamps to grid boundaries.
    """
    n = len(grid)
    idx = np.searchsorted(grid, x) - 1
    idx = np.clip(idx, 0, n - 2)
    dx = grid[1] - grid[0]  # assumes uniform grid
    w = (x - grid[idx]) / dx
    w = np.clip(w, 0.0, 1.0)
    return idx, w
