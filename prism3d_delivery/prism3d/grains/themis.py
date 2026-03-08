"""
THEMIS Dust Model for PRISM-3D.

Implements the Heterogeneous dust Evolution Model for Interstellar Solids
(Jones et al. 2013, 2017; Ysard et al. 2024 — THEMIS 2.0) adapted for
self-consistent coupling to PDR gas chemistry and thermal balance.

Three grain populations:
  1. a-C(:H) nano-particles (a = 0.4–20 nm): aromatic/aliphatic carbon
     → stochastically heated, produce MIR AIB emission
     → dominate photoelectric heating
     → evolve via UV photo-processing (aromatization, destruction)
  2. Large a-C(:H) grains (a = 20 nm – 0.2 μm): carbonaceous grains
     → in thermal equilibrium
     → contribute to FUV/visible extinction
  3. Large a-Sil grains (a = 20 nm – 0.3 μm): amorphous silicates
     → in thermal equilibrium
     → dominate FIR/submm emission

Key physics:
  - Band gap E_g controls aromatic vs aliphatic character
  - UV processing lowers E_g (aliphatic → aromatic)
  - Photo-destruction of nano-particles in strong FUV
  - Size-dependent optical properties Q_abs(a, λ)
  - Full PE heating from charge distribution (replaces BT94)
  - Stochastic heating via temperature probability distribution P(T)
  - Feedback to gas: PE heating, H₂ formation, electron balance, UV opacity

References:
  Jones et al. 2013, A&A 558, A62 — THEMIS framework
  Jones et al. 2017, A&A 602, A46 — Global THEMIS model
  Compiègne et al. 2011, A&A 525, A103 — DustEM code
  Elyajouri et al. 2024, A&A 685, A76 — JWST dust evolution Orion Bar
  Ysard et al. 2024 — THEMIS 2.0
"""

import numpy as np
from ..utils.constants import (
    k_boltz, h_planck, c_light, eV_to_erg,
    e_charge, m_H, pc_cm
)

# Import with correct names
from ..utils.constants import sigma_sb as sigma_SB
from ..utils.constants import m_electron as m_e


# ============================================================
# Physical constants for dust
# ============================================================

# Carbon atom mass
m_C = 12.0 * m_H  # g

# Debye specific heat: C_grain ≈ N_atoms × k_B at high T
# At low T, C ∝ T³ (Debye model)
DEBYE_T_CARBON = 863.0  # K, Debye temperature for graphite
DEBYE_T_SILICATE = 500.0  # K, Debye temperature for silicates

# FUV energy range
E_FUV_MIN = 6.0 * eV_to_erg   # 6 eV
E_FUV_MAX = 13.6 * eV_to_erg  # 13.6 eV
E_FUV_MEAN = 10.0 * eV_to_erg # Mean FUV photon energy

# Standard ISRF FUV flux (Habing field)
F_HABING = 1.6e-3  # erg cm⁻² s⁻¹ (6-13.6 eV)

# Dust-to-gas mass ratio (standard ISM)
DUST_TO_GAS_STANDARD = 0.01

# THEMIS band gap for diffuse ISM carbon nano-grains
EG_DIFFUSE = 0.1   # eV — partially aromatic (UV-processed in diffuse ISM)
EG_DENSE = 2.5     # eV — highly aliphatic (shielded in dense regions)
EG_AROMATIC = 0.0   # eV — fully aromatic (graphitic)


# ============================================================
# THEMIS Grain Population
# ============================================================

class THEMISPopulation:
    """
    A single THEMIS grain population (nano-carbons, large carbons, or silicates).
    
    Each population has a size distribution, material properties, and
    optical properties that depend on size and band gap.
    
    Parameters
    ----------
    name : str
        Population name ('aC_nano', 'aC_large', 'aSil')
    a_min : float
        Minimum grain radius [cm]
    a_max : float
        Maximum grain radius [cm]
    n_bins : int
        Number of size bins
    rho : float
        Material density [g/cm³]
    E_g : float
        Band gap [eV] — controls aromatic/aliphatic character
        Only meaningful for carbonaceous grains.
    slope : float
        Power-law slope of dn/da (default -3.5 for MRN)
    mass_fraction : float
        Fraction of total dust mass in this population
    """
    
    def __init__(self, name, a_min, a_max, n_bins=15,
                 rho=2.24, E_g=0.1, slope=-3.5, mass_fraction=0.1):
        self.name = name
        self.a_min = a_min
        self.a_max = a_max
        self.n_bins = n_bins
        self.rho = rho
        self.E_g = E_g  # Band gap [eV]
        self.slope = slope
        self.mass_fraction = mass_fraction
        
        # Size grid
        self.a_edges = np.logspace(np.log10(a_min), np.log10(a_max), n_bins + 1)
        self.a = np.sqrt(self.a_edges[:-1] * self.a_edges[1:])  # geometric mean
        self.da = self.a_edges[1:] - self.a_edges[:-1]
        
        # Size distribution dn/da per H nucleus [cm⁻¹ per H]
        raw = self.a**slope
        # Normalize to target mass fraction
        mass_integral = np.sum((4./3.) * np.pi * self.a**3 * rho * raw * self.da)
        target_mass = mass_fraction * DUST_TO_GAS_STANDARD * m_H
        self.dn_da = raw * (target_mass / mass_integral) if mass_integral > 0 else raw * 0
        
        # Number of carbon atoms per grain (for nano-carbons)
        self.N_C = (4./3.) * np.pi * self.a**3 * rho / m_C
        
        # Precompute optical properties
        self._compute_optics()
    
    def _compute_optics(self):
        """
        Compute absorption and scattering efficiencies.
        
        Uses analytic approximations calibrated to THEMIS/DustEM tables.
        Q_abs depends on size, wavelength, and band gap.
        """
        # FUV absorption efficiency (integrated 6-13.6 eV)
        # Small grains: Q_abs ≈ 1 (geometric limit for a > λ/2π)
        # Very small grains: Q_abs ∝ a (Rayleigh regime)
        lambda_FUV = 0.1e-4  # ~1000 Å = 0.1 μm representative FUV wavelength
        x = 2 * np.pi * self.a / lambda_FUV  # size parameter
        
        # Carbonaceous grains (higher FUV absorption than silicates)
        if 'aC' in self.name:
            self.Q_abs_FUV = np.where(x > 1, 1.0, x)
            # Band gap effect: lower E_g → more UV absorption
            E_g_factor = 1.0 + 0.5 * (0.1 - min(self.E_g, 0.5))
            self.Q_abs_FUV *= np.clip(E_g_factor, 0.5, 1.5)
        else:
            # Silicates: lower FUV efficiency
            self.Q_abs_FUV = np.where(x > 1, 0.7, 0.7 * x)
        
        # Optical/NIR absorption (for extinction curve)
        lambda_V = 0.55e-4  # V band
        x_V = 2 * np.pi * self.a / lambda_V
        self.Q_abs_V = np.where(x_V > 1, 1.0, x_V**2)
        
        # FIR/submm emissivity: Q_abs ∝ a * (ν/ν₀)^β
        # β ≈ 2 for silicates, 1.5-2 for amorphous carbon
        self.beta_FIR = 2.0 if 'Sil' in self.name else 1.8
        
        # Reference: Q_abs at 250 μm for a = 0.1 μm grain
        self.Q_abs_250_ref = 1e-3 if 'Sil' in self.name else 5e-4
    
    def Q_abs_at(self, i_bin, wavelength_cm):
        """
        Absorption efficiency for size bin i at wavelength(s).
        
        Parameters
        ----------
        i_bin : int
            Size bin index
        wavelength_cm : float or ndarray
            Wavelength(s) [cm]
        
        Returns
        -------
        Q_abs : float or ndarray
        """
        a = self.a[i_bin]
        wl = np.atleast_1d(wavelength_cm)
        x = 2 * np.pi * a / wl
        
        Q = np.empty_like(wl)
        uv = wl < 0.3e-4
        opt = (~uv) & (wl < 5e-4)
        fir = wl >= 5e-4
        
        Q[uv] = self.Q_abs_FUV[i_bin]
        Q[opt] = np.where(x[opt] > 1, 1.0, x[opt]**2)
        
        if np.any(fir):
            nu = c_light / wl[fir]
            nu_ref = c_light / 250e-4
            Q_ref = self.Q_abs_250_ref * (a / 1e-5)
            Q[fir] = Q_ref * (nu / nu_ref)**self.beta_FIR
        
        return Q if len(Q) > 1 else float(Q[0])
    
    def cross_section_FUV(self):
        """Total FUV extinction cross section per H [cm²/H]."""
        return np.sum(np.pi * self.a**2 * self.Q_abs_FUV * self.dn_da * self.da)
    
    def total_surface_area(self):
        """Total grain surface area per H [cm²/H]."""
        return np.sum(4.0 * np.pi * self.a**2 * self.dn_da * self.da)
    
    def total_number(self):
        """Total grain number per H."""
        return np.sum(self.dn_da * self.da)
    
    def mass_per_H(self):
        """Total grain mass per H nucleus [g/H]."""
        return np.sum((4./3.) * np.pi * self.a**3 * self.rho * self.dn_da * self.da)


# ============================================================
# Full THEMIS Dust Model
# ============================================================

class THEMISDust:
    """
    Complete THEMIS dust model with three populations.
    
    Provides:
    - Photoelectric heating from grain charge distribution
    - H₂ formation on grain surfaces
    - Dust temperature (equilibrium for large, stochastic for nano)
    - Dust SED per cell
    - Dust evolution (nano-grain depletion by UV)
    
    Parameters
    ----------
    f_nano : float
        Scaling factor for nano-grain abundance relative to diffuse ISM.
        1.0 = standard, 0.07 = Orion Bar edge (Elyajouri+ 2024)
    E_g : float
        Band gap of nano-carbons [eV]. 0.1 = diffuse ISM, higher = denser/shielded
    """
    
    def __init__(self, f_nano=1.0, E_g=0.1):
        self.f_nano = f_nano
        self.E_g = E_g
        
        # Build three populations following THEMIS diffuse ISM model
        # Mass fractions calibrated to give:
        #   σ_FUV/H ~ 2e-21 cm²
        #   R_H2 ~ 3e-17 cm³/s
        #   dust-to-gas ~ 0.01
        #   n_PAH/n_H ~ 1e-7 (WD01 standard)
        
        # Nano-carbons: normalized to total PAH number = 1e-7 per H
        self.nano_C = THEMISPopulation(
            'aC_nano',
            a_min=4e-8,    # 0.4 nm
            a_max=2e-7,    # 20 nm
            n_bins=12,
            rho=1.6,
            E_g=E_g,
            slope=-3.5,
            mass_fraction=0.005 * f_nano
        )
        # Override: normalize nano-C to total number = 1e-7 per H
        n_target = 1e-7 * f_nano
        n_current = self.nano_C.total_number()
        if n_current > 0:
            self.nano_C.dn_da *= n_target / n_current
        
        self.large_C = THEMISPopulation(
            'aC_large',
            a_min=2e-7, a_max=2e-5, n_bins=10,
            rho=1.6, E_g=0.1, slope=-3.5, mass_fraction=0.30
        )
        
        self.silicate = THEMISPopulation(
            'aSil',
            a_min=2e-7, a_max=3e-5, n_bins=10,
            rho=3.5, slope=-3.5, mass_fraction=0.695
        )
        
        self.populations = [self.nano_C, self.large_C, self.silicate]
    
    # --------------------------------------------------------
    # Photoelectric heating
    # --------------------------------------------------------
    
    def photoelectric_heating(self, n_H, T, G0, x_e):
        """
        Compute PE heating from the full grain charge distribution.
        
        This replaces the Bakes & Tielens (1994) analytic fit with a
        proper WD01-style calculation summing over all grain sizes.
        The key improvement: accounts for nano-grain depletion and
        band-gap-dependent yields.
        
        Parameters
        ----------
        n_H : float
            H number density [cm⁻³]
        T : float
            Gas temperature [K]
        G0 : float
            FUV field [Habing units]
        x_e : float
            Electron fraction
        
        Returns
        -------
        Gamma_PE : float
            PE heating rate [erg cm⁻³ s⁻¹]
        """
        if G0 <= 0 or T <= 0:
            return 0.0
        
        n_e = max(x_e, 1e-10) * n_H
        Gamma_total = 0.0
        
        # FUV photon flux
        phi_UV = F_HABING * G0 / E_FUV_MEAN
        
        # Charging parameter (controls mean grain charge)
        psi = G0 * np.sqrt(T) / n_e
        
        for pop in self.populations:
            # Material properties
            if 'aC' in pop.name:
                W_eV = 4.0 + pop.E_g  # Work function
                Y_0 = 0.14
            else:
                W_eV = 8.0
                Y_0 = 0.05
            
            for i in range(pop.n_bins):
                a = pop.a[i]
                n_gr = pop.dn_da[i] * pop.da[i] * n_H
                
                if n_gr <= 0:
                    continue
                
                sigma_abs = np.pi * a**2 * pop.Q_abs_FUV[i]
                
                # Maximum charge: where IP = photon energy
                Z_max = max((10.0 - W_eV) * a * eV_to_erg / e_charge**2, 0.5)
                
                # Average over charge distribution P(Z)
                # In WD01, the charge distribution is roughly Gaussian
                # centered near Z_mean with width ~1-2 charges.
                # Crucially, the NEUTRAL fraction always contributes.
                
                # Mean charge (saturates at Z_max)
                Z_mean = min(psi * a / 5e-7, Z_max)
                
                # Sum over charges Z = -1, 0, 1, ..., Z_max
                Z_range = np.arange(-1, int(Z_max) + 2)
                
                # Gaussian charge distribution centered on Z_mean
                sigma_Z = max(1.0, np.sqrt(Z_mean + 1))
                P_Z = np.exp(-0.5 * ((Z_range - Z_mean) / sigma_Z)**2)
                P_Z /= np.sum(P_Z)
                
                Gamma_grain = 0.0
                for k, Z in enumerate(Z_range):
                    if P_Z[k] < 1e-6:
                        continue
                    
                    # Yield at this charge
                    Y_Z = Y_0 * max(1.0 - max(Z, 0) / (Z_max + 1), 0.01)
                    
                    # Small grain quantum yield enhancement
                    if a < 1e-7:
                        Y_Z *= min(3.0, 1e-7 / a)
                    
                    # PE electron energy at this charge
                    IP = W_eV * eV_to_erg + max(Z, 0) * e_charge**2 / max(a, 1e-8)
                    E_pe = max(E_FUV_MEAN - IP, 0)
                    
                    Gamma_grain += P_Z[k] * Y_Z * E_pe
                
                Gamma_total += n_gr * sigma_abs * phi_UV * Gamma_grain
        
        return Gamma_total
    
    # --------------------------------------------------------
    # H₂ formation
    # --------------------------------------------------------
    
    def h2_formation_rate(self, T, T_dust):
        """
        H₂ formation rate coefficient on all grain surfaces [cm³/s].
        
        Accounts for:
        - Size-dependent surface area
        - Temperature-dependent sticking
        - Chemisorbed sites on carbonaceous grains (high-T formation)
        - Physisorbed sites on silicates (low-T formation)
        
        Returns R such that formation rate = R × n(H) × n_H [cm⁻³ s⁻¹]
        """
        v_H = np.sqrt(8.0 * k_boltz * T / (np.pi * m_H))
        
        # Sticking coefficient (Hollenbach & McKee 1979, updated)
        S_H = 1.0 / (1.0 + 0.04 * np.sqrt(T + T_dust)
                     + 0.002 * T + 8e-6 * T**2)
        
        R_total = 0.0
        
        for pop in self.populations:
            for i in range(pop.n_bins):
                sigma = np.pi * pop.a[i]**2
                dn_da_i = pop.dn_da[i]
                da_i = pop.da[i]
                
                # Surface recombination efficiency
                if 'Sil' in pop.name:
                    # Silicates: physisorption-dominated
                    # Efficient at T_dust < 20 K, drops sharply above
                    if T_dust < 14:
                        eps = 1.0
                    elif T_dust < 25:
                        eps = np.exp(-(T_dust - 14) / 5.0)
                    else:
                        eps = 0.01
                else:
                    # Carbonaceous: chemisorption allows high-T formation
                    # Cazaux & Tielens (2004): efficient up to ~400 K
                    if T_dust < 50:
                        eps = 0.8
                    elif T_dust < 400:
                        eps = 0.8 * np.exp(-(T_dust - 50) / 200.0)
                    else:
                        eps = 0.01
                    # Eley-Rideal mechanism at very high T
                    # (KOSMA-τ 2022 update)
                    eps = max(eps, 0.005)
                
                R_total += 0.5 * v_H * S_H * eps * sigma * dn_da_i * da_i
        
        return R_total
    
    # --------------------------------------------------------
    # Dust temperature
    # --------------------------------------------------------
    
    def equilibrium_temperature(self, G0, A_V=0):
        """
        Equilibrium dust temperature for large grains.
        
        Balances FUV/optical absorption against FIR emission.
        Returns dict of {population_name: T_dust_array}.
        """
        T_eq = {}
        
        for pop in [self.large_C, self.silicate]:
            # Absorbed power per grain: sigma_abs * F_UV * exp(-1.8 AV)
            # + optical/NIR ISRF contribution
            F_abs = F_HABING * G0 * np.exp(-1.8 * A_V)
            # Add optical ISRF (roughly equal to FUV in energy)
            F_abs += F_HABING * 0.5 * np.exp(-A_V)
            
            P_abs = np.pi * pop.a**2 * pop.Q_abs_FUV * F_abs
            
            # Emitted power: 4π * sigma_SB * T^4 * <Q_abs>_Planck * 4π a²
            # <Q_abs>_Planck ~ Q_0 * (a/a_0) * (T/T_0)^beta
            # Solving: T = [P_abs / (16π² σ_SB a² Q_eff)]^(1/(4+beta))
            
            beta = pop.beta_FIR
            Q_eff_ref = pop.Q_abs_250_ref  # at 250 μm, a = 0.1 μm, T ~ 20K
            # Scale: Q_eff(a,T) ≈ Q_ref * (a/1e-5) * (T/20)^beta
            # This gives: P_abs = 16π² σ_SB a² Q_ref (a/1e-5) T^(4+beta) / 20^beta
            
            denom = 16 * np.pi**2 * sigma_SB * pop.a**2 * Q_eff_ref * (pop.a / 1e-5) / 20.0**beta
            denom = np.maximum(denom, 1e-50)
            
            T = np.power(P_abs / denom, 1.0 / (4.0 + beta))
            T = np.clip(T, 5.0, 2000.0)  # Physical bounds
            
            T_eq[pop.name] = T
        
        return T_eq
    
    def stochastic_temperature_distribution(self, G0, A_V=0, n_T_bins=50):
        """
        Temperature probability distribution P(T) for nano-grains.
        
        Small grains undergo temperature spikes from individual
        UV photon absorptions. The time-averaged P(T) determines
        the MIR emission spectrum (PAH bands, VSG continuum).
        
        Uses the method of Draine & Li (2001): solve the thermal
        balance with discretized temperature bins.
        
        Returns
        -------
        T_grid : ndarray (n_T_bins,)
        P_T : ndarray (n_bins, n_T_bins)
            P(T) for each size bin of nano-carbons
        """
        T_grid = np.logspace(np.log10(5.0), np.log10(2500.0), n_T_bins)
        pop = self.nano_C
        P_T = np.zeros((pop.n_bins, n_T_bins))
        
        F_abs = F_HABING * G0 * np.exp(-1.8 * A_V)
        phi_UV = F_abs / E_FUV_MEAN  # FUV photon flux [cm⁻² s⁻¹]
        
        for i in range(pop.n_bins):
            a = pop.a[i]
            N_C = pop.N_C[i]
            
            if N_C < 10:
                N_C = 10  # Minimum for meaningful heat capacity
            
            # Photon absorption rate
            sigma_abs = np.pi * a**2 * pop.Q_abs_FUV[i]
            R_abs = sigma_abs * phi_UV  # photons/s
            
            if R_abs <= 0:
                P_T[i, 0] = 1.0
                continue
            
            # Heat capacity: C(T) = N_atoms * k_B * f_Debye(T/T_D)
            # Simplified: C ≈ N_C * k_B * min(1, (T/T_D)³) for carbon
            T_D = DEBYE_T_CARBON
            C_grain = N_C * k_boltz * np.minimum(1.0, (T_grid / T_D)**3)
            C_grain = np.maximum(C_grain, k_boltz)  # Floor
            
            # Temperature after absorbing one FUV photon
            # ΔT ≈ E_photon / C(T_base)
            # For small grains: T_peak can be >> T_eq
            
            # Cooling rate: dE/dt = -4π a² σ_SB <Q_abs> T⁴
            # ≈ 4π a² * Q_eff * σ_SB * T^(4+beta)
            Q_eff = pop.Q_abs_250_ref * (a / 1e-5)
            cool_rate = 4 * np.pi * a**2 * Q_eff * sigma_SB * T_grid**(4 + pop.beta_FIR) / 20.0**pop.beta_FIR
            
            # Cooling time: t_cool = C * T / cool_rate
            t_cool = C_grain * T_grid / np.maximum(cool_rate, 1e-50)
            
            # Time between photon absorptions
            t_abs = 1.0 / max(R_abs, 1e-30)
            
            # Steady-state P(T): balance heating spikes with cooling
            # Grains spend most time near T_min (just before next photon)
            # P(T) ∝ t_cool(T) for T near equilibrium,
            # with spikes extending to T_peak after absorption
            
            # Equilibrium temperature (for reference)
            P_abs_grain = sigma_abs * F_abs
            denom_eq = 4 * np.pi * a**2 * Q_eff * sigma_SB / 20.0**pop.beta_FIR
            T_eq_grain = max((P_abs_grain / max(denom_eq, 1e-50))**(1.0/(4+pop.beta_FIR)), 5.0)
            
            # For very small grains (N_C < 200): truly stochastic
            # For larger grains: P(T) → delta(T - T_eq)
            if N_C > 500 or t_abs > 100 * t_cool[n_T_bins // 2]:
                # Thermal equilibrium regime
                idx_eq = np.argmin(np.abs(T_grid - T_eq_grain))
                P_T[i, idx_eq] = 1.0
            else:
                # Stochastic regime
                # T_peak after one photon absorption from T_base ~ 5-10 K
                C_base = N_C * k_boltz * min(1.0, (10.0 / T_D)**3)
                T_peak = min((E_FUV_MEAN / max(C_base, k_boltz))**0.5 * 10.0, 2500.0)
                
                # Simple P(T) model: power law between T_min and T_peak
                # P(T) ∝ T^(-alpha) with alpha ~ 1 + beta
                # (grains cool fast at high T, so spend less time there)
                alpha = 1.0 + pop.beta_FIR
                
                mask = (T_grid >= max(T_eq_grain * 0.5, 5.0)) & (T_grid <= T_peak)
                if np.any(mask):
                    P_T[i, mask] = T_grid[mask]**(-alpha)
                    # Add peak at equilibrium
                    idx_eq = np.argmin(np.abs(T_grid - T_eq_grain))
                    P_T[i, idx_eq] += 10.0 * T_eq_grain**(-alpha)
                else:
                    idx_eq = np.argmin(np.abs(T_grid - T_eq_grain))
                    P_T[i, idx_eq] = 1.0
            
            # Normalize
            norm = np.sum(P_T[i])
            if norm > 0:
                P_T[i] /= norm
        
        return T_grid, P_T
    
    # --------------------------------------------------------
    # Dust SED
    # --------------------------------------------------------
    
    def compute_SED(self, G0, A_V=0, wavelengths=None):
        """
        Compute the dust emission SED for one cell.
        
        Returns emissivity j_ν per H nucleus at each wavelength.
        
        Parameters
        ----------
        G0 : float
            FUV field [Habing]
        A_V : float
            Visual extinction [mag]
        wavelengths : ndarray, optional
            Wavelengths [cm]. Default: 1 μm – 1 mm.
        
        Returns
        -------
        wavelengths : ndarray [cm]
        j_nu : ndarray [erg s⁻¹ Hz⁻¹ sr⁻¹ per H]
        """
        if wavelengths is None:
            wavelengths = np.logspace(np.log10(1e-4), np.log10(0.1), 100)
        
        nu = c_light / wavelengths
        j_nu = np.zeros_like(nu)
        
        # Large grains: thermal equilibrium emission
        T_eq = self.equilibrium_temperature(G0, A_V)
        
        for pop in [self.large_C, self.silicate]:
            T_arr = T_eq[pop.name]
            for i in range(pop.n_bins):
                Q_abs = pop.Q_abs_at(i, wavelengths)
                sigma = np.pi * pop.a[i]**2
                dn_da_i = pop.dn_da[i]
                da_i = pop.da[i]
                
                # Planck function B_ν(T)
                T_i = T_arr[i]
                x = h_planck * nu / (k_boltz * T_i)
                x = np.clip(x, 0, 500)
                B_nu = 2 * h_planck * nu**3 / c_light**2 / (np.exp(x) - 1 + 1e-30)
                
                j_nu += sigma * Q_abs * B_nu * dn_da_i * da_i
        
        # Nano-grains: stochastic emission
        T_grid, P_T = self.stochastic_temperature_distribution(G0, A_V)
        pop = self.nano_C
        
        for i in range(pop.n_bins):
            Q_abs = pop.Q_abs_at(i, wavelengths)
            sigma = np.pi * pop.a[i]**2
            dn_da_i = pop.dn_da[i]
            da_i = pop.da[i]
            
            # Integrate over P(T)
            for j in range(len(T_grid)):
                if P_T[i, j] <= 0:
                    continue
                T_j = T_grid[j]
                x = h_planck * nu / (k_boltz * T_j)
                x = np.clip(x, 0, 500)
                B_nu = 2 * h_planck * nu**3 / c_light**2 / (np.exp(x) - 1 + 1e-30)
                
                dT = T_grid[min(j+1, len(T_grid)-1)] - T_grid[max(j-1, 0)]
                j_nu += sigma * Q_abs * B_nu * P_T[i, j] * dn_da_i * da_i
        
        return wavelengths, j_nu
    
    # --------------------------------------------------------
    # Dust evolution
    # --------------------------------------------------------
    
    def evolve(self, G0, n_H, dt):
        """
        Evolve the dust population based on local conditions.
        
        Implements two key THEMIS evolution processes:
        1. Photo-destruction of nano-grains by FUV (Elyajouri+ 2024)
        2. Band gap evolution (UV aromatization)
        
        Parameters
        ----------
        G0 : float
            FUV field [Habing]
        n_H : float
            Density [cm⁻³] (for accretion in dense regions)
        dt : float
            Time step [s]
        
        Returns
        -------
        f_nano_new : float
            Updated nano-grain fraction
        E_g_new : float
            Updated band gap [eV]
        """
        # 1. Photo-destruction of nano-carbons
        # Timescale from Elyajouri+ (2024): τ_dest ~ 10⁴ yr at G0 = 10⁴
        # τ ∝ 1/G0 for photodestruction
        tau_dest = 1e4 * 3.15e7 / max(G0 / 1e4, 0.01)  # seconds
        
        # Destruction rate
        f_dest = np.exp(-dt / tau_dest)
        
        # 2. Replenishment from large grain fragmentation
        # Shattering in turbulence or UV-driven fragmentation
        # Timescale much longer: ~10⁶ yr
        tau_replenish = 1e6 * 3.15e7
        f_replenish = min(dt / tau_replenish, 0.1)  # Slow
        
        # 3. In dense shielded gas (G0 < 1, n > 10⁴):
        # Nano-grains can re-form via accretion of C from gas
        if G0 < 1 and n_H > 1e4:
            tau_accrete = 1e5 * 3.15e7 * (1e4 / n_H)
            f_accrete = min(dt / tau_accrete, 0.1)
        else:
            f_accrete = 0.0
        
        # Update f_nano
        f_nano_new = self.f_nano * f_dest + (1.0 - self.f_nano) * f_replenish + f_accrete
        f_nano_new = np.clip(f_nano_new, 0.01, 2.0)  # Floor and ceiling
        
        # 4. Band gap evolution: UV lowers E_g (aromatization)
        # In strong FUV: E_g → 0 on timescale ~10³ yr
        # In shielded regions: E_g stays high or increases (hydrogenation)
        tau_aromat = 1e3 * 3.15e7 / max(G0, 0.01)
        E_g_eq = 0.01 if G0 > 10 else min(0.1 + 0.5 * np.log10(max(n_H, 1)), 2.5)
        
        E_g_new = self.E_g + (E_g_eq - self.E_g) * min(dt / tau_aromat, 1.0)
        E_g_new = np.clip(E_g_new, 0.01, 2.5)
        
        return f_nano_new, E_g_new
    
    # --------------------------------------------------------
    # FUV extinction
    # --------------------------------------------------------
    
    def sigma_FUV_per_H(self):
        """Total FUV dust cross section per H nucleus [cm²]."""
        return sum(pop.cross_section_FUV() for pop in self.populations)
    
    def A_V_per_NH(self):
        """Visual extinction per H column [mag cm²]."""
        sigma_V = sum(
            np.sum(np.pi * pop.a**2 * pop.Q_abs_V * pop.dn_da * pop.da)
            for pop in self.populations
        )
        return sigma_V / 1.086  # Convert optical depth to mag
    
    # --------------------------------------------------------
    # Electron balance (PAH-assisted recombination)
    # --------------------------------------------------------
    
    def grain_electron_recombination_rate(self, T, G0, x_e, n_H):
        """
        Electron recombination rate on grain surfaces.
        
        Nano-carbons (PAH analogs) are the dominant electron sink
        in the surface layers of PDRs.
        
        Returns rate coefficient k_gr such that
        dn_e/dt_grain = -k_gr × n_e × n_H [cm⁻³ s⁻¹]
        """
        pop = self.nano_C
        n_e = max(x_e, 1e-10) * n_H
        v_e = np.sqrt(8 * k_boltz * T / (np.pi * m_e))
        
        k_total = 0.0
        for i in range(pop.n_bins):
            a = pop.a[i]
            sigma = np.pi * a**2
            
            # Grain charge (simplified — use mean Z from PE calculation)
            psi = G0 * np.sqrt(T) / n_e if n_e > 0 else 1e10
            Z = min(psi * a / 1e-7, 10)  # Very rough
            
            # Electron sticking enhanced by Coulomb for Z > 0
            if Z > 0:
                S_e = 1.0 + Z * e_charge**2 / (a * k_boltz * T)
            else:
                S_e = max(1.0 + Z * e_charge**2 / (a * k_boltz * T), 0.0)
            
            k_total += v_e * sigma * S_e * pop.dn_da[i] * pop.da[i]
        
        return k_total
    
    # --------------------------------------------------------
    # Summary / diagnostics
    # --------------------------------------------------------
    
    def summary(self):
        """Print a summary of the dust model state."""
        print(f"THEMIS Dust Model:")
        print(f"  Nano-grain fraction: {self.f_nano:.3f} (1.0 = diffuse ISM)")
        print(f"  Band gap E_g: {self.E_g:.2f} eV")
        print(f"  σ_FUV per H: {self.sigma_FUV_per_H():.2e} cm²")
        print(f"  A_V/N_H: {self.A_V_per_NH():.2e} mag cm²")
        for pop in self.populations:
            print(f"  {pop.name}: n/H = {pop.total_number():.2e}, "
                  f"area/H = {pop.total_surface_area():.2e} cm²/H, "
                  f"mass/H = {pop.mass_per_H():.2e} g/H")
        print(f"  R_H2 (T=50K, Td=20K): {self.h2_formation_rate(50, 20):.2e} cm³/s")
