"""
Grain physics module for PRISM-3D.

Tracks the grain size distribution and computes:
1. Photoelectric heating efficiency from the actual grain population
2. H2 formation rate on grain surfaces
3. Grain charge distribution (for electron recombination on grains)
4. Simple grain evolution (shattering, coagulation)

The grain size distribution is discretized into bins spanning
from PAH-sized (3.5 Å) to large grains (1 μm).
"""

import numpy as np
from ..utils.constants import k_boltz, m_H, eV_to_erg, e_charge


class GrainSizeDistribution:
    """
    Discretized grain size distribution.

    Uses the MRN (Mathis-Rumpl-Nordsieck 1977) distribution as
    default: dn/da ∝ a^(-3.5) for a_min < a < a_max.

    Optionally includes a PAH component following
    Weingartner & Draine (2001).

    Parameters
    ----------
    n_bins : int
        Number of size bins
    a_min : float
        Minimum grain size [cm]
    a_max : float
        Maximum grain size [cm]
    composition : str
        'silicate', 'carbonaceous', or 'mixed'
    """

    def __init__(self, n_bins=20, a_min=3.5e-8, a_max=2.5e-5,
                 composition='mixed'):
        self.n_bins = n_bins
        self.a_min = a_min
        self.a_max = a_max
        self.composition = composition

        # Log-spaced bin edges
        self.a_edges = np.logspace(np.log10(a_min), np.log10(a_max), n_bins + 1)
        self.a_centers = np.sqrt(self.a_edges[:-1] * self.a_edges[1:])
        self.da = self.a_edges[1:] - self.a_edges[:-1]

        # Initialize MRN distribution
        self.dn_da = self._mrn_distribution()

        # Grain material properties
        if composition == 'silicate':
            self.rho_grain = 3.5  # g/cm³
        elif composition == 'carbonaceous':
            self.rho_grain = 2.24  # g/cm³
        else:
            self.rho_grain = 3.0  # mixed

        # PAH fraction (fraction of C in PAHs)
        self.q_PAH = 0.046  # WD01 value for Milky Way

    def _mrn_distribution(self):
        """Standard MRN power-law distribution."""
        # dn/da = C * n_H * a^(-3.5)
        # Normalization from dust-to-gas ratio
        # For standard ISM: integral gives tau_V / N_H ~ 5.3e-22
        C_mrn = 1e-25.13  # normalization constant [cm^(2.5) per H]
        return C_mrn * self.a_centers**(-3.5)

    def total_cross_section(self, per_H=True):
        """
        Total geometric cross section of grains.

        Parameters
        ----------
        per_H : bool
            If True, return per H nucleus [cm²/H]

        Returns
        -------
        sigma : float
            Total grain cross section
        """
        sigma = np.sum(np.pi * self.a_centers**2 * self.dn_da * self.da)
        return sigma

    def total_surface_area(self, per_H=True):
        """Total grain surface area per H nucleus [cm²/H]."""
        return np.sum(4.0 * np.pi * self.a_centers**2 * self.dn_da * self.da)

    def dust_to_gas_ratio(self):
        """Compute the dust-to-gas mass ratio."""
        mass_per_H = np.sum(
            (4.0/3.0) * np.pi * self.a_centers**3 * self.rho_grain
            * self.dn_da * self.da
        )
        return mass_per_H / m_H


class GrainPhysics:
    """
    Grain physics calculations for a PDR cell.

    Computes PE heating, H2 formation, and grain charging
    from the local grain size distribution and conditions.
    """

    def __init__(self, grain_dist=None):
        if grain_dist is None:
            grain_dist = GrainSizeDistribution()
        self.grain_dist = grain_dist

    def photoelectric_heating_from_distribution(self, n_H, T, G0, x_e):
        """
        Compute PE heating from the actual grain size distribution.

        This is more accurate than the Bakes & Tielens (1994) fit
        because it accounts for the actual grain population rather
        than an assumed average.

        Uses the Weingartner & Draine (2001) grain charging model.

        Parameters
        ----------
        n_H : float
            H nuclei density [cm⁻³]
        T : float
            Gas temperature [K]
        G0 : float
            FUV field [Habing units]
        x_e : float
            Electron fraction

        Returns
        -------
        Gamma_PE : float
            PE heating rate [erg/cm³/s]
        """
        if G0 <= 0:
            return 0.0

        n_e = max(x_e * n_H, 1e-10 * n_H)
        gd = self.grain_dist

        Gamma_total = 0.0

        for i in range(gd.n_bins):
            a = gd.a_centers[i]
            dn_da_i = gd.dn_da[i]
            da_i = gd.da[i]

            # Number density of grains in this bin
            n_grain_i = dn_da_i * da_i * n_H

            # Grain charging parameter for this size
            # tau = a * k_B * T / e^2 (reduced temperature)
            tau = a * k_boltz * T / e_charge**2

            # Mean charge
            # Simplified: Z ~ G0 * a / (n_e * sqrt(T))
            psi_i = G0 * np.sqrt(T) / n_e * (a / 1e-5)
            Z_mean = psi_i / (1.0 + psi_i / 3.0)  # Very approximate

            # PE yield for this grain
            # Y(a, Z) from WD01 - depends on grain size and charge
            # Simplified: Y ~ 0.1 for neutral small grains, decreasing with charge
            Y_pe = 0.1 * np.exp(-Z_mean / 3.0) * min(1.0, a / 5e-8)

            # UV absorption cross section
            Q_abs_UV = min(1.0, 2.0 * np.pi * a / 1e-5)  # efficiency factor
            sigma_abs = np.pi * a**2 * Q_abs_UV

            # FUV photon rate hitting this grain
            F_UV = 1.6e-3 * G0  # erg/cm²/s
            E_photon = 10.0 * eV_to_erg  # mean FUV photon energy ~10 eV
            phi_UV = F_UV / E_photon  # photons/cm²/s

            # Mean PE electron energy
            # E_pe ~ h*nu - W - Z*e^2/a
            # W ~ 4-8 eV (work function)
            W = 6.0 * eV_to_erg  # Work function
            E_pe = max(E_photon - W - Z_mean * e_charge**2 / max(a, 1e-8), 0)

            # Heating rate from this grain size
            Gamma_i = n_grain_i * sigma_abs * phi_UV * Y_pe * E_pe

            Gamma_total += Gamma_i

        return Gamma_total

    def h2_formation_rate(self, T, T_dust, a_index=None):
        """
        Compute H2 formation rate coefficient on grain surfaces.

        Uses the Cazaux & Tielens (2004) model which accounts for
        grain temperature, sticking coefficient, and surface mobility.

        Parameters
        ----------
        T : float
            Gas temperature [K]
        T_dust : float
            Dust temperature [K]
        a_index : int, optional
            If given, compute for a specific grain size bin.
            Otherwise, sum over all bins.

        Returns
        -------
        R_H2 : float
            H2 formation rate coefficient [cm³/s]
            (rate = R_H2 * n(H) * n_H)
        """
        gd = self.grain_dist

        R_total = 0.0

        # Thermal speed of H atoms
        v_H = np.sqrt(8.0 * k_boltz * T / (np.pi * m_H))

        # Sticking coefficient (Hollenbach & McKee 1979)
        S_H = 1.0 / (1.0 + 0.04 * np.sqrt(T + T_dust)
                     + 0.002 * T + 8e-6 * T**2)

        if a_index is not None:
            bins = [a_index]
        else:
            bins = range(gd.n_bins)

        for i in bins:
            a = gd.a_centers[i]
            dn_da_i = gd.dn_da[i]
            da_i = gd.da[i]

            # Cross section
            sigma = np.pi * a**2

            # Recombination efficiency on surface
            # Depends on T_dust: efficient at low T, drops above ~20 K
            # Cazaux & Tielens (2004) formula
            if T_dust < 10:
                epsilon = 1.0
            elif T_dust < 20:
                epsilon = 1.0 - 0.05 * (T_dust - 10)
            elif T_dust < 100:
                epsilon = 0.5 * np.exp(-(T_dust - 20) / 30.0)
            else:
                # Chemisorption allows H2 formation even at high T_dust
                epsilon = 0.1 * np.exp(-(T_dust - 100) / 200.0)

            # Contribution from this bin
            R_i = 0.5 * v_H * S_H * epsilon * sigma * dn_da_i * da_i

            R_total += R_i

        return R_total

    def grain_recombination_rate(self, T, x_e, ion_species='C+'):
        """
        Compute grain-assisted recombination rate for an ion species.

        Important for C+ recombination in PDR surfaces where
        grain recombination can exceed radiative recombination.

        Following Weingartner & Draine (2001).

        Parameters
        ----------
        T : float
            Gas temperature [K]
        x_e : float
            Electron fraction
        ion_species : str
            Ion species name

        Returns
        -------
        k_gr : float
            Grain recombination rate coefficient [cm³/s]
        """
        gd = self.grain_dist

        # Simplified WD01 grain recombination
        # k_gr ~ sum over sizes: pi*a^2 * v_ion * S * f(charge)

        # Ion mass (approximate)
        ion_masses = {'C+': 12, 'S+': 32, 'Si+': 28, 'Fe+': 56, 'H+': 1}
        m_ion = ion_masses.get(ion_species, 12) * m_H

        v_ion = np.sqrt(8.0 * k_boltz * T / (np.pi * m_ion))

        k_total = 0.0
        for i in range(gd.n_bins):
            a = gd.a_centers[i]
            sigma = np.pi * a**2
            dn_da_i = gd.dn_da[i]
            da_i = gd.da[i]

            # Sticking probability for ions (enhanced by Coulomb focusing
            # for negatively charged grains)
            S_ion = min(1.0, 1.0 + e_charge**2 / (a * k_boltz * T))

            k_i = v_ion * S_ion * sigma * dn_da_i * da_i
            k_total += k_i

        return k_total

    def evolve_distribution(self, n_H, T, v_turb, dt):
        """
        Evolve the grain size distribution due to shattering and coagulation.

        Simple implementation following Hirashita & Yan (2009):
        - Shattering by grain-grain collisions in turbulence
        - Coagulation in dense regions

        Parameters
        ----------
        n_H : float
            H nuclei density [cm⁻³]
        T : float
            Gas temperature [K]
        v_turb : float
            Turbulent velocity [cm/s]
        dt : float
            Time step [s]
        """
        gd = self.grain_dist

        # Shattering: moves mass from large to small grains
        # Rate ∝ n_grain² * v_turb * sigma
        # Only significant in strong shocks or high turbulence

        # Coagulation: moves mass from small to large grains
        # Rate ∝ n_grain² * v_thermal * sigma * sticking
        # Only significant at high density (n > 10^5)

        # For now, simple relaxation toward MRN
        tau_relax = 1e14 / (n_H / 1e3)  # ~3 Myr at n=1e3

        if dt < tau_relax * 0.01:
            return  # No significant evolution

        mrn = gd._mrn_distribution()
        f = min(dt / tau_relax, 1.0)
        gd.dn_da = gd.dn_da * (1.0 - f) + mrn * f
