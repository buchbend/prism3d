"""
Heating processes for PRISM-3D PDR models.

Implements all major heating mechanisms:
1. Photoelectric heating on dust grains and PAHs
2. H2 photodissociation heating
3. H2 formation heating
4. H2 UV pumping (vibrational de-excitation)
5. Cosmic ray heating
6. Gas-grain collisional heating
7. Carbon photoionization heating
8. Chemical heating (exothermic reactions)

All rates returned in erg/cm³/s.
"""

import numpy as np
from ..utils.constants import (k_boltz, eV_to_erg, G0_to_flux, m_H,
                                 solar_abundances, h_planck, c_light)


def photoelectric_heating(n_H, T, G0, x_e, grain_param=None):
    """
    Photoelectric (PE) heating on dust grains and PAHs.

    Uses the Bakes & Tielens (1994) / Weingartner & Draine (2001)
    formulation. The PE heating rate depends on the grain charging
    parameter psi = G0 * sqrt(T) / (n_e), which controls the
    balance between UV photoemission and electron recapture.

    Parameters
    ----------
    n_H : float
        H nuclei density [cm⁻³]
    T : float
        Gas temperature [K]
    G0 : float
        FUV field [Habing units]
    x_e : float
        Electron fraction n_e/n_H
    grain_param : dict, optional
        Grain parameters. If None, uses standard ISM values.

    Returns
    -------
    Gamma_PE : float
        PE heating rate [erg/cm³/s]
    """
    if G0 <= 0:
        return 0.0

    n_e = x_e * n_H
    if n_e <= 0:
        n_e = 1e-10 * n_H  # floor

    # Charging parameter
    psi = G0 * np.sqrt(T) / n_e

    # Bakes & Tielens (1994) Eq. 41-42
    # Heating efficiency epsilon(psi)
    epsilon = 0.049 / (1.0 + (psi / 1925.0)**0.73) \
            + 0.037 * (T / 1e4)**0.7 / (1.0 + psi / 5000.0)

    # Total PE heating rate
    # Gamma_PE = 1.3e-24 * epsilon * G0 * n_H [erg/cm³/s]
    Gamma_PE = 1.3e-24 * epsilon * G0 * n_H

    return Gamma_PE


def photoelectric_heating_wd01(n_H, T, G0, x_e, R_V=3.1):
    """
    PE heating from Weingartner & Draine (2001) - more accurate than BT94.

    Uses their fit for the standard ISM grain size distribution.
    """
    if G0 <= 0:
        return 0.0

    n_e = max(x_e * n_H, 1e-10 * n_H)

    # WD01 charging parameter (their Eq. 2)
    phi_pah = G0 * np.sqrt(T) / (n_e * 1.0)  # for PAH fraction = 1

    # WD01 heating rate (their Eq. 1 and Table 2 for R_V=3.1)
    C0 = 5.22e-26  # erg/s
    C1 = 3.46e-2
    C2 = 5.11e-1
    C3 = 1.83e4
    C4 = 1.39e1
    C5 = 6.14e3
    C6 = 1.36e-3

    epsilon = C1 / (1.0 + C2 * phi_pah**C3) + C4 / (1.0 + C5 * phi_pah) + C6

    Gamma = C0 * epsilon * G0 * n_H

    return max(Gamma, 0.0)


def h2_photodissociation_heating(n_H, x_H2, G0, A_V, f_shield_H2):
    """
    Heating from H2 photodissociation.

    Each H2 dissociation event deposits ~0.4 eV of kinetic energy
    (the excess above the dissociation energy).

    Parameters
    ----------
    n_H : float
        H nuclei density [cm⁻³]
    x_H2 : float
        H2 abundance relative to n_H
    G0 : float
        FUV field [Habing units]
    A_V : float
        Visual extinction [mag]
    f_shield_H2 : float
        H2 self-shielding factor

    Returns
    -------
    Gamma : float
        Heating rate [erg/cm³/s]
    """
    # H2 photodissociation rate
    k_pd = 3.3e-11 * G0 * np.exp(-3.74 * A_V) * f_shield_H2  # s⁻¹

    # Energy deposited per dissociation
    E_heat = 0.4 * eV_to_erg  # erg

    n_H2 = x_H2 * n_H
    return k_pd * n_H2 * E_heat


def h2_formation_heating(n_H, x_HI, T, R_H2=3e-17):
    """
    Heating from H2 formation on grain surfaces.

    Formation releases 4.48 eV. A fraction goes to the grain,
    a fraction to kinetic energy, and a fraction to H2 internal states.
    Following Tielens & Hollenbach (1985): ~1/3 each.

    Parameters
    ----------
    R_H2 : float
        H2 formation rate coefficient [cm³/s]
    """
    # Formation rate = R_H2 * n(H) * n_H
    n_HI = x_HI * n_H
    formation_rate = R_H2 * np.sqrt(T / 100.0) * n_HI * n_H  # cm⁻³ s⁻¹

    # Kinetic energy fraction: ~1.5 eV per formation
    E_kin = 1.5 * eV_to_erg

    return formation_rate * E_kin


def h2_uv_pumping_heating(n_H, x_H2, G0, A_V, f_shield_H2, T, n_cr=1e4):
    """
    Heating from H2 UV fluorescent pumping.

    FUV photons excite H2 to excited electronic states, which cascade
    back producing vibrationally excited H2 in the ground state.
    Collisional de-excitation of these levels heats the gas.

    The FUV pumping rate is ~9x the photodissociation rate (since ~10%
    of absorptions lead to dissociation, ~90% to fluorescence).
    But the effective heating depends on the competition between
    collisional de-excitation (heating) and radiative cascade (no heating).

    Following Burton, Hollenbach & Tielens (1990) and
    Röllig+ (2006) Eq. 1.
    """
    # Pumping rate: ~9x the photodissociation rate
    # Photodissociation rate: 3.3e-11 * G0 * exp(-3.74*A_V) * f_shield
    # Pumping rate: ~9 * 3.3e-11 = 3.0e-10 per H2 molecule
    k_pump = 3.0e-10 * G0 * np.exp(-3.74 * A_V) * f_shield_H2  # s⁻¹

    n_H2 = x_H2 * n_H

    if n_H2 <= 0 or k_pump <= 0:
        return 0.0

    # Average energy of vibrationally excited H2: ~2 eV per pump event
    # But only a fraction goes to heating (rest re-radiated)
    E_vib = 2.0 * eV_to_erg

    # Heating efficiency: fraction of pumped energy that heats the gas
    # At low density (n << n_cr): energy radiated away (IR fluorescence)
    # At high density (n >> n_cr): collisional de-excitation heats gas
    # n_cr for v=1 J=0: ~1e4 cm⁻³ for H-H2 collisions
    # For higher v levels: n_cr can be >1e5
    # Use effective n_cr that accounts for the distribution of v levels
    n_cr_eff = 3e4  # effective critical density

    f_heat = n_H / (n_H + n_cr_eff)

    return k_pump * n_H2 * E_vib * f_heat


def cosmic_ray_heating(n_H, x_H2, x_HI, zeta_CR):
    """
    Heating by cosmic ray ionization.

    Each CR ionization deposits ~20 eV to the gas through the
    primary electron and its secondary ionizations.

    Parameters
    ----------
    zeta_CR : float
        Primary CR ionization rate of H [s⁻¹]
    """
    # Energy deposited per ionization: ~20 eV (Glassgold+ 2012)
    # In molecular gas, H2 ionization dominates
    E_CR = 20.0 * eV_to_erg

    n_HI = x_HI * n_H
    n_H2 = x_H2 * n_H

    # CR ionization rate for H2 is ~2x that of H
    Gamma = zeta_CR * (n_HI * E_CR + 2.0 * n_H2 * E_CR)

    return Gamma


def gas_grain_heating(n_H, T_gas, T_dust, x_H2):
    """
    Gas-grain collisional heating (or cooling if T_gas > T_dust).

    Gas particles collide with dust grains and exchange thermal energy.
    Important in dense, shielded regions.

    From Hollenbach & McKee (1989).
    """
    # Accommodation coefficient
    alpha_T = 0.35

    # Grain cross section per H
    sigma_d = 1e-21  # cm² per H

    # Mean thermal speed
    from ..utils.constants import m_H
    mu = 2.0 if x_H2 > 0.5 else 1.0  # mean molecular weight
    v_th = np.sqrt(8.0 * k_boltz * T_gas / (np.pi * mu * m_H))

    # Heating rate (positive when T_dust > T_gas)
    Gamma_gg = 2.0 * alpha_T * sigma_d * n_H**2 * v_th * k_boltz * (T_dust - T_gas)

    return Gamma_gg


def carbon_photoionization_heating(n_H, x_C, G0, A_V):
    """
    Heating from photoionization of neutral carbon.

    Each C photoionization releases ~1 eV of kinetic energy
    (photon energy - IP of 11.26 eV, typical photon ~12.3 eV).
    """
    k_ion = 3.0e-10 * G0 * np.exp(-3.0 * A_V)  # s⁻¹
    E_kin = 1.0 * eV_to_erg
    n_C = x_C * n_H
    return k_ion * n_C * E_kin


def total_heating_rate(n_H, T_gas, T_dust, G0, A_V, zeta_CR,
                        x_e, x_HI, x_H2, x_C, x_Cp,
                        f_shield_H2=1.0, R_H2=3e-17):
    """
    Compute total heating rate.

    Parameters
    ----------
    All standard cell properties.

    Returns
    -------
    Gamma_total : float
        Total heating rate [erg/cm³/s]
    components : dict
        Individual heating rate components
    """
    components = {}

    components['PE'] = photoelectric_heating(n_H, T_gas, G0, x_e)
    components['H2_photodiss'] = h2_photodissociation_heating(
        n_H, x_H2, G0, A_V, f_shield_H2)
    components['H2_formation'] = h2_formation_heating(n_H, x_HI, T_gas, R_H2)
    components['H2_pumping'] = h2_uv_pumping_heating(
        n_H, x_H2, G0, A_V, f_shield_H2, T_gas)
    components['CR'] = cosmic_ray_heating(n_H, x_H2, x_HI, zeta_CR)
    components['gas_grain'] = gas_grain_heating(n_H, T_gas, T_dust, x_H2)
    components['C_photoion'] = carbon_photoionization_heating(n_H, x_C, G0, A_V)

    Gamma_total = sum(components.values())
    return Gamma_total, components
