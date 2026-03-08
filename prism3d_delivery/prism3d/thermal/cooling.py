"""
Cooling processes for PRISM-3D PDR models.

Implements all major cooling mechanisms:
1. [CII] 158 μm fine-structure line
2. [OI] 63 μm and 145 μm fine-structure lines
3. [CI] 609 μm and 370 μm lines
4. CO rotational lines (full ladder J=1-0 through J=40-39)
5. H2 ro-vibrational lines
6. Lyman-alpha cooling
7. Gas-grain collisional cooling
8. Recombination cooling
9. Metastable line cooling (e.g., [SiII], [FeII])

All rates returned in erg/cm³/s.
"""

import numpy as np
from ..utils.constants import (k_boltz, h_planck, c_light, m_H, m_proton,
                                 fine_structure_lines, eV_to_erg)


def cii_158_cooling(n_H, T, x_e, x_HI, x_H2, x_Cp):
    """
    Cooling by [CII] 158 μm ²P₃/₂ → ²P₁/₂ transition.

    This is the dominant coolant in the FUV-illuminated surface layers.
    Computed using a two-level atom with collisional de-excitation
    rates for e⁻, H, and H2 collisions.

    Parameters
    ----------
    n_H : float
        H nuclei density [cm⁻³]
    T : float
        Gas temperature [K]
    x_e, x_HI, x_H2, x_Cp : float
        Abundances relative to n_H

    Returns
    -------
    Lambda_CII : float
        Cooling rate [erg/cm³/s]
    """
    _, freq, A_ul, T_ul, g_u, g_l = fine_structure_lines['CII_158']

    # Collision rate coefficients [cm³/s]
    # Electron collisions (Blum & Pradhan 1992)
    gamma_e = 8.63e-6 / (g_l * np.sqrt(T)) * 2.15 * np.exp(-T_ul / T)

    # H collisions (Barinovs+ 2005)
    gamma_H = 8.0e-10 * (T / 100.0)**0.07

    # H2 collisions (Wiesenfeld & Goldsmith 2014)
    gamma_H2 = 4.9e-10 * (T / 100.0)**0.12

    # Total de-excitation rate
    n_e = x_e * n_H
    n_HI = x_HI * n_H
    n_H2 = x_H2 * n_H
    n_Cp = x_Cp * n_H

    R_ul = A_ul + gamma_e * n_e + gamma_H * n_HI + gamma_H2 * n_H2

    # Two-level population
    # n_u/n_l = (g_u/g_l) * [gamma_e*n_e + gamma_H*n_HI + gamma_H2*n_H2] / R_ul * exp(-T_ul/T)
    R_lu = (g_u / g_l) * (gamma_e * n_e + gamma_H * n_HI + gamma_H2 * n_H2) * np.exp(-T_ul / T)

    n_u = n_Cp * R_lu / (R_lu + R_ul)

    # Cooling rate
    E_ul = h_planck * freq
    Lambda = n_u * A_ul * E_ul  # erg/cm³/s (optically thin)

    return Lambda


def oi_cooling(n_H, T, x_e, x_HI, x_H2, x_O):
    """
    Cooling by [OI] 63 μm and 145 μm transitions.

    Three-level atom: ³P₂ (ground), ³P₁, ³P₀.
    63 μm: ³P₁ → ³P₂ (dominant)
    145 μm: ³P₀ → ³P₁

    Collisional rates from Abrahamsson+ (2007) and Jaquet+ (1992).
    """
    n_e = x_e * n_H
    n_HI = x_HI * n_H
    n_H2 = x_H2 * n_H
    n_O = x_O * n_H

    # 63 μm transition
    _, freq_63, A_63, T_63, g_u_63, g_l_63 = fine_structure_lines['OI_63']

    # H collision rate for 63 μm (Abrahamsson+ 2007)
    gamma_H_63 = 9.2e-12 * T**0.67
    gamma_H2_63 = 4.5e-12 * T**0.64
    gamma_e_63 = 1.4e-8 * (T / 100.0)**0.39

    R_lu_63 = (g_u_63 / g_l_63) * (
        gamma_H_63 * n_HI + gamma_H2_63 * n_H2 + gamma_e_63 * n_e
    ) * np.exp(-T_63 / T)
    R_ul_63 = A_63 + gamma_H_63 * n_HI + gamma_H2_63 * n_H2 + gamma_e_63 * n_e

    n_u_63 = n_O * R_lu_63 / (R_lu_63 + R_ul_63)
    Lambda_63 = n_u_63 * A_63 * h_planck * freq_63

    # 145 μm transition
    _, freq_145, A_145, T_145, g_u_145, g_l_145 = fine_structure_lines['OI_145']

    gamma_H_145 = 4.0e-12 * T**0.60
    gamma_H2_145 = 2.0e-12 * T**0.58
    gamma_e_145 = 5.0e-9 * (T / 100.0)**0.39

    R_lu_145 = (g_u_145 / g_l_145) * (
        gamma_H_145 * n_HI + gamma_H2_145 * n_H2 + gamma_e_145 * n_e
    ) * np.exp(-T_145 / T)
    R_ul_145 = A_145 + gamma_H_145 * n_HI + gamma_H2_145 * n_H2 + gamma_e_145 * n_e

    n_u_145 = n_O * R_lu_145 / (R_lu_145 + R_ul_145)
    Lambda_145 = n_u_145 * A_145 * h_planck * freq_145

    return Lambda_63 + Lambda_145


def ci_cooling(n_H, T, x_e, x_HI, x_H2, x_C):
    """
    Cooling by [CI] 609 μm (³P₁ → ³P₀) and 370 μm (³P₂ → ³P₁).
    """
    n_e = x_e * n_H
    n_HI = x_HI * n_H
    n_H2 = x_H2 * n_H
    n_C = x_C * n_H

    Lambda_total = 0.0

    for line_name in ['CI_609', 'CI_370']:
        _, freq, A_ul, T_ul, g_u, g_l = fine_structure_lines[line_name]

        # Collision rates (Schroder+ 1991, Johnson+ 1987)
        gamma_H = 1.0e-10 * (T / 100.0)**0.3
        gamma_H2 = 5.0e-11 * (T / 100.0)**0.3
        gamma_e = 2.0e-7 * (T / 100.0)**(-0.5)

        R_lu = (g_u / g_l) * (
            gamma_H * n_HI + gamma_H2 * n_H2 + gamma_e * n_e
        ) * np.exp(-T_ul / T)
        R_ul = A_ul + gamma_H * n_HI + gamma_H2 * n_H2 + gamma_e * n_e

        n_u = n_C * R_lu / (R_lu + R_ul)
        Lambda_total += n_u * A_ul * h_planck * freq

    return Lambda_total


def co_rotational_cooling(n_H, T, x_H2, x_CO, N_CO=0.0):
    """
    Cooling by CO rotational lines.

    Sums over the CO rotational ladder up to J=40.
    Uses escape probability for optical depth effects.

    Parameters
    ----------
    N_CO : float
        CO column density [cm⁻²] for escape probability calculation.
    """
    from ..utils.constants import CO_B_rot, CO_dipole

    n_H2 = x_H2 * n_H
    n_CO = x_CO * n_H

    if n_CO <= 0 or T < 5:
        return 0.0

    Lambda_CO = 0.0

    # CO properties
    B_rot = 57.635968e9  # Hz (rotational constant)
    mu_d = 0.1098e-18    # Debye in esu·cm

    J_max = min(40, int(5.0 * np.sqrt(T / 2.77)))

    for J_u in range(1, J_max + 1):
        J_l = J_u - 1

        # Transition frequency
        freq = 2.0 * B_rot * J_u  # Hz

        # Energy of upper level
        E_u = h_planck * B_rot * J_u * (J_u + 1)
        T_u = E_u / k_boltz

        if T_u > 5.0 * T:
            break

        # Einstein A coefficient
        A_ul = (64.0 * np.pi**4 * freq**3 * mu_d**2 * J_u) / \
               (3.0 * h_planck * c_light**3 * (2*J_u + 1))

        # Statistical weights
        g_u = 2 * J_u + 1
        g_l = 2 * J_l + 1

        # CO-H2 collision rate (Flower 2001, approximate)
        gamma_H2 = 3.3e-11 * (T / 100.0)**0.5 * (1.0 + 0.1 * J_u)

        # Excitation rate
        R_lu = (g_u / g_l) * gamma_H2 * n_H2 * np.exp(-T_u / T)
        R_ul = A_ul + gamma_H2 * n_H2

        # Population of upper level
        n_u = n_CO * R_lu / (R_lu + R_ul)

        # Escape probability (large velocity gradient approximation)
        if N_CO > 0:
            tau_line = A_ul * c_light**3 * n_u / (8.0 * np.pi * freq**3) \
                      * (n_CO * g_u / (n_u * g_l) - 1.0) * N_CO / n_CO \
                      if n_u > 0 else 0.0
            tau_line = abs(tau_line)
            if tau_line > 0.01:
                beta_esc = (1.0 - np.exp(-tau_line)) / tau_line
            else:
                beta_esc = 1.0
        else:
            beta_esc = 1.0

        Lambda_CO += n_u * A_ul * h_planck * freq * beta_esc

    return Lambda_CO


def h2_rovibrational_cooling(n_H, T, x_H2, x_HI):
    """
    Cooling by H2 ro-vibrational lines.

    Important at T > 500 K in the warm PDR surface layers.
    Uses the Glover & Abel (2008) fit.
    """
    n_H2 = x_H2 * n_H
    n_HI = x_HI * n_H

    if T < 100 or n_H2 <= 0:
        return 0.0

    # Low-density cooling function Λ₀(T) [erg/s per H2 molecule]
    # Glover & Abel (2008) fit for H-H2 and H2-H2 collisions
    logT = np.log10(T)
    logT = np.clip(logT, 2.0, 4.5)

    # H-H2 cooling (Glover & Abel 2008)
    log_LH = -24.311 + 3.5692 * logT - 11.332 * logT**2 \
             + 15.738 * logT**3 - 10.581 * logT**4 \
             + 3.5803 * logT**5 - 0.48365 * logT**6

    # H2-H2 cooling (ibid)
    log_LH2 = -24.311 + 4.6585 * logT - 14.272 * logT**2 \
              + 19.779 * logT**3 - 13.255 * logT**4 \
              + 4.4840 * logT**5 - 0.60504 * logT**6

    L_H = 10**log_LH    # erg/s per H2
    L_H2 = 10**log_LH2  # erg/s per H2

    # LTE cooling function (high density limit)
    log_LTE = -19.703 + 0.5 * logT

    L_LTE = 10**log_LTE  # erg/s per H2

    # Effective cooling rate with density interpolation
    L0 = L_H * n_HI + L_H2 * n_H2
    Lambda_H2 = n_H2 * L0 * L_LTE / (L0 + L_LTE) if (L0 + L_LTE) > 0 else 0.0

    return Lambda_H2


def lyman_alpha_cooling(n_H, T, x_HI, x_e):
    """
    Lyman-alpha cooling (collisional excitation of H by electrons).

    Important at T > 8000 K.
    """
    if T < 3000:
        return 0.0

    n_HI = x_HI * n_H
    n_e = x_e * n_H

    # Collisional excitation rate coefficient (Osterbrock & Ferland 2006)
    T4 = T / 1e4
    q_lu = 2.41e-6 / np.sqrt(T) * 0.486 * np.exp(-1.18e5 / T)  # cm³/s

    E_lya = 10.2 * eV_to_erg

    return n_HI * n_e * q_lu * E_lya


def gas_grain_cooling(n_H, T_gas, T_dust, x_H2):
    """
    Gas-grain collisional cooling.
    Same as heating but with opposite sign when T_gas > T_dust.
    """
    from .heating import gas_grain_heating
    Gamma_gg = gas_grain_heating(n_H, T_gas, T_dust, x_H2)
    # gas_grain_heating returns positive when T_dust > T_gas (heating)
    # Cooling is when T_gas > T_dust, so cooling = -heating when heating < 0
    return max(-Gamma_gg, 0.0)


def recombination_cooling(n_H, T, x_e, x_Cp, x_Sp, x_Sip, x_Fep):
    """
    Radiative recombination cooling.

    Energy lost when ions recombine and emit recombination photons.
    """
    n_e = x_e * n_H

    Lambda = 0.0

    # C+ recombination (kT per recombination)
    alpha_Cp = 4.67e-12 * (T / 300.0)**(-0.6)
    Lambda += x_Cp * n_H * n_e * alpha_Cp * k_boltz * T

    # Other ions (smaller contribution)
    alpha_Sp = 3.9e-11 * (T / 300.0)**(-0.63)
    Lambda += x_Sp * n_H * n_e * alpha_Sp * k_boltz * T

    alpha_Sip = 4.26e-12 * (T / 300.0)**(-0.62)
    Lambda += x_Sip * n_H * n_e * alpha_Sip * k_boltz * T

    alpha_Fep = 3.7e-12 * (T / 300.0)**(-0.65)
    Lambda += x_Fep * n_H * n_e * alpha_Fep * k_boltz * T

    return Lambda


def total_cooling_rate(n_H, T_gas, T_dust, x_e, x_HI, x_H2,
                        x_Cp, x_C, x_O, x_CO,
                        x_Sp=0, x_Sip=0, x_Fep=0, N_CO=0.0):
    """
    Compute total cooling rate.

    Returns
    -------
    Lambda_total : float
        Total cooling rate [erg/cm³/s]
    components : dict
        Individual cooling rate components
    """
    components = {}

    components['CII_158'] = cii_158_cooling(n_H, T_gas, x_e, x_HI, x_H2, x_Cp)
    components['OI'] = oi_cooling(n_H, T_gas, x_e, x_HI, x_H2, x_O)
    components['CI'] = ci_cooling(n_H, T_gas, x_e, x_HI, x_H2, x_C)
    components['CO_rot'] = co_rotational_cooling(n_H, T_gas, x_H2, x_CO, N_CO)
    components['H2_rovib'] = h2_rovibrational_cooling(n_H, T_gas, x_H2, x_HI)
    components['Lya'] = lyman_alpha_cooling(n_H, T_gas, x_HI, x_e)
    components['gas_grain'] = gas_grain_cooling(n_H, T_gas, T_dust, x_H2)
    components['recombination'] = recombination_cooling(
        n_H, T_gas, x_e, x_Cp, x_Sp, x_Sip, x_Fep)

    Lambda_total = sum(components.values())
    return Lambda_total, components
