"""
Physical and astronomical constants in CGS units for PRISM-3D.
"""

import numpy as np

# Fundamental constants (CGS)
c_light = 2.99792458e10        # Speed of light [cm/s]
h_planck = 6.62607015e-27      # Planck constant [erg·s]
k_boltz = 1.380649e-16         # Boltzmann constant [erg/K]
m_proton = 1.67262192e-24      # Proton mass [g]
m_electron = 9.10938370e-28    # Electron mass [g]
e_charge = 4.80320451e-10      # Electron charge [esu]
G_grav = 6.67430e-8            # Gravitational constant [cm³/g/s²]
sigma_sb = 5.670374e-5         # Stefan-Boltzmann [erg/cm²/s/K⁴]
a_rad = 4.0 * sigma_sb / c_light  # Radiation constant

# Atomic / molecular
m_H = 1.6735575e-24            # Hydrogen atom mass [g]
m_H2 = 2.0 * m_H               # H2 mass [g]
m_He = 6.6464764e-24            # Helium mass [g]
m_C = 1.9944235e-23             # Carbon mass [g]
m_O = 2.6567629e-23             # Oxygen mass [g]

# Energy conversions
eV_to_erg = 1.602176634e-12     # 1 eV in erg
eV_to_K = 1.160451812e4         # 1 eV in Kelvin
Ry_to_eV = 13.605693            # 1 Rydberg in eV

# Astronomical units
pc_cm = 3.0856776e18            # Parsec in cm
au_cm = 1.49597871e13           # AU in cm
Msun_g = 1.98892e33             # Solar mass in g
Lsun_erg = 3.828e33             # Solar luminosity in erg/s
yr_s = 3.15576e7                # Year in seconds

# ISM parameters
mu_H = 1.4                     # Mean molecular weight per H nucleus (with He)
X_H = 0.7154                   # Hydrogen mass fraction (solar)
Y_He = 0.2703                  # Helium mass fraction (solar)

# FUV radiation field
# Habing field: G0 = 1 corresponds to 1.6e-3 erg/cm²/s integrated 6-13.6 eV
# Draine field: chi = 1 corresponds to 2.7e-3 erg/cm²/s
G0_to_flux = 1.6e-3            # G0=1 flux [erg/cm²/s] (Habing 1968)
chi_to_flux = 2.7e-3           # chi=1 flux [erg/cm²/s] (Draine 1978)
G0_to_chi = G0_to_flux / chi_to_flux  # ~0.59

# Dust properties
sigma_dust_FUV = 1.8e-21       # FUV dust cross section per H [cm²] (at ~1000 Å)
# A_V / N_H = 5.34e-22 mag cm² (Bohlin+ 1978)
AV_per_NH = 5.34e-22           # Visual extinction per H column [mag/cm⁻²]
NH_per_AV = 1.0 / AV_per_NH   # H column per A_V [cm⁻² / mag]
# R_V = A_V / E(B-V), typically 3.1
R_V = 3.1
# FUV extinction relative to visual: A_FUV / A_V
tau_FUV_per_AV = 1.8           # ~1.8 for standard ISM dust

# Standard solar abundances (by number, relative to H)
# Asplund+ (2009) values
solar_abundances = {
    'H':  1.0,
    'He': 8.51e-2,
    'C':  2.69e-4,
    'N':  6.76e-5,
    'O':  4.90e-4,
    'Ne': 8.51e-5,
    'Mg': 3.98e-5,
    'Si': 3.24e-5,
    'S':  1.32e-5,
    'Fe': 3.16e-5,
}

# Gas-phase depletion factors for the ISM (Jenkins 2009, Savage & Sembach 1996)
# These represent the fraction of each element in the gas phase
# (the rest is locked in dust grains)
# For dense molecular clouds, depletion is stronger
ism_depletion = {
    'C':  0.4,    # 40% in gas phase (60% in grains/PAHs)
    'N':  1.0,    # Nitrogen is essentially undepleted
    'O':  0.6,    # 60% in gas phase (O depletion is moderate)
    'Mg': 0.01,   # Heavily depleted into silicates
    'Si': 0.03,   # Heavily depleted into silicates
    'S':  0.4,    # Moderate depletion (debated, 0.1-1.0)
    'Fe': 0.006,  # Very heavily depleted into grains
}

# Gas-phase abundances = solar * depletion factor
# These are what the chemistry should use
gas_phase_abundances = {
    'H':  1.0,
    'He': 8.51e-2,
    'C':  solar_abundances['C'] * ism_depletion['C'],       # 1.08e-4
    'N':  solar_abundances['N'] * ism_depletion['N'],       # 6.76e-5
    'O':  solar_abundances['O'] * ism_depletion['O'],       # 2.94e-4
    'Si': solar_abundances['Si'] * ism_depletion['Si'],     # 9.72e-7
    'S':  solar_abundances['S'] * ism_depletion['S'],       # 5.28e-6
    'Fe': solar_abundances['Fe'] * ism_depletion['Fe'],     # 1.90e-7
}

# Fine-structure line data
# Format: (wavelength_um, frequency_Hz, Einstein_A [s⁻¹], E_upper/k [K], g_upper, g_lower)
fine_structure_lines = {
    'CII_158':  (157.741e-4, 1.9005e12, 2.321e-6, 91.21, 4, 2),
    'OI_63':    (63.184e-4,  4.7448e12, 8.910e-5, 227.71, 3, 5),
    'OI_145':   (145.525e-4, 2.0604e12, 1.750e-5, 326.58, 1, 3),
    'CI_609':   (609.135e-4, 4.9231e11, 7.880e-8, 23.62, 3, 1),
    'CI_370':   (370.415e-4, 8.0929e11, 2.650e-7, 62.46, 5, 3),
}

# CO rotational line data
# CO B_rot = 57635.968 MHz, first few transitions
CO_B_rot = 57.635968e9  # Hz
CO_dipole = 0.1098e-18  # Debye in esu·cm

# H2 parameters
H2_dissociation_energy = 4.478  # eV
H2_binding_energy_grain = 0.04  # eV (physisorption on silicate)


def mean_molecular_weight(x_H2, x_HI, x_e):
    """
    Mean molecular weight per particle given H2, HI fractions and electron fraction.
    x_H2 = n(H2)/n_H, x_HI = n(HI)/n_H, x_e = n(e)/n_H
    """
    # Total particles per H nucleus: x_HI + x_H2 (each H2 = 1 particle) + He/H + x_e
    He_per_H = solar_abundances['He']
    n_particles_per_nH = x_HI + x_H2 + He_per_H + x_e
    # Mean mass per H: (1 + 4*He/H) * m_H
    mean_mass = (1.0 + 4.0 * He_per_H) * m_H
    mu = mean_mass / (n_particles_per_nH * m_H)
    return mu
