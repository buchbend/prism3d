/// Physical and astronomical constants in CGS units.
/// Mirrored from prism3d/utils/constants.py — keep in sync.

// Fundamental constants
pub const C_LIGHT: f64 = 2.99792458e10;       // Speed of light [cm/s]
pub const H_PLANCK: f64 = 6.62607015e-27;     // Planck constant [erg·s]
pub const K_BOLTZ: f64 = 1.380649e-16;        // Boltzmann constant [erg/K]
pub const M_PROTON: f64 = 1.67262192e-24;     // Proton mass [g]
pub const M_ELECTRON: f64 = 9.10938370e-28;   // Electron mass [g]
pub const E_CHARGE: f64 = 4.80320451e-10;     // Electron charge [esu]
pub const SIGMA_SB: f64 = 5.670374e-5;        // Stefan-Boltzmann [erg/cm²/s/K⁴]

// Atomic / molecular
pub const M_H: f64 = 1.6735575e-24;           // Hydrogen atom mass [g]
pub const M_H2: f64 = 2.0 * M_H;             // H2 mass [g]

// Energy conversions
pub const EV_TO_ERG: f64 = 1.602176634e-12;   // 1 eV in erg

// FUV radiation field
pub const G0_TO_FLUX: f64 = 1.6e-3;           // G0=1 flux [erg/cm²/s]

// Dust properties
pub const AV_PER_NH: f64 = 5.34e-22;          // A_V per N_H [mag/cm⁻²]
pub const TAU_FUV_PER_AV: f64 = 1.8;          // FUV extinction / A_V

// CO rotational
pub const CO_B_ROT: f64 = 57.635968e9;        // Hz
pub const CO_DIPOLE: f64 = 0.1098e-18;        // Debye in esu·cm

// Gas-phase abundances (solar × depletion)
pub const X_C_TOTAL: f64 = 1.076e-4;          // 2.69e-4 × 0.4
pub const X_O_TOTAL: f64 = 2.94e-4;           // 4.90e-4 × 0.6
pub const HE_PER_H: f64 = 8.51e-2;            // He/H abundance ratio

// Fine-structure line parameters: (freq_Hz, A_ul, T_ul, g_u, g_l)
pub const CII_158: (f64, f64, f64, f64, f64) = (1.9005e12, 2.321e-6, 91.21, 4.0, 2.0);
pub const OI_63: (f64, f64, f64, f64, f64) = (4.7448e12, 8.910e-5, 227.71, 3.0, 5.0);
pub const OI_145: (f64, f64, f64, f64, f64) = (2.0604e12, 1.750e-5, 326.58, 1.0, 3.0);
pub const CI_609: (f64, f64, f64, f64, f64) = (4.9231e11, 7.880e-8, 23.62, 3.0, 1.0);
pub const CI_370: (f64, f64, f64, f64, f64) = (8.0929e11, 2.650e-7, 62.46, 5.0, 3.0);

// Useful derived constant
pub const PI: f64 = std::f64::consts::PI;
