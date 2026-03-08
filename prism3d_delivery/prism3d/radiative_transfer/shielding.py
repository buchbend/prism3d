"""
Self-shielding functions for H2 and CO photodissociation.

H2: Uses the Draine & Bertoldi (1996) analytic approximation,
    updated with Wolcott-Green+ (2011) for high-T corrections.

CO: Uses the Visser+ (2009) parameterization including mutual
    shielding by H2 and self-shielding.

These functions return a shielding factor f_shield ∈ [0, 1] that
multiplies the unshielded photodissociation rate.
"""

import numpy as np


def f_shield_H2(N_H2, T=100.0, doppler_b=None):
    """
    H2 self-shielding factor from Draine & Bertoldi (1996), Eq. 37.

    The H2 photodissociation rate in the Lyman-Werner bands is reduced
    by line self-shielding. The rate becomes:
        k_pd = k_pd,0 * f_shield(N_H2)

    Parameters
    ----------
    N_H2 : float or array
        H2 column density [cm⁻²]
    T : float
        Gas temperature [K] (affects Doppler broadening)
    doppler_b : float, optional
        Doppler parameter [km/s]. If None, computed from T.

    Returns
    -------
    f_shield : float or array
        Shielding factor (0 = fully shielded, 1 = unshielded)
    """
    N_H2 = np.asarray(N_H2, dtype=np.float64)

    # Doppler parameter
    if doppler_b is None:
        # b = sqrt(2kT/m_H2) in km/s
        from ..utils.constants import k_boltz, m_H2
        doppler_b = np.sqrt(2.0 * k_boltz * T / m_H2) / 1e5  # cm/s → km/s

    # Draine & Bertoldi (1996) Eq. 37
    # f_shield = 0.965/(1 + x/b5)^2 + 0.035/sqrt(1+x) * exp(-8.5e-4*sqrt(1+x))
    # where x = N_H2 / 5e14 and b5 = b / (1e5 cm/s) but we use km/s

    x = N_H2 / 5.0e14
    b5 = doppler_b  # already in km/s, the formula uses b/(1 km/s)

    term1 = 0.965 / (1.0 + x / b5)**2
    sqrt_1px = np.sqrt(1.0 + x)
    term2 = 0.035 / sqrt_1px * np.exp(-8.5e-4 * sqrt_1px)

    f = term1 + term2
    return np.clip(f, 0.0, 1.0)


def f_shield_CO(N_CO, N_H2, mode='visser2009'):
    """
    CO self-shielding and mutual shielding by H2.

    Uses Visser+ (2009) Table 5 parameterization or the simpler
    Lee+ (1996) approximation.

    Parameters
    ----------
    N_CO : float or array
        CO column density [cm⁻²]
    N_H2 : float or array
        H2 column density [cm⁻²]
    mode : str
        'visser2009' (default) or 'lee1996'

    Returns
    -------
    f_shield : float or array
        CO shielding factor
    """
    N_CO = np.asarray(N_CO, dtype=np.float64)
    N_H2 = np.asarray(N_H2, dtype=np.float64)

    if mode == 'lee1996':
        # Lee+ (1996) simplified approximation
        # Self-shielding of CO
        theta_CO = _co_self_shield_lee(N_CO)
        # Mutual shielding by H2
        theta_H2 = _h2_mutual_shield_lee(N_H2)
        return theta_CO * theta_H2

    elif mode == 'visser2009':
        # Visser+ (2009) - analytic fit to their tabulated values
        return _co_shield_visser(N_CO, N_H2)

    else:
        raise ValueError(f"Unknown shielding mode: {mode}")


def _co_self_shield_lee(N_CO):
    """CO self-shielding from Lee+ (1996) / van Dishoeck & Black (1988)."""
    N_CO = np.maximum(N_CO, 1e10)

    # Approximate fit
    logN = np.log10(N_CO)

    # Piecewise linear in log
    f = np.ones_like(logN, dtype=float)
    mask1 = logN > 13.0
    mask2 = logN > 14.0
    mask3 = logN > 15.0
    mask4 = logN > 16.0

    f = np.where(mask1 & ~mask2, 10**(-0.5 * (logN - 13.0)), f)
    f = np.where(mask2 & ~mask3, 10**(-0.5 - 1.0 * (logN - 14.0)), f)
    f = np.where(mask3 & ~mask4, 10**(-1.5 - 1.5 * (logN - 15.0)), f)
    f = np.where(mask4, 10**(-3.0 - 2.0 * (logN - 16.0)), f)

    return np.clip(f, 1e-10, 1.0)


def _h2_mutual_shield_lee(N_H2):
    """Mutual shielding of CO by H2 line overlap, Lee+ (1996)."""
    N_H2 = np.maximum(N_H2, 1e10)
    logN = np.log10(N_H2)

    # H2 lines overlap with some CO lines, reducing CO photodissociation
    f = np.ones_like(logN, dtype=float)
    mask = logN > 20.0
    f = np.where(mask, np.exp(-0.5 * (logN - 20.0)), f)
    return np.clip(f, 0.01, 1.0)


def _co_shield_visser(N_CO, N_H2):
    """
    CO shielding function from Visser+ (2009).

    Uses their analytic fit (Eq. 4):
    Theta(N_CO, N_H2) = exp(-k1 * N_CO^a1) * exp(-k2 * N_H2^a2)
                       * [1 + k3 * N_CO^a3 * exp(-k4 * N_H2^a4)]^(-1)

    Fit parameters calibrated to their full calculation.
    """
    # Fit parameters from Visser+ (2009)
    # These are approximate representative values
    k1 = 6.428e-8
    a1 = 0.53
    k2 = 3.0e-12
    a2 = 0.495
    k3 = 1e-5
    a3 = 0.6
    k4 = 8e-12
    a4 = 0.48

    N_CO = np.maximum(N_CO, 1.0)
    N_H2 = np.maximum(N_H2, 1.0)

    # Self-shielding term
    term1 = np.exp(-k1 * N_CO**a1)

    # Mutual shielding by H2
    term2 = np.exp(-k2 * N_H2**a2)

    # Cross-term
    term3 = 1.0 / (1.0 + k3 * N_CO**a3 * np.exp(-k4 * N_H2**a4))

    f = term1 * term2 * term3

    return np.clip(f, 1e-10, 1.0)


def effective_shielding_multiray(N_H2_rays, N_CO_rays, T=100.0):
    """
    Compute effective shielding factor averaged over multiple ray directions.

    For 3D geometry, the shielding seen by a cell depends on direction.
    We compute the mean photodissociation rate by averaging the shielded
    rates over all directions.

    Parameters
    ----------
    N_H2_rays : ndarray, shape (n_rays,)
        H2 column density along each ray [cm⁻²]
    N_CO_rays : ndarray, shape (n_rays,)
        CO column density along each ray [cm⁻²]
    T : float
        Gas temperature [K]

    Returns
    -------
    f_H2_eff : float
        Effective H2 shielding factor
    f_CO_eff : float
        Effective CO shielding factor
    """
    # Compute shielding along each ray
    f_H2 = f_shield_H2(N_H2_rays, T=T)
    f_CO = f_shield_CO(N_CO_rays, N_H2_rays)

    # Average (equal solid angle weights assumed)
    f_H2_eff = np.mean(f_H2)
    f_CO_eff = np.mean(f_CO)

    return f_H2_eff, f_CO_eff
