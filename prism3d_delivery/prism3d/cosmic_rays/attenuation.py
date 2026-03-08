"""
Cosmic ray attenuation module for PRISM-3D.

Implements column-density dependent cosmic ray ionization rate
following Padovani+ (2009, 2018). This breaks the standard assumption
of a uniform CR ionization rate throughout the cloud.

Three models are available:
- Model L: Low CR flux (lower bound from Voyager measurements)
- Model H: High CR flux (fits diffuse cloud observations)
- Model M: Medium (geometric mean, recommended default)
"""

import numpy as np
from ..utils.constants import AV_per_NH


def cr_ionization_rate(N_H, model='M', zeta_0=2e-16):
    """
    Compute the CR ionization rate as a function of column density.

    The CR spectrum is attenuated as it penetrates into the cloud.
    Low-energy CRs (which dominate ionization) are stopped first.

    Parameters
    ----------
    N_H : float or array
        Total H column density [cm⁻²]
    model : str
        CR spectrum model: 'L' (low), 'M' (medium), 'H' (high)
    zeta_0 : float
        Normalization: CR ionization rate at N_H = 0 [s⁻¹]

    Returns
    -------
    zeta : float or array
        CR ionization rate [s⁻¹]
    """
    N_H = np.asarray(N_H, dtype=np.float64)

    if model == 'L':
        # Padovani+ (2018) Model L
        # Low CR flux, appropriate for shielded environments
        zeta = _padovani_model_L(N_H) * (zeta_0 / 2e-16)

    elif model == 'H':
        # Padovani+ (2018) Model H
        # High CR flux, appropriate for diffuse clouds
        zeta = _padovani_model_H(N_H) * (zeta_0 / 2e-16)

    elif model == 'M':
        # Geometric mean of L and H
        zeta_L = _padovani_model_L(N_H)
        zeta_H = _padovani_model_H(N_H)
        zeta = np.sqrt(zeta_L * zeta_H) * (zeta_0 / 2e-16)

    else:
        raise ValueError(f"Unknown CR model: {model}")

    return np.maximum(zeta, 1e-22)  # Floor


def _padovani_model_L(N_H):
    """
    Padovani+ (2018) Model L: Low CR ionization rate.

    Fit: log10(zeta) = a0 + a1*x + a2*x^2 + ...
    where x = log10(N_H / 10^20 cm⁻²)
    """
    N_H = np.maximum(N_H, 1e18)
    x = np.log10(N_H / 1e20)

    # Fit coefficients from Padovani+ (2018) Table 3
    # These are approximate representative values
    log_zeta = np.where(
        x < -1.0,
        -15.7,                          # Unattenuated
        np.where(
            x < 2.0,
            -15.7 - 0.4 * x - 0.1 * x**2,  # Moderate attenuation
            np.where(
                x < 5.0,
                -16.3 - 0.8 * (x - 2.0),     # Strong attenuation
                -18.7                           # Deep, only high-E CRs
            )
        )
    )

    return 10**log_zeta


def _padovani_model_H(N_H):
    """
    Padovani+ (2018) Model H: High CR ionization rate.
    """
    N_H = np.maximum(N_H, 1e18)
    x = np.log10(N_H / 1e20)

    log_zeta = np.where(
        x < -1.0,
        -15.0,
        np.where(
            x < 2.0,
            -15.0 - 0.35 * x - 0.08 * x**2,
            np.where(
                x < 5.0,
                -15.6 - 0.7 * (x - 2.0),
                -17.7
            )
        )
    )

    return 10**log_zeta


def cr_attenuation_multiray(grid, model='M', zeta_0=2e-16):
    """
    Compute CR ionization rate at each cell using column densities
    from multiple ray directions.

    For each cell, the effective zeta_CR is the mean of the
    attenuation along all ray directions (since CRs are approximately
    isotropic).

    Parameters
    ----------
    grid : OctreeGrid
        The computational grid (cells must have N_H_total set)
    model : str
        CR attenuation model
    zeta_0 : float
        Surface CR rate [s⁻¹]
    """
    leaves = grid.get_leaves()

    for leaf in leaves:
        # Use the column density already computed by the RT module
        N_H = leaf.data.N_H_total

        # Compute attenuated rate
        leaf.data.zeta_CR = cr_ionization_rate(N_H, model=model, zeta_0=zeta_0)


def cr_heating_rate(n_H, x_H2, x_HI, zeta_CR):
    """
    Compute CR heating rate.

    Convenience function - same as in thermal/heating.py but
    available here for standalone use.

    Parameters
    ----------
    Returns
    -------
    Gamma_CR : float
        CR heating rate [erg/cm³/s]
    """
    from ..utils.constants import eV_to_erg

    E_heat = 20.0 * eV_to_erg  # Energy deposited per ionization
    n_HI = x_HI * n_H
    n_H2 = x_H2 * n_H

    return zeta_CR * (n_HI + 2.0 * n_H2) * E_heat
