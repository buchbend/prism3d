"""
Thermal equilibrium solver for PRISM-3D.

Solves the equation Γ(T) = Λ(T) for the equilibrium gas temperature,
given the local density, radiation field, and chemical abundances.

Uses a combination of bisection and Brent's method for robustness.
"""

import numpy as np
from scipy.optimize import brentq
from .heating import total_heating_rate
from .cooling import total_cooling_rate


class ThermalSolver:
    """
    Solve for thermal equilibrium temperature.

    Parameters
    ----------
    T_min : float
        Minimum temperature to consider [K]
    T_max : float
        Maximum temperature to consider [K]
    tol : float
        Relative tolerance for temperature solution
    """

    def __init__(self, T_min=10.0, T_max=1e5, tol=1e-3):
        self.T_min = T_min
        self.T_max = T_max
        self.tol = tol

    def solve(self, n_H, T_dust, G0, A_V, zeta_CR,
              x_e, x_HI, x_H2, x_Cp, x_C, x_O, x_CO,
              x_Sp=0.0, x_Sip=0.0, x_Fep=0.0,
              f_shield_H2=1.0, N_CO=0.0, T_guess=None):
        """
        Find equilibrium temperature where heating = cooling.

        Returns
        -------
        T_eq : float
            Equilibrium temperature [K]
        Gamma : float
            Heating rate at equilibrium [erg/cm³/s]
        Lambda : float
            Cooling rate at equilibrium [erg/cm³/s]
        heat_components : dict
        cool_components : dict
        """

        def net_rate(T):
            """Γ - Λ as a function of T."""
            Gamma, _ = total_heating_rate(
                n_H, T, T_dust, G0, A_V, zeta_CR,
                x_e, x_HI, x_H2, x_C, x_Cp, f_shield_H2
            )
            Lambda, _ = total_cooling_rate(
                n_H, T, T_dust, x_e, x_HI, x_H2,
                x_Cp, x_C, x_O, x_CO,
                x_Sp, x_Sip, x_Fep, N_CO
            )
            return Gamma - Lambda

        # Find temperature bracket
        T_lo = self.T_min
        T_hi = self.T_max

        # Check that there's a sign change
        f_lo = net_rate(T_lo)
        f_hi = net_rate(T_hi)

        if f_lo * f_hi > 0:
            # No sign change - return boundary temperature
            if abs(f_lo) < abs(f_hi):
                T_eq = T_lo
            else:
                T_eq = T_hi
        else:
            # Use Brent's method
            try:
                T_eq = brentq(net_rate, T_lo, T_hi,
                             xtol=self.tol * T_lo, rtol=self.tol,
                             maxiter=200)
            except ValueError:
                # Fall back to bisection
                T_eq = self._bisection(net_rate, T_lo, T_hi)

        # Compute final rates at equilibrium
        Gamma, heat_comp = total_heating_rate(
            n_H, T_eq, T_dust, G0, A_V, zeta_CR,
            x_e, x_HI, x_H2, x_C, x_Cp, f_shield_H2
        )
        Lambda, cool_comp = total_cooling_rate(
            n_H, T_eq, T_dust, x_e, x_HI, x_H2,
            x_Cp, x_C, x_O, x_CO,
            x_Sp, x_Sip, x_Fep, N_CO
        )

        return T_eq, Gamma, Lambda, heat_comp, cool_comp

    def _bisection(self, func, a, b, max_iter=100):
        """Simple bisection fallback."""
        for _ in range(max_iter):
            mid = (a + b) / 2.0
            if func(mid) * func(a) < 0:
                b = mid
            else:
                a = mid
            if (b - a) / a < self.tol:
                break
        return (a + b) / 2.0

    def compute_dust_temperature(self, G0, A_V, n_H=None, T_gas=None):
        """
        Estimate dust temperature from radiative equilibrium.

        Simple approximation: T_dust ∝ G0^(1/4+δ) * exp(-A_V/4)

        For a more accurate calculation, would need full dust SED
        fitting with grain size distribution.

        Parameters
        ----------
        G0 : float
            FUV field [Habing units]
        A_V : float
            Visual extinction [mag]
        n_H : float, optional
            Density for gas-grain coupling correction
        T_gas : float, optional
            Gas temperature for gas-grain coupling
        """
        from ..utils.constants import sigma_sb

        # Dust absorbs FUV and reradiates in IR
        # Simple equilibrium: sigma_abs * F_UV = sigma_SB * T_d^(4+beta)
        # where beta ~ 1.5-2 for grain emissivity index

        # Effective UV flux
        F_UV = 1.6e-3 * G0 * np.exp(-1.8 * A_V)  # erg/cm²/s

        # Add ISRF heating floor (CMB + ISRF)
        F_UV += 1e-5  # erg/cm²/s minimum

        # T_dust from balance
        # Assuming Q_abs(UV)/Q_abs(IR) ~ 1000 and beta=2:
        # T_d = (Q_UV/Q_IR * F_UV / sigma_SB)^(1/6)
        Q_ratio = 1000.0
        T_dust = (Q_ratio * F_UV / sigma_sb) ** (1.0 / 6.0)

        # Floor: CMB temperature
        T_dust = max(T_dust, 2.725)

        # Gas-grain coupling correction at high density
        if n_H is not None and T_gas is not None and n_H > 1e5:
            # At high density, dust approaches gas temperature
            f_couple = min(1.0, n_H / 1e7)
            T_dust = T_dust * (1 - f_couple) + T_gas * f_couple

        return T_dust
