"""Parity tests: Rust thermal solver vs Python reference implementation."""

import numpy as np
import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_rust():
    try:
        from prism3d._rust_backend import has_rust
        return has_rust()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_rust(), reason="Rust extension not built"
)


@pytest.fixture
def solver_8cube():
    """Create an 8^3 solver with one RT pass completed."""
    from prism3d.density_fields import fractal_turbulent
    from prism3d.solver_3d import PDRSolver3D

    box_size = 2.0 * 3.0856776e18  # 2 pc in cm
    result = fractal_turbulent(8, box_size, n_mean=500.0,
                                sigma_ln=1.5, seed=42)
    # fractal_turbulent may return (density, cell_size) or just density
    density = result[0] if isinstance(result, tuple) else result
    solver = PDRSolver3D(density, box_size, G0_external=100.0,
                         zeta_CR_0=2e-16, nside_rt=1)

    # Run one RT pass to get realistic G0/AV fields
    from prism3d.radiative_transfer.fuv_rt_3d import compute_fuv_field_3d
    G0, A_V, N_H, N_H2, N_CO = compute_fuv_field_3d(
        solver.density, solver.G0_external, solver.cell_size,
        x_H2=solver.x_H2, x_CO=solver.x_CO, nside=solver.nside_rt
    )
    solver.G0 = G0
    solver.A_V = A_V
    solver.N_H = N_H
    solver.N_H2 = N_H2
    solver.N_CO = N_CO

    # Set analytical IC and update dust
    solver._set_analytical_ic()
    solver._update_dust_heating()

    return solver


def test_thermal_parity(solver_8cube):
    """Rust and Python thermal solvers agree on 8^3 grid."""
    from prism3d.radiative_transfer.shielding import f_shield_H2
    from prism3d.solver_3d import _net_heating_ctx

    solver = solver_8cube
    f_sh_H2 = f_shield_H2(solver.N_H2, T=np.mean(solver.T_gas))

    # ── Python path ──
    ctx = solver._prepare_thermal_ctx(f_sh_H2)
    T_lo = np.full_like(solver.T_gas, 10.0)
    T_hi = np.full_like(solver.T_gas, 1e5)
    for _ in range(10):
        T_mid = np.sqrt(T_lo * T_hi)
        net = _net_heating_ctx(T_mid, ctx)
        pos = net > 0
        T_lo = np.where(pos, T_mid, T_lo)
        T_hi = np.where(~pos, T_mid, T_hi)
    T = np.sqrt(T_lo * T_hi)
    for _ in range(3):
        f = _net_heating_ctx(T, ctx)
        pos = f > 0
        T_lo = np.where(pos, np.maximum(T, T_lo), T_lo)
        T_hi = np.where(~pos, np.minimum(T, T_hi), T_hi)
        eps = T * 0.01
        fp = _net_heating_ctx(T + eps, ctx)
        dfdT = (fp - f) / eps
        dT = -f / np.where(np.abs(dfdT) > 1e-50, dfdT, -1e-50)
        dT = np.clip(dT, -(T - T_lo), T_hi - T)
        dT = np.clip(dT, -0.5 * T, 0.5 * T)
        T = T + dT
    T_python = T.copy()

    # ── Rust path ──
    from prism3d._rust_backend import get_rust
    from prism3d.utils.constants import fine_structure_lines
    core = get_rust()

    (Gamma_fixed, gg_coeff, T_dust_ctx,
     n_e, n_HI, n_H2, n_CO, nCp_nH, nO_nH, nC_nH, mu, nH,
     CII_f, CII_A, CII_T, CII_gu, CII_gl,
     OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl,
     OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl,
     CI_consts, co_J) = ctx

    to_f64_flat = lambda a: np.ascontiguousarray(a, dtype=np.float64).ravel()
    t_out = np.zeros(solver.n_cells, dtype=np.float64)
    cii_p = np.array([CII_f, CII_A, CII_T, CII_gu, CII_gl], dtype=np.float64)
    oi63_p = np.array([OI63_f, OI63_A, OI63_T, OI63_gu, OI63_gl], dtype=np.float64)
    oi145_p = np.array([OI145_f, OI145_A, OI145_T, OI145_gu, OI145_gl], dtype=np.float64)
    ci_p = np.array([v for tup in CI_consts for v in tup], dtype=np.float64)

    core.solve_thermal_vec(
        to_f64_flat(Gamma_fixed),
        to_f64_flat(gg_coeff),
        to_f64_flat(T_dust_ctx),
        to_f64_flat(n_e),
        to_f64_flat(n_HI),
        to_f64_flat(n_H2),
        to_f64_flat(n_CO),
        to_f64_flat(nCp_nH),
        to_f64_flat(nO_nH),
        to_f64_flat(nC_nH),
        cii_p, oi63_p, oi145_p, ci_p,
        10, 3,
        t_out,
    )
    T_rust = t_out.reshape(solver.T_gas.shape)

    # ── Compare ──
    rel_diff = np.abs(T_rust - T_python) / np.maximum(T_python, 10.0)
    max_rel = np.max(rel_diff)
    mean_rel = np.mean(rel_diff)
    print(f"\nThermal parity: max_rel={max_rel:.2e}, mean_rel={mean_rel:.2e}")
    print(f"  T_python range: {T_python.min():.1f} – {T_python.max():.1f} K")
    print(f"  T_rust range:   {T_rust.min():.1f} – {T_rust.max():.1f} K")

    # Allow up to 1% relative difference (float64 arithmetic order differences)
    np.testing.assert_allclose(T_rust, T_python, rtol=0.01, atol=0.1,
                                err_msg="Rust thermal solver deviates from Python")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
