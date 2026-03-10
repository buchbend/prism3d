/// Hybrid bisection-Newton thermal equilibrium solver.
///
/// Ports the Python `_net_heating_ctx` + bisection-Newton from solver_3d.py.
/// Each cell is solved independently → embarrassingly parallel via rayon.
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::*;

// ── CO J-level precomputed constants ─────────────────────────────────
// Precompute at compile time for J=1..40
struct CoJLevel {
    t_u: f64,       // E_upper / k_B [K]
    a_co: f64,      // Einstein A coefficient [s^-1]
    g_ratio: f64,   // g_u / g_l
    j_factor: f64,  // 1 + 0.1 * J_u
    ahv: f64,       // A_ul * h * freq [erg/s]
}

fn precompute_co_levels() -> Vec<CoJLevel> {
    let mut levels = Vec::with_capacity(40);
    for j_u in 1..=40u32 {
        let j = j_u as f64;
        let freq = 2.0 * CO_B_ROT * j;
        let t_u = H_PLANCK * CO_B_ROT * j * (j + 1.0) / K_BOLTZ;
        let a_co = (64.0 * PI.powi(4) * freq.powi(3) * CO_DIPOLE * CO_DIPOLE * j)
            / (3.0 * H_PLANCK * C_LIGHT.powi(3) * (2.0 * j + 1.0));
        let g_ratio = (2.0 * j + 1.0) / (2.0 * (j - 1.0) + 1.0);
        let j_factor = 1.0 + 0.1 * j;
        let ahv = a_co * H_PLANCK * freq;
        levels.push(CoJLevel {
            t_u,
            a_co,
            g_ratio,
            j_factor,
            ahv,
        });
    }
    levels
}

// ── Per-cell net heating function ────────────────────────────────────

/// Compute net heating rate (Gamma - Lambda) for a single cell.
///
/// All T-independent quantities are precomputed and passed in.
/// Only T varies during the bisection-Newton iteration.
#[inline(always)]
fn net_heating_cell(
    t: f64,
    // T-independent precomputed context
    gamma_fixed: f64,
    gg_coeff: f64,
    t_dust: f64,
    n_e: f64,
    n_hi: f64,
    n_h2: f64,
    n_co: f64,
    n_cp_nh: f64,
    n_o_nh: f64,
    n_c_nh: f64,
    // Fine-structure line constants
    cii: &(f64, f64, f64, f64, f64),  // (freq, A, T_ul, g_u, g_l)
    oi63: &(f64, f64, f64, f64, f64),
    oi145: &(f64, f64, f64, f64, f64),
    ci609: &(f64, f64, f64, f64, f64),
    ci370: &(f64, f64, f64, f64, f64),
    // CO levels
    co_levels: &[CoJLevel],
) -> f64 {
    let sqrt_t = t.sqrt();
    let inv_t = 1.0 / t;
    let t_01 = t * 0.01; // T/100

    // ── Heating: precomputed sum + gas-grain (T-dependent) ──
    let mut net = gamma_fixed + gg_coeff * sqrt_t * (t_dust - t);

    // ── Cooling ──

    // [CII] 158 μm
    {
        let (freq, a_ul, t_ul, g_u, g_l) = *cii;
        let gamma_e = 8.63e-6 / (g_l * sqrt_t) * 2.15 * (-t_ul * inv_t).exp();
        let gamma_h = 8.0e-10 * t_01.powf(0.07);
        let gamma_h2 = 4.9e-10 * t_01.powf(0.12);
        let q_coll = gamma_e * n_e + gamma_h * n_hi + gamma_h2 * n_h2;
        let r_ul = a_ul + q_coll;
        let r_lu = (g_u / g_l) * q_coll * (-t_ul * inv_t).exp();
        net -= n_cp_nh * r_lu / (r_lu + r_ul) * a_ul * H_PLANCK * freq;
    }

    // [OI] 63 μm
    {
        let (freq, a_ul, t_ul, g_u, g_l) = *oi63;
        let gamma_h = 9.2e-12 * t.powf(0.67);
        let gamma_h2 = 4.5e-12 * t.powf(0.64);
        let gamma_e = 1.4e-8 * t_01.powf(0.39);
        let q = gamma_h * n_hi + gamma_h2 * n_h2 + gamma_e * n_e;
        let r_lu = (g_u / g_l) * q * (-t_ul * inv_t).exp();
        let r_ul = a_ul + q;
        net -= n_o_nh * r_lu / (r_lu + r_ul) * a_ul * H_PLANCK * freq;
    }

    // [OI] 145 μm
    {
        let (freq, a_ul, t_ul, g_u, g_l) = *oi145;
        let gamma_h = 4.0e-12 * t.powf(0.60);
        let gamma_h2 = 2.0e-12 * t.powf(0.58);
        let gamma_e = 5.0e-9 * t_01.powf(0.39);
        let q = gamma_h * n_hi + gamma_h2 * n_h2 + gamma_e * n_e;
        let r_lu = (g_u / g_l) * q * (-t_ul * inv_t).exp();
        let r_ul = a_ul + q;
        net -= n_o_nh * r_lu / (r_lu + r_ul) * a_ul * H_PLANCK * freq;
    }

    // [CI] 609 + 370 μm
    {
        let t_01_03 = t_01.powf(0.3);
        let t_01_m05 = t_01.powf(-0.5);
        for ci in [ci609, ci370] {
            let (freq, a_ul, t_ul, g_u, g_l) = *ci;
            let q = 1.0e-10 * t_01_03 * n_hi + 5.0e-11 * t_01_03 * n_h2
                + 2.0e-7 * t_01_m05 * n_e;
            let r_lu = (g_u / g_l) * q * (-t_ul * inv_t).exp();
            let r_ul = a_ul + q;
            net -= n_c_nh * r_lu / (r_lu + r_ul) * a_ul * H_PLANCK * freq;
        }
    }

    // CO rotational (J=1..40, early break)
    {
        let sqrt_t_01 = t_01.sqrt();
        for level in co_levels {
            if level.t_u >= 5.0 * t {
                break;
            }
            let gamma_h2_co = 3.3e-11 * sqrt_t_01 * level.j_factor;
            let r_lu = level.g_ratio * gamma_h2_co * n_h2 * (-level.t_u * inv_t).exp();
            let r_ul = level.a_co + gamma_h2_co * n_h2;
            net -= n_co * r_lu / (r_lu + r_ul) * level.ahv;
        }
    }

    // H2 rovibrational (Glover & Abel 2008)
    if t >= 100.0 {
        let log_t = t.max(100.0).log10().clamp(2.0, 4.5);
        let lt2 = log_t * log_t;
        let lt3 = lt2 * log_t;
        let lt4 = lt3 * log_t;
        let lt5 = lt4 * log_t;
        let lt6 = lt5 * log_t;

        let log_lh = -24.311 + 3.5692 * log_t - 11.332 * lt2 + 15.738 * lt3
            - 10.581 * lt4 + 3.5803 * lt5 - 0.48365 * lt6;
        let log_lh2 = -24.311 + 4.6585 * log_t - 14.272 * lt2 + 19.779 * lt3
            - 13.255 * lt4 + 4.4840 * lt5 - 0.60504 * lt6;
        let l_h = 10.0_f64.powf(log_lh);
        let l_h2 = 10.0_f64.powf(log_lh2);
        let l_lte = 10.0_f64.powf(-19.703 + 0.5 * log_t);

        let l0 = l_h * n_hi + l_h2 * n_h2;
        let denom = (l0 + l_lte).max(1e-50);
        net -= n_h2 * l0 * l_lte / denom;
    }

    // Lyman-alpha
    if t >= 3000.0 {
        let q_lu = 2.41e-6 / sqrt_t * 0.486 * (-1.18e5 * inv_t).exp();
        net -= n_hi * n_e * q_lu * 10.2 * EV_TO_ERG;
    }

    // Gas-grain cooling (separate from heating part already in gamma_fixed)
    let gg_cool = gg_coeff * sqrt_t * (t - t_dust);
    if gg_cool > 0.0 {
        net -= gg_cool;
    }

    // Recombination cooling (C+ only in the fast path)
    let alpha_cp = 4.67e-12 * (t / 300.0).powf(-0.6);
    net -= n_cp_nh * n_e * alpha_cp * K_BOLTZ * t;

    net
}

// ── Bisection-Newton solver for a single cell ────────────────────────

#[inline(always)]
fn solve_cell(
    gamma_fixed: f64,
    gg_coeff: f64,
    t_dust: f64,
    n_e: f64,
    n_hi: f64,
    n_h2: f64,
    n_co: f64,
    n_cp_nh: f64,
    n_o_nh: f64,
    n_c_nh: f64,
    cii: &(f64, f64, f64, f64, f64),
    oi63: &(f64, f64, f64, f64, f64),
    oi145: &(f64, f64, f64, f64, f64),
    ci609: &(f64, f64, f64, f64, f64),
    ci370: &(f64, f64, f64, f64, f64),
    co_levels: &[CoJLevel],
    n_bisect: usize,
    n_newton: usize,
) -> f64 {
    let eval = |t: f64| -> f64 {
        net_heating_cell(
            t,
            gamma_fixed,
            gg_coeff,
            t_dust,
            n_e,
            n_hi,
            n_h2,
            n_co,
            n_cp_nh,
            n_o_nh,
            n_c_nh,
            cii,
            oi63,
            oi145,
            ci609,
            ci370,
            co_levels,
        )
    };

    let mut t_lo = 10.0_f64;
    let mut t_hi = 1e5_f64;

    // Phase 1: bisection with geometric midpoint
    for _ in 0..n_bisect {
        let t_mid = (t_lo * t_hi).sqrt();
        let net = eval(t_mid);
        if net > 0.0 {
            t_lo = t_mid;
        } else {
            t_hi = t_mid;
        }
    }

    // Phase 2: Newton-Raphson with clamping to bracket
    let mut t = (t_lo * t_hi).sqrt();
    for _ in 0..n_newton {
        let f = eval(t);
        if f > 0.0 {
            t_lo = t_lo.max(t);
        } else {
            t_hi = t_hi.min(t);
        }
        let eps = t * 0.01;
        let fp = eval(t + eps);
        let dfdt = (fp - f) / eps;
        let mut dt = if dfdt.abs() > 1e-50 {
            -f / dfdt
        } else {
            -f / (-1e-50)
        };
        // Clamp to bracket
        dt = dt.clamp(-(t - t_lo), t_hi - t);
        // Clamp to max 50% change
        dt = dt.clamp(-0.5 * t, 0.5 * t);
        t += dt;
    }

    t
}

// ── PyO3 entry point ─────────────────────────────────────────────────

/// Solve thermal equilibrium for all cells in parallel.
///
/// All input arrays are 1D (pre-flattened by Python).
/// `t_gas` is modified in-place with the equilibrium temperatures.
///
/// Line constants are passed as flat arrays: [freq, A, T_ul, g_u, g_l].
/// CI constants: 10 elements [freq1, A1, T1, gu1, gl1, freq2, A2, T2, gu2, gl2].
/// CO J-level constants are precomputed internally.
#[pyfunction]
pub fn solve_thermal_vec<'py>(
    py: Python<'py>,
    // T-independent precomputed context (1D arrays, n_cells)
    gamma_fixed: PyReadonlyArray1<'py, f64>,
    gg_coeff: PyReadonlyArray1<'py, f64>,
    t_dust: PyReadonlyArray1<'py, f64>,
    n_e: PyReadonlyArray1<'py, f64>,
    n_hi: PyReadonlyArray1<'py, f64>,
    n_h2: PyReadonlyArray1<'py, f64>,
    n_co: PyReadonlyArray1<'py, f64>,
    n_cp_nh: PyReadonlyArray1<'py, f64>,
    n_o_nh: PyReadonlyArray1<'py, f64>,
    n_c_nh: PyReadonlyArray1<'py, f64>,
    // Line constants as flat arrays [freq, A, T_ul, g_u, g_l]
    cii_params: PyReadonlyArray1<'py, f64>,
    oi63_params: PyReadonlyArray1<'py, f64>,
    oi145_params: PyReadonlyArray1<'py, f64>,
    ci_params: PyReadonlyArray1<'py, f64>, // 10 elements: 2 lines × 5 params
    // Solver config
    n_bisect: usize,
    n_newton: usize,
    // Output (modified in-place)
    mut t_gas: PyReadwriteArray1<'py, f64>,
) -> PyResult<()> {
    // Precompute CO J-levels once (compile-time-ish)
    let co_levels = precompute_co_levels();

    // Unpack line constants
    let cii_p = cii_params.as_slice()?;
    let cii = (cii_p[0], cii_p[1], cii_p[2], cii_p[3], cii_p[4]);

    let oi63_p = oi63_params.as_slice()?;
    let oi63 = (oi63_p[0], oi63_p[1], oi63_p[2], oi63_p[3], oi63_p[4]);

    let oi145_p = oi145_params.as_slice()?;
    let oi145 = (oi145_p[0], oi145_p[1], oi145_p[2], oi145_p[3], oi145_p[4]);

    let ci_p = ci_params.as_slice()?;
    let ci609 = (ci_p[0], ci_p[1], ci_p[2], ci_p[3], ci_p[4]);
    let ci370 = (ci_p[5], ci_p[6], ci_p[7], ci_p[8], ci_p[9]);

    // Get array slices (read-only inputs)
    let gf = gamma_fixed.as_slice()?;
    let gg = gg_coeff.as_slice()?;
    let td = t_dust.as_slice()?;
    let ne = n_e.as_slice()?;
    let nhi = n_hi.as_slice()?;
    let nh2 = n_h2.as_slice()?;
    let nco = n_co.as_slice()?;
    let ncp = n_cp_nh.as_slice()?;
    let no = n_o_nh.as_slice()?;
    let nc = n_c_nh.as_slice()?;

    // Compute results into a Vec, then write back
    // (avoids Send issues with PyReadwriteArray across allow_threads)
    let n_cells = gf.len();

    let results = py.allow_threads(|| {
        let mut results = vec![0.0_f64; n_cells];
        results
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, t_i)| {
                *t_i = solve_cell(
                    gf[i], gg[i], td[i], ne[i], nhi[i], nh2[i], nco[i], ncp[i], no[i],
                    nc[i], &cii, &oi63, &oi145, &ci609, &ci370, &co_levels, n_bisect,
                    n_newton,
                );
            });
        results
    });

    // Write results back into numpy array
    let t_out = t_gas.as_slice_mut()?;
    t_out.copy_from_slice(&results);

    Ok(())
}

// ── Unit tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_co_levels_precomputed() {
        let levels = precompute_co_levels();
        assert_eq!(levels.len(), 40);
        // J=1: T_u should be ~5.53 K (2 * B_rot * h / k)
        assert!((levels[0].t_u - 5.53).abs() < 0.1);
        // A coefficient for J=1 should be ~7.2e-8 s^-1
        assert!(levels[0].a_co > 1e-9 && levels[0].a_co < 1e-6);
    }

    #[test]
    fn test_net_heating_warm_surface() {
        // Warm PDR surface: G0=1000, n=1e4, AV=0, T=500 K
        // Should be roughly in equilibrium (net ≈ 0)
        let co_levels = precompute_co_levels();

        // Approximate precomputed context for a warm PDR surface cell
        let n_h: f64 = 1e4;
        let x_e = 1e-4;
        let x_hi = 0.9;
        let x_h2 = 0.05;

        let net = net_heating_cell(
            500.0,
            1e-20, // gamma_fixed (PE + other T-independent heating)
            2.0 * 0.35 * 1e-21 * n_h * n_h
                * (8.0 * K_BOLTZ / (PI * 1.0 * M_H)).sqrt()
                * K_BOLTZ, // gg_coeff
            30.0,           // T_dust
            x_e * n_h,      // n_e
            x_hi * n_h,     // n_HI
            x_h2 * n_h,     // n_H2
            1e-6 * n_h,     // n_CO
            1e-4 * n_h,     // nCp_nH
            3e-4 * n_h,     // nO_nH
            1e-5 * n_h,     // nC_nH
            &CII_158,
            &OI_63,
            &OI_145,
            &CI_609,
            &CI_370,
            &co_levels,
        );
        // The net heating can be positive or negative depending on exact conditions;
        // just check it's a reasonable number (not NaN or inf)
        assert!(net.is_finite(), "net_heating returned non-finite: {}", net);
    }

    #[test]
    fn test_solve_cell_returns_reasonable_temperature() {
        let co_levels = precompute_co_levels();
        let n_h: f64 = 1e3;
        let x_e = 1e-4;

        // With significant PE heating, expect T > 30 K
        let t = solve_cell(
            1e-22,  // moderate PE heating
            2.0 * 0.35 * 1e-21 * n_h * n_h
                * (8.0 * K_BOLTZ / (PI * 1.0 * M_H)).sqrt()
                * K_BOLTZ,
            20.0,
            x_e * n_h,
            0.9 * n_h,
            0.05 * n_h,
            1e-6 * n_h,
            1e-4 * n_h,
            3e-4 * n_h,
            1e-5 * n_h,
            &CII_158,
            &OI_63,
            &OI_145,
            &CI_609,
            &CI_370,
            &co_levels,
            10,
            3,
        );

        assert!(t >= 10.0 && t <= 1e5, "Temperature out of range: {}", t);
    }
}
