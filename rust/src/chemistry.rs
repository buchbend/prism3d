/// Vectorized explicit Euler chemistry integrator.
///
/// Ports the Python `_solve_chemistry_vec` from solver_3d.py.
/// Each cell is independent → embarrassingly parallel via rayon.
use numpy::PyReadwriteArray1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Solve chemistry for a single cell over n_substeps.
/// Returns the final abundances.
#[inline(always)]
fn solve_cell_chemistry(
    n_h: f64,
    k_pd_base: f64,
    k_ci: f64,
    alpha_cp: f64,
    k_co_total: f64,
    r_h2_half_nh: f64,
    r_cr_base: f64,
    x_c_total: f64,
    mut x_hi: f64,
    mut x_h2: f64,
    mut x_cp: f64,
    mut x_c: f64,
    mut x_co: f64,
    x_hcop: f64,
    n_substeps: usize,
    conv_tol: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    // x_o, x_e are derived
    let mut dt = 1e10_f64;

    for step in 0..n_substeps {
        let x_e = (x_cp + x_hcop).max(1e-30);

        // H2 formation/destruction
        let r_h2_form = r_h2_half_nh * x_hi;
        let r_pd = k_pd_base * x_h2;
        let r_cr = r_cr_base * x_h2;
        let dx_h2 = r_h2_form - r_pd - r_cr;
        let dx_h = -2.0 * dx_h2;

        // Carbon chemistry
        let alpha_ne = alpha_cp * x_e * n_h;
        let x_oh = 1e-9 + 9e-8 * (x_h2 * 10.0).clamp(0.0, 1.0);
        let dx_cp = k_ci * x_c - alpha_ne * x_cp + k_co_total * x_co;
        let dx_co = 1e-10 * x_c * x_oh * n_h - k_co_total * x_co;
        let dx_c = -k_ci * x_c + alpha_ne * x_cp - 1e-10 * x_c * x_oh * n_h;

        // Adaptive dt per cell
        let max_rate = (dx_h2.abs() / x_h2.max(1e-20))
            .max((dx_cp.abs() / x_cp.max(1e-20)).max(dx_co.abs() / x_co.max(1e-20)));

        let dt_eff = if max_rate * dt > 0.1 {
            0.1 / max_rate.max(1e-30)
        } else {
            dt
        };

        x_hi = (x_hi + dx_h * dt_eff).max(1e-30);
        x_h2 = (x_h2 + dx_h2 * dt_eff).max(1e-30);
        x_cp = (x_cp + dx_cp * dt_eff).max(1e-30);
        x_c = (x_c + dx_c * dt_eff).max(1e-30);
        x_co = (x_co + dx_co * dt_eff).max(1e-30);

        // H conservation: x_H + 2*x_H2 = 1
        let h_sum = x_hi + 2.0 * x_h2;
        x_hi /= h_sum;
        x_h2 /= h_sum;

        // C conservation (float64 sum is exact enough)
        let c_sum = x_cp + x_c + x_co;
        if c_sum > 0.0 {
            let scale = x_c_total / c_sum;
            x_cp *= scale;
            x_c *= scale;
            x_co *= scale;
        }

        // Convergence check every 20 steps
        if step > 0 && step % 20 == 0 {
            // We'd need old values to check convergence properly.
            // For the per-cell Euler, we use a simplified check:
            // if the adaptive dt is at max (1e16), rates are tiny → converged.
            if dt >= 1e15 && max_rate * dt < conv_tol {
                break;
            }
        }

        dt = (dt * 3.0).min(1e16);
    }

    let x_o = (crate::constants::X_O_TOTAL - x_co).max(1e-30);
    let x_e = (x_cp + x_hcop).max(1e-30);

    (x_hi, x_h2, x_cp, x_c, x_co, x_o, x_e)
}

/// Solve chemistry for all cells in parallel.
///
/// All input/output arrays are 1D (pre-flattened by Python).
/// Rate coefficients are precomputed by Python and passed in.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn solve_chemistry_euler<'py>(
    py: Python<'py>,
    // Precomputed rate coefficient arrays (1D, n_cells)
    n_h: PyReadonlyArray1<'py, f64>,
    k_pd_base: PyReadonlyArray1<'py, f64>,
    k_ci: PyReadonlyArray1<'py, f64>,
    alpha_cp: PyReadonlyArray1<'py, f64>,
    k_co_total: PyReadonlyArray1<'py, f64>,
    r_h2_half_nh: PyReadonlyArray1<'py, f64>,
    r_cr_base: PyReadonlyArray1<'py, f64>,
    // Scalar parameters
    x_c_total: f64,
    // State arrays (read-write, modified in place)
    mut x_hi: PyReadwriteArray1<'py, f64>,
    mut x_h2: PyReadwriteArray1<'py, f64>,
    mut x_cp: PyReadwriteArray1<'py, f64>,
    mut x_c: PyReadwriteArray1<'py, f64>,
    mut x_co: PyReadwriteArray1<'py, f64>,
    mut x_o: PyReadwriteArray1<'py, f64>,
    mut x_e: PyReadwriteArray1<'py, f64>,
    x_hcop: PyReadonlyArray1<'py, f64>,
    // Config
    n_substeps: usize,
    conv_tol: f64,
) -> PyResult<usize> {
    let nh = n_h.as_slice()?;
    let kpd = k_pd_base.as_slice()?;
    let kci = k_ci.as_slice()?;
    let acp = alpha_cp.as_slice()?;
    let kco = k_co_total.as_slice()?;
    let rh2 = r_h2_half_nh.as_slice()?;
    let rcr = r_cr_base.as_slice()?;
    let hcop = x_hcop.as_slice()?;

    // Read current state
    let hi_in = x_hi.as_slice()?.to_vec();
    let h2_in = x_h2.as_slice()?.to_vec();
    let cp_in = x_cp.as_slice()?.to_vec();
    let c_in = x_c.as_slice()?.to_vec();
    let co_in = x_co.as_slice()?.to_vec();

    let n_cells = nh.len();

    // Run in parallel
    let results: Vec<(f64, f64, f64, f64, f64, f64, f64)> = py.allow_threads(|| {
        (0..n_cells)
            .into_par_iter()
            .map(|i| {
                solve_cell_chemistry(
                    nh[i], kpd[i], kci[i], acp[i], kco[i], rh2[i], rcr[i],
                    x_c_total,
                    hi_in[i], h2_in[i], cp_in[i], c_in[i], co_in[i], hcop[i],
                    n_substeps, conv_tol,
                )
            })
            .collect()
    });

    // Write results back
    let hi_out = x_hi.as_slice_mut()?;
    let h2_out = x_h2.as_slice_mut()?;
    let cp_out = x_cp.as_slice_mut()?;
    let c_out = x_c.as_slice_mut()?;
    let co_out = x_co.as_slice_mut()?;
    let o_out = x_o.as_slice_mut()?;
    let e_out = x_e.as_slice_mut()?;

    for (i, r) in results.iter().enumerate() {
        hi_out[i] = r.0;
        h2_out[i] = r.1;
        cp_out[i] = r.2;
        c_out[i] = r.3;
        co_out[i] = r.4;
        o_out[i] = r.5;
        e_out[i] = r.6;
    }

    Ok(n_cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h_conservation() {
        let (x_hi, x_h2, _, _, _, _, _) = solve_cell_chemistry(
            1e3,    // n_H
            1e-15,  // k_pd_base (low UV)
            1e-14,  // k_ci
            4.67e-12, // alpha_cp
            1e-14,  // k_co_total
            3e-17 * 1e3 * 0.5, // r_h2_half_nh
            2.5e-17, // r_cr_base
            1.076e-4, // x_c_total
            0.5,    // x_hi
            0.25,   // x_h2
            5e-5,   // x_cp
            1e-5,   // x_c
            4e-5,   // x_co
            1e-10,  // x_hcop
            200,    // n_substeps
            1e-4,   // conv_tol
        );
        let h_sum = x_hi + 2.0 * x_h2;
        assert!((h_sum - 1.0).abs() < 1e-10, "H conservation violated: {}", h_sum);
    }

    #[test]
    fn test_c_conservation() {
        let (_, _, x_cp, x_c, x_co, _, _) = solve_cell_chemistry(
            1e3, 1e-15, 1e-14, 4.67e-12, 1e-14,
            3e-17 * 1e3 * 0.5, 2.5e-17,
            1.076e-4,
            0.5, 0.25, 5e-5, 1e-5, 4e-5, 1e-10,
            200, 1e-4,
        );
        let c_sum = x_cp + x_c + x_co;
        assert!(
            (c_sum - 1.076e-4).abs() / 1.076e-4 < 1e-6,
            "C conservation violated: {} vs {}", c_sum, 1.076e-4
        );
    }
}
