/// Vectorized THEMIS PE heating table lookup.
///
/// Ports the Python `THEMISTables.pe_heating_vec` from grains/themis_tables.py.
/// Per-cell independent → embarrassingly parallel via rayon.
///
/// The PE table is precomputed by Python (charge-distribution physics).
/// This module only does the fast bilinear interpolation + summation
/// over grain bins at runtime.
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray3;
use numpy::PyReadwriteArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Habing flux constant [erg cm⁻² s⁻¹]
const F_HABING: f64 = 1.6e-3;
/// Mean FUV photon energy [erg]
const E_FUV_MEAN: f64 = 10.0 * 1.602176634e-12;

/// Compute interpolation index and weight for uniform grid.
#[inline(always)]
fn interp_idx_weight(x: f64, grid_min: f64, grid_step: f64, n: usize) -> (usize, f64) {
    let t = (x - grid_min) / grid_step;
    let idx = t.floor() as i64;
    let idx = idx.max(0).min((n as i64) - 2) as usize;
    let w = ((x - (grid_min + idx as f64 * grid_step)) / grid_step).clamp(0.0, 1.0);
    (idx, w)
}

/// Vectorized THEMIS PE heating rate [erg cm⁻³ s⁻¹].
///
/// All per-cell arrays are 1D (pre-flattened by Python).
/// The PE table shape is (n_bins, n_psi, n_Eg), built by Python.
///
/// Parameters:
/// - pe_table: precomputed table (n_bins × n_psi × n_Eg)
/// - log_psi_min, log_psi_step: uniform grid params for log10(psi) axis
/// - eg_min, eg_step: uniform grid params for E_g axis
/// - bin_sigma_abs, bin_dn_da_da: per-bin grain properties (n_bins)
/// - bin_is_nano: per-bin flag (1.0 = nano, 0.0 = not)
/// - g0, t_gas, x_e, n_h, f_nano, e_g: per-cell arrays (n_cells)
/// - gamma_pe_out: output array (n_cells), written in-place
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pe_heating_vec<'py>(
    py: Python<'py>,
    pe_table: PyReadonlyArray3<'py, f64>,
    log_psi_min: f64,
    log_psi_step: f64,
    n_psi: usize,
    eg_min: f64,
    eg_step: f64,
    n_eg: usize,
    bin_sigma_abs: PyReadonlyArray1<'py, f64>,
    bin_dn_da_da: PyReadonlyArray1<'py, f64>,
    bin_is_nano: PyReadonlyArray1<'py, f64>,
    g0: PyReadonlyArray1<'py, f64>,
    t_gas: PyReadonlyArray1<'py, f64>,
    x_e: PyReadonlyArray1<'py, f64>,
    n_h: PyReadonlyArray1<'py, f64>,
    f_nano: PyReadonlyArray1<'py, f64>,
    e_g: PyReadonlyArray1<'py, f64>,
    mut gamma_pe_out: PyReadwriteArray1<'py, f64>,
) -> PyResult<()> {
    // Extract slices
    let table = pe_table.as_slice()?;
    let sigma_abs = bin_sigma_abs.as_slice()?;
    let dn_da_da = bin_dn_da_da.as_slice()?;
    let is_nano = bin_is_nano.as_slice()?;
    let g0_s = g0.as_slice()?;
    let t_s = t_gas.as_slice()?;
    let xe_s = x_e.as_slice()?;
    let nh_s = n_h.as_slice()?;
    let fn_s = f_nano.as_slice()?;
    let eg_s = e_g.as_slice()?;

    let n_cells = g0_s.len();
    let n_bins = sigma_abs.len();
    let table_stride_bin = n_psi * n_eg; // stride for one bin in flattened table

    // Compute results in parallel
    let results: Vec<f64> = py.allow_threads(|| {
        (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let nh_i = nh_s[i];
                let g0_i = g0_s[i];
                let t_i = t_s[i];
                let xe_i = xe_s[i];
                let fn_i = fn_s[i];
                let eg_i = eg_s[i];

                // Charging parameter psi = G0 * sqrt(T) / n_e
                let n_e = (xe_i * nh_i).max(1e-10 * nh_i);
                let psi = g0_i * t_i.sqrt() / n_e;
                let log_psi = psi.max(0.1).log10();

                // Photon flux
                let phi_uv = F_HABING * g0_i / E_FUV_MEAN;

                // Interpolation indices
                let (psi_idx, psi_w) =
                    interp_idx_weight(log_psi, log_psi_min, log_psi_step, n_psi);
                let (eg_idx, eg_w) = interp_idx_weight(eg_i, eg_min, eg_step, n_eg);

                let mut gamma = 0.0_f64;

                for ib in 0..n_bins {
                    // Bilinear interpolation from flattened table
                    let base = ib * table_stride_bin;
                    let v00 = table[base + psi_idx * n_eg + eg_idx];
                    let v10 = table[base + (psi_idx + 1) * n_eg + eg_idx];
                    let v01 = table[base + psi_idx * n_eg + (eg_idx + 1)];
                    let v11 = table[base + (psi_idx + 1) * n_eg + (eg_idx + 1)];

                    let ye = v00 * (1.0 - psi_w) * (1.0 - eg_w)
                        + v10 * psi_w * (1.0 - eg_w)
                        + v01 * (1.0 - psi_w) * eg_w
                        + v11 * psi_w * eg_w;

                    let n_gr = dn_da_da[ib];
                    let contrib = n_gr * nh_i * sigma_abs[ib] * phi_uv * ye;

                    if is_nano[ib] > 0.5 {
                        gamma += fn_i * contrib;
                    } else {
                        gamma += contrib;
                    }
                }

                gamma
            })
            .collect()
    });

    // Write results back
    let out = gamma_pe_out.as_slice_mut()?;
    out.copy_from_slice(&results);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp_idx_weight() {
        // Middle of grid
        let (idx, w) = interp_idx_weight(3.0, 0.0, 1.0, 10);
        assert_eq!(idx, 3);
        assert!((w - 0.0).abs() < 1e-10);

        // Between points
        let (idx, w) = interp_idx_weight(3.5, 0.0, 1.0, 10);
        assert_eq!(idx, 3);
        assert!((w - 0.5).abs() < 1e-10);

        // Clamped low
        let (idx, w) = interp_idx_weight(-1.0, 0.0, 1.0, 10);
        assert_eq!(idx, 0);
        assert_eq!(w, 0.0);

        // Clamped high
        let (idx, w) = interp_idx_weight(100.0, 0.0, 1.0, 10);
        assert_eq!(idx, 8);
        assert_eq!(w, 1.0);
    }
}
