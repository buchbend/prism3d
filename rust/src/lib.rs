use pyo3::prelude::*;

mod chemistry;
mod constants;
mod pe_table;
mod thermal;

/// PRISM-3D Rust core: accelerated physics kernels.
///
/// Provides drop-in replacements for the Python thermal solver,
/// chemistry integrator, PE table lookup, and RT cumsum.
/// All functions operate on numpy arrays in-place via PyO3.
#[pymodule]
fn _prism3d_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(thermal::solve_thermal_vec, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::solve_chemistry_euler, m)?)?;
    m.add_function(wrap_pyfunction!(pe_table::pe_heating_vec, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the version of the Rust core.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
