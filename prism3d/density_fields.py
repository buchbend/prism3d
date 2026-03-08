"""
3D density field generators for PRISM-3D.

Provides structured and turbulent density fields for 3D PDR modeling:
- Uniform slab (for benchmark comparison)
- Clumpy medium (spherical clumps in inter-clump medium)
- Fractal/turbulent (log-normal density PDF, power-law power spectrum)
- From external data (e.g., hydro simulation snapshots)
"""

import numpy as np


def uniform_slab(n_cells, box_size, n_H=1e3):
    """
    Uniform density slab illuminated from the -x face.
    
    Parameters
    ----------
    n_cells : int
        Cells per dimension (total cells = n_cells³).
    box_size : float
        Physical box size [cm].
    n_H : float
        Uniform hydrogen density [cm⁻³].
    
    Returns
    -------
    density : ndarray, shape (n_cells, n_cells, n_cells)
        Density field [cm⁻³].
    cell_size : float
        Cell edge length [cm].
    """
    density = np.full((n_cells, n_cells, n_cells), n_H, dtype=np.float64)
    cell_size = box_size / n_cells
    return density, cell_size


def clumpy_medium(n_cells, box_size, n_interclump=100, n_clump=1e4,
                   n_clumps=20, clump_radius_frac=0.05, seed=42):
    """
    Clumpy density field: dense spherical clumps in a diffuse medium.
    
    Parameters
    ----------
    n_interclump : float
        Inter-clump density [cm⁻³].
    n_clump : float
        Clump peak density [cm⁻³].
    n_clumps : int
        Number of clumps.
    clump_radius_frac : float
        Clump radius as fraction of box size.
    seed : int
        Random seed.
    """
    rng = np.random.RandomState(seed)
    cell_size = box_size / n_cells
    density = np.full((n_cells, n_cells, n_cells), n_interclump, dtype=np.float64)
    
    # Generate clump positions and sizes
    clump_centers = rng.rand(n_clumps, 3) * box_size
    clump_radii = clump_radius_frac * box_size * (0.5 + rng.rand(n_clumps))
    
    # Cell coordinates
    x = np.arange(n_cells) * cell_size + 0.5 * cell_size
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    
    for ic in range(n_clumps):
        cx, cy, cz = clump_centers[ic]
        rc = clump_radii[ic]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
        # Gaussian clump profile
        clump_profile = n_clump * np.exp(-0.5 * (dist / rc)**2)
        density = np.maximum(density, clump_profile)
    
    return density, cell_size


def fractal_turbulent(n_cells, box_size, n_mean=1e3, sigma_ln=1.5,
                       spectral_index=-3.7, seed=42):
    """
    Fractal turbulent density field with log-normal PDF.
    
    Generates a density field with:
    - Log-normal density PDF (characteristic of supersonic turbulence)
    - Power-law power spectrum P(k) ∝ k^spectral_index
    
    Parameters
    ----------
    n_mean : float
        Mean density [cm⁻³].
    sigma_ln : float
        Width of log-normal PDF. sigma_ln = sqrt(ln(1 + b²*M²))
        where b ~ 0.5 and M is the Mach number.
        sigma_ln = 1.5 corresponds to Mach ~ 5 turbulence.
    spectral_index : float
        Power spectrum slope. -11/3 for Kolmogorov, -4 for Burgers.
        Default -3.7 is typical for molecular clouds.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    cell_size = box_size / n_cells
    
    # Generate random Fourier modes with the desired power spectrum
    # Create 3D k-space
    kx = np.fft.fftfreq(n_cells, d=cell_size)
    ky = np.fft.fftfreq(n_cells, d=cell_size)
    kz = np.fft.fftfreq(n_cells, d=cell_size)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Power spectrum amplitude
    amplitude = K**(spectral_index / 2.0)
    amplitude[0, 0, 0] = 0.0  # No DC component
    
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, (n_cells, n_cells, n_cells))
    
    # Generate Gaussian random field in Fourier space
    random_field_k = amplitude * np.exp(1j * phases)
    
    # Transform to real space
    gaussian_field = np.real(np.fft.ifftn(random_field_k))
    
    # Normalize to desired sigma
    gaussian_field -= np.mean(gaussian_field)
    if np.std(gaussian_field) > 0:
        gaussian_field *= sigma_ln / np.std(gaussian_field)
    
    # Convert to log-normal density
    # ln(n/n_mean) = gaussian_field - sigma²/2 (so that <n> = n_mean)
    log_density = gaussian_field - sigma_ln**2 / 2.0
    density = n_mean * np.exp(log_density)
    
    # Enforce minimum density
    density = np.maximum(density, 1.0)  # Floor at 1 cm⁻³
    
    return density, cell_size


def density_field_stats(density):
    """Print statistics of a density field."""
    print(f"  Shape: {density.shape}")
    print(f"  Mean:  {np.mean(density):.2e} cm⁻³")
    print(f"  Med:   {np.median(density):.2e} cm⁻³")
    print(f"  Min:   {np.min(density):.2e} cm⁻³")
    print(f"  Max:   {np.max(density):.2e} cm⁻³")
    print(f"  σ(ln): {np.std(np.log(density)):.2f}")
    total_mass_proxy = np.sum(density)
    print(f"  Total: {total_mass_proxy:.2e} (sum)")
