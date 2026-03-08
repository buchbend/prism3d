"""
PRISM-3D Observation-to-Model Pipeline.

Converts observational data (FITS cubes, continuum maps) into
3D density fields for PRISM-3D modeling, then provides quantitative
comparison between model predictions and observations.

Workflow:
  1. Read observation (FITS continuum or line cube)
  2. Derive column density map from dust continuum SED
  3. Convert 2D column density → 3D density field
  4. Set boundary conditions (G₀, geometry)
  5. Run PRISM-3D
  6. Generate matched synthetic observations
  7. Compute residuals and χ² maps

Supports three observation → 3D inversion methods:
  a) Geometric: assume slab/cylinder/filament geometry
  b) Velocity-informed: use line-of-sight velocity from CO/[CII]
  c) Statistical: match observed N_H PDF with turbulent 3D field

Can also ingest FLASH/AREPO/RAMSES simulation snapshots as density fields.
"""

import numpy as np
import os


# ============================================================
# FITS I/O (uses astropy if available, numpy fallback)
# ============================================================

def read_fits(filepath):
    """
    Read a FITS file and return data + header.
    
    Returns
    -------
    data : ndarray
    header : dict
        WCS and metadata
    """
    try:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = dict(hdul[0].header)
        return data, header
    except ImportError:
        print("WARNING: astropy not installed. Install with: pip install astropy")
        print("Attempting numpy-only FITS read (limited)...")
        # Minimal FITS reader for simple 2D images
        with open(filepath, 'rb') as f:
            raw = f.read()
        # FITS header is ASCII in 2880-byte blocks
        header = {}
        pos = 0
        while pos < len(raw):
            block = raw[pos:pos+2880]
            if b'END' in block[:80]:
                break
            for i in range(0, 2880, 80):
                card = block[i:i+80].decode('ascii', errors='replace')
                if '=' in card[:8]:
                    key = card[:8].strip()
                    val = card[10:].split('/')[0].strip().strip("'")
                    header[key] = val
            pos += 2880
        
        # Data starts after header
        naxis1 = int(header.get('NAXIS1', 0))
        naxis2 = int(header.get('NAXIS2', 0))
        bitpix = int(header.get('BITPIX', -64))
        
        data_start = pos + 2880
        dtype = {-64: '>f8', -32: '>f4', 16: '>i2', 32: '>i4'}[bitpix]
        n_pix = naxis1 * naxis2
        data = np.frombuffer(raw[data_start:data_start + n_pix * abs(bitpix)//8],
                            dtype=dtype).reshape(naxis2, naxis1)
        return data.astype(np.float64), header


def write_fits(data, filepath, header=None):
    """Write data to a FITS file."""
    try:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        if header:
            for k, v in header.items():
                try:
                    hdu.header[k] = v
                except Exception:
                    pass
        hdu.writeto(filepath, overwrite=True)
    except ImportError:
        # Fallback: save as npz
        np.savez(filepath.replace('.fits', '.npz'), data=data, header=header or {})
        print(f"  (Saved as .npz — install astropy for FITS support)")


# ============================================================
# Column Density from Dust Continuum
# ============================================================

def column_density_from_continuum(flux_map, wavelength_um, T_dust=20.0,
                                    kappa_0=0.04, beta=1.8):
    """
    Derive H column density from dust continuum emission.
    
    Uses the modified blackbody:
      I_ν = B_ν(T_dust) × κ_ν × Σ_dust = B_ν(T) × κ_ν × (D/G) × μ × m_H × N_H
    
    Parameters
    ----------
    flux_map : ndarray (2D)
        Observed flux [MJy/sr] or [Jy/beam]
    wavelength_um : float
        Observation wavelength [μm]
    T_dust : float or ndarray
        Dust temperature [K] (from SED fitting or assumed)
    kappa_0 : float
        Dust opacity at 250 μm [cm²/g of dust]
    beta : float
        Emissivity spectral index
    
    Returns
    -------
    N_H : ndarray (2D)
        H column density [cm⁻²]
    """
    from ..utils.constants import k_boltz, h_planck, c_light, m_H
    
    lam_cm = wavelength_um * 1e-4
    nu = c_light / lam_cm
    nu_0 = c_light / 250e-4  # 250 μm reference
    
    kappa_nu = kappa_0 * (nu / nu_0)**beta  # cm²/g of dust
    
    # Planck function B_ν(T)
    x = h_planck * nu / (k_boltz * T_dust)
    x = np.clip(x, 0, 500)
    B_nu = 2 * h_planck * nu**3 / c_light**2 / (np.exp(x) - 1 + 1e-30)
    
    # Convert flux from MJy/sr to cgs
    I_nu = flux_map * 1e-17  # 1 MJy/sr = 1e-17 erg/s/cm²/Hz/sr
    
    # N_H = I_ν / (B_ν × κ_ν × D/G × μ × m_H)
    D2G = 0.01  # dust-to-gas ratio
    mu = 1.4    # mean molecular weight per H
    
    N_H = I_nu / (B_nu * kappa_nu * D2G * mu * m_H + 1e-50)
    N_H = np.maximum(N_H, 0)
    
    return N_H


def sed_fit_temperature(flux_maps, wavelengths_um, beta=1.8):
    """
    Fit dust temperature from multi-wavelength continuum.
    
    Uses the ratio of two bands to determine T_dust,
    then derives N_H from the longer wavelength.
    
    Parameters
    ----------
    flux_maps : list of 2D arrays
        Flux maps at each wavelength [MJy/sr]
    wavelengths_um : list of float
        Wavelengths [μm]
    
    Returns
    -------
    T_dust : ndarray (2D) [K]
    N_H : ndarray (2D) [cm⁻²]
    """
    from ..utils.constants import k_boltz, h_planck, c_light
    
    if len(flux_maps) < 2:
        raise ValueError("Need at least 2 bands for SED fitting")
    
    # Use ratio of two bands to get temperature
    # I₁/I₂ = (ν₁/ν₂)^(3+β) × (exp(hν₂/kT)-1) / (exp(hν₁/kT)-1)
    I1, I2 = flux_maps[0], flux_maps[1]
    nu1 = c_light / (wavelengths_um[0] * 1e-4)
    nu2 = c_light / (wavelengths_um[1] * 1e-4)
    
    ratio = np.maximum(I1, 1e-10) / np.maximum(I2, 1e-10)
    
    # Solve for T iteratively
    T = np.full_like(I1, 20.0)
    for _ in range(10):
        x1 = h_planck * nu1 / (k_boltz * T)
        x2 = h_planck * nu2 / (k_boltz * T)
        x1 = np.clip(x1, 0, 200)
        x2 = np.clip(x2, 0, 200)
        model_ratio = (nu1/nu2)**(3+beta) * (np.exp(x2)-1) / (np.exp(x1)-1 + 1e-30)
        
        # Newton correction
        T = T * (model_ratio / ratio)**0.3
        T = np.clip(T, 5, 200)
    
    # Get N_H from longer wavelength band
    N_H = column_density_from_continuum(I2, wavelengths_um[1], T_dust=T, beta=beta)
    
    return T, N_H


# ============================================================
# 2D Column Density → 3D Density Field
# ============================================================

def column_to_3d_slab(N_H_map, depth_pc, n_cells_z, pixel_scale_pc=None):
    """
    Convert a 2D column density map to a 3D density field
    assuming a uniform slab geometry.
    
    n_H(x, y, z) = N_H(x, y) / L_depth
    
    Parameters
    ----------
    N_H_map : ndarray (nx, ny)
        Column density [cm⁻²]
    depth_pc : float
        Assumed depth of the slab [pc]
    n_cells_z : int
        Number of cells along the depth axis
    
    Returns
    -------
    density : ndarray (nx, ny, nz)
    """
    from ..utils.constants import pc_cm
    
    L = depth_pc * pc_cm
    n_H_mean = N_H_map / L  # cm⁻³
    
    nx, ny = N_H_map.shape
    density = np.zeros((nx, ny, n_cells_z))
    for iz in range(n_cells_z):
        density[:, :, iz] = n_H_mean
    
    return density


def column_to_3d_turbulent(N_H_map, depth_pc, n_cells_z, sigma_ln=1.0, seed=42):
    """
    Convert 2D column density to 3D using turbulent substructure.
    
    The column density at each (x,y) pixel is distributed along z
    following a log-normal density profile with the observed mean.
    This preserves the total column while adding realistic 3D clumpiness.
    
    Parameters
    ----------
    N_H_map : ndarray (nx, ny)
    depth_pc : float
    n_cells_z : int
    sigma_ln : float
        Log-normal width for z-structure (0 = uniform, 1.5 = Mach 5)
    """
    from ..utils.constants import pc_cm
    
    L = depth_pc * pc_cm
    dz = L / n_cells_z
    
    nx, ny = N_H_map.shape
    rng = np.random.RandomState(seed)
    
    density = np.zeros((nx, ny, n_cells_z))
    
    for ix in range(nx):
        for iy in range(ny):
            # Mean density along this line of sight
            n_mean = N_H_map[ix, iy] / L
            
            if n_mean <= 0:
                continue
            
            # Generate log-normal z-profile
            z_noise = rng.normal(0, sigma_ln, n_cells_z)
            z_profile = np.exp(z_noise - sigma_ln**2 / 2)
            z_profile *= n_mean / np.mean(z_profile)  # Normalize to match N_H
            
            density[ix, iy, :] = z_profile
    
    density = np.maximum(density, 1.0)  # Floor
    return density


def column_to_3d_velocity(N_H_map, velocity_cube, depth_pc, n_cells_z):
    """
    Convert 2D column density to 3D using velocity information.
    
    The velocity cube (e.g., CO or [CII] position-position-velocity)
    provides depth structure through Doppler mapping: each velocity
    channel corresponds to a different z-position under the assumption
    of ordered motion (inflow, outflow, rotation).
    
    Parameters
    ----------
    N_H_map : ndarray (nx, ny)
    velocity_cube : ndarray (nx, ny, n_v)
        Intensity as a function of position and velocity
    depth_pc : float
    n_cells_z : int
    
    Returns
    -------
    density : ndarray (nx, ny, nz)
    """
    from ..utils.constants import pc_cm
    
    L = depth_pc * pc_cm
    nx, ny = N_H_map.shape
    n_v = velocity_cube.shape[2]
    
    density = np.zeros((nx, ny, n_cells_z))
    
    for ix in range(nx):
        for iy in range(ny):
            # Use velocity profile as proxy for z-structure
            v_profile = velocity_cube[ix, iy, :]
            v_profile = np.maximum(v_profile, 0)
            
            # Resample to n_cells_z
            z_indices = np.linspace(0, n_v - 1, n_cells_z)
            z_profile = np.interp(z_indices, np.arange(n_v), v_profile)
            
            # Normalize so column matches N_H
            total = np.sum(z_profile) * (L / n_cells_z)
            if total > 0:
                density[ix, iy, :] = z_profile * N_H_map[ix, iy] / total
    
    density = np.maximum(density, 1.0)
    return density


# ============================================================
# FLASH / Hydro Simulation Reader
# ============================================================

def read_flash_hdf5(filepath, field='dens'):
    """
    Read a FLASH AMR simulation snapshot (HDF5 format).
    
    Returns the density field resampled to a uniform grid.
    
    Parameters
    ----------
    filepath : str
        Path to FLASH checkpoint or plot file
    field : str
        Field name ('dens', 'temp', 'velx', etc.)
    
    Returns
    -------
    data : ndarray (3D)
    cell_size : float [cm]
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for FLASH files: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        # FLASH stores data in block-structured AMR
        if field in f:
            data = f[field][:]
        elif f'/{field}' in f:
            data = f[f'/{field}'][:]
        else:
            # Try yt-style
            available = list(f.keys())
            raise KeyError(f"Field '{field}' not found. Available: {available}")
        
        # Get grid metadata
        if 'bounding box' in f:
            bbox = f['bounding box'][:]
            box_size = bbox[0, 1] - bbox[0, 0]
        else:
            box_size = 1.0  # Assume code units
    
    if data.ndim == 4:
        # FLASH block format: (n_blocks, nxb, nyb, nzb)
        # Simple: take the first block or reshape
        n_blocks = data.shape[0]
        nb = data.shape[1]
        n_side = int(round(n_blocks**(1/3)))
        data = data.reshape(n_side, nb, n_side, nb, n_side, nb)
        data = data.transpose(0, 1, 2, 3, 4, 5).reshape(n_side*nb, n_side*nb, n_side*nb)
    
    cell_size = box_size / data.shape[0]
    return data, cell_size


def read_arepo_hdf5(filepath):
    """
    Read an AREPO Voronoi mesh snapshot.
    
    Resamples to a uniform Cartesian grid for PRISM-3D.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        coords = f['PartType0/Coordinates'][:]
        density = f['PartType0/Density'][:]
        
        # Determine box size
        box = f['Header'].attrs.get('BoxSize', np.max(coords))
    
    # Resample to uniform grid using nearest-neighbor
    n = 64  # Default resolution
    grid = np.zeros((n, n, n))
    cell_size = box / n
    
    ix = np.clip((coords[:, 0] / cell_size).astype(int), 0, n-1)
    iy = np.clip((coords[:, 1] / cell_size).astype(int), 0, n-1)
    iz = np.clip((coords[:, 2] / cell_size).astype(int), 0, n-1)
    
    # Average density in each cell
    count = np.zeros((n, n, n))
    np.add.at(grid, (ix, iy, iz), density)
    np.add.at(count, (ix, iy, iz), 1)
    mask = count > 0
    grid[mask] /= count[mask]
    grid[~mask] = np.median(density)
    
    return grid, cell_size


# ============================================================
# Model-Observation Comparison
# ============================================================

def compare_maps(observed, predicted, mask=None,
                  obs_beam_arcsec=None, model_cell_arcsec=None):
    """
    Quantitative comparison between observed and predicted maps.
    
    Hybrid beam-matching strategy:
      - If obs beam > model cell: convolve MODEL to obs beam
        (observation is coarser → degrade model to match)
      - If obs beam < model cell: convolve OBSERVATION to model cell
        (model is coarser → degrade observation to match)
      - Comparison always happens at the resolution of the coarser partner
    
    Parameters
    ----------
    observed, predicted : ndarray (2D)
    mask : ndarray (2D, bool), optional
    obs_beam_arcsec : float, optional
        Native beam of the observation [arcsec]
    model_cell_arcsec : float, optional
        Model cell size [arcsec] — effective model resolution
    
    Returns
    -------
    stats : dict
        Includes 'comparison_beam' showing what resolution was used
    """
    from scipy.ndimage import gaussian_filter
    
    model_beam = model_cell_arcsec * 2.5 if model_cell_arcsec else None
    comparison_beam = None
    resolution_status = 'no_beam_info'
    
    if obs_beam_arcsec is not None and model_beam is not None:
        if obs_beam_arcsec > model_beam:
            # Observation is coarser → convolve MODEL up to obs beam
            # Kernel: σ² = σ_obs² - σ_model²
            sigma_obs = obs_beam_arcsec / 2.355
            sigma_model = model_beam / 2.355
            sigma_kernel = np.sqrt(sigma_obs**2 - sigma_model**2)
            sigma_pix = sigma_kernel / model_cell_arcsec
            if sigma_pix > 0.3:
                predicted = gaussian_filter(predicted.copy(), sigma=sigma_pix)
            comparison_beam = obs_beam_arcsec
            resolution_status = 'model_convolved_to_obs'
        else:
            # Model is coarser → convolve OBSERVATION down to model beam
            # Kernel: σ² = σ_model² - σ_obs²
            sigma_obs = obs_beam_arcsec / 2.355
            sigma_model = model_beam / 2.355
            sigma_kernel = np.sqrt(sigma_model**2 - sigma_obs**2)
            sigma_pix = sigma_kernel / model_cell_arcsec
            if sigma_pix > 0.3:
                observed = gaussian_filter(observed.copy(), sigma=sigma_pix)
            comparison_beam = model_beam
            resolution_status = 'obs_convolved_to_model'
    
    if mask is None:
        mask = ((observed > 0) & (predicted > 0) & 
                np.isfinite(observed) & np.isfinite(predicted))
    
    obs_v = observed[mask]
    pred_v = predicted[mask]
    
    if len(obs_v) < 3:
        return {'chi2_reduced': np.inf, 'n_pixels': 0,
                'resolution_status': resolution_status}
    
    residual = pred_v - obs_v
    sigma = 0.1 * np.abs(obs_v) + 1e-30
    chi2 = np.sum((residual / sigma)**2) / len(obs_v)
    
    log_obs = np.log10(np.maximum(obs_v, 1e-30))
    log_pred = np.log10(np.maximum(pred_v, 1e-30))
    corr = np.corrcoef(log_obs, log_pred)[0, 1] if len(obs_v) > 2 else 0
    
    scale = np.median(pred_v) / np.median(obs_v) if np.median(obs_v) > 0 else 0
    
    return {
        'chi2_reduced': float(chi2),
        'correlation': float(corr),
        'scale_factor': float(scale),
        'residual_rms': float(np.std(residual)),
        'n_pixels': int(len(obs_v)),
        'resolution_status': resolution_status,
        'comparison_beam_arcsec': comparison_beam,
        'obs_beam_arcsec': obs_beam_arcsec,
        'model_beam_arcsec': model_beam,
    }


def full_comparison(solver, observed_maps, distance_pc=414, los_axis=2,
                     beam_sizes=None, pixel_arcsec=None):
    """
    Run a complete model-vs-observation comparison.
    
    Model predictions are convolved to the observation beam before
    comparison. Beam sizes can be provided per-map or will use defaults.
    
    Parameters
    ----------
    solver : PDRSolver3D
    observed_maps : dict
        {'CII_158': array, 'PACS160': array, ...}
    beam_sizes : dict, optional
        {'CII_158': 11.2, 'PACS160': 11.4, ...} in arcsec
    pixel_arcsec : float, optional
        Pixel size. If None, computed from solver grid and distance.
    """
    from .jwst_pipeline import (
        line_emission_map, dust_continuum_map, column_density_maps,
        LINE_BEAMS, JWST_BANDS, HERSCHEL_BANDS, ALMA_BANDS
    )
    from ..utils.constants import pc_cm
    
    if pixel_arcsec is None:
        pixel_arcsec = (solver.cell_size / (distance_pc * pc_cm)) * 206265
    
    cell_arcsec = pixel_arcsec  # Model cell size = pixel size
    
    # Default beam sizes from instrument catalogs
    default_beams = {**LINE_BEAMS}
    for band_name, band in {**JWST_BANDS, **HERSCHEL_BANDS, **ALMA_BANDS}.items():
        default_beams[band_name] = band['beam']
    if beam_sizes:
        default_beams.update(beam_sizes)
    
    results = {}
    
    for key, obs_map in observed_maps.items():
        if not isinstance(obs_map, np.ndarray) or obs_map.ndim != 2:
            continue
        
        # Generate model prediction
        clean_key = key.replace('line_', '').replace('dust_', '')
        
        if clean_key in ['CII_158', 'OI_63', 'OI_145', 'CI_609', 'CI_370'] or \
           clean_key.startswith('CO_') or clean_key.startswith('H2_'):
            try:
                pred = line_emission_map(solver, clean_key, los_axis)
            except (ValueError, KeyError):
                continue
        elif clean_key in ['PACS70', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500', 'Band7']:
            band_wavelengths = {
                'PACS70': 70, 'PACS160': 160, 'SPIRE250': 250,
                'SPIRE350': 350, 'SPIRE500': 500, 'Band7': 850,
            }
            lam = band_wavelengths.get(clean_key, 160)
            pred = dust_continuum_map(solver, lam, los_axis)
        else:
            continue
        
        # Match array shapes if needed
        if pred.shape != obs_map.shape:
            from scipy.ndimage import zoom
            factors = [o / p for o, p in zip(obs_map.shape, pred.shape)]
            pred = zoom(pred, factors, order=1)
        
        # Get beam size for this observable
        beam = default_beams.get(clean_key, default_beams.get(key, None))
        
        results[key] = compare_maps(
            obs_map, pred,
            obs_beam_arcsec=beam,
            model_cell_arcsec=cell_arcsec,
        )
        results[key]['obs_beam_arcsec'] = beam
    
    # Resolution summary: what grid is needed for each observable?
    from ..utils.constants import pc_cm
    cell_arcsec = (solver.cell_size / (distance_pc * pc_cm)) * 206265
    
    results['_resolution_summary'] = {
        'model_cell_arcsec': float(cell_arcsec),
        'model_cell_AU': float(cell_arcsec * distance_pc),
        'model_n': solver.nx,
        'model_box_pc': float(solver.box_size / pc_cm),
    }
    
    needs_higher_res = []
    for key, stats in results.items():
        if key.startswith('_'):
            continue
        status = stats.get('resolution_status', 'unknown')
        beam = stats.get('beam_arcsec')
        if status == 'underresolved' and beam:
            # How many cells would we need to resolve this beam?
            n_needed = int(np.ceil(solver.box_size / pc_cm / 
                          (beam * distance_pc * 1.496e13 / 3.086e18) * 2))
            needs_higher_res.append(f"{key}: beam={beam}\" → need ≥{n_needed}³")
    
    results['_resolution_summary']['underresolved'] = needs_higher_res
    
    return results
