#!/usr/bin/env python3
"""
PRISM-3D Data Preparation — Orion Bar.

Reads downloaded JWST/Herschel FITS data and prepares it on a grid
that exactly matches the PRISM-3D model output, enabling direct
pixel-by-pixel comparison.

Pipeline:
  1. Read FITS cubes (NIRSpec IFU, MIRI MRS, Herschel maps)
  2. Define the Orion Bar extraction region (RA/Dec box)
  3. Reproject all data to a common Cartesian grid
  4. Extract emission line maps from spectral cubes
  5. Derive column density and dust temperature from continuum
  6. Save as a single .npz file with matched dimensions

Output: orion_bar_prepared.npz containing:
  - Grid metadata (n_pix, pixel_pc, extent_pc)
  - Line maps: CII_158, OI_63, H2_1-0_S1, CO_v1-0, PAH_7.7, PAH_11.3
  - Continuum maps: F770W, F1130W, PACS70, PACS160, SPIRE250
  - Derived: N_H, T_dust, A_V
  - Model config matching the observation grid

Usage:
  python prepare_data.py --input ./orion_bar_data --output ./prepared
  python prepare_data.py --input ./orion_bar_data --n-pix 32 --output ./prepared

Then in PRISM-3D:
  from prism3d.observations.from_observations import full_comparison
  import numpy as np
  data = np.load('prepared/orion_bar_prepared.npz', allow_pickle=True)
  # data['CII_158'], data['N_H'], data['model_config'], etc.
"""

import argparse
import os
import sys
import json
import glob
import numpy as np


# ============================================================
# Orion Bar Region Definition
# ============================================================

# The PDRs4All NIRSpec mosaic covers a strip across the Bar.
# We define the extraction box to match this strip.
ORION_BAR_REGION = {
    'ra_center': 83.83167,     # deg — center of NIRSpec mosaic
    'dec_center': -5.41611,    # deg
    'width_arcsec': 40.0,      # across the bar (RA direction)
    'height_arcsec': 25.0,     # along the bar (Dec direction)
    'pa_deg': 140.0,           # Position angle of the bar ridge
    'distance_pc': 414.0,
}

# Convert arcsec to pc at 414 pc: 1" = 414 AU = 0.00201 pc
ARCSEC_TO_PC = 414.0 * 1.496e13 / 3.086e18  # 0.00201 pc/arcsec

# Key emission lines and their rest wavelengths [μm]
LINE_CATALOG = {
    # From NIRSpec
    'H2_1-0_S1':  {'lam': 2.1218, 'dlam': 0.003, 'instrument': 'nirspec'},
    'H2_1-0_S0':  {'lam': 2.2233, 'dlam': 0.003, 'instrument': 'nirspec'},
    'H2_1-0_Q3':  {'lam': 2.4237, 'dlam': 0.003, 'instrument': 'nirspec'},
    'Br_alpha':   {'lam': 4.0512, 'dlam': 0.005, 'instrument': 'nirspec'},
    # From MIRI MRS
    'H2_0-0_S7':  {'lam': 5.511,  'dlam': 0.01, 'instrument': 'miri'},
    'H2_0-0_S5':  {'lam': 6.909,  'dlam': 0.01, 'instrument': 'miri'},
    'H2_0-0_S3':  {'lam': 9.665,  'dlam': 0.01, 'instrument': 'miri'},
    'H2_0-0_S1':  {'lam': 17.035, 'dlam': 0.02, 'instrument': 'miri'},
    'NeII_12.8':  {'lam': 12.814, 'dlam': 0.02, 'instrument': 'miri'},
    'PAH_7.7':    {'lam': 7.7,    'dlam': 0.5,  'instrument': 'miri', 'type': 'band'},
    'PAH_11.3':   {'lam': 11.3,   'dlam': 0.3,  'instrument': 'miri', 'type': 'band'},
    'PAH_3.3':    {'lam': 3.3,    'dlam': 0.05, 'instrument': 'nirspec', 'type': 'band'},
}


def prepare_data(input_dir, output_dir, n_pix=32, verbose=True):
    """
    Main preparation pipeline.
    
    Parameters
    ----------
    input_dir : str
        Directory with downloaded data (from download_data.py)
    output_dir : str
        Output directory for prepared arrays
    n_pix : int
        Spatial pixels per dimension for the output grid
    """
    os.makedirs(output_dir, exist_ok=True)
    
    region = ORION_BAR_REGION
    pixel_arcsec = region['width_arcsec'] / n_pix
    pixel_pc = pixel_arcsec * ARCSEC_TO_PC
    box_pc = region['width_arcsec'] * ARCSEC_TO_PC
    
    if verbose:
        print("="*60)
        print("  PRISM-3D Data Preparation — Orion Bar")
        print("="*60)
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Grid:   {n_pix} × {n_pix} pixels")
        print(f"  Pixel:  {pixel_arcsec:.2f}\" = {pixel_pc*1000:.1f} mpc")
        print(f"  FOV:    {region['width_arcsec']:.0f}\" × {region['height_arcsec']:.0f}\"")
        print(f"  Box:    {box_pc:.3f} pc")
    
    prepared = {
        'n_pix': n_pix,
        'pixel_arcsec': pixel_arcsec,
        'pixel_pc': pixel_pc,
        'box_pc': box_pc,
        'distance_pc': region['distance_pc'],
        'ra_center': region['ra_center'],
        'dec_center': region['dec_center'],
    }
    
    # --------------------------------------------------------
    # Step 1: Try to load JWST spectral cubes
    # --------------------------------------------------------
    cube_dir = os.path.join(input_dir, 'jwst_cubes')
    nirspec_cubes = sorted(glob.glob(os.path.join(cube_dir, '**/*nirspec*s3d*.fits'), recursive=True))
    miri_cubes = sorted(glob.glob(os.path.join(cube_dir, '**/*miri*mrs*s3d*.fits'), recursive=True))
    
    has_cubes = len(nirspec_cubes) > 0 or len(miri_cubes) > 0
    
    if has_cubes and verbose:
        print(f"\n  Found {len(nirspec_cubes)} NIRSpec + {len(miri_cubes)} MIRI cubes")
    
    if has_cubes:
        line_maps = extract_line_maps_from_cubes(
            nirspec_cubes, miri_cubes, region, n_pix, verbose)
        prepared.update(line_maps)
    
    # --------------------------------------------------------
    # Step 2: Try to load template spectra (always available)
    # --------------------------------------------------------
    template_dir = os.path.join(input_dir, 'templates')
    templates = load_templates(template_dir, verbose)
    if templates:
        prepared['templates'] = templates
    
    # --------------------------------------------------------
    # Step 3: Load Herschel continuum maps
    # --------------------------------------------------------
    herschel_dir = os.path.join(input_dir, 'herschel')
    herschel_maps = load_herschel(herschel_dir, region, n_pix, verbose)
    if herschel_maps:
        prepared.update(herschel_maps)
    
    # --------------------------------------------------------
    # Step 4: If no real data, generate synthetic reference
    # --------------------------------------------------------
    if not has_cubes and not herschel_maps:
        if verbose:
            print("\n  No observational data found. Generating synthetic reference.")
            print("  (Run download_data.py first for real data)")
        prepared.update(generate_synthetic_reference(n_pix, box_pc, region))
    
    # --------------------------------------------------------
    # Step 5: Derive column density and dust temperature
    # --------------------------------------------------------
    if 'PACS160' in prepared and 'SPIRE250' in prepared:
        T_dust, N_H = derive_dust_properties(
            prepared['PACS160'], prepared.get('SPIRE250'),
            wavelengths=[160, 250])
        prepared['T_dust_obs'] = T_dust
        prepared['N_H_obs'] = N_H
        if verbose:
            print(f"  Derived T_dust: {np.min(T_dust):.0f}–{np.max(T_dust):.0f} K")
            print(f"  Derived N_H: {np.min(N_H):.1e}–{np.max(N_H):.1e} cm⁻²")
    
    # --------------------------------------------------------
    # Step 5b: Hybrid beam matching for ALL maps
    # --------------------------------------------------------
    # Two cases:
    #   A) Observation finer than model → convolve OBS to model beam
    #      Store as {name}_matched — compare directly with model output
    #   B) Observation coarser than model → leave OBS as-is
    #      At comparison time, the MODEL gets convolved to obs beam
    #      Store as {name}_matched = native (unchanged)
    #      Store {name}_needs_model_conv = True
    
    model_beam = pixel_arcsec * 2.5  # Effective model beam (~Nyquist)
    
    native_beams = {
        'CII_158': 11.2, 'OI_63': 4.5, 'OI_145': 11.0,
        'CI_609': 9.4, 'CI_370': 18.0,
        'H2_1-0_S1': 0.10, 'H2_1-0_S0': 0.10, 'Br_alpha': 0.13,
        'H2_0-0_S7': 0.20, 'H2_0-0_S5': 0.25, 'H2_0-0_S3': 0.33,
        'H2_0-0_S1': 0.55, 'NeII_12.8': 0.40,
        'PAH_3.3': 0.11, 'PAH_7.7': 0.25, 'PAH_11.3': 0.36,
        'CO_1-0': 2.0, 'CO_2-1': 1.0, 'CO_3-2': 0.5,
        'PACS70': 5.6, 'PACS160': 11.4, 'SPIRE250': 18.1,
        'SPIRE350': 24.9, 'SPIRE500': 36.4,
        'T_dust_obs': 11.4, 'N_H_obs': 11.4, 'A_V_obs': 11.4,
    }
    
    obs_maps = {k: v for k, v in prepared.items()
                if isinstance(v, np.ndarray) and v.ndim == 2}
    
    if obs_maps and verbose:
        print(f"\n  Hybrid beam matching (model beam = {model_beam:.1f}\"):")
    
    from scipy.ndimage import gaussian_filter
    
    obs_finer = []   # Obs convolved down to model
    obs_coarser = [] # Model will be convolved up to obs at comparison time
    obs_matched_count = 0
    
    for name, image in obs_maps.items():
        obs_beam = native_beams.get(name, 0)
        
        if obs_beam < model_beam:
            # Case A: observation is finer → convolve obs to model beam
            sigma_obs = obs_beam / 2.355
            sigma_model = model_beam / 2.355
            sigma_kernel = np.sqrt(max(sigma_model**2 - sigma_obs**2, 0))
            sigma_pix = sigma_kernel / pixel_arcsec
            
            if sigma_pix > 0.3:
                prepared[f'{name}_matched'] = gaussian_filter(image, sigma=sigma_pix)
            else:
                prepared[f'{name}_matched'] = image.copy()
            
            prepared[f'{name}_compare_beam'] = np.array(model_beam)
            obs_finer.append(f"{name} ({obs_beam:.2f}\" → {model_beam:.1f}\")")
        else:
            # Case B: observation is coarser → keep as-is
            # At comparison time, model gets convolved to this beam
            prepared[f'{name}_matched'] = image.copy()
            prepared[f'{name}_compare_beam'] = np.array(obs_beam)
            obs_coarser.append(f"{name} ({obs_beam:.1f}\" — model convolved to match)")
        
        obs_matched_count += 1
    
    prepared['model_beam_arcsec'] = np.array(model_beam)
    
    if verbose and obs_maps:
        print(f"    Obs → model beam ({len(obs_finer)} maps):")
        for s in obs_finer:
            print(f"      ↓ {s}")
        print(f"    Model → obs beam at comparison ({len(obs_coarser)} maps):")
        for s in obs_coarser:
            print(f"      ↑ {s}")
    
    # --------------------------------------------------------
    # Step 6: Create matching PRISM-3D config
    # --------------------------------------------------------
    model_config = {
        'n': n_pix,
        'box': float(box_pc),
        'G0': 26000,
        'n_mean': 50000,
        'sigma_ln': 0.8,
        'zeta_cr': 1e-16,
        'nside': 2,
        'max_iter': 10,
        'distance_pc': 414,
        'output': os.path.join(output_dir, 'prism3d_orion_bar'),
        'pixel_arcsec': float(pixel_arcsec),
    }
    prepared['model_config'] = model_config
    
    config_path = os.path.join(output_dir, 'matched_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # --------------------------------------------------------
    # Step 7: Save everything
    # --------------------------------------------------------
    save_dict = {}
    for k, v in prepared.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (int, float, str)):
            save_dict[k] = np.array(v)
        elif isinstance(v, dict):
            save_dict[k] = np.array(json.dumps(v))
    
    npz_path = os.path.join(output_dir, 'orion_bar_prepared.npz')
    np.savez_compressed(npz_path, **save_dict)
    
    if verbose:
        n_maps = sum(1 for v in prepared.values() if isinstance(v, np.ndarray) and v.ndim == 2)
        print(f"\n  Saved: {npz_path}")
        print(f"  Contains: {n_maps} 2D maps, grid {n_pix}×{n_pix}")
        print(f"  Matching config: {config_path}")
        print(f"\n  To run PRISM-3D on this grid:")
        print(f"    python -m prism3d.run --config {config_path}")
    
    return prepared


# ============================================================
# FITS Cube Processing
# ============================================================

def extract_line_maps_from_cubes(nirspec_cubes, miri_cubes, region, n_pix, verbose):
    """Extract emission line maps from JWST IFU cubes."""
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError:
        if verbose:
            print("  astropy required for FITS cube processing")
        return {}
    
    maps = {}
    all_cubes = [(f, 'nirspec') for f in nirspec_cubes] + [(f, 'miri') for f in miri_cubes]
    
    for cube_path, instrument in all_cubes:
        if verbose:
            print(f"  Processing: {os.path.basename(cube_path)}")
        
        try:
            hdu = fits.open(cube_path)
            sci = hdu['SCI'] if 'SCI' in hdu else hdu[1]
            data = sci.data  # (n_wave, ny, nx)
            wcs = WCS(sci.header)
            
            if data is None or data.ndim != 3:
                hdu.close()
                continue
            
            n_wave, ny, nx = data.shape
            
            # Get wavelength axis
            if wcs.has_spectral:
                wave = wcs.spectral.pixel_to_world(np.arange(n_wave))
                wave_um = wave.to('um').value
            else:
                # Fallback: use header keywords
                crval3 = sci.header.get('CRVAL3', 1.0)
                cdelt3 = sci.header.get('CDELT3', 0.001)
                crpix3 = sci.header.get('CRPIX3', 1)
                wave_um = (crval3 + (np.arange(n_wave) - crpix3 + 1) * cdelt3) * 1e6
                if wave_um[0] > 100:  # Probably in meters
                    wave_um *= 1e6
            
            # Extract lines that fall in this cube's wavelength range
            for line_name, line_info in LINE_CATALOG.items():
                if line_info['instrument'] != instrument:
                    continue
                
                lam = line_info['lam']
                dlam = line_info['dlam']
                
                # Check if line is in this cube
                if lam < wave_um[0] or lam > wave_um[-1]:
                    continue
                
                # Extract line map
                is_band = line_info.get('type') == 'band'
                
                if is_band:
                    # Integrate over the band
                    mask = (wave_um >= lam - dlam) & (wave_um <= lam + dlam)
                else:
                    # Peak - continuum for narrow lines
                    line_mask = (wave_um >= lam - dlam) & (wave_um <= lam + dlam)
                    cont_mask = (
                        ((wave_um >= lam - 3*dlam) & (wave_um < lam - dlam)) |
                        ((wave_um > lam + dlam) & (wave_um <= lam + 3*dlam))
                    )
                    mask = line_mask
                
                if not np.any(mask):
                    continue
                
                line_map = np.nanmean(data[mask], axis=0)
                
                if not is_band and np.any(cont_mask):
                    cont_map = np.nanmean(data[cont_mask], axis=0)
                    line_map = line_map - cont_map
                
                # Reproject to target grid
                line_map_regrid = reproject_to_grid(line_map, wcs, region, n_pix)
                
                if line_map_regrid is not None:
                    maps[line_name] = line_map_regrid
                    if verbose:
                        print(f"    ✓ {line_name} ({lam:.3f} μm)")
            
            hdu.close()
            
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed: {e}")
    
    return maps


def reproject_to_grid(image_2d, wcs_in, region, n_pix):
    """
    Reproject a 2D image onto the target Cartesian grid.
    
    Uses simple nearest-neighbor interpolation from WCS coordinates
    to the target pixel grid.
    """
    try:
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        # Fallback: just resize with scipy
        from scipy.ndimage import zoom
        if image_2d is None:
            return None
        factors = [n_pix / s for s in image_2d.shape]
        return zoom(image_2d, factors, order=1)
    
    if image_2d is None:
        return None
    
    ny, nx = image_2d.shape
    
    # Target grid: centered on region, in arcsec offsets
    half_w = region['width_arcsec'] / 2
    half_h = region['height_arcsec'] / 2
    
    # Create target pixel centers in RA/Dec
    da = np.linspace(-half_w, half_w, n_pix) / 3600  # deg
    dd = np.linspace(-half_h, half_h, n_pix) / 3600
    DA, DD = np.meshgrid(da, dd)
    
    target_ra = region['ra_center'] + DA / np.cos(np.radians(region['dec_center']))
    target_dec = region['dec_center'] + DD
    
    # Map to input pixel coordinates
    try:
        # Use spatial WCS only (drop spectral axis if present)
        if wcs_in.naxis > 2:
            wcs_2d = wcs_in.celestial
        else:
            wcs_2d = wcs_in
        
        pix_coords = wcs_2d.world_to_pixel(
            SkyCoord(target_ra * u.deg, target_dec * u.deg))
        px, py = pix_coords[0], pix_coords[1]
        
        # Nearest-neighbor sampling
        ix = np.clip(np.round(px).astype(int), 0, nx - 1)
        iy = np.clip(np.round(py).astype(int), 0, ny - 1)
        
        output = image_2d[iy, ix]
        output = np.nan_to_num(output, nan=0.0)
        return output
        
    except Exception:
        # Fallback: simple resize
        from scipy.ndimage import zoom
        factors = [n_pix / s for s in image_2d.shape]
        return zoom(np.nan_to_num(image_2d), factors, order=1)


# ============================================================
# Template Spectra
# ============================================================

def load_templates(template_dir, verbose):
    """Load PDRs4All template spectra."""
    if not os.path.exists(template_dir):
        return None
    
    templates = {}
    fits_files = sorted(glob.glob(os.path.join(template_dir, '**/*.fits'), recursive=True))
    
    if not fits_files:
        # Try .dat files
        dat_files = sorted(glob.glob(os.path.join(template_dir, '**/*.dat'), recursive=True))
        for f in dat_files:
            name = os.path.basename(f).replace('.dat', '')
            try:
                data = np.loadtxt(f, comments='#')
                if data.ndim == 2 and data.shape[1] >= 2:
                    templates[name] = {'wavelength': data[:, 0], 'flux': data[:, 1]}
                    if verbose:
                        print(f"  ✓ Template: {name} ({len(data)} points)")
            except Exception:
                pass
        return templates if templates else None
    
    try:
        from astropy.io import fits as afits
        for f in fits_files:
            name = os.path.basename(f).replace('.fits', '')
            hdu = afits.open(f)
            for ext in range(len(hdu)):
                if hdu[ext].data is not None:
                    d = hdu[ext].data
                    if hasattr(d.dtype, 'names') and d.dtype.names:
                        cols = d.dtype.names
                        wl_col = [c for c in cols if 'wave' in c.lower() or 'lambda' in c.lower()]
                        fl_col = [c for c in cols if 'flux' in c.lower() or 'data' in c.lower() or 'sb' in c.lower()]
                        if wl_col and fl_col:
                            templates[name] = {
                                'wavelength': np.array(d[wl_col[0]], dtype=float),
                                'flux': np.array(d[fl_col[0]], dtype=float),
                            }
                            if verbose:
                                print(f"  ✓ Template: {name} ({len(d)} points, "
                                      f"{d[wl_col[0]][0]:.2f}–{d[wl_col[0]][-1]:.2f} μm)")
                            break
            hdu.close()
    except ImportError:
        pass
    
    return templates if templates else None


# ============================================================
# Herschel Maps
# ============================================================

def load_herschel(herschel_dir, region, n_pix, verbose):
    """Load and reproject Herschel maps."""
    if not os.path.exists(herschel_dir):
        return {}
    
    fits_files = sorted(glob.glob(os.path.join(herschel_dir, '**/*.fits'), recursive=True))
    if not fits_files:
        return {}
    
    maps = {}
    try:
        from astropy.io import fits as afits
        from astropy.wcs import WCS
        
        for f in fits_files:
            hdu = afits.open(f)
            name = os.path.basename(f).replace('.fits', '')
            
            # Identify band from filename or header
            header = hdu[0].header if hdu[0].data is not None else hdu[1].header
            data = hdu[0].data if hdu[0].data is not None else hdu[1].data
            
            band = None
            for key in ['PACS70', 'PACS160', 'SPIRE250', 'SPIRE350', 'SPIRE500']:
                if key.lower() in name.lower() or key.lower() in str(header).lower():
                    band = key
                    break
            
            if band is None:
                # Guess from wavelength in header
                wave = header.get('WAVELNTH', header.get('LAMBDA', 0))
                if 60 < wave < 80: band = 'PACS70'
                elif 140 < wave < 180: band = 'PACS160'
                elif 220 < wave < 280: band = 'SPIRE250'
            
            if band and data is not None and data.ndim == 2:
                wcs = WCS(header)
                regridded = reproject_to_grid(data, wcs, region, n_pix)
                if regridded is not None:
                    maps[band] = regridded
                    if verbose:
                        print(f"  ✓ {band}: {data.shape} → {n_pix}×{n_pix}")
            
            hdu.close()
    except ImportError:
        if verbose:
            print("  astropy required for Herschel map processing")
    
    return maps


# ============================================================
# Derived Properties
# ============================================================

def derive_dust_properties(flux_160, flux_250=None, wavelengths=[160, 250]):
    """Derive T_dust and N_H from Herschel continuum."""
    from prism3d.observations.from_observations import column_density_from_continuum
    
    if flux_250 is not None:
        # Two-band SED fit
        from prism3d.observations.from_observations import sed_fit_temperature
        T_dust, N_H = sed_fit_temperature(
            [flux_160, flux_250], wavelengths, beta=1.8)
    else:
        # Single band, assumed temperature
        T_dust = np.full_like(flux_160, 40.0)  # Typical for Orion Bar
        N_H = column_density_from_continuum(flux_160, 160, T_dust=40.0)
    
    return T_dust, N_H


# ============================================================
# Synthetic Reference (when no real data available)
# ============================================================

def generate_synthetic_reference(n_pix, box_pc, region):
    """
    Generate synthetic observation maps for testing the pipeline
    when real data hasn't been downloaded yet.
    
    Uses physically motivated profiles based on published Orion Bar
    observations.
    """
    maps = {}
    
    x = np.linspace(0, 1, n_pix)  # Across the bar
    y = np.linspace(0, 1, n_pix)  # Along the bar
    X, Y = np.meshgrid(x, y)
    
    # Distance from illuminated edge (x=0)
    d = X  # 0 at IF, 1 at deep molecular zone
    
    # Add turbulent fluctuations
    rng = np.random.RandomState(42)
    turb = 0.15 * rng.randn(n_pix, n_pix)
    
    # [CII] 158 μm: peaks at the PDR surface, drops into molecular zone
    # Typical: 1e-3 to 5e-3 erg/s/cm²/sr for the Orion Bar
    maps['CII_158'] = 3e-3 * np.exp(-3 * d) * (1 + turb) * (0.8 + 0.4 * np.sin(5*Y))
    
    # H2 1-0 S(1): peaks at the dissociation front
    maps['H2_1-0_S1'] = 5e-4 * np.exp(-((d - 0.25)/0.08)**2) * (1 + turb)
    
    # CO J=3-2: rises in the molecular zone
    maps['CO_3-2'] = 1e-4 * (1 - np.exp(-5 * d)) * (1 + turb)
    
    # PAH 7.7 μm: peaks at the IF, drops with nano-grain depletion
    maps['PAH_7.7'] = 1e2 * np.exp(-1.5 * d) * (1 + turb)  # MJy/sr
    
    # PAH 11.3 μm
    maps['PAH_11.3'] = 0.5e2 * np.exp(-1.5 * d) * (1 + turb)
    
    # Herschel-like continuum
    maps['PACS70'] = 50 * (0.3 + 0.7 * np.exp(-d)) * (1 + turb * 0.5)  # MJy/sr
    maps['PACS160'] = 30 * (0.5 + 0.5 * np.exp(-0.5 * d)) * (1 + turb * 0.5)
    maps['SPIRE250'] = 15 * (0.6 + 0.4 * np.exp(-0.3 * d)) * (1 + turb * 0.3)
    
    # Derived properties
    maps['T_dust_obs'] = 35 + 30 * np.exp(-2 * d) + 3 * turb
    maps['N_H_obs'] = 1e22 * (0.5 + 2 * d) * (1 + turb)
    maps['A_V_obs'] = maps['N_H_obs'] * 5.3e-22  # Standard conversion
    
    # Ensure all positive
    for k in maps:
        maps[k] = np.maximum(maps[k], 0)
    
    maps['is_synthetic'] = np.array(True)
    
    # Beam sizes for each observable [arcsec FWHM]
    # These are needed to convolve model predictions before comparison
    maps['beam_CII_158'] = np.array(11.2)      # Herschel PACS
    maps['beam_OI_63'] = np.array(4.5)         # Herschel PACS
    maps['beam_H2_1-0_S1'] = np.array(0.10)    # JWST NIRSpec
    maps['beam_CO_3-2'] = np.array(0.5)        # ALMA Band 7
    maps['beam_PAH_7.7'] = np.array(0.25)      # JWST MIRI Ch1
    maps['beam_PAH_11.3'] = np.array(0.36)     # JWST MIRI Ch2
    maps['beam_PACS70'] = np.array(5.6)        # Herschel PACS
    maps['beam_PACS160'] = np.array(11.4)      # Herschel PACS
    maps['beam_SPIRE250'] = np.array(18.1)     # Herschel SPIRE
    
    return maps


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM-3D Data Preparation — Orion Bar')
    parser.add_argument('--input', type=str, default='./orion_bar_data',
                        help='Input directory (from download_data.py)')
    parser.add_argument('--output', type=str, default='./prepared',
                        help='Output directory')
    parser.add_argument('--n-pix', type=int, default=32,
                        help='Spatial pixels per dimension (default: 32)')
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, n_pix=args.n_pix, verbose=True)


if __name__ == '__main__':
    main()
