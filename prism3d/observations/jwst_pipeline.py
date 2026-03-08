"""
Synthetic Observation Pipeline for PRISM-3D.

Generates mock observations from 3D PDR models:
  1. Dust continuum maps (Herschel, ALMA, JWST MIRI bands)
  2. Fine-structure line maps ([CII] 158μm, [OI] 63μm)
  3. CO rotational line maps
  4. Column density maps (N_H2, N_CO, A_V)
  5. Dust SED integrated along LOS
  6. Beam convolution for telescope comparison

These are integrated along a chosen line-of-sight (LOS) through
the 3D model to produce 2D maps directly comparable to observations.
"""

import numpy as np
from ..utils.constants import (k_boltz, h_planck, c_light, m_H, eV_to_erg,
                                 pc_cm)


# ============================================================
# Instrument definitions
# ============================================================

# JWST NIRCam/MIRI bands (central wavelength, bandwidth, beam FWHM in arcsec)
JWST_BANDS = {
    'F335M':  {'lam_um': 3.35,  'dlam_um': 0.35,  'beam': 0.11},
    'F470N':  {'lam_um': 4.70,  'dlam_um': 0.05,  'beam': 0.15},
    'F770W':  {'lam_um': 7.7,   'dlam_um': 2.2,   'beam': 0.25},
    'F1130W': {'lam_um': 11.3,  'dlam_um': 0.7,   'beam': 0.36},
    'F1500W': {'lam_um': 15.0,  'dlam_um': 3.0,   'beam': 0.49},
    'F2550W': {'lam_um': 25.5,  'dlam_um': 4.0,   'beam': 0.82},
}

# Herschel PACS/SPIRE bands
HERSCHEL_BANDS = {
    'PACS70':   {'lam_um': 70,   'dlam_um': 25,   'beam': 5.6},
    'PACS160':  {'lam_um': 160,  'dlam_um': 85,   'beam': 11.4},
    'SPIRE250': {'lam_um': 250,  'dlam_um': 75,   'beam': 18.1},
    'SPIRE350': {'lam_um': 350,  'dlam_um': 100,  'beam': 24.9},
    'SPIRE500': {'lam_um': 500,  'dlam_um': 200,  'beam': 36.4},
}

# ALMA bands
ALMA_BANDS = {
    'Band6':  {'lam_um': 1300,  'dlam_um': 200,  'beam': 0.5},
    'Band7':  {'lam_um': 850,   'dlam_um': 100,  'beam': 0.3},
}

# Fine-structure lines (with beam convolution)
LINE_BEAMS = {
    # JWST MIRI MRS beam sizes (diffraction-limited)
    'CII_158':  11.2,   # Herschel PACS [arcsec] — NOT a JWST line
    'OI_63':    4.5,    # Herschel PACS
    'OI_145':   11.0,   # Herschel PACS
    'CI_609':   9.4,    # Herschel SPIRE FTS (or ground-based)
    'CI_370':   18.0,   # Herschel SPIRE FTS
    # JWST NIRSpec IFU lines
    'H2_1-0_S1': 0.10,  # NIRSpec at 2.12 μm
    'H2_1-0_S0': 0.10,
    'Br_alpha':  0.13,   # NIRSpec at 4.05 μm
    # JWST MIRI MRS lines
    'H2_0-0_S7': 0.20,  # MIRI Ch1 at 5.5 μm
    'H2_0-0_S5': 0.25,  # MIRI Ch1 at 6.9 μm
    'H2_0-0_S3': 0.33,  # MIRI Ch2 at 9.7 μm
    'H2_0-0_S1': 0.55,  # MIRI Ch3 at 17 μm
    'NeII_12.8': 0.40,  # MIRI Ch2 at 12.8 μm
    # PAH bands
    'PAH_3.3':  0.11,   # NIRSpec
    'PAH_7.7':  0.25,   # MIRI Ch1
    'PAH_11.3': 0.36,   # MIRI Ch2
    # CO rotational (ground-based / ALMA)
    'CO_1-0':   2.0,    # ALMA Band 3 (typical)
    'CO_2-1':   1.0,    # ALMA Band 6
    'CO_3-2':   0.5,    # ALMA Band 7
}

# Fine-structure lines
LINES = {    'CII_158':  {'lam_um': 157.74, 'E_up_K': 91.2,   'A_ul': 2.3e-6,
                  'g_u': 4, 'species': 'C+'},
    'OI_63':    {'lam_um': 63.18,  'E_up_K': 228.0,  'A_ul': 8.95e-5,
                  'g_u': 3, 'species': 'O'},
    'OI_145':   {'lam_um': 145.5,  'E_up_K': 326.6,  'A_ul': 1.75e-5,
                  'g_u': 1, 'species': 'O'},
    'CI_609':   {'lam_um': 609.14, 'E_up_K': 23.6,   'A_ul': 7.93e-8,
                  'g_u': 3, 'species': 'C'},
    'CI_370':   {'lam_um': 370.42, 'E_up_K': 62.5,   'A_ul': 2.68e-7,
                  'g_u': 5, 'species': 'C'},
}

# CO rotational lines
CO_LINES = {}
B_CO = 57.636e9  # Hz (rotational constant)
for J in range(1, 11):
    nu = 2 * B_CO * J  # Hz
    lam = c_light / nu * 1e4  # μm
    E_up = h_planck * B_CO * J * (J + 1) / k_boltz  # K
    A_ul = 1.07e-7 * J**3 / (2 * J + 1)  # Approximate A coefficient
    CO_LINES[f'CO_{J}-{J-1}'] = {
        'lam_um': lam, 'E_up_K': E_up, 'A_ul': A_ul,
        'g_u': 2 * J + 1, 'J_up': J, 'species': 'CO'
    }


# ============================================================
# Core integration routines
# ============================================================

def integrate_column(solver, quantity, los_axis=2):
    """
    Integrate a quantity along the line of sight.
    
    Parameters
    ----------
    solver : PDRSolver3D
        The 3D PDR model
    quantity : str or ndarray
        Name of solver attribute or 3D array to integrate
    los_axis : int
        LOS axis (0=x, 1=y, 2=z)
    
    Returns
    -------
    column : ndarray (2D)
        Integrated column [quantity × cm]
    """
    if isinstance(quantity, str):
        data = getattr(solver, quantity)
    else:
        data = quantity
    
    return np.sum(data, axis=los_axis) * solver.cell_size


def column_density_maps(solver, los_axis=2):
    """
    Compute column density maps for all major species.
    
    Returns
    -------
    maps : dict of 2D arrays
        N_H, N_H2, N_CO, N_Cp, N_e, A_V_total
    """
    ds = solver.cell_size
    maps = {}
    
    # Total H column
    maps['N_H'] = np.sum(solver.density, axis=los_axis) * ds
    
    # H2 column
    maps['N_H2'] = np.sum(solver.density * solver.x_H2 * 2, axis=los_axis) * ds
    
    # CO column
    maps['N_CO'] = np.sum(solver.density * solver.x_CO, axis=los_axis) * ds
    
    # C+ column
    maps['N_Cp'] = np.sum(solver.density * solver.x_Cp, axis=los_axis) * ds
    
    # Electron column
    maps['N_e'] = np.sum(solver.density * solver.x_e, axis=los_axis) * ds
    
    # Total visual extinction through the cloud
    from ..utils.constants import AV_per_NH
    maps['A_V_total'] = maps['N_H'] * AV_per_NH
    
    return maps


def dust_continuum_map(solver, lam_um, los_axis=2):
    """
    Compute a dust continuum emission map at a given wavelength.
    
    Integrates the thermal dust emission along the LOS.
    Uses modified blackbody with local T_dust and density.
    
    Parameters
    ----------
    solver : PDRSolver3D
    lam_um : float
        Wavelength [μm]
    los_axis : int
    
    Returns
    -------
    I_nu : ndarray (2D)
        Specific intensity [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]
    """
    lam_cm = lam_um * 1e-4
    nu = c_light / lam_cm
    
    # Dust opacity per H: kappa_nu = kappa_0 * (nu/nu_0)^beta
    # Using standard ISM values
    nu_0 = c_light / (250e-4)  # 250 μm reference
    kappa_0 = 0.04  # cm²/g at 250 μm (Hildebrand 1983)
    beta = 1.8
    kappa_nu = kappa_0 * (nu / nu_0)**beta
    
    # Emissivity per cell: j_nu = n_H * m_H * D/G * kappa_nu * B_nu(T_dust)
    D2G = 0.01  # dust-to-gas ratio
    
    # Build 3D emissivity
    x = h_planck * nu / (k_boltz * solver.T_dust)
    x = np.clip(x, 0, 500)
    B_nu = 2 * h_planck * nu**3 / c_light**2 / (np.exp(x) - 1 + 1e-30)
    
    j_nu = solver.density * m_H * D2G * kappa_nu * B_nu  # erg/s/cm³/Hz/sr
    
    # Integrate along LOS
    I_nu = np.sum(j_nu, axis=los_axis) * solver.cell_size
    
    return I_nu


def line_emission_map(solver, line_name, los_axis=2):
    """
    Compute a fine-structure line emission map.
    
    Uses the thin-limit emissivity with escape probability correction
    for [CII] and [OI].
    
    Parameters
    ----------
    solver : PDRSolver3D
    line_name : str
        Line identifier (e.g., 'CII_158', 'OI_63', 'CO_1-0')
    
    Returns
    -------
    I_line : ndarray (2D)
        Line intensity [erg s⁻¹ cm⁻² sr⁻¹]
    """
    if line_name in LINES:
        line = LINES[line_name]
    elif line_name in CO_LINES:
        line = CO_LINES[line_name]
    else:
        raise ValueError(f"Unknown line: {line_name}")
    
    E_up = line['E_up_K']
    A_ul = line['A_ul']
    g_u = line['g_u']
    lam = line['lam_um'] * 1e-4  # cm
    nu = c_light / lam
    species = line['species']
    
    # Get abundance of the emitting species
    species_map = {
        'C+': solver.x_Cp, 'O': solver.x_O, 'C': solver.x_C,
        'CO': solver.x_CO,
    }
    x_species = species_map.get(species, np.zeros_like(solver.density))
    
    # Number density of emitting species
    n_species = solver.density * x_species
    
    # Fraction in upper level (LTE approximation)
    # For two-level atom: n_u/n_total = (g_u/g_total) * exp(-E_up/T) / Z(T)
    # Simplified partition function
    T = solver.T_gas
    
    if species in ('C+', 'O', 'C'):
        # Fine-structure: use Boltzmann with correct g
        g_total = {'C+': 6, 'O': 9, 'C': 9}[species]  # Total statistical weight
        f_up = (g_u / g_total) * np.exp(-E_up / np.maximum(T, 5.0))
    elif species == 'CO':
        # CO rotational: LTE with partition function Z = kT/(hB)
        J = line.get('J_up', 1)
        Z_rot = k_boltz * T / (h_planck * B_CO)
        Z_rot = np.maximum(Z_rot, 1.0)
        f_up = g_u * np.exp(-E_up / np.maximum(T, 5.0)) / Z_rot
    else:
        f_up = np.exp(-E_up / np.maximum(T, 5.0))
    
    # Emissivity: j_line = n_u * A_ul * h*nu / (4π)
    n_u = n_species * f_up
    j_line = n_u * A_ul * h_planck * nu / (4 * np.pi)  # erg/s/cm³/sr
    
    # Lower level population (for optical depth)
    if species in ('C+', 'O', 'C'):
        g_total_dict = {'C+': 6, 'O': 9, 'C': 9}
        g_l = g_total_dict[species] - g_u
        n_l = n_species * (g_l / g_total_dict[species])
    elif species == 'CO':
        J_up = line.get('J_up', 1)
        g_l = 2 * (J_up - 1) + 1 if J_up > 0 else 1
        E_low = h_planck * B_CO * (J_up - 1) * J_up / k_boltz if J_up > 0 else 0
        Z_rot2 = np.maximum(k_boltz * T / (h_planck * B_CO), 1.0)
        n_l = n_species * g_l * np.exp(-E_low / np.maximum(T, 5.0)) / Z_rot2
    else:
        n_l = n_species * 0.5
    
    # Frequency-integrated absorption coefficient
    kappa_0 = A_ul * c_light**2 / (8 * np.pi * nu**2)
    g_ratio = g_u / max(g_l if species in ('C+','O','C','CO') else 1, 1)
    kappa_line = kappa_0 * np.maximum(n_l * g_ratio - n_u, 0)
    
    # Line profile width (thermal + 1 km/s turbulent)
    m_sp = {'C+': 12, 'O': 16, 'C': 12, 'CO': 28}.get(species, 12) * m_H
    v_turb = 1e5  # 1 km/s default
    sigma_v = np.sqrt(k_boltz * T / m_sp + v_turb**2)
    # Profile value at line center: φ(0) = 1/(σ_v √(2π))
    phi_0 = 1.0 / (sigma_v * np.sqrt(2 * np.pi))  # in (cm/s)⁻¹
    # Convert to Hz⁻¹: φ_Hz = φ_v × (c/ν)
    phi_Hz = phi_0 * c_light / nu
    
    # Optical depth per cell at line center
    dtau = kappa_line * phi_Hz * solver.cell_size
    
    # LOS radiative transfer: I(s+ds) = I(s)*exp(-dτ) + S*(1-exp(-dτ))
    nx, ny, nz = solver.density.shape
    if los_axis == 0:
        n1, n2, n_los = ny, nz, nx
    elif los_axis == 1:
        n1, n2, n_los = nx, nz, ny
    else:
        n1, n2, n_los = nx, ny, nz
    
    I_line = np.zeros((n1, n2))
    tau_total = np.zeros((n1, n2))  # Accumulated optical depth
    ds = solver.cell_size
    
    for i_los in range(n_los):
        if los_axis == 0:
            j_s = j_line[i_los, :, :]
            dt_s = dtau[i_los, :, :]
        elif los_axis == 1:
            j_s = j_line[:, i_los, :]
            dt_s = dtau[:, i_los, :]
        else:
            j_s = j_line[:, :, i_los]
            dt_s = dtau[:, :, i_los]
        
        dt_s = np.minimum(dt_s, 50)
        
        # Escape probability: β(τ) = (1 - exp(-τ)) / τ
        # For τ << 1: β → 1 (optically thin, all photons escape)
        # For τ >> 1: β → 1/τ (only surface layer contributes)
        # This gives the correct frequency-integrated intensity in both limits
        beta_esc = np.where(dt_s > 1e-3,
                            (1 - np.exp(-dt_s)) / (dt_s + 1e-30),
                            1.0 - 0.5 * dt_s)  # Taylor expansion for small τ
        
        # Attenuate existing background by this cell's absorption
        I_line *= np.exp(-dt_s)
        
        # Add local emission × escape probability
        I_line += j_s * ds * beta_esc
        
        tau_total += dt_s
    
    return I_line


def convolve_beam(image, beam_fwhm_arcsec, pixel_size_arcsec):
    """
    Convolve an image with a Gaussian beam.
    
    Parameters
    ----------
    image : ndarray (2D)
    beam_fwhm_arcsec : float
    pixel_size_arcsec : float
    
    Returns
    -------
    convolved : ndarray (2D)
    """
    from scipy.ndimage import gaussian_filter
    
    sigma_pix = beam_fwhm_arcsec / (2.355 * pixel_size_arcsec)
    if sigma_pix < 0.5:
        return image  # Beam smaller than pixel
    return gaussian_filter(image, sigma=sigma_pix)


def convolve_to_common_beam(maps, native_beams, target_beam_arcsec,
                              pixel_size_arcsec, warnings=None):
    """
    Convolve all maps to a common beam (the model resolution).
    
    For each map, the convolution kernel is the quadrature difference
    between the target beam and the native beam:
      σ_kernel² = σ_target² - σ_native²
    
    If the native beam is already larger than the target, the map
    is returned unchanged (cannot sharpen). A warning is emitted.
    
    Parameters
    ----------
    maps : dict
        {name: 2D array}
    native_beams : dict
        {name: beam_fwhm_arcsec} — original instrument beam
    target_beam_arcsec : float
        Target beam FWHM to convolve everything to
    pixel_size_arcsec : float
        Pixel size of the maps
    warnings : list, optional
        Append warning strings here
    
    Returns
    -------
    convolved_maps : dict
        All maps convolved to target_beam
    """
    from scipy.ndimage import gaussian_filter
    
    if warnings is None:
        warnings = []
    
    convolved = {}
    sigma_target = target_beam_arcsec / 2.355
    
    for name, image in maps.items():
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            convolved[name] = image
            continue
        
        native_beam = native_beams.get(name, 0)
        sigma_native = native_beam / 2.355
        
        if native_beam >= target_beam_arcsec:
            # Native beam already coarser than target — cannot sharpen
            convolved[name] = image
            if native_beam > target_beam_arcsec * 1.2:
                warnings.append(
                    f"{name}: native beam {native_beam:.1f}\" > model "
                    f"resolution {target_beam_arcsec:.1f}\" — observation "
                    f"is coarser than model, spatial info lost")
        else:
            # Convolve with the quadrature-difference kernel
            sigma_kernel = np.sqrt(sigma_target**2 - sigma_native**2)
            sigma_pix = sigma_kernel / pixel_size_arcsec
            
            if sigma_pix > 0.3:
                convolved[name] = gaussian_filter(image, sigma=sigma_pix)
            else:
                convolved[name] = image
        
    return convolved


# ============================================================
# High-level observation generator
# ============================================================

def generate_observations(solver, distance_pc=414, los_axis=2):
    """
    Generate a complete set of synthetic observations from a 3D model.
    
    Parameters
    ----------
    solver : PDRSolver3D
        Converged 3D PDR model
    distance_pc : float
        Distance to source [pc]. Default: 414 pc (Orion)
    los_axis : int
        Line-of-sight axis
    
    Returns
    -------
    obs : dict
        Dictionary containing all maps, with metadata
    """
    obs = {}
    
    # Pixel size in arcsec
    pix_cm = solver.cell_size
    pix_arcsec = (pix_cm / (distance_pc * pc_cm)) * 206265
    obs['pixel_size_arcsec'] = pix_arcsec
    obs['distance_pc'] = distance_pc
    
    # Column densities
    cols = column_density_maps(solver, los_axis)
    obs.update(cols)
    
    # Dust continuum at key wavelengths
    for band_name, band in {**JWST_BANDS, **HERSCHEL_BANDS, **ALMA_BANDS}.items():
        lam = band['lam_um']
        I_nu = dust_continuum_map(solver, lam, los_axis)
        
        # Convert to MJy/sr for FIR, μJy/arcsec² for MIR
        if lam > 50:
            obs[f'dust_{band_name}'] = I_nu * 1e23 * 1e-6  # MJy/sr
        else:
            obs[f'dust_{band_name}'] = I_nu * 1e23 * 1e6 * (pix_arcsec)**2  # μJy/pix
        
        # Beam-convolved version
        beam = band['beam']
        obs[f'dust_{band_name}_conv'] = convolve_beam(
            obs[f'dust_{band_name}'], beam, pix_arcsec)
    
    # Fine-structure lines (with beam convolution)
    for line_name in ['CII_158', 'OI_63', 'CI_609']:
        I_line = line_emission_map(solver, line_name, los_axis)
        obs[f'line_{line_name}'] = I_line
        beam = LINE_BEAMS.get(line_name, 1.0)
        obs[f'line_{line_name}_conv'] = convolve_beam(I_line, beam, pix_arcsec)
    
    # CO lines (J=1-0, 2-1, 3-2)
    for J in [1, 2, 3]:
        line_name = f'CO_{J}-{J-1}'
        I_line = line_emission_map(solver, line_name, los_axis)
        obs[f'line_{line_name}'] = I_line
        beam = LINE_BEAMS.get(line_name, 1.0)
        obs[f'line_{line_name}_conv'] = convolve_beam(I_line, beam, pix_arcsec)
    
    # Dust evolution maps (projected)
    obs['f_nano_proj'] = np.mean(solver.f_nano, axis=los_axis)
    obs['E_g_proj'] = np.mean(solver.E_g, axis=los_axis)
    obs['Gamma_PE_proj'] = np.sum(solver.Gamma_PE, axis=los_axis) * solver.cell_size
    
    # Resolution diagnostic: which observables are resolved by this model?
    cell_AU = pix_arcsec * distance_pc
    obs['_resolution'] = {
        'model_cell_arcsec': pix_arcsec,
        'model_cell_AU': cell_AU,
        'resolved': [],     # beam > 2 × cell
        'marginal': [],     # cell < beam < 2 × cell
        'underresolved': [],  # beam < cell
    }
    all_beams = {**{k: v['beam'] for k, v in 
                    {**JWST_BANDS, **HERSCHEL_BANDS, **ALMA_BANDS}.items()},
                  **LINE_BEAMS}
    for name, beam in sorted(all_beams.items()):
        ratio = beam / pix_arcsec
        if ratio >= 2.0:
            obs['_resolution']['resolved'].append(f"{name} ({beam}\")")
        elif ratio >= 0.5:
            obs['_resolution']['marginal'].append(f"{name} ({beam}\")")
        else:
            obs['_resolution']['underresolved'].append(f"{name} ({beam}\")")
    
    return obs


def plot_observations(obs, solver, savepath=None):
    """
    Generate a publication-quality multi-panel observation figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    pix = obs['pixel_size_arcsec']
    nx = obs['N_H'].shape[0]
    ext_arcsec = [0, nx * pix, 0, nx * pix]
    ext_pc = [0, solver.box_size / pc_cm, 0, solver.box_size / pc_cm]
    
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('PRISM-3D Synthetic Observations\n'
                 f'3D Turbulent Cloud with THEMIS Dust Evolution '
                 f'(G₀={solver.G0_external:.0f}, d={obs["distance_pc"]} pc)',
                 fontsize=15, fontweight='bold')
    
    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.4)
    
    def implot(ax, data, title, cmap='viridis', log=False, units='', ext=ext_pc):
        if log and np.any(data > 0):
            vmin = np.percentile(data[data > 0], 5) if np.any(data > 0) else 1e-30
            im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap,
                          norm=LogNorm(vmin=max(vmin, 1e-30)))
        else:
            im = ax.imshow(data.T, origin='lower', extent=ext, cmap=cmap)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        if units:
            cb.set_label(units, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('x [pc]', fontsize=8)
        ax.set_ylabel('y [pc]', fontsize=8)
        ax.tick_params(labelsize=7)
        return im
    
    # Row 1: Column densities
    ax = fig.add_subplot(gs[0, 0])
    implot(ax, obs['N_H'], r'$N_H$ [cm$^{-2}$]', 'bone_r', log=True, units='cm⁻²')
    
    ax = fig.add_subplot(gs[0, 1])
    implot(ax, obs['N_H2'], r'$N(H_2)$ [cm$^{-2}$]', 'YlGnBu', log=True, units='cm⁻²')
    
    ax = fig.add_subplot(gs[0, 2])
    implot(ax, obs['N_CO'], r'$N(CO)$ [cm$^{-2}$]', 'YlGn', log=True, units='cm⁻²')
    
    ax = fig.add_subplot(gs[0, 3])
    implot(ax, obs['N_Cp'], r'$N(C^+)$ [cm$^{-2}$]', 'YlOrRd', log=True, units='cm⁻²')
    
    ax = fig.add_subplot(gs[0, 4])
    implot(ax, obs['A_V_total'], r'$A_V$ [mag]', 'Greys', units='mag')
    
    # Row 2: Dust continuum
    dust_panels = [
        ('dust_F770W', 'JWST 7.7 μm (PAH)', 'inferno'),
        ('dust_F1130W', 'JWST 11.3 μm (PAH)', 'inferno'),
        ('dust_PACS70', 'Herschel 70 μm', 'hot'),
        ('dust_PACS160', 'Herschel 160 μm', 'hot'),
        ('dust_Band7', 'ALMA 850 μm', 'magma'),
    ]
    for i, (key, title, cmap) in enumerate(dust_panels):
        ax = fig.add_subplot(gs[1, i])
        if key in obs:
            implot(ax, obs[key], title, cmap, log=True)
    
    # Row 3: Line emission
    line_panels = [
        ('line_CII_158', '[CII] 158 μm', 'RdYlBu_r'),
        ('line_OI_63', '[OI] 63 μm', 'PuRd'),
        ('line_CI_609', '[CI] 609 μm', 'BuGn'),
        ('line_CO_1-0', 'CO(1-0)', 'YlGn'),
        ('line_CO_3-2', 'CO(3-2)', 'YlOrBr'),
    ]
    for i, (key, title, cmap) in enumerate(line_panels):
        ax = fig.add_subplot(gs[2, i])
        if key in obs:
            data = obs[key]
            if np.any(data > 0):
                implot(ax, data, title, cmap, log=True, units='erg/s/cm²/sr')
            else:
                implot(ax, data + 1e-30, title, cmap, units='erg/s/cm²/sr')
    
    # Row 4: Dust evolution + diagnostics
    dust_evo_panels = [
        ('f_nano_proj', 'Nano-grain fraction\n(THEMIS)', 'RdYlGn', False),
        ('E_g_proj', 'Band gap E_g [eV]\n(aromatic↔aliphatic)', 'coolwarm', False),
        ('Gamma_PE_proj', 'PE Heating\n(integrated)', 'hot', True),
    ]
    for i, (key, title, cmap, log) in enumerate(dust_evo_panels):
        ax = fig.add_subplot(gs[3, i])
        if key in obs:
            implot(ax, obs[key], title, cmap, log=log)
    
    # Add a text box with model parameters
    ax = fig.add_subplot(gs[3, 3:])
    ax.axis('off')
    info = (
        f"Model: {solver.nx}³ cells, {solver.box_size/pc_cm:.1f} pc\n"
        f"G₀ = {solver.G0_external:.0f}, ζ_CR = {solver.zeta_CR_0:.1e} s⁻¹\n"
        f"n_H = {np.min(solver.density):.0f}–{np.max(solver.density):.0f} cm⁻³\n"
        f"T_gas = {np.min(solver.T_gas):.0f}–{np.max(solver.T_gas):.0f} K\n"
        f"T_dust = {np.min(solver.T_dust):.0f}–{np.max(solver.T_dust):.0f} K\n"
        f"f_nano = {np.min(solver.f_nano):.3f}–{np.max(solver.f_nano):.3f}\n"
        f"THEMIS dust evolution: ON\n"
        f"d = {obs['distance_pc']} pc, pixel = {pix:.1f}″"
    )
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Model Parameters', fontsize=10, fontweight='bold')
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {savepath}")
    else:
        plt.show()
    
    return fig
