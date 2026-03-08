"""
PRISM-3D Spectral Line Cube Generator.

Produces position-position-velocity (PPV) cubes and extracted spectra
from 3D PDR models. Each cell contributes emission at a velocity
determined by:
  - Thermal broadening: Δv_th = sqrt(2 k T / m)
  - Turbulent broadening: Δv_turb (from density field or user input)
  - Bulk velocity: v_los (from velocity field or zero for static models)

Output:
  - PPV cubes in (x, y, v) for any emission line
  - Template spectra integrated over user-defined apertures
  - Spectral line profiles (intensity vs velocity)
  - Multi-line spectra (stacking lines across wavelength)

Supports: [CII] 158μm, [OI] 63/145μm, [CI] 609/370μm,
          CO J=1-0 through J=10-9, H₂ rovibrational lines
"""

import numpy as np
from ..utils.constants import k_boltz, h_planck, c_light, m_H
from .jwst_pipeline import LINES, CO_LINES, LINE_BEAMS


# Molecular/atomic masses [amu]
SPECIES_MASS = {
    'C+': 12, 'O': 16, 'C': 12, 'CO': 28,
    'H2': 2, 'H': 1, 'HCO+': 29,
}

# Rest frequencies [Hz]
LINE_FREQ = {}
for name, info in {**LINES, **CO_LINES}.items():
    LINE_FREQ[name] = c_light / (info['lam_um'] * 1e-4)


def compute_ppv_cube(solver, line_name, los_axis=2,
                      n_vel=64, vel_range_kms=10.0,
                      v_turb_kms=1.0, velocity_field=None):
    """
    Compute a position-position-velocity (PPV) cube for an emission line.
    
    Parameters
    ----------
    solver : PDRSolver3D
        Converged 3D model
    line_name : str
        Line identifier (e.g. 'CII_158', 'CO_1-0')
    los_axis : int
        Line-of-sight axis (0, 1, or 2)
    n_vel : int
        Number of velocity channels
    vel_range_kms : float
        Half-width of velocity axis [km/s]
    v_turb_kms : float
        Microturbulent velocity [km/s] (added in quadrature to thermal)
    velocity_field : ndarray (nx, ny, nz), optional
        Bulk LOS velocity field [cm/s]. If None, all cells at v=0.
    
    Returns
    -------
    ppv : dict with keys:
        'cube': ndarray (n1, n2, n_vel) — PPV data cube [erg/s/cm²/sr/(km/s)]
        'vel_axis': ndarray (n_vel,) — velocity channels [km/s]
        'freq_axis': ndarray (n_vel,) — frequency channels [Hz]
        'line_name': str
        'rest_freq': float [Hz]
        'integrated_map': ndarray (n1, n2) — velocity-integrated intensity
    """
    # Get line properties
    if line_name in LINES:
        line = LINES[line_name]
    elif line_name in CO_LINES:
        line = CO_LINES[line_name]
    else:
        raise ValueError(f"Unknown line: {line_name}. "
                         f"Available: {list(LINES.keys()) + list(CO_LINES.keys())}")
    
    E_up = line['E_up_K']
    A_ul = line['A_ul']
    g_u = line['g_u']
    lam_cm = line['lam_um'] * 1e-4
    nu_0 = c_light / lam_cm
    species = line['species']
    m_species = SPECIES_MASS.get(species, 12) * m_H  # grams
    
    # Velocity grid
    vel_axis = np.linspace(-vel_range_kms, vel_range_kms, n_vel)  # km/s
    dv = vel_axis[1] - vel_axis[0]  # km/s
    
    # Frequency axis: ν = ν₀ (1 - v/c)
    freq_axis = nu_0 * (1 - vel_axis * 1e5 / c_light)
    
    # Get abundance of emitting species
    species_map = {
        'C+': solver.x_Cp, 'O': solver.x_O, 'C': solver.x_C,
        'CO': solver.x_CO, 'H2': solver.x_H2, 'H': solver.x_HI,
    }
    x_species = species_map.get(species, np.zeros_like(solver.density))
    n_species = solver.density * x_species
    
    # Upper level population (LTE)
    T = solver.T_gas
    
    if species in ('C+', 'O', 'C'):
        g_total = {'C+': 6, 'O': 9, 'C': 9}[species]
        f_up = (g_u / g_total) * np.exp(-E_up / np.maximum(T, 5.0))
    elif species == 'CO':
        from .jwst_pipeline import B_CO
        Z_rot = k_boltz * T / (h_planck * B_CO)
        Z_rot = np.maximum(Z_rot, 1.0)
        f_up = g_u * np.exp(-E_up / np.maximum(T, 5.0)) / Z_rot
    else:
        f_up = np.exp(-E_up / np.maximum(T, 5.0))
    
    n_u = n_species * f_up  # Upper level density [cm⁻³]
    
    # Lower level population (for absorption / optical depth)
    # For two-level: n_l = n_species - n_u (approximately)
    # More precisely: n_l = n_species * (g_l/g_total) * exp(-E_low/T) / Z
    # For ground-state lines: E_low ~ 0, so n_l ~ n_species * g_l/g_total
    if species in ('C+', 'O', 'C'):
        g_total = {'C+': 6, 'O': 9, 'C': 9}[species]
        g_l = g_total - g_u  # Remaining statistical weight
        n_l = n_species * (g_l / g_total)
    elif species == 'CO':
        J_up = line.get('J_up', 1)
        g_l = 2 * (J_up - 1) + 1 if J_up > 0 else 1
        from .jwst_pipeline import B_CO
        Z_rot = k_boltz * T / (h_planck * B_CO)
        Z_rot = np.maximum(Z_rot, 1.0)
        E_low = h_planck * B_CO * (J_up - 1) * J_up / k_boltz if J_up > 0 else 0
        n_l = n_species * g_l * np.exp(-E_low / np.maximum(T, 5.0)) / Z_rot
    else:
        n_l = n_species * 0.5
    
    # Emissivity: j = n_u × A_ul × hν / (4π)  [erg/s/cm³/sr]
    j_line = n_u * A_ul * h_planck * nu_0 / (4 * np.pi)
    
    # Absorption coefficient: κ_ν = (hν/4π) × (n_l B_lu - n_u B_ul) × φ(ν)
    # Using Einstein relations: B_ul = A_ul c² / (2hν³), B_lu = (g_u/g_l) B_ul
    # κ_line = (hν/4π) × A_ul × c² / (2hν³) × (n_l g_u/g_l - n_u)
    #        = (A_ul c² / (8π ν²)) × (n_l g_u/g_l - n_u)
    # This is the frequency-integrated absorption; the per-channel κ includes φ(v)
    kappa_0 = A_ul * c_light**2 / (8 * np.pi * nu_0**2)  # cm² Hz (Einstein coeff part)
    if species in ('C+', 'O', 'C') or species == 'CO':
        g_ratio = g_u / max(g_l, 1)
    else:
        g_ratio = 1.0
    # Net absorption (stimulated emission subtracted)
    kappa_line = kappa_0 * np.maximum(n_l * g_ratio - n_u, 0)  # cm⁻¹ Hz
    
    # Line width per cell: thermal + turbulent (quadrature)
    # σ_v = sqrt(kT/m + v_turb²) — THIS IS SPECIES-DEPENDENT
    sigma_th = np.sqrt(k_boltz * T / m_species)  # cm/s, thermal only
    sigma_turb = v_turb_kms * 1e5  # cm/s
    sigma_v = np.sqrt(sigma_th**2 + sigma_turb**2)  # total
    sigma_v_kms = sigma_v / 1e5  # km/s
    
    # Build PPV cube with proper radiative transfer
    nx, ny, nz = solver.density.shape
    if los_axis == 0:
        n1, n2, n_los = ny, nz, nx
    elif los_axis == 1:
        n1, n2, n_los = nx, nz, ny
    else:
        n1, n2, n_los = nx, ny, nz
    
    ppv_cube = np.zeros((n1, n2, n_vel))
    tau_cube = np.zeros((n1, n2, n_vel))  # Optical depth cube
    ds = solver.cell_size
    
    # LOS radiative transfer: I_ν(s+ds) = I_ν(s) × exp(-dτ) + S_ν × (1 - exp(-dτ))
    # where S_ν = j_ν / κ_ν (source function) and dτ = κ_ν × φ(v) × ds
    
    for i_los in range(n_los):
        if los_axis == 0:
            j_slab = j_line[i_los, :, :]
            k_slab = kappa_line[i_los, :, :]
            sigma_slab = sigma_v_kms[i_los, :, :]
            v_slab = velocity_field[i_los, :, :] / 1e5 if velocity_field is not None else 0
        elif los_axis == 1:
            j_slab = j_line[:, i_los, :]
            k_slab = kappa_line[:, i_los, :]
            sigma_slab = sigma_v_kms[:, i_los, :]
            v_slab = velocity_field[:, i_los, :] / 1e5 if velocity_field is not None else 0
        else:
            j_slab = j_line[:, :, i_los]
            k_slab = kappa_line[:, :, i_los]
            sigma_slab = sigma_v_kms[:, :, i_los]
            v_slab = velocity_field[:, :, i_los] / 1e5 if velocity_field is not None else 0
        
        for iv, v in enumerate(vel_axis):
            dv_shifted = v - v_slab
            # Line profile φ(v) normalized so ∫φ dv = 1 (in km/s)
            phi = np.exp(-0.5 * (dv_shifted / np.maximum(sigma_slab, 0.01))**2)
            phi /= (np.maximum(sigma_slab, 0.01) * np.sqrt(2 * np.pi))
            
            # Optical depth increment: dτ = κ_0 × φ(v) × ds
            # κ_0 is in cm⁻¹ Hz; φ is in (km/s)⁻¹; convert: φ_Hz = φ_v × c/ν₀
            phi_Hz = phi / (1e5) * c_light / nu_0  # Convert (km/s)⁻¹ → Hz⁻¹
            dtau = k_slab * phi_Hz * ds
            dtau = np.minimum(dtau, 50)  # Prevent overflow
            
            # Source function S = j/κ (= Planck function in LTE)
            S = np.where(k_slab > 0, j_slab / (k_slab * phi_Hz + 1e-50), 0)
            
            # Formal solution in velocity space:
            #   dI_v/ds = j*φ_v - κ_ν * I_v
            # where κ_ν = kappa_line * phi_Hz [cm⁻¹] (absorption at this velocity)
            # and j*φ_v [erg/s/cm³/sr/(km/s)] is emissivity per velocity channel
            #
            # Source function: S_v = j*φ_v / κ_ν = j*φ_v / (kappa_line * phi_Hz)
            # dtau = κ_ν * ds = kappa_line * phi_Hz * ds
            #
            # I_v(s+ds) = I_v(s)*exp(-dτ) + S_v*(1-exp(-dτ))
            # For dτ→0: I_v += j*φ_v*ds   (thin limit)
            # For dτ→∞: I_v → S_v          (thick/saturated)
            
            exp_tau = np.exp(-dtau)
            
            # Emissivity per velocity channel
            j_v = j_slab * phi  # [erg/s/cm³/sr/(km/s)]
            
            # Source function in velocity units
            kappa_nu = k_slab * phi_Hz  # [cm⁻¹]
            S_v = np.where(kappa_nu > 1e-50,
                           j_v / kappa_nu,  # [erg/s/cm²/sr/(km/s)]
                           0)
            
            # Formal solution
            ppv_cube[:, :, iv] = ppv_cube[:, :, iv] * exp_tau + S_v * (1 - exp_tau)
            tau_cube[:, :, iv] += dtau
    
    # Velocity-integrated intensity (moment 0)
    integrated_map = np.sum(ppv_cube, axis=2) * dv  # erg/s/cm²/sr
    
    return {
        'cube': ppv_cube,
        'tau_cube': tau_cube,
        'vel_axis': vel_axis,
        'freq_axis': freq_axis,
        'dv_kms': dv,
        'line_name': line_name,
        'rest_freq_Hz': nu_0,
        'wavelength_um': line['lam_um'],
        'integrated_map': integrated_map,
        'species': species,
        'peak_tau': float(np.max(tau_cube)),
        'mean_tau_center': float(np.mean(tau_cube[:, :, n_vel//2])),
    }


def extract_spectrum(ppv, x_pix=None, y_pix=None, aperture_pix=None):
    """
    Extract a spectrum from a PPV cube.
    
    Parameters
    ----------
    ppv : dict
        Output from compute_ppv_cube()
    x_pix, y_pix : int, optional
        Center pixel. Default: center of map.
    aperture_pix : int, optional
        Aperture radius in pixels. Default: single pixel.
    
    Returns
    -------
    spectrum : dict
        'velocity': ndarray [km/s]
        'intensity': ndarray [erg/s/cm²/sr/(km/s)]
        'peak': float — peak intensity
        'fwhm': float — FWHM [km/s]
        'integrated': float — integrated intensity [erg/s/cm²/sr]
    """
    cube = ppv['cube']
    n1, n2, nv = cube.shape
    
    if x_pix is None:
        x_pix = n1 // 2
    if y_pix is None:
        y_pix = n2 // 2
    
    if aperture_pix is None or aperture_pix <= 0:
        spec = cube[x_pix, y_pix, :]
    else:
        # Sum over circular aperture
        yy, xx = np.mgrid[:n1, :n2]
        mask = ((xx - x_pix)**2 + (yy - y_pix)**2) <= aperture_pix**2
        spec = np.mean(cube[mask], axis=0)
    
    vel = ppv['vel_axis']
    dv = ppv['dv_kms']
    
    # Measure line properties
    peak = np.max(spec)
    integrated = np.sum(spec) * dv
    
    # FWHM
    if peak > 0:
        half_max = peak / 2
        above = spec >= half_max
        if np.any(above):
            indices = np.where(above)[0]
            fwhm = (indices[-1] - indices[0]) * dv
        else:
            fwhm = 0
    else:
        fwhm = 0
    
    # Centroid velocity
    if integrated > 0:
        v_centroid = np.sum(vel * spec) / np.sum(spec)
    else:
        v_centroid = 0
    
    return {
        'velocity': vel,
        'intensity': spec,
        'peak': peak,
        'fwhm_kms': fwhm,
        'integrated': integrated,
        'v_centroid_kms': v_centroid,
        'line_name': ppv['line_name'],
        'wavelength_um': ppv['wavelength_um'],
    }


def multi_line_spectrum(solver, line_names=None, los_axis=2,
                         x_pix=None, y_pix=None, aperture_pix=None,
                         v_turb_kms=1.0):
    """
    Generate spectra for multiple lines from the same aperture.
    
    Parameters
    ----------
    solver : PDRSolver3D
    line_names : list of str, optional
        Lines to compute. Default: standard PDR diagnostic set.
    
    Returns
    -------
    spectra : dict of dict
        {line_name: spectrum_dict} for each line
    """
    if line_names is None:
        line_names = [
            'CII_158', 'OI_63', 'CI_609',
            'CO_1-0', 'CO_2-1', 'CO_3-2',
        ]
    
    spectra = {}
    for line_name in line_names:
        try:
            ppv = compute_ppv_cube(solver, line_name, los_axis=los_axis,
                                    v_turb_kms=v_turb_kms)
            spec = extract_spectrum(ppv, x_pix=x_pix, y_pix=y_pix,
                                    aperture_pix=aperture_pix)
            spectra[line_name] = spec
        except (ValueError, KeyError) as e:
            pass
    
    return spectra


def plot_spectra(spectra, savepath=None):
    """
    Plot multi-line spectra in a publication-quality figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_lines = len(spectra)
    if n_lines == 0:
        return
    
    n_cols = min(3, n_lines)
    n_rows = (n_lines + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_lines == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    
    fig.suptitle('PRISM-3D: Predicted Line Spectra', fontsize=14, fontweight='bold')
    
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4',
              '#795548', '#607D8B', '#E91E63', '#3F51B5']
    
    for i, (name, spec) in enumerate(spectra.items()):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        color = colors[i % len(colors)]
        ax.plot(spec['velocity'], spec['intensity'], color=color, linewidth=1.5)
        ax.fill_between(spec['velocity'], 0, spec['intensity'],
                        alpha=0.15, color=color)
        
        ax.set_xlabel('Velocity [km/s]', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        
        lam = spec['wavelength_um']
        ax.set_title(f"{name}  ({lam:.1f} μm)", fontsize=11, fontweight='bold')
        
        # Annotate line properties
        props = (f"Peak: {spec['peak']:.2e}\n"
                 f"FWHM: {spec['fwhm_kms']:.2f} km/s\n"
                 f"∫I dv: {spec['integrated']:.2e}")
        ax.text(0.97, 0.95, props, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.2)
    
    # Hide empty panels
    for i in range(n_lines, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_ppv_cube(ppv, savepath=None):
    """
    Plot a PPV cube: moment 0, moment 1, channel maps, and spectrum.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    cube = ppv['cube']
    vel = ppv['vel_axis']
    dv = ppv['dv_kms']
    
    # Moments
    mom0 = np.sum(cube, axis=2) * dv  # Integrated intensity
    
    with np.errstate(invalid='ignore'):
        vel_3d = vel[np.newaxis, np.newaxis, :]
        mom1 = np.sum(cube * vel_3d, axis=2) / np.maximum(np.sum(cube, axis=2), 1e-30)
    
    # Central spectrum
    cx, cy = cube.shape[0] // 2, cube.shape[1] // 2
    central_spec = cube[cx, cy, :]
    
    # Channel maps (pick 6 evenly spaced channels)
    n_chan = min(6, len(vel))
    chan_idx = np.linspace(0, len(vel)-1, n_chan, dtype=int)
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"PRISM-3D: {ppv['line_name']} PPV Cube "
                 f"({ppv['wavelength_um']:.1f} μm)",
                 fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(3, n_chan, hspace=0.35, wspace=0.3)
    
    # Row 1: moment 0 + moment 1 + spectrum
    ax0 = fig.add_subplot(gs[0, :2])
    im = ax0.imshow(mom0.T, origin='lower', cmap='magma')
    plt.colorbar(im, ax=ax0, label='Integrated [erg/s/cm²/sr]', shrink=0.8)
    ax0.set_title('Moment 0 (integrated intensity)')
    
    ax1 = fig.add_subplot(gs[0, 2:4])
    im = ax1.imshow(mom1.T, origin='lower', cmap='RdBu_r',
                    vmin=-5, vmax=5)
    plt.colorbar(im, ax=ax1, label='v_centroid [km/s]', shrink=0.8)
    ax1.set_title('Moment 1 (velocity field)')
    
    ax_sp = fig.add_subplot(gs[0, 4:])
    ax_sp.plot(vel, central_spec, 'k-', linewidth=1.5)
    ax_sp.fill_between(vel, 0, central_spec, alpha=0.2)
    ax_sp.set_xlabel('Velocity [km/s]')
    ax_sp.set_ylabel('Intensity')
    ax_sp.set_title('Central pixel spectrum')
    ax_sp.grid(True, alpha=0.2)
    
    # Row 2-3: channel maps
    for i, ci in enumerate(chan_idx):
        row = 1 + i // n_chan
        col = i % n_chan
        ax = fig.add_subplot(gs[1 + i // n_chan, i % n_chan])
        
        vmin_ch = np.percentile(cube[:, :, ci][cube[:, :, ci] > 0], 5) if np.any(cube[:, :, ci] > 0) else 0
        ax.imshow(cube[:, :, ci].T, origin='lower', cmap='inferno')
        ax.set_title(f'v = {vel[ci]:.1f} km/s', fontsize=9)
        ax.tick_params(labelsize=7)
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig
