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


def embedded_star_cloud(n_cells, box_size, n_cloud=3000, n_cavity=10,
                        R_cavity_frac=0.15, n_shell_factor=10,
                        shell_width_frac=0.02, sigma_ln=1.2,
                        star_offset=(0.0, 0.0, -0.05),
                        blowout_axis=2, blowout_thin=0.4,
                        seed=42):
    """
    Molecular cloud with an embedded young O star carving out a cavity.

    Creates a turbulent cloud with:
    - A roughly spherical cavity around the star (cleared by stellar wind/HII)
    - A dense swept-up shell at the cavity wall
    - Asymmetry along one axis (the blowout direction) where the shell
      is thinner and the star is about to break through
    - Turbulent substructure in the ambient cloud with dense pillars
      protruding into the cavity

    Parameters
    ----------
    n_cells : int
        Cells per dimension.
    box_size : float
        Physical box size [cm].
    n_cloud : float
        Mean ambient cloud density [cm⁻³].
    n_cavity : float
        Cavity interior density [cm⁻³] (hot ionised gas).
    R_cavity_frac : float
        Cavity radius as fraction of box size.
    n_shell_factor : float
        Shell peak density as multiple of n_cloud.
    shell_width_frac : float
        Shell thickness as fraction of box size.
    sigma_ln : float
        Turbulence amplitude (log-normal width).
    star_offset : tuple (3,)
        Star offset from box centre, in fractions of box size.
    blowout_axis : int
        Axis along which the cavity is about to blow out (0, 1, or 2).
    blowout_thin : float
        Thinning factor for the shell on the blowout side (0–1).
        0 = no thinning, 1 = shell completely removed.
    seed : int
        Random seed.

    Returns
    -------
    density : ndarray (n_cells, n_cells, n_cells)
    cell_size : float
    star_pos_cm : ndarray (3,)
        Star position in physical coordinates [cm].
    """
    rng = np.random.RandomState(seed)
    cell_size = box_size / n_cells

    # Star position
    centre = 0.5 * box_size
    sx = centre + star_offset[0] * box_size
    sy = centre + star_offset[1] * box_size
    sz = centre + star_offset[2] * box_size
    star_pos = np.array([sx, sy, sz])

    # Cell coordinates
    x = (np.arange(n_cells) + 0.5) * cell_size
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Distance from star
    r = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
    R_cav = R_cavity_frac * box_size
    w_shell = shell_width_frac * box_size

    # ── Anisotropic cavity: elongated toward blowout direction ────
    # Increase effective cavity radius along blowout axis
    disp = [X - sx, Y - sy, Z - sz]
    blowout_disp = disp[blowout_axis]
    # Blowout side: positive displacement along blowout axis
    blowout_frac = np.clip(blowout_disp / (R_cav + 1e-10), 0, 2)
    # Effective cavity radius: larger on blowout side
    R_eff = R_cav * (1.0 + 0.3 * blowout_frac)

    # ── Turbulent ambient cloud ──────────────────────────────────
    kx = np.fft.fftfreq(n_cells, d=cell_size)
    ky = np.fft.fftfreq(n_cells, d=cell_size)
    kz = np.fft.fftfreq(n_cells, d=cell_size)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0

    amplitude = K**(-3.7 / 2.0)
    amplitude[0, 0, 0] = 0.0
    phases = rng.uniform(0, 2 * np.pi, (n_cells, n_cells, n_cells))
    gaussian = np.real(np.fft.ifftn(amplitude * np.exp(1j * phases)))
    gaussian -= np.mean(gaussian)
    if np.std(gaussian) > 0:
        gaussian *= sigma_ln / np.std(gaussian)

    cloud = n_cloud * np.exp(gaussian - sigma_ln**2 / 2)

    # ── Swept-up shell ───────────────────────────────────────────
    # Gaussian shell profile at r = R_eff
    shell_profile = np.exp(-0.5 * ((r - R_eff) / w_shell)**2)

    # Thin the shell on the blowout side
    thin_factor = 1.0 - blowout_thin * np.clip(blowout_frac, 0, 1)
    shell = n_cloud * n_shell_factor * shell_profile * thin_factor

    # ── Assemble: cavity interior + shell + cloud exterior ───────
    # Smooth cavity profile (tanh transition at R_eff)
    cavity_mask = 0.5 * (1.0 + np.tanh((r - R_eff) / (0.5 * w_shell)))
    # 0 inside cavity, 1 outside
    density = n_cavity + cavity_mask * cloud + shell

    # ── Pillars / fingers of dense gas protruding into cavity ────
    # A few elongated structures pointing radially inward
    n_pillars = 4
    for _ in range(n_pillars):
        # Random direction from star
        theta = np.arccos(2 * rng.rand() - 1)
        phi = 2 * np.pi * rng.rand()
        px = np.sin(theta) * np.cos(phi)
        py = np.sin(theta) * np.sin(phi)
        pz = np.cos(theta)

        # Pillar spine: line from R_cav inward by ~0.5 R_cav
        pillar_len = (0.3 + 0.4 * rng.rand()) * R_cav
        pillar_base_r = R_cav
        pillar_width = (0.02 + 0.02 * rng.rand()) * box_size

        # Distance from pillar spine for each cell
        # Spine runs from r=pillar_base_r to r=pillar_base_r - pillar_len
        # along direction (px, py, pz)
        base = star_pos + pillar_base_r * np.array([px, py, pz])
        tip = star_pos + (pillar_base_r - pillar_len) * np.array([px, py, pz])
        # Project each cell onto the spine
        spine_vec = tip - base
        spine_len = np.linalg.norm(spine_vec)
        spine_hat = spine_vec / spine_len
        to_cell = np.stack([X - base[0], Y - base[1], Z - base[2]], axis=-1)
        t = np.sum(to_cell * spine_hat, axis=-1)
        t = np.clip(t, 0, spine_len)
        closest = base + t[..., np.newaxis] * spine_hat
        dist_to_spine = np.sqrt((X - closest[..., 0])**2 +
                                (Y - closest[..., 1])**2 +
                                (Z - closest[..., 2])**2)
        pillar = 5 * n_cloud * np.exp(-0.5 * (dist_to_spine / pillar_width)**2)
        # Fade pillar beyond base (into cloud, less visible)
        pillar *= np.clip(1.0 - t / spine_len, 0, 1)
        density = np.maximum(density, pillar)

    density = np.maximum(density, 1.0)  # Floor

    return density, cell_size, star_pos


def turbulent_velocity_field(n_cells, box_size, density, sigma_ln=1.5,
                              spectral_index=-3.7, zeta=0.0,
                              T_mean=50.0, mu=2.35, seed=42):
    """
    Generate a 3-component turbulent velocity field correlated with density.

    Uses the same Fourier-space power spectrum approach as the density field,
    with Helmholtz decomposition to set the solenoidal/compressive ratio and
    density-velocity anti-correlation (dense clumps are slow).

    Parameters
    ----------
    n_cells : int
        Grid resolution per dimension.
    box_size : float
        Physical box size [cm].
    density : ndarray (n_cells, n_cells, n_cells)
        Density field [cm⁻³] for density-velocity correlation.
    sigma_ln : float
        Log-normal width of density PDF (sets Mach number).
    spectral_index : float
        Power spectrum slope (default -3.7 for molecular clouds).
    zeta : float
        Compressive fraction (0 = purely solenoidal, 1 = purely compressive).
        Molecular clouds are solenoidal-dominated (zeta ~ 0).
    T_mean : float
        Mean gas temperature [K] for sound speed calculation.
    mu : float
        Mean molecular weight [amu].
    seed : int
        Random seed.

    Returns
    -------
    velocity : ndarray (3, n_cells, n_cells, n_cells)
        Velocity field [cm/s].
    """
    from .utils.constants import k_boltz, m_H

    rng = np.random.RandomState(seed + 1000)  # Offset from density seed
    cell_size = box_size / n_cells

    # Sound speed
    c_s = np.sqrt(k_boltz * T_mean / (mu * m_H))  # cm/s

    # Mach number from sigma_ln: sigma_ln² = ln(1 + b²M²), b ~ 0.5
    b = 0.5
    M2 = (np.exp(sigma_ln**2) - 1) / b**2
    Mach = np.sqrt(M2)
    v_rms = Mach * c_s

    # k-space grid
    kx = np.fft.fftfreq(n_cells, d=cell_size)
    ky = np.fft.fftfreq(n_cells, d=cell_size)
    kz = np.fft.fftfreq(n_cells, d=cell_size)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0

    # Power spectrum amplitude
    amplitude = K**(spectral_index / 2.0)
    amplitude[0, 0, 0] = 0.0

    # Generate 3 independent Fourier velocity components
    vel = np.zeros((3, n_cells, n_cells, n_cells))
    for comp in range(3):
        phases = rng.uniform(0, 2 * np.pi, (n_cells, n_cells, n_cells))
        field_k = amplitude * np.exp(1j * phases)
        vel[comp] = np.real(np.fft.ifftn(field_k))

    # Helmholtz decomposition in k-space: split into solenoidal + compressive
    # Compressive: v_c = k̂(k̂·v)   Solenoidal: v_s = v - v_c
    vel_k = np.fft.fftn(vel, axes=(1, 2, 3))
    K_safe = np.where(K > 0, K, 1.0)
    khat = np.array([KX, KY, KZ]) / K_safe[np.newaxis, :, :, :]

    # k̂·v in Fourier space
    kdotv = np.sum(khat * vel_k, axis=0)

    # Compressive component
    vel_comp = khat * kdotv[np.newaxis, :, :, :]
    # Solenoidal component
    vel_sol = vel_k - vel_comp

    # Mix: v = sqrt(1-zeta)*v_sol + sqrt(zeta)*v_comp
    vel_k_mixed = np.sqrt(1 - zeta) * vel_sol + np.sqrt(zeta) * vel_comp
    vel_k_mixed[:, 0, 0, 0] = 0  # No bulk motion

    vel = np.real(np.fft.ifftn(vel_k_mixed, axes=(1, 2, 3)))

    # Normalize to target v_rms
    current_rms = np.sqrt(np.mean(vel**2))
    if current_rms > 0:
        vel *= v_rms / current_rms

    # Density-velocity anti-correlation: dense clumps are slow
    n_mean = np.mean(density)
    weight = np.exp(-0.5 * np.log(density / n_mean))
    for comp in range(3):
        vel[comp] *= weight

    # Re-normalize after weighting
    current_rms = np.sqrt(np.mean(vel**2))
    if current_rms > 0:
        vel *= v_rms / current_rms

    print(f"  Velocity field: Mach={Mach:.1f}, v_rms={v_rms/1e5:.2f} km/s, "
          f"c_s={c_s/1e5:.2f} km/s, zeta={zeta:.1f}")

    return vel


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
