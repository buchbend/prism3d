"""
Vectorized 3D FUV radiative transfer on regular grids.

This is the performance-critical RT module for 3D PDR calculations.
Operates on numpy arrays (not the octree) for maximum speed.

For each cell, traces rays in N_ray directions (HEALPix) to the
grid boundary, accumulating dust optical depth and column densities
of H₂ and CO for self-shielding.

Key optimization: rays along each direction are independent →
embarrassingly parallel. On a regular grid, ray marching uses
simple DDA (digital differential analyzer) stepping.
"""

import numpy as np
from ..radiative_transfer.fuv_rt import healpix_directions
from ..utils.constants import AV_per_NH, sigma_dust_FUV


def directional_G0(G0, source_direction, nside=1, background=0.0):
    """
    Create a per-ray G0 array for directional illumination.

    Rays whose direction points back toward the source (i.e. the ray
    enters from the source side) receive the full G0.  The weighting
    follows a cosine fall-off so that the total integrated flux on a
    surface normal to the source equals G0.

    Parameters
    ----------
    G0 : float
        FUV field strength from the source [Habing].
    source_direction : array-like, shape (3,)
        Unit vector pointing FROM the cloud TOWARD the source.
        Example: source at -x → source_direction = (-1, 0, 0).
    nside : int
        HEALPix resolution (must match the RT call).
    background : float
        Isotropic background FUV field added to all rays [Habing].

    Returns
    -------
    G0_rays : ndarray, shape (n_rays,)
        Per-ray external G0 values.
    """
    directions, weights = healpix_directions(nside)
    n_rays = len(directions)
    src = np.asarray(source_direction, dtype=np.float64)
    src /= np.linalg.norm(src)

    # Each ray direction d points toward where that ray *comes from*.
    # A ray illuminates a cell if d is aligned with src (both point
    # toward the source).  cos_theta = d · src > 0 means illuminated.
    cos_theta = directions @ src  # (n_rays,)

    # The RT loop accumulates:  G0_total = Σ G0_rays[i] * w[i] / (4π)
    # For a surface perpendicular to the source at AV=0 we want
    # G0_total = G0.  With a cos-weighted angular distribution:
    #   G0_rays[i] = G0 * cos_theta[i] * 4π / Σ(cos[j>0] * w[j])
    illuminated = cos_theta > 0
    ws = np.sum(np.where(illuminated, cos_theta * weights, 0.0))
    if ws > 0:
        G0_rays = np.where(illuminated,
                           G0 * cos_theta * (4.0 * np.pi) / ws, 0.0)
    else:
        # Fallback: isotropic
        G0_rays = np.full(n_rays, G0)

    G0_rays += background
    return G0_rays


def compute_fuv_field_point_source(density, star_pos_cm, L_FUV, cell_size,
                                    x_H2=None, x_CO=None):
    """
    Compute the FUV field from an embedded point source (e.g. O star).

    For each cell, traces a ray from the star position to the cell,
    accumulating dust optical depth.  The local G₀ is then:

        G₀(r) = L_FUV / (4π r² G₀→flux) × exp(−1.8 A_V)

    Uses a 6-ray axis-projected approximation: cells are grouped by
    dominant axis direction from the star, then cumulative sums give
    the column density from star to each cell.

    Parameters
    ----------
    density : ndarray (nx, ny, nz)
        Hydrogen number density [cm⁻³].
    star_pos_cm : array-like (3,)
        Star position in physical coordinates [cm].
    L_FUV : float
        FUV luminosity of the star [erg/s] (integrated 6–13.6 eV).
    cell_size : float
        Cell edge length [cm].
    x_H2 : ndarray (nx, ny, nz), optional
    x_CO : ndarray (nx, ny, nz), optional

    Returns
    -------
    G0, A_V, N_H, N_H2, N_CO : ndarrays (nx, ny, nz)
    """
    nx, ny, nz = density.shape
    if x_H2 is None:
        x_H2 = np.zeros_like(density)
    if x_CO is None:
        x_CO = np.zeros_like(density)

    sx, sy, sz = np.asarray(star_pos_cm, dtype=np.float64)

    # Cell centre coordinates
    cx = (np.arange(nx) + 0.5) * cell_size
    cy = (np.arange(ny) + 0.5) * cell_size
    cz = (np.arange(nz) + 0.5) * cell_size
    CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing='ij')

    # Distance from star (avoid div-by-0 at star cell)
    dx = CX - sx
    dy = CY - sy
    dz = CZ - sz
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.5 * cell_size)

    # Star grid index (nearest cell)
    is_ = int(np.clip(sx / cell_size, 0, nx - 1))
    js_ = int(np.clip(sy / cell_size, 0, ny - 1))
    ks_ = int(np.clip(sz / cell_size, 0, nz - 1))

    # Column density layers
    nH_layer = density * cell_size
    nH2_layer = density * x_H2 * 2.0 * cell_size
    nCO_layer = density * x_CO * cell_size

    # Accumulate column density from star outward along each axis direction.
    # For each of 6 half-spaces (±x, ±y, ±z from star), cumsum from star.
    NH = np.zeros_like(density)
    NH2 = np.zeros_like(density)
    NCO = np.zeros_like(density)

    # +x direction (cells with i > is_)
    if is_ < nx - 1:
        NH[is_ + 1:, :, :] = np.cumsum(nH_layer[is_:nx - 1, :, :], axis=0)
        NH2[is_ + 1:, :, :] = np.cumsum(nH2_layer[is_:nx - 1, :, :], axis=0)
        NCO[is_ + 1:, :, :] = np.cumsum(nCO_layer[is_:nx - 1, :, :], axis=0)
    # -x direction (cells with i < is_)
    if is_ > 0:
        NH[:is_, :, :] = np.cumsum(nH_layer[is_:0:-1, :, :], axis=0)[::-1]
        NH2[:is_, :, :] = np.cumsum(nH2_layer[is_:0:-1, :, :], axis=0)[::-1]
        NCO[:is_, :, :] = np.cumsum(nCO_layer[is_:0:-1, :, :], axis=0)[::-1]

    # For y and z axes, we ADD the y/z contribution (multi-axis path)
    NH_y = np.zeros_like(density)
    NH2_y = np.zeros_like(density)
    NCO_y = np.zeros_like(density)
    if js_ < ny - 1:
        NH_y[:, js_ + 1:, :] = np.cumsum(nH_layer[:, js_:ny - 1, :], axis=1)
        NH2_y[:, js_ + 1:, :] = np.cumsum(nH2_layer[:, js_:ny - 1, :], axis=1)
        NCO_y[:, js_ + 1:, :] = np.cumsum(nCO_layer[:, js_:ny - 1, :], axis=1)
    if js_ > 0:
        NH_y[:, :js_, :] = np.cumsum(nH_layer[:, js_:0:-1, :], axis=1)[:, ::-1, :]
        NH2_y[:, :js_, :] = np.cumsum(nH2_layer[:, js_:0:-1, :], axis=1)[:, ::-1, :]
        NCO_y[:, :js_, :] = np.cumsum(nCO_layer[:, js_:0:-1, :], axis=1)[:, ::-1, :]

    NH_z = np.zeros_like(density)
    NH2_z = np.zeros_like(density)
    NCO_z = np.zeros_like(density)
    if ks_ < nz - 1:
        NH_z[:, :, ks_ + 1:] = np.cumsum(nH_layer[:, :, ks_:nz - 1], axis=2)
        NH2_z[:, :, ks_ + 1:] = np.cumsum(nH2_layer[:, :, ks_:nz - 1], axis=2)
        NCO_z[:, :, ks_ + 1:] = np.cumsum(nCO_layer[:, :, ks_:nz - 1], axis=2)
    if ks_ > 0:
        NH_z[:, :, :ks_] = np.cumsum(nH_layer[:, :, ks_:0:-1], axis=2)[:, :, ::-1]
        NH2_z[:, :, :ks_] = np.cumsum(nH2_layer[:, :, ks_:0:-1], axis=2)[:, :, ::-1]
        NCO_z[:, :, :ks_] = np.cumsum(nCO_layer[:, :, ks_:0:-1], axis=2)[:, :, ::-1]

    # Pick the dominant-axis column for each cell.
    # The dominant axis is the one with the largest |displacement| from star.
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    abs_dz = np.abs(dz)
    dom_x = (abs_dx >= abs_dy) & (abs_dx >= abs_dz)
    dom_y = (abs_dy > abs_dx) & (abs_dy >= abs_dz)
    dom_z = ~dom_x & ~dom_y

    # Path length correction: actual radial distance vs axis-projected distance
    # factor = r / (dominant_displacement) — accounts for oblique paths
    path_corr = np.ones_like(r)
    path_corr[dom_x] = r[dom_x] / np.maximum(abs_dx[dom_x], 0.5 * cell_size)
    path_corr[dom_y] = r[dom_y] / np.maximum(abs_dy[dom_y], 0.5 * cell_size)
    path_corr[dom_z] = r[dom_z] / np.maximum(abs_dz[dom_z], 0.5 * cell_size)
    path_corr = np.minimum(path_corr, 3.0)  # Cap for cells near axes

    # Select column density from the dominant axis, apply path correction
    NH_tot = np.where(dom_x, NH, np.where(dom_y, NH_y, NH_z)) * path_corr
    NH2_tot = np.where(dom_x, NH2, np.where(dom_y, NH2_y, NH2_z)) * path_corr
    NCO_tot = np.where(dom_x, NCO, np.where(dom_y, NCO_y, NCO_z)) * path_corr

    # Midpoint correction
    NH_tot = np.maximum(NH_tot - 0.5 * nH_layer * path_corr, 0.0)
    NH2_tot = np.maximum(NH2_tot - 0.5 * nH2_layer * path_corr, 0.0)
    NCO_tot = np.maximum(NCO_tot - 0.5 * nCO_layer * path_corr, 0.0)

    AV = NH_tot * AV_per_NH

    # G₀ = L_FUV / (4π r² × G₀→flux) × exp(−τ_FUV)
    from ..utils.constants import G0_to_flux
    G0_unatt = L_FUV / (4.0 * np.pi * r**2 * G0_to_flux)
    G0 = G0_unatt * np.exp(-1.8 * AV)

    return G0, AV, NH_tot, NH2_tot, NCO_tot


def compute_fuv_field_3d(density, G0_external, cell_size,
                          x_H2=None, x_CO=None, nside=1):
    """
    Compute the FUV field at every cell in a 3D regular grid.
    
    Parameters
    ----------
    density : ndarray, shape (nx, ny, nz)
        Hydrogen number density [cm⁻³].
    G0_external : float or ndarray
        External FUV field [Habing]. If scalar, isotropic.
        If array shape (n_rays,), directional illumination.
    cell_size : float
        Cell edge length [cm].
    x_H2 : ndarray, shape (nx, ny, nz), optional
        H₂ abundance fraction. For column density calculation.
    x_CO : ndarray, shape (nx, ny, nz), optional
        CO abundance fraction.
    nside : int
        HEALPix resolution. n_rays = 12 * nside².
    
    Returns
    -------
    G0 : ndarray, shape (nx, ny, nz)
        FUV field at each cell [Habing].
    A_V : ndarray, shape (nx, ny, nz)
        Visual extinction (minimum over all ray directions) [mag].
    N_H : ndarray, shape (nx, ny, nz)
        H column density along minimum-AV ray [cm⁻²].
    N_H2 : ndarray, shape (nx, ny, nz)
        H₂ column density along minimum-AV ray [cm⁻²].
    N_CO : ndarray, shape (nx, ny, nz)
        CO column density along minimum-AV ray [cm⁻²].
    """
    nx, ny, nz = density.shape
    directions, weights = healpix_directions(nside)
    n_rays = len(directions)
    
    if x_H2 is None:
        x_H2 = np.zeros_like(density)
    if x_CO is None:
        x_CO = np.zeros_like(density)
    
    # Initialize outputs
    G0 = np.zeros_like(density)
    AV_min = np.full_like(density, 1e10)
    NH_best = np.zeros_like(density)
    NH2_best = np.zeros_like(density)
    NCO_best = np.zeros_like(density)
    
    # External G0 per ray direction
    if np.isscalar(G0_external):
        G0_ext_rays = np.full(n_rays, G0_external)
    else:
        G0_ext_rays = np.asarray(G0_external)
    
    # For each ray direction, march through the grid
    for iray in range(n_rays):
        d = directions[iray]
        w = weights[iray] / (4.0 * np.pi)
        G0_ext = G0_ext_rays[iray] if len(G0_ext_rays) > iray else G0_ext_rays[0]
        
        # Compute column densities along this ray direction for all cells
        # using vectorized cumulative sums along the ray axis
        AV_ray, NH_ray, NH2_ray, NCO_ray = _trace_ray_direction(
            density, x_H2, x_CO, d, cell_size
        )
        
        # Attenuated FUV from this direction
        G0_dir = G0_ext * np.exp(-1.8 * AV_ray)  # tau_FUV ≈ 1.8 * AV
        G0 += G0_dir * w
        
        # Track minimum AV direction (dominant illumination)
        mask = AV_ray < AV_min
        AV_min = np.where(mask, AV_ray, AV_min)
        NH_best = np.where(mask, NH_ray, NH_best)
        NH2_best = np.where(mask, NH2_ray, NH2_best)
        NCO_best = np.where(mask, NCO_ray, NCO_best)
    
    return G0, AV_min, NH_best, NH2_best, NCO_best


def _trace_ray_direction(density, x_H2, x_CO, direction, cell_size):
    """
    Compute column densities along a single ray direction for ALL cells.
    
    Uses a vectorized approach: project the grid along the dominant
    axis of the ray direction, then use cumulative sums.
    
    For non-axis-aligned rays, we approximate by decomposing the path
    into the 3 axis components and weighting by the direction cosines.
    This is the "6-ray" approximation used by Glover & Mac Low (2007).
    
    Parameters
    ----------
    density : ndarray (nx, ny, nz)
    x_H2 : ndarray (nx, ny, nz)
    x_CO : ndarray (nx, ny, nz)
    direction : ndarray (3,)
        Unit vector pointing TOWARD the radiation source.
    cell_size : float
    
    Returns
    -------
    AV, NH, NH2, NCO : ndarrays (nx, ny, nz)
    """
    nx, ny, nz = density.shape
    dx, dy, dz = direction
    
    # Determine dominant axis
    abs_d = np.abs([dx, dy, dz])
    dominant = np.argmax(abs_d)
    
    # Path length factor: ds = cell_size / |cos(theta_dominant)|
    # This accounts for oblique rays traversing more of each cell
    path_factor = 1.0 / max(abs_d[dominant], 0.01)
    ds = cell_size * path_factor
    
    # Column density per cell layer
    nH_layer = density * ds  # [cm⁻²] per cell
    nH2_layer = density * x_H2 * 2.0 * ds  # N(H2) per cell
    nCO_layer = density * x_CO * ds
    
    # Cumulative sum along the dominant axis, in the direction
    # the ray enters the grid
    sign = 1 if [dx, dy, dz][dominant] < 0 else -1  # Ray comes from + or - side
    
    if dominant == 0:
        if sign > 0:
            NH = np.cumsum(nH_layer, axis=0)
        else:
            NH = np.cumsum(nH_layer[::-1], axis=0)[::-1]
        # Midpoint correction: subtract half the current cell
        NH = NH - 0.5 * nH_layer
    elif dominant == 1:
        if sign > 0:
            NH = np.cumsum(nH_layer, axis=1)
        else:
            NH = np.cumsum(nH_layer[:, ::-1], axis=1)[:, ::-1]
        NH = NH - 0.5 * nH_layer
    else:
        if sign > 0:
            NH = np.cumsum(nH_layer, axis=2)
        else:
            NH = np.cumsum(nH_layer[:, :, ::-1], axis=2)[:, :, ::-1]
        NH = NH - 0.5 * nH_layer
    
    NH = np.maximum(NH, 0.0)
    
    # Same for H2 and CO
    if dominant == 0:
        if sign > 0:
            NH2 = np.cumsum(nH2_layer, axis=0)
            NCO = np.cumsum(nCO_layer, axis=0)
        else:
            NH2 = np.cumsum(nH2_layer[::-1], axis=0)[::-1]
            NCO = np.cumsum(nCO_layer[::-1], axis=0)[::-1]
    elif dominant == 1:
        if sign > 0:
            NH2 = np.cumsum(nH2_layer, axis=1)
            NCO = np.cumsum(nCO_layer, axis=1)
        else:
            NH2 = np.cumsum(nH2_layer[:, ::-1], axis=1)[:, ::-1]
            NCO = np.cumsum(nCO_layer[:, ::-1], axis=1)[:, ::-1]
    else:
        if sign > 0:
            NH2 = np.cumsum(nH2_layer, axis=2)
            NCO = np.cumsum(nCO_layer, axis=2)
        else:
            NH2 = np.cumsum(nH2_layer[:, :, ::-1], axis=2)[:, :, ::-1]
            NCO = np.cumsum(nCO_layer[:, :, ::-1], axis=2)[:, :, ::-1]
    
    NH2 = np.maximum(NH2 - 0.5 * nH2_layer, 0.0)
    NCO = np.maximum(NCO - 0.5 * nCO_layer, 0.0)
    
    AV = NH * AV_per_NH
    
    return AV, NH, NH2, NCO
