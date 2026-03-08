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
