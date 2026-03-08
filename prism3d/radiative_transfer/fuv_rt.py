"""
FUV Radiative Transfer for PRISM-3D.

Implements multi-ray FUV continuum transfer using ray tracing along
HEALPix-defined directions. Computes the attenuated FUV field G0(x)
at every cell in the 3D grid.

Features:
- HEALPix-based angular discretization (uniform solid angle coverage)
- Ray tracing through octree grid
- Dust continuum attenuation
- H2 and CO self-shielding applied as post-processing factors
"""

import numpy as np
from ..utils.constants import (AV_per_NH, tau_FUV_per_AV, sigma_dust_FUV)


def healpix_directions(nside=2):
    """
    Generate approximately uniform directions on the sphere using
    a simplified HEALPix-like scheme.

    For nside=1: 12 directions
    For nside=2: 48 directions
    For nside=4: 192 directions

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter. Number of pixels = 12 * nside^2.

    Returns
    -------
    directions : ndarray, shape (N_pix, 3)
        Unit vectors for each ray direction.
    weights : ndarray, shape (N_pix,)
        Solid angle weight for each direction [sr].
    """
    n_pix = 12 * nside**2
    directions = np.zeros((n_pix, 3))
    weights = np.full(n_pix, 4.0 * np.pi / n_pix)

    # Use a Fibonacci spiral for approximately uniform distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_pix):
        # Distribute points uniformly in cos(theta)
        cos_theta = 1.0 - 2.0 * (i + 0.5) / n_pix
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = golden_angle * i

        directions[i, 0] = sin_theta * np.cos(phi)
        directions[i, 1] = sin_theta * np.sin(phi)
        directions[i, 2] = cos_theta

    return directions, weights


class FUVRadiativeTransfer:
    """
    FUV continuum radiative transfer module.

    Traces rays from external radiation sources through the grid,
    computing the attenuated FUV field at each cell.

    Parameters
    ----------
    grid : OctreeGrid
        The computational grid
    G0_external : float or callable
        External FUV field. If float, isotropic illumination.
        If callable, G0_external(direction) returns the FUV field
        for each ray direction.
    nside : int
        HEALPix resolution for ray directions.
    """

    def __init__(self, grid, G0_external=1.0, nside=2):
        self.grid = grid
        self.G0_external = G0_external
        self.nside = nside

        # Generate ray directions
        self.directions, self.weights = healpix_directions(nside)
        self.n_rays = len(self.directions)

    def compute_fuv_field(self):
        """
        Compute the FUV field G0 at every leaf cell by tracing rays.

        For each cell, traces rays in all HEALPix directions back to
        the grid boundary, accumulating optical depth. The attenuated
        field is averaged over all directions.

        Also computes column densities N_H, N_H2, N_CO for shielding.
        """
        leaves = self.grid.get_leaves()
        n_cells = len(leaves)

        # Initialize
        G0_cells = np.zeros(n_cells)
        N_H_cells = np.zeros(n_cells)
        N_H2_cells = np.zeros(n_cells)
        N_CO_cells = np.zeros(n_cells)
        AV_cells = np.zeros(n_cells)

        for i, leaf in enumerate(leaves):
            G0_total = 0.0
            # Track minimum A_V (dominant direction) for shielding
            min_AV = np.inf
            best_N_H = 0.0
            best_N_H2 = 0.0
            best_N_CO = 0.0

            for j in range(self.n_rays):
                direction = self.directions[j]
                weight = self.weights[j]

                # Get external G0 for this direction
                if callable(self.G0_external):
                    G0_ext = self.G0_external(direction)
                else:
                    G0_ext = self.G0_external

                # Trace ray from cell to boundary
                tau_dust, N_H, N_H2, N_CO = self._trace_ray(
                    leaf.center, -direction, leaf.size
                )

                # Attenuated FUV field from this direction
                G0_dir = G0_ext * np.exp(-tau_dust)

                # Weight by solid angle
                G0_total += G0_dir * weight / (4.0 * np.pi)

                # Track the direction with minimum extinction
                # (dominant illumination direction)
                A_V_ray = tau_dust / tau_FUV_per_AV
                if A_V_ray < min_AV:
                    min_AV = A_V_ray
                    best_N_H = N_H
                    best_N_H2 = N_H2
                    best_N_CO = N_CO

            G0_cells[i] = G0_total
            AV_cells[i] = min_AV if min_AV < np.inf else 0.0
            N_H_cells[i] = best_N_H
            N_H2_cells[i] = best_N_H2
            N_CO_cells[i] = best_N_CO

        # Write back to cells
        for i, leaf in enumerate(leaves):
            leaf.data.G0 = G0_cells[i]
            leaf.data.A_V = AV_cells[i]
            leaf.data.N_H_total = N_H_cells[i]
            leaf.data.N_H2 = N_H2_cells[i]
            leaf.data.N_CO = N_CO_cells[i]

        return G0_cells, AV_cells

    def _trace_ray(self, origin, direction, cell_size):
        """
        Trace a single ray from origin along direction to the grid boundary.

        Uses DDA (Digital Differential Analyzer) ray marching through
        the octree.

        Parameters
        ----------
        origin : ndarray
            Starting point [cm]
        direction : ndarray
            Ray direction (unit vector, pointing outward from cell)
        cell_size : float
            Size of the starting cell [cm]

        Returns
        -------
        tau_dust : float
            Accumulated dust optical depth in FUV
        N_H : float
            Total H column density [cm⁻²]
        N_H2 : float
            H2 column density [cm⁻²]
        N_CO : float
            CO column density [cm⁻²]
        """
        tau_dust = 0.0
        N_H = 0.0
        N_H2 = 0.0
        N_CO = 0.0

        # Step size: fraction of current cell size
        step_size = cell_size * 0.5
        max_steps = 10000
        pos = origin.copy()

        half_box = self.grid.box_size / 2.0
        center = self.grid.center

        for _ in range(max_steps):
            pos += direction * step_size

            # Check if we've left the grid
            if np.any(np.abs(pos - center) > half_box):
                break

            # Find the cell at this position
            cell = self.grid.find_cell(pos)
            if cell is None:
                break

            # Path length through this cell (approximate)
            ds = min(step_size, cell.size)

            # Accumulate optical depth and column densities
            n_H = cell.data.n_H
            tau_dust += n_H * sigma_dust_FUV * ds
            N_H += n_H * ds
            N_H2 += n_H * cell.data.x_H2 * ds * 2.0  # n(H2) = x_H2 * n_H
            N_CO += n_H * cell.data.x_CO * ds

            # Adapt step size to cell size
            step_size = cell.size * 0.5

        return tau_dust, N_H, N_H2, N_CO

    def compute_1d_column_densities(self, axis=0, direction=1):
        """
        Compute column densities along a single axis (for 1D slab geometry).
        Much faster than full ray tracing for benchmarking.

        Parameters
        ----------
        axis : int
            Axis along which to integrate (0=x, 1=y, 2=z)
        direction : int
            +1 or -1, direction of illumination along axis
        """
        from ..utils.constants import AV_per_NH, tau_FUV_per_AV

        leaves = self.grid.get_leaves()

        # Sort by position along axis
        positions = np.array([l.center[axis] for l in leaves])
        if direction > 0:
            order = np.argsort(positions)
        else:
            order = np.argsort(-positions)

        # Accumulate column density
        N_H_cum = 0.0
        N_H2_cum = 0.0
        N_CO_cum = 0.0

        for idx in order:
            leaf = leaves[idx]
            ds = leaf.size  # path through cell

            # Column density at cell center (midpoint rule)
            N_H_mid = N_H_cum + 0.5 * leaf.data.n_H * ds
            N_H2_mid = N_H2_cum + 0.5 * leaf.data.n_H * leaf.data.x_H2 * 2.0 * ds
            N_CO_mid = N_CO_cum + 0.5 * leaf.data.n_H * leaf.data.x_CO * ds

            leaf.data.N_H_total = N_H_mid
            leaf.data.N_H2 = N_H2_mid
            leaf.data.N_CO = N_CO_mid
            leaf.data.A_V = N_H_mid * AV_per_NH

            # FUV field
            if callable(self.G0_external):
                G0_ext = self.G0_external(np.array([direction, 0, 0]))
            else:
                G0_ext = self.G0_external

            leaf.data.G0 = G0_ext * np.exp(-leaf.data.A_V * tau_FUV_per_AV)

            # Accumulate for next cell
            N_H_cum += leaf.data.n_H * ds
            N_H2_cum += leaf.data.n_H * leaf.data.x_H2 * 2.0 * ds
            N_CO_cum += leaf.data.n_H * leaf.data.x_CO * ds
