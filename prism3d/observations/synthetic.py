"""
Synthetic observations module for PRISM-3D.

Generates observable quantities from model results:
1. Fine-structure line intensity maps ([CII], [OI], [CI])
2. CO rotational line maps (all J transitions)
3. H2 ro-vibrational line maps
4. Dust continuum emission SEDs
5. Column density maps
6. Beam convolution for telescope comparison

All intensities in erg/s/cm²/sr unless noted.
"""

import numpy as np
from ..utils.constants import (h_planck, c_light, k_boltz,
                                 fine_structure_lines, pc_cm)


class SyntheticObserver:
    """
    Generate synthetic observations from a PRISM-3D model.

    Parameters
    ----------
    grid : OctreeGrid
        The model grid with converged solution
    distance : float
        Distance to the source [cm]. Default: 400 pc (Orion)
    inclination : float
        Viewing angle [degrees]. 0 = face-on, 90 = edge-on
    """

    def __init__(self, grid, distance=400*pc_cm, inclination=0.0):
        self.grid = grid
        self.distance = distance
        self.inclination = inclination * np.pi / 180.0

    def line_intensity_map(self, line_name, axis=0, n_pixels=64):
        """
        Generate a 2D intensity map for a given emission line.

        Integrates emissivity along the line of sight.

        Parameters
        ----------
        line_name : str
            Line identifier (e.g., 'CII_158', 'OI_63', 'CO_1-0')
        axis : int
            Line-of-sight axis (0=x, 1=y, 2=z)
        n_pixels : int
            Number of pixels per side in the output map

        Returns
        -------
        intensity : ndarray, shape (n_pixels, n_pixels)
            Integrated line intensity [erg/s/cm²/sr]
        x_coords : ndarray
            Physical coordinates of pixel centers [cm]
        y_coords : ndarray
            Physical coordinates of pixel centers [cm]
        """
        leaves = self.grid.get_leaves()

        # Set up pixel grid
        half_box = self.grid.box_size / 2.0
        x_edges = np.linspace(-half_box, half_box, n_pixels + 1)
        y_edges = np.linspace(-half_box, half_box, n_pixels + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        intensity = np.zeros((n_pixels, n_pixels))

        # Determine which axes map to image plane
        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1

        for leaf in leaves:
            # Compute emissivity of this cell
            emissivity = self._cell_emissivity(leaf, line_name)

            if emissivity <= 0:
                continue

            # Find pixel(s) this cell maps to
            pos1 = leaf.center[ax1] + half_box
            pos2 = leaf.center[ax2] + half_box

            # Pixel indices
            i1 = int(pos1 / self.grid.box_size * n_pixels)
            i2 = int(pos2 / self.grid.box_size * n_pixels)

            i1 = np.clip(i1, 0, n_pixels - 1)
            i2 = np.clip(i2, 0, n_pixels - 1)

            # Add contribution: emissivity * path length
            ds = leaf.size  # path through cell [cm]
            intensity[i1, i2] += emissivity * ds

        # Convert to erg/s/cm²/sr (already in volume emissivity * ds = surface brightness)
        # The factor of 1/(4π) converts isotropic emission to per steradian
        intensity /= (4.0 * np.pi)

        return intensity, x_centers, y_centers

    def _cell_emissivity(self, cell, line_name):
        """
        Compute volume emissivity [erg/cm³/s] for a given line in a cell.

        Parameters
        ----------
        cell : OctreeNode
            Grid cell
        line_name : str
            Line identifier

        Returns
        -------
        j : float
            Volume emissivity [erg/cm³/s]
        """
        d = cell.data
        n_H = d.n_H
        T = d.T_gas

        if line_name == 'CII_158':
            from ..thermal.cooling import cii_158_cooling
            return cii_158_cooling(n_H, T, d.x_e, d.x_HI, d.x_H2, d.x_Cp)

        elif line_name == 'OI_63':
            from ..thermal.cooling import oi_cooling
            # Return only the 63 μm component
            # Simplified: assume 63 μm dominates at typical PDR conditions
            total = oi_cooling(n_H, T, d.x_e, d.x_HI, d.x_H2, d.x_O)
            return total * 0.85  # ~85% in 63 μm

        elif line_name == 'OI_145':
            from ..thermal.cooling import oi_cooling
            total = oi_cooling(n_H, T, d.x_e, d.x_HI, d.x_H2, d.x_O)
            return total * 0.15

        elif line_name == 'CI_609':
            from ..thermal.cooling import ci_cooling
            total = ci_cooling(n_H, T, d.x_e, d.x_HI, d.x_H2, d.x_C)
            return total * 0.4

        elif line_name == 'CI_370':
            from ..thermal.cooling import ci_cooling
            total = ci_cooling(n_H, T, d.x_e, d.x_HI, d.x_H2, d.x_C)
            return total * 0.6

        elif line_name.startswith('CO_'):
            from ..thermal.cooling import co_rotational_cooling
            total = co_rotational_cooling(n_H, T, d.x_H2, d.x_CO, d.N_CO)
            # For specific J transitions, would need per-line calculation
            return total

        elif line_name == 'H2_S1':
            # H2 v=0-0 S(1) 17.03 μm
            if T < 300:
                return 0.0
            n_H2 = d.x_H2 * n_H
            T_ul = 1015.0  # K
            A_ul = 4.76e-10  # s⁻¹
            g_u = 21
            g_l = 9
            f_u = g_u * np.exp(-T_ul / T)
            Q = sum((2*J+1) * np.exp(-J*(J+1)*85.4/T) for J in range(20))
            n_u = n_H2 * f_u / Q
            return n_u * A_ul * h_planck * c_light / (17.03e-4)

        return 0.0

    def column_density_map(self, species='H2', axis=0, n_pixels=64):
        """
        Generate a column density map.

        Parameters
        ----------
        species : str
            Species name: 'H', 'H2', 'CO', 'C+', 'C', 'O', 'e'
        axis : int
            Line-of-sight axis

        Returns
        -------
        N_map : ndarray, shape (n_pixels, n_pixels)
            Column density [cm⁻²]
        """
        leaves = self.grid.get_leaves()
        half_box = self.grid.box_size / 2.0

        N_map = np.zeros((n_pixels, n_pixels))

        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1

        # Species abundance attribute mapping
        species_attr = {
            'H': 'x_HI', 'H2': 'x_H2', 'CO': 'x_CO',
            'C+': 'x_Cp', 'C': 'x_C', 'O': 'x_O', 'e': 'x_e',
            'OH': 'x_OH', 'H2O': 'x_H2O',
        }

        attr = species_attr.get(species, None)
        if attr is None:
            raise ValueError(f"Unknown species: {species}")

        for leaf in leaves:
            pos1 = leaf.center[ax1] + half_box
            pos2 = leaf.center[ax2] + half_box

            i1 = int(pos1 / self.grid.box_size * n_pixels)
            i2 = int(pos2 / self.grid.box_size * n_pixels)
            i1 = np.clip(i1, 0, n_pixels - 1)
            i2 = np.clip(i2, 0, n_pixels - 1)

            x_species = getattr(leaf.data, attr)
            n_species = x_species * leaf.data.n_H

            # For H2, column density is n(H2) * ds, but total H in H2 is 2*n(H2)
            if species == 'H2':
                n_species *= 2.0  # Convention: N(H2) counts molecules

            N_map[i1, i2] += n_species * leaf.size

        return N_map

    def dust_sed(self, cell, wavelengths=None):
        """
        Compute the dust SED for a single cell.

        Simple modified blackbody model. For full accuracy, would
        use DustEM with the local grain size distribution.

        Parameters
        ----------
        cell : OctreeNode
            Grid cell
        wavelengths : ndarray, optional
            Wavelengths [cm]. Default: 1 μm to 1 mm.

        Returns
        -------
        flux : ndarray
            Flux density [erg/s/cm²/Hz/sr]
        wavelengths : ndarray
            Wavelengths [cm]
        """
        if wavelengths is None:
            wavelengths = np.logspace(-4, -1, 200)  # 1 μm to 1 mm

        T_d = cell.data.T_dust
        freq = c_light / wavelengths

        # Modified blackbody: B_nu(T_d) * kappa_nu * rho_dust * ds
        # kappa_nu ∝ nu^beta with beta ~ 1.8
        beta_dust = 1.8
        kappa_0 = 10.0  # cm²/g at 250 μm
        nu_0 = c_light / 250e-4  # reference frequency

        kappa_nu = kappa_0 * (freq / nu_0)**beta_dust

        # Planck function
        x = h_planck * freq / (k_boltz * T_d)
        x = np.clip(x, 0, 500)
        B_nu = 2.0 * h_planck * freq**3 / c_light**2 / (np.exp(x) - 1.0)

        # Dust mass column through cell
        dust_to_gas = 0.01
        Sigma_dust = dust_to_gas * cell.data.n_H * m_H * cell.size  # g/cm²

        # Specific intensity
        I_nu = B_nu * (1.0 - np.exp(-kappa_nu * Sigma_dust))

        return I_nu, wavelengths

    def convolve_beam(self, image, beam_fwhm_arcsec, pixel_size_arcsec):
        """
        Convolve an image with a Gaussian beam.

        Parameters
        ----------
        image : ndarray
            2D image array
        beam_fwhm_arcsec : float
            Beam FWHM [arcsec]
        pixel_size_arcsec : float
            Pixel size [arcsec]

        Returns
        -------
        convolved : ndarray
            Beam-convolved image
        """
        from scipy.ndimage import gaussian_filter

        sigma_pix = beam_fwhm_arcsec / pixel_size_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return gaussian_filter(image, sigma=sigma_pix)
