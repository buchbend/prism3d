"""
3D Adaptive Mesh Refinement (AMR) Octree Grid for PRISM-3D.

Supports:
- Uniform and adaptively refined grids
- Cell neighbor finding for gradient calculations
- HEALPix-based ray casting directions
- Refinement criteria based on chemical gradients
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class CellData:
    """
    Physical state data for a single grid cell.
    All quantities in CGS unless noted.
    """
    # Gas properties
    n_H: float = 1e3              # Total H nuclei density [cm⁻³]
    T_gas: float = 100.0          # Gas temperature [K]
    T_dust: float = 20.0          # Dust temperature [K]
    v_turb: float = 1e5           # Microturbulent velocity [cm/s] (1 km/s)

    # Radiation field
    G0: float = 1.0               # FUV field in Habing units
    A_V: float = 0.0              # Visual extinction from illuminating source [mag]

    # Cosmic ray ionization rate
    zeta_CR: float = 2e-16        # CR ionization rate [s⁻¹]

    # Chemical abundances (relative to n_H)
    # These are the key species tracked at full resolution
    x_e: float = 1e-4             # Electron fraction
    x_HI: float = 1.0             # Atomic H
    x_H2: float = 0.0             # Molecular H2
    x_Cp: float = 0.0             # C+
    x_C: float = 0.0              # C
    x_CO: float = 0.0             # CO
    x_O: float = 0.0              # O
    x_OH: float = 0.0             # OH
    x_H2O: float = 0.0            # H2O
    x_CHx: float = 0.0            # CH, CH2 (lumped)
    x_OHx: float = 0.0            # OH, H2O (lumped in reduced network)
    x_Sp: float = 0.0             # S+
    x_S: float = 0.0              # S
    x_Si: float = 0.0             # Si
    x_Sip: float = 0.0            # Si+
    x_Fe: float = 0.0             # Fe
    x_Fep: float = 0.0            # Fe+
    x_HCOp: float = 0.0           # HCO+
    x_Hp: float = 0.0             # H+
    x_Hep: float = 0.0            # He+
    x_Mp: float = 0.0             # Generic metal ion (for charge balance)

    # Heating / cooling rates [erg/cm³/s] (diagnostic)
    Gamma_total: float = 0.0      # Total heating rate
    Lambda_total: float = 0.0     # Total cooling rate

    # Grain properties
    G_PE_efficiency: float = 0.05 # Photoelectric heating efficiency
    R_H2_form: float = 3e-17     # H2 formation rate coefficient [cm³/s]

    # Column densities along primary ray [cm⁻²] (for shielding)
    N_H_total: float = 0.0
    N_H2: float = 0.0
    N_CO: float = 0.0

    # Convergence tracking
    converged: bool = False
    delta_T: float = 1e10         # Last temperature change
    delta_x: float = 1e10         # Max abundance change

    def get_abundances_array(self):
        """Return abundances as numpy array for vectorized operations."""
        return np.array([
            self.x_e, self.x_HI, self.x_H2, self.x_Cp, self.x_C,
            self.x_CO, self.x_O, self.x_OH, self.x_H2O, self.x_CHx,
            self.x_OHx, self.x_Sp, self.x_S, self.x_Si, self.x_Sip,
            self.x_Fe, self.x_Fep, self.x_HCOp, self.x_Hp, self.x_Hep,
            self.x_Mp
        ])

    def set_abundances_from_array(self, arr):
        """Set abundances from numpy array."""
        (self.x_e, self.x_HI, self.x_H2, self.x_Cp, self.x_C,
         self.x_CO, self.x_O, self.x_OH, self.x_H2O, self.x_CHx,
         self.x_OHx, self.x_Sp, self.x_S, self.x_Si, self.x_Sip,
         self.x_Fe, self.x_Fep, self.x_HCOp, self.x_Hp, self.x_Hep,
         self.x_Mp) = arr

    def n_total(self):
        """Total particle number density [cm⁻³]."""
        from ..utils.constants import solar_abundances
        He_per_H = solar_abundances['He']
        return self.n_H * (self.x_HI + self.x_H2 + He_per_H + self.x_e)


# Species index mapping
SPECIES_NAMES = [
    'e', 'HI', 'H2', 'C+', 'C', 'CO', 'O', 'OH', 'H2O', 'CHx',
    'OHx', 'S+', 'S', 'Si', 'Si+', 'Fe', 'Fe+', 'HCO+', 'H+', 'He+', 'M+'
]
SPECIES_INDEX = {name: i for i, name in enumerate(SPECIES_NAMES)}
N_SPECIES = len(SPECIES_NAMES)


class OctreeNode:
    """
    Single node in the octree. Can be a leaf (contains data) or internal
    (has 8 children).
    """
    __slots__ = ['level', 'index', 'center', 'size', 'children', 'data',
                 'parent', 'is_leaf']

    def __init__(self, center, size, level=0, parent=None):
        self.center = np.array(center, dtype=np.float64)  # [cm]
        self.size = float(size)                             # cell edge length [cm]
        self.level = level
        self.parent = parent
        self.children = None   # None = leaf, list of 8 = internal
        self.data = CellData()
        self.is_leaf = True

    @property
    def volume(self):
        return self.size ** 3

    def contains(self, point):
        half = self.size / 2.0
        return all(abs(point[i] - self.center[i]) <= half for i in range(3))

    def refine(self):
        """Split this cell into 8 children."""
        if not self.is_leaf:
            return
        self.children = []
        self.is_leaf = False
        quarter = self.size / 4.0
        child_size = self.size / 2.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    offset = np.array([
                        (2*i - 1) * quarter,
                        (2*j - 1) * quarter,
                        (2*k - 1) * quarter
                    ])
                    child = OctreeNode(
                        self.center + offset,
                        child_size,
                        level=self.level + 1,
                        parent=self
                    )
                    # Inherit parent data
                    child.data = CellData()
                    child.data.n_H = self.data.n_H
                    child.data.T_gas = self.data.T_gas
                    child.data.T_dust = self.data.T_dust
                    child.data.G0 = self.data.G0
                    child.data.A_V = self.data.A_V
                    child.data.zeta_CR = self.data.zeta_CR
                    child.data.set_abundances_from_array(
                        self.data.get_abundances_array()
                    )
                    self.children.append(child)
        # Clear parent data (no longer a leaf)
        self.data = None


class OctreeGrid:
    """
    3D AMR octree grid for PDR calculations.

    The grid covers a cubic domain. Cells can be adaptively refined
    based on physical criteria (chemical gradients, density jumps, etc.)

    Parameters
    ----------
    box_size : float
        Physical size of the domain [cm]
    n_base : int
        Base resolution (number of cells per dimension at level 0).
        Must be a power of 2.
    center : array-like
        Center of the domain [cm]. Default: origin.
    max_level : int
        Maximum refinement level. Each level doubles resolution.
    """

    def __init__(self, box_size, n_base=32, center=None, max_level=5):
        self.box_size = float(box_size)
        self.n_base = n_base
        self.max_level = max_level

        if center is None:
            center = np.zeros(3)
        self.center = np.array(center, dtype=np.float64)

        self.base_cell_size = self.box_size / self.n_base

        # Build base grid
        self.root_nodes = []
        half_box = self.box_size / 2.0
        half_cell = self.base_cell_size / 2.0

        for i in range(n_base):
            for j in range(n_base):
                for k in range(n_base):
                    cell_center = self.center + np.array([
                        -half_box + half_cell + i * self.base_cell_size,
                        -half_box + half_cell + j * self.base_cell_size,
                        -half_box + half_cell + k * self.base_cell_size,
                    ])
                    node = OctreeNode(cell_center, self.base_cell_size, level=0)
                    self.root_nodes.append(node)

        self._leaf_cache = None
        self._leaf_cache_valid = False

    @property
    def n_cells(self):
        """Total number of leaf cells."""
        return len(self.get_leaves())

    def get_leaves(self):
        """Get all leaf nodes (cells with data)."""
        if self._leaf_cache_valid and self._leaf_cache is not None:
            return self._leaf_cache

        leaves = []
        stack = list(self.root_nodes)
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                stack.extend(node.children)

        self._leaf_cache = leaves
        self._leaf_cache_valid = True
        return leaves

    def invalidate_cache(self):
        self._leaf_cache_valid = False

    def find_cell(self, point):
        """Find the leaf cell containing a given point."""
        point = np.asarray(point)
        # Find which root node
        half_box = self.box_size / 2.0
        idx = np.floor(
            (point - self.center + half_box) / self.base_cell_size
        ).astype(int)
        idx = np.clip(idx, 0, self.n_base - 1)
        root_idx = idx[0] * self.n_base**2 + idx[1] * self.n_base + idx[2]

        if root_idx < 0 or root_idx >= len(self.root_nodes):
            return None

        node = self.root_nodes[root_idx]
        while not node.is_leaf:
            # Determine which octant
            octant = 0
            if point[0] >= node.center[0]: octant += 4
            if point[1] >= node.center[1]: octant += 2
            if point[2] >= node.center[2]: octant += 1
            node = node.children[octant]
        return node

    def refine_by_gradient(self, quantity='x_H2', threshold=0.1):
        """
        Refine cells where the gradient of a quantity exceeds threshold.
        Uses neighbor differences as a proxy for gradients.
        """
        leaves = self.get_leaves()
        to_refine = []

        for leaf in leaves:
            if leaf.level >= self.max_level:
                continue
            val = getattr(leaf.data, quantity)
            # Check neighbors along each axis
            for axis in range(3):
                for direction in [-1, 1]:
                    offset = np.zeros(3)
                    offset[axis] = direction * leaf.size
                    neighbor_pos = leaf.center + offset
                    neighbor = self.find_cell(neighbor_pos)
                    if neighbor is not None and neighbor.is_leaf:
                        neighbor_val = getattr(neighbor.data, quantity)
                        if abs(val - neighbor_val) > threshold:
                            to_refine.append(leaf)
                            break
                else:
                    continue
                break

        for node in to_refine:
            node.refine()

        if to_refine:
            self.invalidate_cache()

        return len(to_refine)

    def get_cell_arrays(self):
        """
        Extract all cell data as structure-of-arrays for vectorized computation.
        Returns dict of numpy arrays.
        """
        leaves = self.get_leaves()
        n = len(leaves)

        arrays = {
            'centers': np.zeros((n, 3)),
            'sizes': np.zeros(n),
            'n_H': np.zeros(n),
            'T_gas': np.zeros(n),
            'T_dust': np.zeros(n),
            'G0': np.zeros(n),
            'A_V': np.zeros(n),
            'zeta_CR': np.zeros(n),
            'abundances': np.zeros((n, N_SPECIES)),
        }

        for i, leaf in enumerate(leaves):
            arrays['centers'][i] = leaf.center
            arrays['sizes'][i] = leaf.size
            arrays['n_H'][i] = leaf.data.n_H
            arrays['T_gas'][i] = leaf.data.T_gas
            arrays['T_dust'][i] = leaf.data.T_dust
            arrays['G0'][i] = leaf.data.G0
            arrays['A_V'][i] = leaf.data.A_V
            arrays['zeta_CR'][i] = leaf.data.zeta_CR
            arrays['abundances'][i] = leaf.data.get_abundances_array()

        return arrays

    def set_cell_arrays(self, arrays):
        """Write structure-of-arrays data back to cell objects."""
        leaves = self.get_leaves()
        for i, leaf in enumerate(leaves):
            leaf.data.n_H = arrays['n_H'][i]
            leaf.data.T_gas = arrays['T_gas'][i]
            leaf.data.T_dust = arrays['T_dust'][i]
            leaf.data.G0 = arrays['G0'][i]
            leaf.data.A_V = arrays['A_V'][i]
            leaf.data.zeta_CR = arrays['zeta_CR'][i]
            leaf.data.set_abundances_from_array(arrays['abundances'][i])

    def setup_1d_slab(self, n_H, G0_surface, N_cells=None):
        """
        Initialize as a 1D plane-parallel slab (for benchmarking).
        Illuminated from the -x face.

        Parameters
        ----------
        n_H : float or callable
            Gas density [cm⁻³]. If callable, n_H(x) where x in [0, box_size].
        G0_surface : float
            FUV field at the illuminated surface in Habing units.
        """
        from ..utils.constants import AV_per_NH
        leaves = self.get_leaves()

        for leaf in leaves:
            # Distance from illuminated face
            x = leaf.center[0] - (self.center[0] - self.box_size/2.0)

            if callable(n_H):
                leaf.data.n_H = n_H(x)
            else:
                leaf.data.n_H = float(n_H)

            # Column density to illuminated face (simple integral)
            N_H_col = leaf.data.n_H * x  # Approximate
            leaf.data.A_V = N_H_col * AV_per_NH
            leaf.data.G0 = G0_surface * np.exp(-1.8 * leaf.data.A_V)
            leaf.data.N_H_total = N_H_col

            # Initialize chemistry: surface = atomic, deep = molecular
            if leaf.data.A_V < 0.5:
                # Atomic surface
                leaf.data.x_HI = 1.0
                leaf.data.x_H2 = 0.0
                leaf.data.x_Cp = 2.69e-4
                leaf.data.x_C = 0.0
                leaf.data.x_CO = 0.0
                leaf.data.x_O = 4.90e-4
                leaf.data.T_gas = 500.0
            else:
                # Molecular interior
                fmol = min(1.0, leaf.data.A_V / 5.0)
                leaf.data.x_HI = 1.0 - fmol
                leaf.data.x_H2 = fmol / 2.0
                leaf.data.x_Cp = 2.69e-4 * (1.0 - fmol)
                leaf.data.x_C = 2.69e-4 * fmol * 0.3
                leaf.data.x_CO = 2.69e-4 * fmol * 0.7
                leaf.data.x_O = 4.90e-4 - leaf.data.x_CO
                leaf.data.T_gas = 500.0 - 400.0 * fmol

    def __repr__(self):
        return (f"OctreeGrid(box={self.box_size:.2e} cm, "
                f"n_base={self.n_base}, leaves={self.n_cells})")
