"""
Chemical species and reaction network for PRISM-3D.

Implements a PDR-focused network with ~30 species and ~200 reactions,
based on the UMIST/KIDA databases with PDR-specific additions.

The network covers:
- H/H2 chemistry (formation, destruction, self-shielding)
- C+/C/CO chemistry (main carbon network)
- O/OH/H2O oxygen chemistry
- S, Si, Fe ionization balance
- Key molecular ions: HCO+, CH+, OH+, H3+
- Charge balance with PAH/grain charging
"""

import numpy as np
from ..utils.constants import k_boltz, eV_to_K


# ============================================================
# Species definitions
# ============================================================

class Species:
    """A chemical species with its properties."""

    def __init__(self, name, mass_amu, charge=0, n_atoms=1, is_grain=False):
        self.name = name
        self.mass_amu = mass_amu
        self.charge = charge
        self.n_atoms = n_atoms
        self.is_grain = is_grain

    def __repr__(self):
        return f"Species({self.name})"


# Define all species in the network
SPECIES_CATALOG = {
    # Atoms and ions
    'H':     Species('H',     1.008,  0, 1),
    'H+':    Species('H+',    1.008,  1, 1),
    'H-':    Species('H-',    1.008, -1, 1),
    'He':    Species('He',    4.003,  0, 1),
    'He+':   Species('He+',   4.003,  1, 1),
    'C':     Species('C',    12.011,  0, 1),
    'C+':    Species('C+',   12.011,  1, 1),
    'O':     Species('O',    15.999,  0, 1),
    'O+':    Species('O+',   15.999,  1, 1),
    'S':     Species('S',    32.06,   0, 1),
    'S+':    Species('S+',   32.06,   1, 1),
    'Si':    Species('Si',   28.086,  0, 1),
    'Si+':   Species('Si+', 28.086,  1, 1),
    'Fe':    Species('Fe',   55.845,  0, 1),
    'Fe+':   Species('Fe+', 55.845,  1, 1),
    'e-':    Species('e-',    5.486e-4, -1, 0),
    'M+':    Species('M+',   24.0,   1, 1),  # generic metal

    # Molecules
    'H2':    Species('H2',    2.016,  0, 2),
    'H2+':   Species('H2+',   2.016,  1, 2),
    'H3+':   Species('H3+',   3.024,  1, 3),
    'CO':    Species('CO',   28.010,  0, 2),
    'OH':    Species('OH',   17.007,  0, 2),
    'OH+':   Species('OH+', 17.007,  1, 2),
    'H2O':   Species('H2O', 18.015,  0, 3),
    'H2O+':  Species('H2O+',18.015,  1, 3),
    'H3O+':  Species('H3O+',19.023,  1, 4),
    'CH':    Species('CH',   13.019,  0, 2),
    'CH+':   Species('CH+', 13.019,  1, 2),
    'CH2':   Species('CH2', 14.027,  0, 3),
    'CH2+':  Species('CH2+',14.027,  1, 3),
    'CO+':   Species('CO+', 28.010,  1, 2),
    'HCO+':  Species('HCO+',29.018,  1, 3),
    'HOC+':  Species('HOC+',29.018,  1, 3),
    'O2':    Species('O2',   31.998,  0, 2),
    'CS':    Species('CS',   44.076,  0, 2),
    'HCS+':  Species('HCS+',45.084,  1, 3),
    'N':     Species('N',    14.007,  0, 1),
    'N+':    Species('N+',   14.007,  1, 1),
    'CN':    Species('CN',   26.018,  0, 2),
    'HCN':   Species('HCN', 27.026,  0, 3),

    # Grain / PAH pseudo-species
    'PAH':   Species('PAH',  100.0,  0, 1, is_grain=True),
    'PAH+':  Species('PAH+', 100.0,  1, 1, is_grain=True),
    'PAH-':  Species('PAH-', 100.0, -1, 1, is_grain=True),
}


# ============================================================
# Reaction types
# ============================================================

class ReactionType:
    """Enumeration of reaction types."""
    TWO_BODY = 'two_body'               # A + B → C + D
    PHOTODISSOCIATION = 'photodiss'     # A + hν → B + C
    PHOTOIONIZATION = 'photoion'         # A + hν → A+ + e-
    COSMIC_RAY = 'cosmic_ray'           # A + CR → products
    CR_INDUCED_PHOTO = 'cr_photo'       # A + CR_photon → products
    GRAIN_SURFACE = 'grain_surface'     # A + B + grain → C + grain
    RECOMBINATION = 'recomb'            # A+ + e- → A + hν (radiative)
    GRAIN_RECOMB = 'grain_recomb'       # A+ + e- + grain → A + grain
    CHARGE_EXCHANGE = 'charge_exchange' # A+ + B → A + B+
    THREE_BODY = 'three_body'           # A + B + M → C + M


class Reaction:
    """
    A single chemical reaction with its rate coefficient.

    Rate coefficient is parameterized as:
        k(T) = alpha * (T/300)^beta * exp(-gamma/T)
    for two-body reactions, or:
        k = alpha * G0 * exp(-gamma * A_V)
    for photoreactions, or:
        k = alpha * zeta_CR
    for cosmic ray reactions.
    """

    def __init__(self, reactants, products, alpha, beta=0.0, gamma=0.0,
                 reaction_type=ReactionType.TWO_BODY,
                 T_range=(10, 41000), name=None):
        self.reactants = list(reactants)
        self.products = list(products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reaction_type = reaction_type
        self.T_range = T_range
        self.name = name or self._auto_name()

    def _auto_name(self):
        r = ' + '.join(self.reactants)
        p = ' + '.join(self.products)
        return f"{r} → {p}"

    def rate(self, T=None, G0=None, A_V=None, zeta_CR=None, n_H=None):
        """
        Compute rate coefficient.

        For two-body: returns k [cm³/s]
        For photo: returns k [s⁻¹] (already includes G0 and shielding)
        For CR: returns k [s⁻¹]
        """
        if self.reaction_type in (ReactionType.TWO_BODY,
                                   ReactionType.RECOMBINATION,
                                   ReactionType.CHARGE_EXCHANGE):
            T_clamped = np.clip(T, self.T_range[0], self.T_range[1])
            return self.alpha * (T_clamped / 300.0)**self.beta * np.exp(-self.gamma / T_clamped)

        elif self.reaction_type in (ReactionType.PHOTODISSOCIATION,
                                     ReactionType.PHOTOIONIZATION):
            return self.alpha * G0 * np.exp(-self.gamma * A_V)

        elif self.reaction_type == ReactionType.COSMIC_RAY:
            return self.alpha * zeta_CR

        elif self.reaction_type == ReactionType.CR_INDUCED_PHOTO:
            # CR-induced UV photons, rate ~ alpha * zeta_CR / (1 - omega)
            return self.alpha * zeta_CR

        elif self.reaction_type == ReactionType.THREE_BODY:
            T_clamped = np.clip(T, self.T_range[0], self.T_range[1])
            return self.alpha * (T_clamped / 300.0)**self.beta * np.exp(-self.gamma / T_clamped) * n_H

        elif self.reaction_type == ReactionType.GRAIN_SURFACE:
            # H2 formation on grains is handled separately
            return self.alpha * (T / 300.0)**self.beta

        elif self.reaction_type == ReactionType.GRAIN_RECOMB:
            T_clamped = np.clip(T, self.T_range[0], self.T_range[1])
            return self.alpha * (T_clamped / 300.0)**self.beta

        return 0.0

    def __repr__(self):
        return f"Reaction({self.name}, α={self.alpha:.2e})"


# ============================================================
# Build the PDR chemical network
# ============================================================

def build_pdr_network():
    """
    Construct the standard PRISM-3D chemical network.

    Returns a list of Reaction objects. Rate coefficients are from
    UMIST 2012 / KIDA 2014 where available, supplemented with
    Glover+ (2010) for H2 chemistry and Visser+ (2009) for CO.

    This is a reduced but complete PDR network covering the key
    species and transitions needed for thermal and chemical balance.
    """
    reactions = []
    R = Reaction  # shorthand

    # =============================================
    # H2 formation and destruction
    # =============================================

    # H + H + grain → H2 + grain (Jura 1975, Cazaux & Tielens 2004)
    # Rate ~ 3e-17 * sqrt(T/100) * n_H * x_HI * n_H  [per H nucleus]
    # The actual rate is R_H2 * n(H) * n_H where R_H2 ~ 3e-17 cm³/s
    reactions.append(R(
        ['H', 'H'], ['H2'],
        alpha=3.0e-17, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_SURFACE,
        name='H2_formation_grain'
    ))

    # H2 + hν → H + H (photodissociation, Solomon process)
    # Base rate 3.3e-11 s⁻¹ in unshielded Draine field
    # Self-shielding handled separately
    reactions.append(R(
        ['H2'], ['H', 'H'],
        alpha=3.3e-11, beta=0.0, gamma=3.74,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='H2_photodiss'
    ))

    # H2 + CR → H + H (+ e)
    # H2 + CR → H2+ + e- (primary ionization channel)
    # Rate: ~2.5 * zeta (Woodall+ 2007, UMIST2012)
    # The dissociative channel H2 -> H + H+ + e is negligible
    # and is NOT included per the Röllig benchmark convention.
    reactions.append(R(
        ['H2'], ['H2+', 'e-'],
        alpha=2.5, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.COSMIC_RAY,
        name='H2_CR_ionization'
    ))

    # =============================================
    # H ionization / recombination
    # =============================================

    # H + CR → H+ + e-
    reactions.append(R(
        ['H'], ['H+', 'e-'],
        alpha=0.46, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.COSMIC_RAY,
        name='H_CR_ionization'
    ))

    # H+ + e- → H + hν (radiative recombination, case B)
    reactions.append(R(
        ['H+', 'e-'], ['H'],
        alpha=2.753e-14, beta=-0.745, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H_recombination'
    ))

    # H + e- → H- + hν
    reactions.append(R(
        ['H', 'e-'], ['H-'],
        alpha=1.4e-18, beta=0.928, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H_minus_formation'
    ))

    # H- + H → H2 + e- (gas-phase H2 formation)
    reactions.append(R(
        ['H-', 'H'], ['H2', 'e-'],
        alpha=4.0e-9, beta=-0.02, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H2_gasphase_Hminus'
    ))

    # =============================================
    # Carbon chemistry: C+ / C / CO
    # =============================================

    # C + hν → C+ + e- (photoionization, IP = 11.26 eV)
    reactions.append(R(
        ['C'], ['C+', 'e-'],
        alpha=3.0e-10, beta=0.0, gamma=3.0,
        reaction_type=ReactionType.PHOTOIONIZATION,
        name='C_photoionization'
    ))

    # C+ + e- → C + hν (radiative recombination)
    reactions.append(R(
        ['C+', 'e-'], ['C'],
        alpha=4.67e-12, beta=-0.6, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='Cp_recombination'
    ))

    # CO + hν → C + O (photodissociation)
    # Self-shielding handled separately
    reactions.append(R(
        ['CO'], ['C', 'O'],
        alpha=2.0e-10, beta=0.0, gamma=3.53,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='CO_photodiss'
    ))

    # C+ + OH → CO + H+ (main CO formation route at low A_V)
    reactions.append(R(
        ['C+', 'OH'], ['CO', 'H+'],
        alpha=7.7e-10, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CO_form_CpOH'
    ))

    # C+ + H2O → HCO+ + H
    reactions.append(R(
        ['C+', 'H2O'], ['HCO+', 'H'],
        alpha=9.0e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='HCOp_form_CpH2O'
    ))

    # C + OH → CO + H
    reactions.append(R(
        ['C', 'OH'], ['CO', 'H'],
        alpha=1.0e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CO_form_COH'
    ))

    # CH + O → CO + H
    reactions.append(R(
        ['CH', 'O'], ['CO', 'H'],
        alpha=6.59e-11, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CO_form_CHO'
    ))

    # C+ + H2 → CH+ + H (endothermic, needs 4640 K)
    reactions.append(R(
        ['C+', 'H2'], ['CH+', 'H'],
        alpha=1.0e-10, beta=0.0, gamma=4640.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CHp_form'
    ))

    # CH+ + H2 → CH2+ + H
    reactions.append(R(
        ['CH+', 'H2'], ['CH2+', 'H'],
        alpha=1.2e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CH2p_form'
    ))

    # CH+ + e- → C + H
    reactions.append(R(
        ['CH+', 'e-'], ['C', 'H'],
        alpha=1.5e-7, beta=-0.42, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='CHp_recomb'
    ))

    # =============================================
    # Oxygen chemistry: OH / H2O
    # =============================================

    # O + H+ → O+ + H (charge exchange, nearly resonant)
    reactions.append(R(
        ['O', 'H+'], ['O+', 'H'],
        alpha=7.31e-10, beta=0.23, gamma=225.9,
        reaction_type=ReactionType.CHARGE_EXCHANGE,
        name='O_Hp_CE'
    ))

    # O+ + H → O + H+ (reverse)
    reactions.append(R(
        ['O+', 'H'], ['O', 'H+'],
        alpha=5.66e-10, beta=0.36, gamma=-8.6,
        reaction_type=ReactionType.CHARGE_EXCHANGE,
        name='Op_H_CE'
    ))

    # O + H3+ → OH+ + H2
    reactions.append(R(
        ['O', 'H3+'], ['OH+', 'H2'],
        alpha=8.4e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='OHp_form_OH3p'
    ))

    # OH+ + H2 → H2O+ + H
    reactions.append(R(
        ['OH+', 'H2'], ['H2O+', 'H'],
        alpha=1.01e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H2Op_form'
    ))

    # H2O+ + H2 → H3O+ + H
    reactions.append(R(
        ['H2O+', 'H2'], ['H3O+', 'H'],
        alpha=6.4e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H3Op_form'
    ))

    # H3O+ + e- → OH + H + H (dissociative recombination)
    reactions.append(R(
        ['H3O+', 'e-'], ['OH', 'H', 'H'],
        alpha=1.08e-7, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H3Op_recomb_OH'
    ))

    # H3O+ + e- → H2O + H
    reactions.append(R(
        ['H3O+', 'e-'], ['H2O', 'H'],
        alpha=6.02e-8, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H3Op_recomb_H2O'
    ))

    # OH + hν → O + H
    reactions.append(R(
        ['OH'], ['O', 'H'],
        alpha=3.5e-10, beta=0.0, gamma=1.7,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='OH_photodiss'
    ))

    # H2O + hν → OH + H
    reactions.append(R(
        ['H2O'], ['OH', 'H'],
        alpha=5.9e-10, beta=0.0, gamma=1.7,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='H2O_photodiss'
    ))

    # O + H2 → OH + H (key neutral-neutral, barrier)
    reactions.append(R(
        ['O', 'H2'], ['OH', 'H'],
        alpha=3.14e-13, beta=2.7, gamma=3150.0,
        reaction_type=ReactionType.TWO_BODY,
        name='OH_form_OH2'
    ))

    # OH + H → O + H2 (reverse)
    reactions.append(R(
        ['OH', 'H'], ['O', 'H2'],
        alpha=6.99e-14, beta=2.8, gamma=1950.0,
        reaction_type=ReactionType.TWO_BODY,
        name='OH_destr_OHH'
    ))

    # OH + H2 → H2O + H
    reactions.append(R(
        ['OH', 'H2'], ['H2O', 'H'],
        alpha=2.05e-12, beta=1.52, gamma=1736.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H2O_form_OHH2'
    ))

    # =============================================
    # HCO+ chemistry
    # =============================================

    # CO + H3+ → HCO+ + H2
    reactions.append(R(
        ['CO', 'H3+'], ['HCO+', 'H2'],
        alpha=1.36e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='HCOp_form_COH3p'
    ))

    # HCO+ + e- → CO + H (dissociative recombination)
    reactions.append(R(
        ['HCO+', 'e-'], ['CO', 'H'],
        alpha=2.4e-7, beta=-0.69, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='HCOp_recomb'
    ))

    # HCO+ + hν → CO + H+
    reactions.append(R(
        ['HCO+'], ['CO+', 'H'],
        alpha=1.5e-10, beta=0.0, gamma=2.5,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='HCOp_photodiss'
    ))

    # =============================================
    # H3+ chemistry (critical ion)
    # =============================================

    # H2+ + H2 → H3+ + H
    reactions.append(R(
        ['H2+', 'H2'], ['H3+', 'H'],
        alpha=2.08e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H3p_form'
    ))

    # H3+ + e- → H2 + H (dissociative recombination)
    reactions.append(R(
        ['H3+', 'e-'], ['H2', 'H'],
        alpha=4.36e-8, beta=-0.52, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H3p_recomb_H2'
    ))

    # H3+ + e- → H + H + H
    reactions.append(R(
        ['H3+', 'e-'], ['H', 'H', 'H'],
        alpha=4.36e-8, beta=-0.52, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H3p_recomb_3H'
    ))

    # H3+ + CO → HCO+ + H2  (already included above)

    # H3+ + C → CH+ + H2
    reactions.append(R(
        ['H3+', 'C'], ['CH+', 'H2'],
        alpha=2.0e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CHp_form_H3pC'
    ))

    # =============================================
    # Sulfur chemistry
    # =============================================

    # S + hν → S+ + e-
    reactions.append(R(
        ['S'], ['S+', 'e-'],
        alpha=6.1e-10, beta=0.0, gamma=2.6,
        reaction_type=ReactionType.PHOTOIONIZATION,
        name='S_photoion'
    ))

    # S+ + e- → S + hν
    reactions.append(R(
        ['S+', 'e-'], ['S'],
        alpha=3.9e-11, beta=-0.63, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='Sp_recomb'
    ))

    # =============================================
    # Silicon chemistry
    # =============================================

    # Si + hν → Si+ + e-
    reactions.append(R(
        ['Si'], ['Si+', 'e-'],
        alpha=4.5e-9, beta=0.0, gamma=2.6,
        reaction_type=ReactionType.PHOTOIONIZATION,
        name='Si_photoion'
    ))

    # Si+ + e- → Si + hν
    reactions.append(R(
        ['Si+', 'e-'], ['Si'],
        alpha=4.26e-12, beta=-0.62, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='Sip_recomb'
    ))

    # =============================================
    # Iron chemistry
    # =============================================

    # Fe + hν → Fe+ + e-
    reactions.append(R(
        ['Fe'], ['Fe+', 'e-'],
        alpha=2.6e-9, beta=0.0, gamma=2.8,
        reaction_type=ReactionType.PHOTOIONIZATION,
        name='Fe_photoion'
    ))

    # Fe+ + e- → Fe + hν (grain-assisted recombination dominates)
    reactions.append(R(
        ['Fe+', 'e-'], ['Fe'],
        alpha=3.7e-12, beta=-0.65, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='Fep_recomb'
    ))

    # =============================================
    # CO+ reactions
    # =============================================

    # CO+ + H → CO + H+
    reactions.append(R(
        ['CO+', 'H'], ['CO', 'H+'],
        alpha=7.5e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='COp_H'
    ))

    # CO+ + H2 → HCO+ + H
    reactions.append(R(
        ['CO+', 'H2'], ['HCO+', 'H'],
        alpha=7.5e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='COp_H2_HCOp'
    ))

    # =============================================
    # He chemistry
    # =============================================

    # He + CR → He+ + e-
    reactions.append(R(
        ['He'], ['He+', 'e-'],
        alpha=0.5, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.COSMIC_RAY,
        name='He_CR_ion'
    ))

    # He+ + CO → C+ + O + He
    reactions.append(R(
        ['He+', 'CO'], ['C+', 'O', 'He'],
        alpha=1.6e-9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='Hep_CO_destruct'
    ))

    # He+ + e- → He + hν
    reactions.append(R(
        ['He+', 'e-'], ['He'],
        alpha=1.0e-12, beta=-0.64, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='Hep_recomb'
    ))

    # =============================================
    # PAH / grain-assisted recombination
    # =============================================

    # C+ + PAH- → C + PAH (grain-assisted C+ recombination)
    # Rate coefficient ~ 4.4e-12 * sqrt(T/100) cm³/s (Wolfire+ 2003)
    reactions.append(R(
        ['C+', 'PAH-'], ['C', 'PAH'],
        alpha=4.4e-12, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Cp_PAH_recomb'
    ))

    # S+ + PAH- → S + PAH
    reactions.append(R(
        ['S+', 'PAH-'], ['S', 'PAH'],
        alpha=4.4e-12, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Sp_PAH_recomb'
    ))

    # H+ + PAH- → H + PAH
    reactions.append(R(
        ['H+', 'PAH-'], ['H', 'PAH'],
        alpha=4.4e-12, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Hp_PAH_recomb'
    ))

    # Si+ + PAH- → Si + PAH
    reactions.append(R(
        ['Si+', 'PAH-'], ['Si', 'PAH'],
        alpha=4.4e-12, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Sip_PAH_recomb'
    ))

    # Fe+ + PAH- → Fe + PAH
    reactions.append(R(
        ['Fe+', 'PAH-'], ['Fe', 'PAH'],
        alpha=4.4e-12, beta=0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Fep_PAH_recomb'
    ))

    # =============================================
    # CR-induced photodissociation / ionization
    # (Critical in shielded gas where external UV is gone)
    # =============================================

    # CO + CR_photon → C + O (Heays+ 2017, Gredel+ 1989)
    # Rate ~ 6 * zeta per CO molecule (updated from original 15)
    reactions.append(R(
        ['CO'], ['C', 'O'],
        alpha=6.0, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.CR_INDUCED_PHOTO,
        name='CO_CR_photodiss'
    ))

    # H2O + CR_photon → OH + H
    reactions.append(R(
        ['H2O'], ['OH', 'H'],
        alpha=6.0, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.CR_INDUCED_PHOTO,
        name='H2O_CR_photodiss'
    ))

    # OH + CR_photon → O + H
    reactions.append(R(
        ['OH'], ['O', 'H'],
        alpha=5.0, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.CR_INDUCED_PHOTO,
        name='OH_CR_photodiss'
    ))

    # C + CR_photon → C+ + e-
    reactions.append(R(
        ['C'], ['C+', 'e-'],
        alpha=3.9, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.CR_INDUCED_PHOTO,
        name='C_CR_photoion'
    ))

    # =============================================
    # O2 chemistry (needed for CO formation via C + O2)
    # =============================================

    # O + OH → O2 + H
    reactions.append(R(
        ['O', 'OH'], ['O2', 'H'],
        alpha=3.5e-11, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='O2_form_OOH'
    ))

    # C + O2 → CO + O (important CO formation at depth!)
    reactions.append(R(
        ['C', 'O2'], ['CO', 'O'],
        alpha=5.56e-11, beta=0.41, gamma=-26.9,
        reaction_type=ReactionType.TWO_BODY,
        name='CO_form_CO2'
    ))

    # O2 + hν → O + O
    reactions.append(R(
        ['O2'], ['O', 'O'],
        alpha=7.9e-10, beta=0.0, gamma=1.8,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='O2_photodiss'
    ))

    # O2 + CR_photon → O + O
    reactions.append(R(
        ['O2'], ['O', 'O'],
        alpha=6.0, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.CR_INDUCED_PHOTO,
        name='O2_CR_photodiss'
    ))

    # O2 + C+ → CO+ + O (another CO formation pathway)
    reactions.append(R(
        ['O2', 'C+'], ['CO+', 'O'],
        alpha=3.8e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='COp_form_O2Cp'
    ))

    # =============================================
    # Additional CH chemistry
    # =============================================

    # CH + H → C + H2
    reactions.append(R(
        ['CH', 'H'], ['C', 'H2'],
        alpha=2.7e-11, beta=0.38, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CH_destr_H'
    ))

    # CH + hν → C + H
    reactions.append(R(
        ['CH'], ['C', 'H'],
        alpha=9.2e-10, beta=0.0, gamma=1.7,
        reaction_type=ReactionType.PHOTODISSOCIATION,
        name='CH_photodiss'
    ))

    # CH2+ + e- → CH + H or C + H2
    reactions.append(R(
        ['CH2+', 'e-'], ['CH', 'H'],
        alpha=1.6e-7, beta=-0.6, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='CH2p_recomb_CH'
    ))

    # CH2+ + e- → C + H + H
    reactions.append(R(
        ['CH2+', 'e-'], ['C', 'H', 'H'],
        alpha=4.0e-7, beta=-0.6, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='CH2p_recomb_C'
    ))

    # CH + O → HCO+ + e-  (minor but contributes to HCO+)
    reactions.append(R(
        ['CH', 'O'], ['HCO+', 'e-'],
        alpha=2.0e-11, beta=0.44, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='HCOp_form_CHO'
    ))

    # =============================================
    # Additional CO+ / HCO+ routes
    # =============================================

    # CO+ + e- → C + O
    reactions.append(R(
        ['CO+', 'e-'], ['C', 'O'],
        alpha=2.0e-7, beta=-0.48, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='COp_recomb'
    ))

    # C + H2O → HCO + H (→ simplify as CO production)
    # Actually: C + H2O has multiple channels, main one → HCO + H
    # We simplify as net CO production
    reactions.append(R(
        ['C', 'H2O'], ['CO', 'H', 'H'],
        alpha=1.2e-10, beta=0.0, gamma=0.0,
        reaction_type=ReactionType.TWO_BODY,
        name='CO_form_CH2O'
    ))

    # =============================================
    # H2 collisional dissociation (important at high T)
    # =============================================

    # H2 + H → H + H + H (collisional dissociation)
    reactions.append(R(
        ['H2', 'H'], ['H', 'H', 'H'],
        alpha=6.67e-12, beta=0.0, gamma=52000.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H2_coll_dissoc_H'
    ))

    # H2 + H2 → H2 + H + H
    reactions.append(R(
        ['H2', 'H2'], ['H2', 'H', 'H'],
        alpha=5.5e-29, beta=0.0, gamma=52000.0,
        reaction_type=ReactionType.TWO_BODY,
        name='H2_coll_dissoc_H2'
    ))

    # =============================================
    # Metal ion depletion onto grains (important for
    # controlling x_e at depth)
    # =============================================

    # Fe+ + grain → Fe + grain (depletion)
    reactions.append(R(
        ['Fe+', 'e-'], ['Fe'],
        alpha=5.0e-12, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Fep_grain_recomb'
    ))

    # Si+ + grain → Si + grain
    reactions.append(R(
        ['Si+', 'e-'], ['Si'],
        alpha=5.0e-12, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.GRAIN_RECOMB,
        name='Sip_grain_recomb'
    ))

    # =============================================
    # H3+ + O chain completion
    # =============================================

    # H3+ + O → OH+ + H2 (already have this)
    # But also need H3+ + O → H2O+ + H (minor channel)

    # OH+ + e- → O + H
    reactions.append(R(
        ['OH+', 'e-'], ['O', 'H'],
        alpha=3.75e-8, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='OHp_recomb'
    ))

    # H2O+ + e- → O + H + H
    reactions.append(R(
        ['H2O+', 'e-'], ['O', 'H', 'H'],
        alpha=3.05e-7, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H2Op_recomb_O'
    ))

    # H2O+ + e- → OH + H
    reactions.append(R(
        ['H2O+', 'e-'], ['OH', 'H'],
        alpha=8.6e-8, beta=-0.5, gamma=0.0,
        reaction_type=ReactionType.RECOMBINATION,
        name='H2Op_recomb_OH'
    ))

    return reactions


class ChemicalNetwork:
    """
    Container for the chemical network.
    Provides methods for computing formation/destruction rates.
    """

    def __init__(self, reactions=None):
        if reactions is None:
            reactions = build_pdr_network()
        self.reactions = reactions

        # Build species list from reactions
        species_set = set()
        for rxn in reactions:
            species_set.update(rxn.reactants)
            species_set.update(rxn.products)
        self.species_names = sorted(species_set)
        self.species_index = {s: i for i, s in enumerate(self.species_names)}
        self.n_species = len(self.species_names)

    def compute_rates(self, abundances, n_H, T, G0, A_V, zeta_CR,
                      f_shield_H2=1.0, f_shield_CO=1.0):
        """
        Compute formation and destruction rates for all species.

        Parameters
        ----------
        abundances : dict
            Species name → abundance relative to n_H
        n_H : float
            H nuclei density [cm⁻³]
        T : float
            Gas temperature [K]
        G0 : float
            FUV field [Habing units]
        A_V : float
            Visual extinction [mag]
        zeta_CR : float
            Cosmic ray ionization rate [s⁻¹]
        f_shield_H2 : float
            H2 self-shielding factor (0 to 1)
        f_shield_CO : float
            CO self-shielding factor (0 to 1)

        Returns
        -------
        dndt : dict
            Species name → time derivative of abundance [s⁻¹]
        """
        dndt = {s: 0.0 for s in self.species_names}

        for rxn in self.reactions:
            # Get rate coefficient
            k = rxn.rate(T=T, G0=G0, A_V=A_V, zeta_CR=zeta_CR, n_H=n_H)

            # Apply self-shielding
            if rxn.name == 'H2_photodiss':
                k *= f_shield_H2
            elif rxn.name == 'CO_photodiss':
                k *= f_shield_CO

            # Compute reaction rate [cm⁻³ s⁻¹ / n_H]
            if rxn.reaction_type in (ReactionType.TWO_BODY,
                                      ReactionType.RECOMBINATION,
                                      ReactionType.CHARGE_EXCHANGE,
                                      ReactionType.GRAIN_RECOMB):
                # Rate = k * n(A) * n(B) = k * x_A * x_B * n_H²
                # but we want dx/dt, so divide by n_H
                x_A = abundances.get(rxn.reactants[0], 0.0)
                x_B = abundances.get(rxn.reactants[1], 0.0)
                rate = k * x_A * x_B * n_H

            elif rxn.reaction_type in (ReactionType.PHOTODISSOCIATION,
                                        ReactionType.PHOTOIONIZATION,
                                        ReactionType.COSMIC_RAY,
                                        ReactionType.CR_INDUCED_PHOTO):
                # Unimolecular: rate = k * n(A) → dx/dt = k * x_A
                x_A = abundances.get(rxn.reactants[0], 0.0)
                rate = k * x_A

            elif rxn.reaction_type == ReactionType.GRAIN_SURFACE:
                # H2 formation: special case
                x_H = abundances.get('H', 0.0)
                rate = k * x_H * n_H * 0.5  # Factor 0.5: 2 H atoms per H2

            elif rxn.reaction_type == ReactionType.THREE_BODY:
                x_A = abundances.get(rxn.reactants[0], 0.0)
                x_B = abundances.get(rxn.reactants[1], 0.0)
                rate = k * x_A * x_B * n_H

            else:
                rate = 0.0

            # Update formation/destruction
            for r in rxn.reactants:
                if r in dndt:
                    dndt[r] -= rate
            for p in rxn.products:
                if p in dndt:
                    dndt[p] += rate

        return dndt

    def __repr__(self):
        return (f"ChemicalNetwork({len(self.reactions)} reactions, "
                f"{self.n_species} species)")
