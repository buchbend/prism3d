"""
Benchmark-specific parameter overrides for the Röllig+ (2007) comparison.

This module provides the EXACT conventions used in the benchmark:
- AV_per_NH = 6.289e-22 (not Bohlin+ 1978 value)
- Photo-rates in Draine-field units with semi-infinite slab factor
- H/He/C/O only chemistry (no metals)
- Standardized elemental abundances

These overrides are applied when running benchmark models to ensure
we compare apples to apples.
"""

import numpy as np

# ============================================================
# Röllig benchmark dust properties
# ============================================================
AV_PER_NH_BENCHMARK = 6.289e-22   # mag cm²  (benchmark convention)
NH_PER_AV_BENCHMARK = 1.0 / AV_PER_NH_BENCHMARK

# FUV extinction parameter (tau_FUV = gamma_d * AV)
# gamma_d varies by reaction in the benchmark
GAMMA_DUST = {
    'H2': 3.74,   # H2 photodissociation
    'CO': 3.53,   # CO photodissociation  
    'C':  3.02,   # C photoionization
    'S':  2.6,    # (not in benchmark but kept for non-benchmark use)
    'Si': 2.6,
    'Fe': 2.8,
    'default': 1.8,
}

# ============================================================
# Röllig benchmark photo-rates
# In units of s⁻¹ per unit Draine field (χ=1)
# At the surface of a semi-infinite slab, the effective rate is:
#   k = k_0 * chi / 2 * exp(-gamma_d * AV)
# The /2 comes from half the sky being blocked by the cloud.
# ============================================================
PHOTO_RATES_CHI1 = {
    'H2_photodiss': 5.18e-11,   # s⁻¹ at χ=1, unshielded
    'CO_photodiss': 2.0e-10,    # s⁻¹ at χ=1
    'C_photoion':   3.0e-10,    # s⁻¹ at χ=1
    'S_photoion':   6.1e-10,    # not in benchmark
    'Si_photoion':  4.5e-9,     # not in benchmark
}

# ============================================================
# Röllig benchmark abundances (gas-phase, relative to n_H)
# ============================================================
BENCHMARK_ABUNDANCES = {
    'H':  1.0,
    'He': 0.1,
    'C':  1.4e-4,
    'O':  3.0e-4,
}

# ============================================================
# Röllig benchmark H2 formation rate
# ============================================================
R_H2_BENCHMARK = 3.0e-17  # cm³ s⁻¹ (constant, no T dependence)

# ============================================================
# Röllig benchmark cosmic ray rate
# ============================================================
ZETA_CR_BENCHMARK = 5.0e-17  # s⁻¹

# ============================================================
# CR-induced photodissociation rates per ζ
# ============================================================
CR_PHOTODISS_RATES = {
    'CO': 6.0,    # rate = 6 * ζ per CO
    'H2O': 6.0,
    'OH': 5.0,
    'C_ion': 3.9,  # C + CRγ → C+ + e
}


def apply_benchmark_overrides():
    """
    Override the global constants in prism3d.utils.constants 
    with benchmark-specific values.
    
    Call this BEFORE setting up benchmark models.
    """
    from prism3d.utils import constants
    
    # Override AV_per_NH
    constants.AV_per_NH = AV_PER_NH_BENCHMARK
    constants.NH_per_AV = NH_PER_AV_BENCHMARK
    
    # Override gas_phase_abundances
    constants.gas_phase_abundances = {
        'H':  1.0,
        'He': 0.1,
        'C':  1.4e-4,
        'O':  3.0e-4,
        'S':  0.0,     # Not in benchmark
        'Si': 0.0,
        'Fe': 0.0,
        'N':  0.0,
    }
    
    # Also update solar_abundances to match benchmark
    # (the chemistry solver uses these for conservation)
    constants.solar_abundances['C'] = 1.4e-4
    constants.solar_abundances['O'] = 3.0e-4
    constants.solar_abundances['He'] = 0.1
    
    print("Applied Röllig benchmark overrides:")
    print(f"  AV_per_NH = {constants.AV_per_NH:.3e}")
    print(f"  C/H = {constants.gas_phase_abundances['C']:.1e}")
    print(f"  O/H = {constants.gas_phase_abundances['O']:.1e}")
    print(f"  S/H = {constants.gas_phase_abundances['S']:.1e} (disabled)")


def build_benchmark_network():
    """
    Build a chemical network using EXACTLY the Röllig benchmark 
    photo-rate conventions.
    
    Key differences from default network:
    1. Photo-rates use chi (Draine) not G0 (Habing)
    2. Semi-infinite slab factor of 1/2 at surface
    3. AV extinction uses benchmark gamma_d values
    4. Only H/He/C/O species (metals removed)
    
    The returned network expects G0 inputs in Habing units
    but internally converts to chi for rate calculations.
    """
    from prism3d.chemistry.network import (
        Reaction, ReactionType, ChemicalNetwork, build_pdr_network
    )
    
    # Start with the full network
    reactions = build_pdr_network()
    
    # Filter out metal reactions (S, Si, Fe) for pure benchmark
    metal_species = {'S', 'S+', 'Si', 'Si+', 'Fe', 'Fe+', 
                     'CS', 'HCS+', 'N', 'N+', 'CN', 'HCN'}
    
    benchmark_reactions = []
    for rxn in reactions:
        reactant_set = set(rxn.reactants)
        product_set = set(rxn.products)
        all_species = reactant_set | product_set
        
        if all_species & metal_species:
            continue  # Skip metal reactions
        
        # Fix photo-rates to benchmark conventions
        if rxn.name == 'H2_photodiss':
            # benchmark: 5.18e-11 * chi * exp(-3.74 * AV)
            # Our convention: alpha * G0 * exp(-gamma * AV)
            # G0 = chi / 1.71, so alpha_G0 = 5.18e-11 * 1.71 / 2 = 4.43e-11
            # The /2 is the semi-infinite slab factor
            rxn.alpha = 5.18e-11 * 1.71 / 2.0
            rxn.gamma = 3.74
            
        elif rxn.name == 'CO_photodiss':
            rxn.alpha = 2.0e-10 * 1.71 / 2.0
            rxn.gamma = 3.53
            
        elif rxn.name == 'C_photoionization':
            rxn.alpha = 3.0e-10 * 1.71 / 2.0
            rxn.gamma = 3.02
            
        elif rxn.name == 'OH_photodiss':
            rxn.alpha = 3.5e-10 * 1.71 / 2.0
            rxn.gamma = 1.7
            
        elif rxn.name == 'H2O_photodiss':
            rxn.alpha = 5.9e-10 * 1.71 / 2.0
            rxn.gamma = 1.7
            
        elif rxn.name == 'O2_photodiss':
            rxn.alpha = 7.9e-10 * 1.71 / 2.0
            rxn.gamma = 1.8
        
        elif rxn.name == 'CH_photodiss':
            rxn.alpha = 9.2e-10 * 1.71 / 2.0
            rxn.gamma = 1.7
        
        # Fix H2 formation rate to benchmark constant
        elif rxn.name == 'H2_formation_grain':
            rxn.alpha = R_H2_BENCHMARK
            rxn.beta = 0.0  # No T dependence in benchmark
        
        benchmark_reactions.append(rxn)
    
    # Also remove PAH reactions (no PAHs in benchmark)
    benchmark_reactions = [
        r for r in benchmark_reactions 
        if not any(sp.startswith('PAH') for sp in r.reactants + r.products)
    ]
    
    return ChemicalNetwork(benchmark_reactions)
