"""
NASA 9-Coefficient Thermodynamic Database Parser.

Parses the CEA-format thermo.inp file containing NASA 9-polynomial
thermodynamic data for chemical species.
"""

import re
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TempInterval:
    """Single temperature interval with NASA9 coefficients."""
    T_low: float
    T_high: float
    n_coeff: int
    exponents: List[float]
    coeffs: List[float]       # a1..a7 (or fewer)
    integration: List[float]  # b1, b2 (enthalpy and entropy integration constants)


@dataclass 
class Species:
    """Chemical species with thermodynamic data."""
    name: str
    description: str
    n_intervals: int
    reference: str
    elements: Dict[str, float]   # element_symbol -> atom_count
    phase: int                   # 0=gas, 1=solid/crystal, 2=liquid
    mol_weight: float            # g/mol
    hf298: float                 # J/mol (heat of formation at 298.15 K)
    intervals: List[TempInterval] = field(default_factory=list)
    is_reactant_only: bool = False

    @property
    def is_gas(self) -> bool:
        return self.phase == 0

    @property
    def is_condensed(self) -> bool:
        return self.phase != 0

    @property
    def phase_str(self) -> str:
        return {0: "gas", 1: "solid", 2: "liquid"}.get(self.phase, "unknown")


def _parse_fortran_float(s: str) -> float:
    """Parse a Fortran-style float (D exponent -> E exponent)."""
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace('D', 'E').replace('d', 'e')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_coeff_line(line: str) -> List[float]:
    """Parse a line of 5 NASA9 coefficients (each 16 chars wide)."""
    coeffs = []
    padded = line.ljust(80)
    for i in range(5):
        segment = padded[i*16:(i+1)*16]
        coeffs.append(_parse_fortran_float(segment))
    return coeffs


def parse_thermo_file(filepath: str) -> Dict[str, Species]:
    """
    Parse NASA9 thermo.inp file.
    
    Returns a dict mapping species name -> Species object.
    Only species with valid thermodynamic data (n_intervals > 0) are included.
    Species from the "reactants only" section are marked accordingly.
    """
    species_dict: Dict[str, Species] = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the start of thermodynamic data (after "thermo" line)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'thermo':
            start_idx = i + 2  # skip "thermo" and the global T range line
            break
    
    in_reactants_section = False
    idx = start_idx
    
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('!'):
            idx += 1
            continue
        
        # Check for section markers
        if stripped.startswith('END PRODUCTS'):
            in_reactants_section = True
            idx += 1
            continue
        if stripped.startswith('END REACTANTS'):
            break
        
        # --- Parse species header (line 1) ---
        species_name = line[:18].strip()
        description = line[18:].strip()
        
        if not species_name:
            idx += 1
            continue
        
        idx += 1
        if idx >= len(lines):
            break
        
        # --- Parse composition line (line 2) ---
        comp_line = lines[idx]
        if len(comp_line) < 50:
            idx += 1
            continue
        
        padded = comp_line.ljust(80)
        
        try:
            n_intervals = int(padded[0:2].strip())
        except (ValueError, IndexError):
            idx += 1
            continue
        
        reference = padded[3:9].strip()
        
        # Parse 5 element pairs (each 8 chars starting at position 10)
        elements = {}
        for i in range(5):
            start = 10 + i * 8
            sym = padded[start:start+2].strip()
            try:
                count = float(padded[start+2:start+8].strip())
            except (ValueError, IndexError):
                count = 0.0
            if sym and abs(count) > 1e-10 and sym != 'E':
                # Skip electron entries (E) as they are for ions
                elements[sym.upper()] = count
        
        # Phase
        try:
            phase = int(padded[50:52].strip())
        except (ValueError, IndexError):
            phase = 0
        
        # Molecular weight
        try:
            mol_weight = float(padded[52:65].strip())
        except (ValueError, IndexError):
            mol_weight = 0.0
        
        # Heat of formation at 298.15 K (J/mol)
        try:
            hf298 = float(padded[65:80].strip())
        except (ValueError, IndexError):
            hf298 = 0.0
        
        sp = Species(
            name=species_name,
            description=description,
            n_intervals=n_intervals,
            reference=reference,
            elements=elements,
            phase=phase,
            mol_weight=mol_weight,
            hf298=hf298,
            is_reactant_only=in_reactants_section
        )
        
        idx += 1
        
        # --- Parse temperature intervals ---
        for interval_idx in range(n_intervals):
            if idx >= len(lines):
                break
            
            # Temperature range line
            t_line = lines[idx].ljust(80)
            
            try:
                T_low = float(t_line[0:11].strip())
                T_high = float(t_line[11:22].strip())
                n_coeff = int(t_line[22:23].strip()) if t_line[22:23].strip() else 7
            except (ValueError, IndexError):
                idx += 1
                continue
            
            # Parse exponents (8 values, each 5 chars)
            exponents = []
            for i in range(8):
                start = 24 + i * 5
                try:
                    exponents.append(float(t_line[start:start+5].strip()))
                except (ValueError, IndexError):
                    exponents.append(0.0)
            
            idx += 1
            if idx >= len(lines):
                break
            
            # Coefficient line 1: a1..a5
            coeff_line1 = _parse_coeff_line(lines[idx])
            idx += 1
            
            if idx >= len(lines):
                break
            
            # Coefficient line 2: a6, a7, [empty], b1, b2
            coeff_line2 = _parse_coeff_line(lines[idx])
            idx += 1
            
            # Build coefficient arrays
            # Standard NASA9: 7 coefficients + 2 integration constants
            coeffs = coeff_line1[:5] + coeff_line2[:2]  # a1..a7
            integration = [coeff_line2[3], coeff_line2[4]]  # b1, b2
            
            interval = TempInterval(
                T_low=T_low,
                T_high=T_high,
                n_coeff=n_coeff,
                exponents=exponents,
                coeffs=coeffs,
                integration=integration
            )
            sp.intervals.append(interval)
        
        # Store species (skip duplicates, keep first occurrence)
        if species_name not in species_dict and n_intervals > 0:
            species_dict[species_name] = sp
    
    return species_dict


def get_products_for_elements(
    species_dict: Dict[str, Species],
    element_set: set,
    include_condensed: bool = True,
    T: float = None
) -> List[Species]:
    """
    Select all product species whose elements are a subset of the given element set.
    
    Args:
        species_dict: Parsed species database
        element_set: Set of element symbols present in reactants
        include_condensed: Whether to include condensed (solid/liquid) species
        T: Temperature (K) for filtering valid temperature ranges
    
    Returns:
        List of candidate product species
    """
    products = []
    for name, sp in species_dict.items():
        # Skip reactant-only species
        if sp.is_reactant_only:
            continue
        
        # Skip ions (species with 'E' in elements from original data or +/- in name)
        if '+' in name or '-' in name:
            continue
            
        # Skip electron
        if name == 'e-':
            continue
        
        # Check if all elements of this species are in our set
        sp_elements = set(sp.elements.keys())
        if not sp_elements.issubset(element_set):
            continue
        
        # Skip condensed species if not requested
        if sp.is_condensed and not include_condensed:
            continue
        
        # Check temperature range validity if T provided
        if T is not None and sp.intervals:
            T_min = min(iv.T_low for iv in sp.intervals)
            T_max = max(iv.T_high for iv in sp.intervals)
            if T < T_min - 100 or T > T_max + 100:
                continue
        
        products.append(sp)
    
    return products


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python nasa9_parser.py <thermo.inp>")
        sys.exit(1)
    
    db = parse_thermo_file(sys.argv[1])
    print(f"Parsed {len(db)} species with valid thermodynamic data.")
    
    # Show a few examples
    for name in ['H2O', 'O2', 'N2', 'CO2', 'CH4', 'H2', 'CO', 'OH']:
        if name in db:
            sp = db[name]
            print(f"\n{sp.name}: MW={sp.mol_weight:.4f}, Hf298={sp.hf298:.1f} J/mol, "
                  f"phase={sp.phase_str}, elements={sp.elements}")
            for iv in sp.intervals:
                print(f"  T: {iv.T_low:.1f} - {iv.T_high:.1f} K, "
                      f"{iv.n_coeff} coefficients")
