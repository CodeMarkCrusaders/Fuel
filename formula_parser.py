"""
Chemical Formula and Reaction Parser.

Parses chemical formulas like "H2O", "CH4", "C2H5OH"
and reaction expressions like "2H2 + O2", "CH4 + 2O2", "1.5 H2 + 0.5 N2".

Supports:
- Standard formulas: H2O, CO2, CH4, NH3
- Complex formulas: C2H5OH, Ca(OH)2
- Reaction expressions: coefficient * formula separated by '+'
- Parentheses in formulas
"""

import re
from typing import Dict, List, Tuple


def parse_formula(formula: str) -> Dict[str, float]:
    """
    Parse a single chemical formula string into element counts.
    
    Examples:
        'H2O' -> {'H': 2.0, 'O': 1.0}
        'C2H5OH' -> {'C': 2.0, 'H': 6.0, 'O': 1.0}
        'Ca(OH)2' -> {'Ca': 1.0, 'O': 2.0, 'H': 2.0}
    """
    elements: Dict[str, float] = {}
    
    def _parse_group(s: str, multiplier: float = 1.0):
        i = 0
        while i < len(s):
            if s[i] == '(':
                # Find matching closing parenthesis
                depth = 1
                j = i + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                    j += 1
                # j now points after ')'
                inner = s[i+1:j-1]
                # Read number after ')'
                num_str = ''
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    num_str += s[j]
                    j += 1
                group_mult = float(num_str) if num_str else 1.0
                _parse_group(inner, multiplier * group_mult)
                i = j
            elif s[i].isupper():
                # Element symbol
                elem = s[i]
                i += 1
                while i < len(s) and s[i].islower():
                    elem += s[i]
                    i += 1
                # Read number
                num_str = ''
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    num_str += s[i]
                    i += 1
                count = float(num_str) if num_str else 1.0
                elem_upper = elem[0].upper() + elem[1:].upper() if len(elem) > 1 else elem.upper()
                elements[elem_upper] = elements.get(elem_upper, 0.0) + count * multiplier
            else:
                i += 1
    
    _parse_group(formula.strip())
    return elements


def parse_reaction_string(reaction_str: str) -> List[Tuple[float, str, Dict[str, float]]]:
    """
    Parse a reaction string (left side of equation).
    
    Supports formats:
        "2H2 + O2"
        "CH4 + 2 O2" 
        "1.5H2 + 0.5N2"
        "3.76N2 + O2"
        "C2H5OH + 3O2"
    
    Returns:
        List of (coefficient, formula_string, element_dict) tuples
    """
    components = []
    
    # Split by '+'
    parts = reaction_str.split('+')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Try to extract leading coefficient
        # Patterns: "2H2O", "2 H2O", "2.5H2O", "2.5 H2O", "H2O"
        match = re.match(r'^(\d+\.?\d*)\s*([A-Z].*)$', part)
        if match:
            coeff = float(match.group(1))
            formula = match.group(2).strip()
        else:
            coeff = 1.0
            formula = part
        
        elem_dict = parse_formula(formula)
        components.append((coeff, formula, elem_dict))
    
    return components


def get_total_elements(components: List[Tuple[float, str, Dict[str, float]]]) -> Dict[str, float]:
    """
    Calculate total element composition from a list of reaction components.
    
    Args:
        components: List from parse_reaction_string()
    
    Returns:
        Dict of total element moles: {element: total_moles}
    """
    total = {}
    for coeff, formula, elem_dict in components:
        for elem, count in elem_dict.items():
            total[elem] = total.get(elem, 0.0) + coeff * count
    return total


def format_elements(elements: Dict[str, float]) -> str:
    """Format element dict as a readable string."""
    parts = []
    for elem, count in sorted(elements.items()):
        if abs(count - round(count)) < 1e-6:
            parts.append(f"{elem}{int(round(count))}" if round(count) != 1 else elem)
        else:
            parts.append(f"{elem}{count:.4f}")
    return ' '.join(parts)


if __name__ == "__main__":
    # Test cases
    test_formulas = ["H2O", "CO2", "CH4", "C2H5OH", "N2O4", "NH3", "H2SO4"]
    print("Formula parsing tests:")
    for f in test_formulas:
        print(f"  {f} -> {parse_formula(f)}")
    
    print("\nReaction parsing tests:")
    test_reactions = [
        "2H2 + O2",
        "CH4 + 2O2",
        "1.5H2 + 0.5N2",
        "C2H5OH + 3O2",
        "3.76N2 + O2",
    ]
    for r in test_reactions:
        components = parse_reaction_string(r)
        total = get_total_elements(components)
        print(f"  '{r}':")
        for coeff, formula, elems in components:
            print(f"    {coeff} × {formula} = {elems}")
        print(f"    Total elements: {total}")
