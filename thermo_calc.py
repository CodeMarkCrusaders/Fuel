"""
NASA 9-Coefficient Thermodynamic Property Calculator.

Calculates Cp, H, S, G from NASA 9-polynomial coefficients.
Reference: NASA TP-2002-211556 (McBride, Gordon, Reno)
"""

import math
from typing import Optional
from nasa9_parser import Species, TempInterval

# Universal gas constant (J/(mol·K))
R_UNIVERSAL = 8.314462618


def _find_interval(species: Species, T: float) -> Optional[TempInterval]:
    """Find the temperature interval containing T for a given species."""
    for iv in species.intervals:
        if iv.T_low <= T <= iv.T_high:
            return iv
    
    # If T is outside all intervals, use the closest one (extrapolation)
    if species.intervals:
        if T < species.intervals[0].T_low:
            return species.intervals[0]
        if T > species.intervals[-1].T_high:
            return species.intervals[-1]
    
    return None


def cp_over_R(species: Species, T: float) -> float:
    """
    Compute Cp/R for a species at temperature T.
    
    NASA 9-coeff polynomial:
    Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
    """
    iv = _find_interval(species, T)
    if iv is None:
        return 2.5  # default for monatomic gas
    
    a = iv.coeffs
    return (a[0] * T**(-2) + a[1] * T**(-1) + a[2] + 
            a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4)


def h_over_RT(species: Species, T: float) -> float:
    """
    Compute H/(RT) for a species at temperature T.
    
    H/(RT) = -a1*T^-2 + a2*ln(T)/T + a3 + a4*T/2 + a5*T^2/3 
             + a6*T^3/4 + a7*T^4/5 + b1/T
    """
    iv = _find_interval(species, T)
    if iv is None:
        return 2.5
    
    a = iv.coeffs
    b1 = iv.integration[0]
    
    return (-a[0] * T**(-2) + a[1] * math.log(T) / T + a[2] + 
            a[3] * T / 2.0 + a[4] * T**2 / 3.0 + 
            a[5] * T**3 / 4.0 + a[6] * T**4 / 5.0 + b1 / T)


def s_over_R(species: Species, T: float) -> float:
    """
    Compute S/R for a species at temperature T (standard state, 1 bar).
    
    S/R = -a1*T^-2/2 - a2*T^-1 + a3*ln(T) + a4*T + a5*T^2/2 
          + a6*T^3/3 + a7*T^4/4 + b2
    """
    iv = _find_interval(species, T)
    if iv is None:
        return 0.0
    
    a = iv.coeffs
    b2 = iv.integration[1]
    
    return (-a[0] * T**(-2) / 2.0 - a[1] * T**(-1) + a[2] * math.log(T) + 
            a[3] * T + a[4] * T**2 / 2.0 + 
            a[5] * T**3 / 3.0 + a[6] * T**4 / 4.0 + b2)


def g_over_RT(species: Species, T: float) -> float:
    """
    Compute dimensionless standard Gibbs function G0/(RT) = H/(RT) - S/R.
    
    This is the standard-state chemical potential divided by RT.
    """
    return h_over_RT(species, T) - s_over_R(species, T)


def cp_J(species: Species, T: float) -> float:
    """Cp in J/(mol·K)."""
    return cp_over_R(species, T) * R_UNIVERSAL


def h_J(species: Species, T: float) -> float:
    """H in J/mol."""
    return h_over_RT(species, T) * R_UNIVERSAL * T


def s_J(species: Species, T: float) -> float:
    """S in J/(mol·K) at standard state (1 bar)."""
    return s_over_R(species, T) * R_UNIVERSAL


def g_J(species: Species, T: float) -> float:
    """G in J/mol at standard state (1 bar)."""
    return g_over_RT(species, T) * R_UNIVERSAL * T


if __name__ == "__main__":
    from nasa9_parser import parse_thermo_file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python thermo_calc.py <thermo.inp> [temperature_K]")
        sys.exit(1)
    
    db = parse_thermo_file(sys.argv[1])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 298.15
    
    print(f"\nThermodynamic properties at T = {T:.2f} K:")
    print(f"{'Species':<12} {'Cp/R':>10} {'H/RT':>10} {'S/R':>10} {'G/RT':>10} "
          f"{'Cp(J/molK)':>12} {'H(kJ/mol)':>12} {'S(J/molK)':>12} {'G(kJ/mol)':>12}")
    print("-" * 120)
    
    for name in ['H2', 'O2', 'N2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'O', 'NO', 'CH4']:
        if name in db:
            sp = db[name]
            cpr = cp_over_R(sp, T)
            hrt = h_over_RT(sp, T)
            sr = s_over_R(sp, T)
            grt = g_over_RT(sp, T)
            print(f"{name:<12} {cpr:10.4f} {hrt:10.4f} {sr:10.4f} {grt:10.4f} "
                  f"{cp_J(sp, T):12.4f} {h_J(sp, T)/1000:12.4f} {s_J(sp, T):12.4f} {g_J(sp, T)/1000:12.4f}")
