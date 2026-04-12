#!/usr/bin/env python3
"""
Example calculations to verify the equilibrium solver.

Runs several well-known equilibrium test cases and displays results.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nasa9_parser import parse_thermo_file
from equilibrium import run_batch, print_result, find_thermo_db


def run_example(name, reactants, T, P_atm, db):
    """Run a single example and print results."""
    P = P_atm * 101325.0
    
    print(f"\n{'#' * 80}")
    print(f"#  Пример: {name}")
    print(f"#  Реагенты: {reactants}")
    print(f"#  T = {T:.0f} K, P = {P_atm} атм")
    print(f"{'#' * 80}")
    
    t0 = time.time()
    result = run_batch(reactants, T=T, P=P, verbose=False)
    elapsed = time.time() - t0
    
    print_result(result, db)
    print(f"\n  Время расчёта: {elapsed:.3f} сек")
    
    gas = result.get_gas_species()
    print(f"\n  Основные газовые компоненты (мольная доля > 0.01%):")
    for sp_name, moles, xi in gas:
        if xi > 0.0001:
            print(f"    {sp_name:<20s} {xi*100:10.4f} %")
    
    return result


def main():
    print("=" * 80)
    print("    ПРИМЕРЫ РАСЧЁТОВ ХИМИЧЕСКОГО РАВНОВЕСИЯ")
    print("    NASA 9-Coefficient Database / Gibbs Minimization")
    print("=" * 80)
    
    db = parse_thermo_file(find_thermo_db())
    
    # Example 1: H2/O2 stoichiometric at 3000K
    run_example(
        "Водород + Кислород (стехиометрия) при 3000K",
        "2H2 + O2", T=3000.0, P_atm=1.0, db=db
    )
    
    # Example 2: H2/O2 at lower temperature
    run_example(
        "Водород + Кислород при 1500K",
        "2H2 + O2", T=1500.0, P_atm=1.0, db=db
    )
    
    # Example 3: CH4/Air combustion
    run_example(
        "Метан + Воздух (стехиометрия) при 2000K",
        "1CH4 + 2O2 + 7.52N2", T=2000.0, P_atm=1.0, db=db
    )
    
    # Example 4: CO/O2 equilibrium
    run_example(
        "Оксид углерода + Кислород при 2500K",
        "2CO + O2", T=2500.0, P_atm=1.0, db=db
    )
    
    # Example 5: H2/O2 at high pressure
    run_example(
        "Водород + Кислород при 3000K, 100 атм",
        "2H2 + O2", T=3000.0, P_atm=100.0, db=db
    )
    
    # Example 6: N2 dissociation
    run_example(
        "Диссоциация азота при 5000K",
        "1N2", T=5000.0, P_atm=1.0, db=db
    )
    
    print("\n" + "=" * 80)
    print("    Все примеры завершены!")
    print("=" * 80)


if __name__ == "__main__":
    main()
