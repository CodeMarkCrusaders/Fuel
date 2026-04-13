#!/usr/bin/env python3
# несколько тестовых расчётов для проверки решателя

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nasa9_parser import parse_thermo_file
from equilibrium import run_batch, print_result, find_thermo_db


def run_example(name, reactants, T, P_atm, db):
    P = P_atm * 101325.0
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"# {reactants},  T={T:.0f} К,  P={P_atm} атм")
    print(f"{'#'*70}")

    t0 = time.time()
    result = run_batch(reactants, T=T, P=P, verbose=False)
    print_result(result, db)
    print(f"  время: {time.time()-t0:.2f} с")

    # выводим компоненты с долей > 0.01%
    major = [(n, xi) for n, _, xi in result.get_gas_species() if xi > 1e-4]
    if major:
        print("  Основные компоненты:")
        for n, xi in major:
            print(f"    {n:<20} {xi*100:.4f} %")


def main():
    print("=" * 70)
    print("  Тестовые расчёты химического равновесия")
    print("=" * 70)

    db = parse_thermo_file(find_thermo_db())

    run_example("H2 + O2 при 3000 К",       "2H2 + O2",          3000, 1.0, db)
    run_example("H2 + O2 при 1500 К",       "2H2 + O2",          1500, 1.0, db)
    run_example("CH4 + воздух при 2000 К",  "1CH4 + 2O2 + 7.52N2", 2000, 1.0, db)
    run_example("CO + O2 при 2500 К",       "2CO + O2",          2500, 1.0, db)
    run_example("H2 + O2 при 3000 К, 100 атм", "2H2 + O2",       3000, 100.0, db)
    run_example("N2 диссоциация при 5000 К", "1N2",              5000, 1.0, db)

    print("\n" + "=" * 70)
    print("  Готово.")
    print("=" * 70)


if __name__ == "__main__":
    main()
