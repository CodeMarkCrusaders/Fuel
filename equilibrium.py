#!/usr/bin/env python3
# расчёт химического равновесия (метод минимизации энергии Гиббса)
# база данных: NASA CEA, 9-коэффициентные полиномы
#
# запуск интерактивно:  python equilibrium.py
# пакетный режим:       python equilibrium.py -r "2H2 + O2" -T 3000 -P "1 atm"

import os
import sys
import math
import numpy as np
from typing import Dict, List, Optional

from nasa9_parser import parse_thermo_file, get_products_for_elements, Species
from thermo_calc import g_over_RT, R_UNIVERSAL
from formula_parser import parse_reaction_string, get_total_elements
from gibbs_solver import solve_equilibrium, EquilibriumResult


# папка с базами данных лежит рядом со скриптом
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_THERMO_DB = os.path.join(DATA_DIR, 'thermo.inp')


def find_thermo_db() -> str:
    # ищем thermo.inp: сначала в текущей папке, потом в data/
    if os.path.exists('thermo.inp'):
        return 'thermo.inp'
    if os.path.exists(DEFAULT_THERMO_DB):
        return DEFAULT_THERMO_DB
    raise FileNotFoundError(
        "Не найден thermo.inp. Положите файл в папку data/ рядом со скриптом."
    )


def select_candidate_species(
    species_db: Dict[str, Species],
    element_set: set,
    T: float,
    include_condensed: bool = True,
    max_gas: int = 50,
    verbose: bool = False,
) -> List[Species]:
    """Выбирает вещества-кандидаты для расчёта равновесия.

    Сначала добавляем «обязательные» (одноатомные, двухатомные, основные
    продукты горения), остальные добираем по убыванию G0/RT до лимита.
    """
    all_cands = get_products_for_elements(
        species_db, element_set, include_condensed=include_condensed, T=T
    )
    gas_cands = [sp for sp in all_cands if sp.is_gas]
    cond_cands = [sp for sp in all_cands if sp.is_condensed]

    # считаем G0/RT и сортируем (меньше = выгоднее)
    gas_g = []
    for sp in gas_cands:
        try:
            gas_g.append((sp, g_over_RT(sp, T)))
        except Exception:
            gas_g.append((sp, 1e10))
    gas_g.sort(key=lambda x: x[1])

    # набор обязательных веществ
    must_have = set()
    for el in element_set:
        must_have.add(el)
        must_have.add(el + '2')
    must_have.update({
        'H2O', 'CO2', 'CO', 'OH', 'NO', 'HO2', 'H2O2', 'NH3',
        'HCN', 'HCO', 'CH4', 'CH3', 'CH2O', 'N2O', 'NO2', 'O3',
        'HCHO,formaldehy', 'C2H2,acetylene', 'C2H4', 'C2H6',
        'HCOOH', 'HNO', 'NH2',
    })

    selected, seen = [], set()
    # сначала обязательные
    for sp, g in gas_g:
        if sp.name in must_have:
            selected.append(sp)
            seen.add(sp.name)
    # потом остальные по G0
    for sp, g in gas_g:
        if sp.name not in seen:
            if len(selected) >= max_gas:
                break
            selected.append(sp)
            seen.add(sp.name)

    if verbose:
        print(f"  газов: {len(selected)}, конденсата: {len(cond_cands)}")

    return selected + cond_cands


def parse_pressure(s: str) -> float:
    """Преобразует строку давления в Па. Без единиц — считается атм."""
    s = s.strip().lower()
    units = {'atm': 101325.0, 'bar': 1e5, 'kpa': 1e3, 'mpa': 1e6, 'psi': 6894.757, 'pa': 1.0}
    for unit, factor in units.items():
        if s.endswith(unit):
            return float(s[:-len(unit)].strip()) * factor
    try:
        return float(s) * 101325.0  # нет единиц — атм
    except ValueError:
        raise ValueError(f"Непонятное давление: '{s}'. Пример: '1 atm', '2 bar', '101325 Pa'")


def print_result(result: EquilibriumResult, species_db: Dict[str, Species]) -> None:
    print("\n" + "=" * 70)
    print("  РЕЗУЛЬТАТЫ РАСЧЁТА ХИМИЧЕСКОГО РАВНОВЕСИЯ")
    print("=" * 70)

    print(f"\n  T = {result.T:.2f} К  ({result.T-273.15:.2f} °C)")
    print(f"  P = {result.P:.0f} Па  ({result.P/101325:.5f} атм)")
    conv = "сошлось ✓" if result.converged else "не сошлось ✗"
    print(f"  {conv}  (итераций: {result.iterations}, невязка: {result.residual:.2e})")

    print(f"\n  Элементный состав (моль):")
    for el, n in sorted(result.elements.items()):
        print(f"    {el}: {n:.6f}")

    gas = result.get_gas_species()
    if gas:
        # считаем массовые доли
        total_mass = 0.0
        masses = {}
        for name, moles, xi in gas:
            mw = species_db[name].mol_weight if name in species_db else 28.0
            masses[name] = moles * mw
            total_mass += masses[name]

        print(f"\n  ГАЗОВАЯ ФАЗА  (молей: {result.total_moles:.6f})")
        print(f"  {'Компонент':<20} {'моль':>14} {'мол.доля':>14} {'масс.доля':>14}")
        print("  " + "-" * 64)
        for name, moles, xi in gas:
            if xi > 1e-10:
                wf = masses[name] / total_mass if total_mass > 0 else 0.0
                print(f"  {name:<20} {moles:>14.6e} {xi:>14.6e} {wf:>14.6e}")
        print("  " + "-" * 64)
        print(f"  {'Итого':<20} {result.total_moles:>14.6e}")

        avg_mw = total_mass / result.total_moles if result.total_moles > 0 else 0.0
        print(f"\n  Средняя М газа: {avg_mw:.3f} г/моль")

    cond = result.get_condensed_species()
    if cond:
        print(f"\n  КОНДЕНСАТ:")
        phases = {0: "газ", 1: "тв.", 2: "жидк."}
        for name, moles in cond:
            idx = result.species_names.index(name)
            print(f"  {name:<25} {moles:.6e}  {phases.get(result.phase[idx], '?')}")

    print("\n" + "=" * 70)


def run_interactive():
    print("=" * 70)
    print("  Расчёт химического равновесия (минимизация Гиббса, NASA CEA)")
    print("=" * 70)

    print("\n  Загружаем базу данных...")
    try:
        db_path = find_thermo_db()
        species_db = parse_thermo_file(db_path)
        ng = sum(1 for sp in species_db.values() if sp.is_gas)
        nc = sum(1 for sp in species_db.values() if sp.is_condensed)
        print(f"  Загружено: {len(species_db)} веществ ({ng} газов, {nc} конденс.)")
    except FileNotFoundError as e:
        print(f"  Ошибка: {e}")
        sys.exit(1)

    while True:
        print("\n" + "-" * 70)
        print("  Введите реагенты (левая часть уравнения реакции):")
        print("  Примеры: 2H2 + O2  /  CH4 + 2O2  /  C2H5OH + 3O2 + 11.28N2")
        print("  (q — выход)\n")

        s = input("  Реагенты: ").strip()
        if s.lower() in ('q', 'quit', 'exit', 'выход'):
            print("  Выход.")
            break
        if not s:
            continue

        try:
            components = parse_reaction_string(s)
            total_elements = get_total_elements(components)
        except Exception as e:
            print(f"  Ошибка разбора: {e}")
            continue

        print("  Распознано:")
        for coeff, formula, elems in components:
            print(f"    {coeff:.4g} × {formula}")
        print(f"  Элементы: {total_elements}")

        try:
            T = float(input("\n  Температура (К): ").strip())
            if T <= 0:
                raise ValueError("температура должна быть > 0")
        except ValueError as e:
            print(f"  Ошибка: {e}")
            continue

        try:
            P = parse_pressure(input("  Давление (напр. '1 atm', '1 bar', '101325 Pa'): ").strip())
        except ValueError as e:
            print(f"  Ошибка: {e}")
            continue

        inc_cond = input("\n  Учитывать конденсат? (y/n, по умолч. y): ").strip().lower()
        include_condensed = inc_cond not in ('n', 'no', 'нет')

        verbose = input("  Подробный лог? (y/n, по умолч. n): ").strip().lower() in ('y', 'yes', 'да')

        print(f"\n  Ищем вещества-продукты...")
        candidates = select_candidate_species(
            species_db, set(total_elements.keys()), T,
            include_condensed=include_condensed, verbose=True
        )
        if not candidates:
            print("  Нет подходящих веществ в базе!")
            continue

        print(f"  Решаем задачу равновесия при T={T:.0f} К, P={P:.0f} Па...")
        result = solve_equilibrium(
            species_list=candidates,
            element_abundances=total_elements,
            T=T, P=P,
            include_condensed=include_condensed,
            verbose=verbose,
        )
        print_result(result, species_db)

        print("\n  Enter — новый расчёт, q — выход.")


def run_batch(
    reactants: str,
    T: float,
    P: float,
    thermo_db_path: str = None,
    include_condensed: bool = True,
    verbose: bool = False,
) -> EquilibriumResult:
    """Пакетный режим — вызов из кода или скрипта."""
    if thermo_db_path is None:
        thermo_db_path = find_thermo_db()

    species_db = parse_thermo_file(thermo_db_path)
    components = parse_reaction_string(reactants)
    total_elements = get_total_elements(components)

    candidates = select_candidate_species(
        species_db, set(total_elements.keys()), T,
        include_condensed=include_condensed, verbose=verbose,
    )
    result = solve_equilibrium(
        species_list=candidates,
        element_abundances=total_elements,
        T=T, P=P,
        include_condensed=include_condensed,
        verbose=verbose,
    )
    if verbose:
        print_result(result, species_db)
    return result


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Расчёт химического равновесия")
    cli.add_argument('--reactants', '-r', type=str)
    cli.add_argument('--temperature', '-T', type=float)
    cli.add_argument('--pressure', '-P', type=str)
    cli.add_argument('--thermo-db', type=str)
    cli.add_argument('--no-condensed', action='store_true')
    cli.add_argument('--verbose', '-v', action='store_true')
    args = cli.parse_args()

    if args.reactants and args.temperature and args.pressure:
        P = parse_pressure(args.pressure)
        db_path = args.thermo_db or find_thermo_db()
        species_db = parse_thermo_file(db_path)
        result = run_batch(
            reactants=args.reactants,
            T=args.temperature,
            P=P,
            thermo_db_path=args.thermo_db,
            include_condensed=not args.no_condensed,
            verbose=args.verbose,
        )
        print_result(result, species_db)
    else:
        run_interactive()
