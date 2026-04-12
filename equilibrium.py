#!/usr/bin/env python3
"""
Chemical Equilibrium Calculator
================================

Determines equilibrium composition of a gas mixture at specified temperature 
and pressure using Gibbs free energy minimization with NASA 9-coefficient
thermodynamic database.

Method: Gordon-McBride (NASA CEA) algorithm
Database: NASA Glenn Research Center CEA thermodynamic data

Usage:
    python equilibrium.py
    
The program will prompt for:
    1. Reactant species (left side of reaction equation)
    2. Temperature (K)
    3. Pressure (Pa, bar, or atm)

Products are automatically determined from the thermodynamic database
based on the elements present in the reactants.

Author: Fuel_Equilibrium Project
"""

import os
import sys
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from nasa9_parser import parse_thermo_file, get_products_for_elements, Species
from thermo_calc import g_over_RT, h_over_RT, s_over_R, cp_over_R, R_UNIVERSAL, g_J
from formula_parser import parse_reaction_string, get_total_elements, format_elements
from gibbs_solver import solve_equilibrium, EquilibriumResult


# Default path to thermodynamic database
DEFAULT_THERMO_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thermo.inp')


def find_thermo_db() -> str:
    """Locate the thermo.inp database file."""
    # Check current directory
    if os.path.exists('thermo.inp'):
        return 'thermo.inp'
    # Check script directory
    if os.path.exists(DEFAULT_THERMO_DB):
        return DEFAULT_THERMO_DB
    # Check data subdirectory
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'thermo.inp')
    if os.path.exists(data_path):
        return data_path
    raise FileNotFoundError(
        "Cannot find thermo.inp database. Place it in the current directory "
        "or in the same directory as this script."
    )


def select_species_for_equilibrium(
    species_db: Dict[str, Species],
    element_set: set,
    T: float,
    include_condensed: bool = True,
    max_gas_species: int = 50,
    verbose: bool = False
) -> List[Species]:
    """
    Automatically select candidate product species for equilibrium calculation.
    
    Strategy:
    1. Include all gas species that contain only the specified elements
    2. Filter by Gibbs energy (remove species with very high g0/RT)
    3. Sort by thermodynamic favorability (lower G)
    4. Limit total number for computational efficiency
    5. Always include elemental and common species
    """
    candidates = get_products_for_elements(
        species_db, element_set, include_condensed=include_condensed, T=T
    )
    
    gas_cands = [sp for sp in candidates if sp.is_gas]
    cond_cands = [sp for sp in candidates if sp.is_condensed]
    
    # Compute g0/RT for each candidate to filter by thermodynamic relevance
    species_g0 = []
    for sp in gas_cands:
        try:
            g = g_over_RT(sp, T)
            species_g0.append((sp, g))
        except Exception:
            species_g0.append((sp, 1e10))
    
    # Sort by g0/RT (lower = more thermodynamically favorable)
    species_g0.sort(key=lambda x: x[1])
    
    # Always include monatomic and diatomic species (elements)
    essential_names = set()
    for elem in element_set:
        # Monatomic
        essential_names.add(elem)
        # Common diatomics
        essential_names.add(f"{elem}2")
    # Always include common combustion species
    common_species = {'H2O', 'CO2', 'CO', 'OH', 'NO', 'HO2', 'H2O2', 'NH3',
                      'HCN', 'HCO', 'CH4', 'CH3', 'CH2O', 'N2O', 'NO2', 'O3',
                      'HCHO,formaldehy', 'C2H2,acetylene', 'C2H4', 'C2H6',
                      'HCOOH', 'HNO', 'NH2'}
    essential_names.update(common_species)
    
    # Build final list: essential species + top by g0/RT
    selected = []
    selected_names = set()
    
    # First pass: add essential species
    for sp, g in species_g0:
        if sp.name in essential_names:
            selected.append(sp)
            selected_names.add(sp.name)
    
    # Second pass: add remaining by g0/RT until limit
    for sp, g in species_g0:
        if sp.name not in selected_names:
            if len(selected) >= max_gas_species:
                break
            selected.append(sp)
            selected_names.add(sp.name)
    
    gas_cands = selected
    
    if verbose:
        print(f"\nSelected {len(gas_cands)} gas species and {len(cond_cands)} condensed species.")
        if len(gas_cands) <= 60:
            print("Gas species: " + ", ".join(sp.name for sp in gas_cands))
    
    return gas_cands + cond_cands


def parse_pressure(p_str: str) -> float:
    """Parse pressure string with units to Pa."""
    p_str = p_str.strip().lower()
    
    if p_str.endswith('atm'):
        return float(p_str[:-3].strip()) * 101325.0
    elif p_str.endswith('bar'):
        return float(p_str[:-3].strip()) * 1e5
    elif p_str.endswith('kpa'):
        return float(p_str[:-3].strip()) * 1e3
    elif p_str.endswith('mpa'):
        return float(p_str[:-3].strip()) * 1e6
    elif p_str.endswith('psi'):
        return float(p_str[:-3].strip()) * 6894.757
    elif p_str.endswith('pa'):
        return float(p_str[:-2].strip())
    else:
        # Default: assume atm
        try:
            val = float(p_str)
            return val * 101325.0  # assume atm
        except ValueError:
            raise ValueError(f"Cannot parse pressure: '{p_str}'. "
                           f"Use format like '1 atm', '1.013 bar', '101325 Pa'")


def print_result(result: EquilibriumResult, species_db: Dict[str, Species]) -> None:
    """Print equilibrium results in a formatted table."""
    
    print("\n" + "=" * 80)
    print("         РЕЗУЛЬТАТЫ РАСЧЁТА ХИМИЧЕСКОГО РАВНОВЕСИЯ")
    print("=" * 80)
    
    print(f"\n  Температура:  {result.T:.2f} K ({result.T - 273.15:.2f} °C)")
    print(f"  Давление:     {result.P:.2f} Па ({result.P/101325:.6f} атм, {result.P/1e5:.6f} бар)")
    print(f"  Сходимость:   {'Да ✓' if result.converged else 'Нет ✗'} "
          f"({result.iterations} итераций, невязка: {result.residual:.2e})")
    
    # Element composition
    print(f"\n  Элементный состав (моль):")
    for elem, moles in sorted(result.elements.items()):
        print(f"    {elem:>4s}: {moles:12.6f}")
    
    # Gas phase results
    gas_species = result.get_gas_species()
    if gas_species:
        print(f"\n  {'─' * 76}")
        print(f"  ГАЗОВАЯ ФАЗА (всего: {result.total_moles:.6f} моль)")
        print(f"  {'─' * 76}")
        print(f"  {'Компонент':<20s} {'Моль':<15s} {'Мольн.доля':<15s} {'Масс.доля':<15s}")
        print(f"  {'─' * 76}")
        
        # Calculate total mass for mass fractions
        total_mass = 0.0
        species_masses = {}
        for name, moles, xi in gas_species:
            if name in species_db:
                mass = moles * species_db[name].mol_weight
            else:
                mass = moles * 28.0  # fallback
            species_masses[name] = mass
            total_mass += mass
        
        for name, moles, xi in gas_species:
            if xi > 1e-10:
                mass_frac = species_masses[name] / total_mass if total_mass > 0 else 0
                print(f"  {name:<20s} {moles:<15.8e} {xi:<15.8e} {mass_frac:<15.8e}")
        
        # Summary stats
        print(f"  {'─' * 76}")
        print(f"  {'Итого':<20s} {result.total_moles:<15.8e} {'1.00000000':<15s}")
        
        avg_mw = total_mass / result.total_moles if result.total_moles > 0 else 0
        print(f"\n  Средняя молярная масса газовой фазы: {avg_mw:.4f} г/моль")
    
    # Condensed phase results
    cond_species = result.get_condensed_species()
    if cond_species:
        print(f"\n  {'─' * 76}")
        print(f"  КОНДЕНСИРОВАННАЯ ФАЗА")
        print(f"  {'─' * 76}")
        print(f"  {'Компонент':<25s} {'Моль':<15s} {'Фаза':<10s}")
        print(f"  {'─' * 76}")
        
        for name, moles in cond_species:
            idx = result.species_names.index(name)
            phase = {0: "газ", 1: "тв.", 2: "жидк."}.get(result.phase[idx], "?")
            print(f"  {name:<25s} {moles:<15.8e} {phase:<10s}")
    
    print(f"\n{'=' * 80}")


def run_interactive():
    """Run the equilibrium calculator in interactive mode."""
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    РАСЧЁТ ХИМИЧЕСКОГО РАВНОВЕСИЯ МЕТОДОМ МИНИМИЗАЦИИ ЭНЕРГИИ ГИББСА   ║")
    print("║    NASA 9-Coefficient Thermodynamic Database                         ║")
    print("║    Gordon-McBride Algorithm (NASA CEA Method)                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Load database
    print("\n  Загрузка термодинамической базы данных...")
    try:
        db_path = find_thermo_db()
        species_db = parse_thermo_file(db_path)
        n_gas = sum(1 for sp in species_db.values() if sp.is_gas)
        n_cond = sum(1 for sp in species_db.values() if sp.is_condensed)
        print(f"  ✓ Загружено {len(species_db)} видов: {n_gas} газовых, {n_cond} конденсированных")
    except FileNotFoundError as e:
        print(f"  ✗ Ошибка: {e}")
        sys.exit(1)
    
    while True:
        print("\n" + "-" * 70)
        
        # Get reactants
        print("\n  Введите левую часть уравнения реакции (реагенты).")
        print("  Примеры: 2H2 + O2")
        print("           CH4 + 2O2")
        print("           C2H5OH + 3O2 + 11.28N2")
        print("           1H2 + 0.5O2")
        print("  ('q' для выхода)\n")
        
        reactant_str = input("  Реагенты: ").strip()
        if reactant_str.lower() in ('q', 'quit', 'exit', 'выход'):
            print("\n  До свидания!")
            break
        
        if not reactant_str:
            print("  ✗ Пустой ввод. Попробуйте снова.")
            continue
        
        # Parse reactants
        try:
            components = parse_reaction_string(reactant_str)
            total_elements = get_total_elements(components)
        except Exception as e:
            print(f"  ✗ Ошибка разбора формулы: {e}")
            continue
        
        print(f"\n  Реагенты:")
        for coeff, formula, elems in components:
            print(f"    {coeff:.4g} × {formula} ({elems})")
        print(f"  Общий элементный состав: {total_elements}")
        
        # Get temperature
        try:
            T_str = input("\n  Температура (K): ").strip()
            T = float(T_str)
            if T <= 0:
                raise ValueError("Temperature must be positive")
        except ValueError as e:
            print(f"  ✗ Ошибка: {e}")
            continue
        
        # Get pressure
        print("  Давление (примеры: '1 atm', '1 bar', '101325 Pa', или число в атм):")
        try:
            P_str = input("  Давление: ").strip()
            P = parse_pressure(P_str)
        except ValueError as e:
            print(f"  ✗ Ошибка: {e}")
            continue
        
        # Options
        include_condensed_str = input("\n  Учитывать конденсированную фазу? (y/n, default=y): ").strip().lower()
        include_condensed = include_condensed_str not in ('n', 'no', 'нет')
        
        verbose_str = input("  Подробный вывод итераций? (y/n, default=n): ").strip().lower()
        verbose = verbose_str in ('y', 'yes', 'да')
        
        # Select product species
        element_set = set(total_elements.keys())
        print(f"\n  Поиск возможных продуктов для элементов: {element_set}...")
        
        candidate_species = select_species_for_equilibrium(
            species_db, element_set, T,
            include_condensed=include_condensed,
            verbose=True
        )
        
        if not candidate_species:
            print("  ✗ Не найдено подходящих продуктов в базе данных!")
            continue
        
        # Solve equilibrium
        print(f"\n  Решение задачи химического равновесия...")
        print(f"  T = {T:.2f} K, P = {P:.2f} Pa ({P/101325:.4f} atm)")
        
        result = solve_equilibrium(
            species_list=candidate_species,
            element_abundances=total_elements,
            T=T,
            P=P,
            include_condensed=include_condensed,
            verbose=verbose
        )
        
        # Print results
        print_result(result, species_db)
        
        # Ask for another calculation
        print("\n  Нажмите Enter для нового расчёта или 'q' для выхода.")


def run_batch(reactants: str, T: float, P: float, 
              thermo_db_path: str = None,
              include_condensed: bool = True,
              verbose: bool = False) -> EquilibriumResult:
    """
    Run equilibrium calculation in batch mode (non-interactive).
    
    Args:
        reactants: Reaction string (e.g., "2H2 + O2")
        T: Temperature in Kelvin
        P: Pressure in Pascals
        thermo_db_path: Path to thermo.inp file
        include_condensed: Whether to include condensed phases
        verbose: Print iteration details
    
    Returns:
        EquilibriumResult
    """
    if thermo_db_path is None:
        thermo_db_path = find_thermo_db()
    
    species_db = parse_thermo_file(thermo_db_path)
    
    components = parse_reaction_string(reactants)
    total_elements = get_total_elements(components)
    element_set = set(total_elements.keys())
    
    candidate_species = select_species_for_equilibrium(
        species_db, element_set, T,
        include_condensed=include_condensed,
        verbose=verbose
    )
    
    result = solve_equilibrium(
        species_list=candidate_species,
        element_abundances=total_elements,
        T=T,
        P=P,
        include_condensed=include_condensed,
        verbose=verbose
    )
    
    if verbose:
        print_result(result, species_db)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chemical Equilibrium Calculator using Gibbs Energy Minimization"
    )
    parser.add_argument('--reactants', '-r', type=str, default=None,
                       help='Reactant species (e.g., "2H2 + O2")')
    parser.add_argument('--temperature', '-T', type=float, default=None,
                       help='Temperature in Kelvin')
    parser.add_argument('--pressure', '-P', type=str, default=None,
                       help='Pressure (e.g., "1 atm", "1 bar", "101325 Pa")')
    parser.add_argument('--thermo-db', type=str, default=None,
                       help='Path to thermo.inp database file')
    parser.add_argument('--no-condensed', action='store_true',
                       help='Exclude condensed phase species')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with iteration details')
    
    args = parser.parse_args()
    
    if args.reactants and args.temperature and args.pressure:
        # Batch mode
        P = parse_pressure(args.pressure)
        db_path = args.thermo_db or find_thermo_db()
        species_db = parse_thermo_file(db_path)
        result = run_batch(
            reactants=args.reactants,
            T=args.temperature,
            P=P,
            thermo_db_path=args.thermo_db,
            include_condensed=not args.no_condensed,
            verbose=args.verbose
        )
        
        # Print result
        print_result(result, species_db)
    else:
        # Interactive mode
        run_interactive()
