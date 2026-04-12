#!/usr/bin/env python3
"""
Главный модуль расчёта химического равновесия.

Метод: минимизация энергии Гиббса (алгоритм Гордона–Макбрайда, NASA CEA).
База данных: NASA Glenn Research Center, 9-коэффициентные полиномы.

Запуск в интерактивном режиме:
    python equilibrium.py

Запуск из командной строки (пакетный режим):
    python equilibrium.py --reactants "2H2 + O2" --temperature 3000 --pressure "1 atm"

Авторы: Fuel_Equilibrium Project
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


# Путь к базе данных по умолчанию (рядом со скриптом)
DEFAULT_THERMO_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thermo.inp')


def find_thermo_db() -> str:
    """
    Ищет файл thermo.inp в нескольких возможных местах.

    Порядок поиска:
      1. Текущая рабочая директория
      2. Директория скрипта
      3. Поддиректория data/ рядом со скриптом
    """
    if os.path.exists('thermo.inp'):
        return 'thermo.inp'
    if os.path.exists(DEFAULT_THERMO_DB):
        return DEFAULT_THERMO_DB
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'thermo.inp')
    if os.path.exists(data_path):
        return data_path

    raise FileNotFoundError(
        "Не найден файл thermo.inp. Поместите его в текущую директорию "
        "или рядом с этим скриптом."
    )


def select_candidate_species(
    species_db: Dict[str, Species],
    element_set: set,
    T: float,
    include_condensed: bool = True,
    max_gas_species: int = 50,
    verbose: bool = False,
) -> List[Species]:
    """
    Автоматически выбирает вещества-кандидаты для расчёта равновесия.

    Стратегия отбора:
      1. Берём все вещества, состоящие только из нужных элементов.
      2. Для газов вычисляем G⁰/RT при заданной температуре.
      3. Сначала добавляем «обязательные» вещества: одноатомные, двухатомные
         и часто встречающиеся в горении.
      4. Дополняем список остальными газами в порядке возрастания G⁰/RT
         (термодинамически выгодные идут первыми) до достижения лимита.
      5. Добавляем все подходящие конденсированные фазы.

    Параметры:
        species_db:        База данных веществ
        element_set:       Набор элементов реагентов (например {'C', 'H', 'O'})
        T:                 Температура, К
        include_condensed: Учитывать конденсат
        max_gas_species:   Максимальное число газовых компонентов
        verbose:           Печатать список выбранных веществ
    """
    all_candidates = get_products_for_elements(
        species_db, element_set, include_condensed=include_condensed, T=T
    )

    gas_candidates = [sp for sp in all_candidates if sp.is_gas]
    cond_candidates = [sp for sp in all_candidates if sp.is_condensed]

    # Вычисляем G⁰/RT для каждого газового кандидата
    gas_with_gibbs = []
    for sp in gas_candidates:
        try:
            g = g_over_RT(sp, T)
        except Exception:
            g = 1e10  # если не удалось вычислить, ставим заведомо большое значение
        gas_with_gibbs.append((sp, g))

    # Сортируем: меньшее G⁰/RT означает большую термодинамическую стабильность
    gas_with_gibbs.sort(key=lambda pair: pair[1])

    # Формируем набор «обязательных» имён: элементарные формы и распространённые молекулы
    essential_names = set()
    for elem in element_set:
        essential_names.add(elem)        # одноатомный: H, O, N, C
        essential_names.add(f"{elem}2")  # двухатомный: H2, O2, N2

    # Наиболее важные вещества в реакциях горения
    common_combustion_species = {
        'H2O', 'CO2', 'CO', 'OH', 'NO', 'HO2', 'H2O2', 'NH3',
        'HCN', 'HCO', 'CH4', 'CH3', 'CH2O', 'N2O', 'NO2', 'O3',
        'HCHO,formaldehy', 'C2H2,acetylene', 'C2H4', 'C2H6',
        'HCOOH', 'HNO', 'NH2',
    }
    essential_names.update(common_combustion_species)

    # Шаг 1: добавляем обязательные вещества
    selected = []
    selected_names = set()
    for sp, g in gas_with_gibbs:
        if sp.name in essential_names:
            selected.append(sp)
            selected_names.add(sp.name)

    # Шаг 2: добираем оставшиеся газы по термодинамической выгодности
    for sp, g in gas_with_gibbs:
        if sp.name not in selected_names:
            if len(selected) >= max_gas_species:
                break
            selected.append(sp)
            selected_names.add(sp.name)

    if verbose:
        print(f"\n  Выбрано {len(selected)} газовых и {len(cond_candidates)} конденсированных веществ.")
        if len(selected) <= 60:
            print("  Газовые: " + ", ".join(sp.name for sp in selected))

    return selected + cond_candidates


def parse_pressure(pressure_str: str) -> float:
    """
    Преобразует строку с давлением в Па.

    Принимаемые форматы:
        '1 atm'      -> 101325 Па
        '1.013 bar'  -> 101300 Па
        '101325 Pa'  -> 101325 Па
        '100 kPa'    -> 100000 Па
        '0.5 MPa'    -> 500000 Па
        '14.7 psi'   -> ~101325 Па
        '1'          -> 101325 Па (число без единиц трактуется как атм)
    """
    s = pressure_str.strip().lower()

    unit_table = {
        'atm': 101325.0,
        'bar': 1e5,
        'kpa': 1e3,
        'mpa': 1e6,
        'psi': 6894.757,
        'pa':  1.0,
    }

    for unit, factor in unit_table.items():
        if s.endswith(unit):
            value_str = s[: -len(unit)].strip()
            return float(value_str) * factor

    # Нет единиц измерения — считаем атмосферами
    try:
        return float(s) * 101325.0
    except ValueError:
        raise ValueError(
            f"Не удалось разобрать давление: '{pressure_str}'. "
            "Примеры: '1 atm', '1.013 bar', '101325 Pa'."
        )


def print_result(result: EquilibriumResult, species_db: Dict[str, Species]) -> None:
    """Выводит результаты расчёта в виде форматированной таблицы."""

    print("\n" + "=" * 80)
    print("         РЕЗУЛЬТАТЫ РАСЧЁТА ХИМИЧЕСКОГО РАВНОВЕСИЯ")
    print("=" * 80)

    print(f"\n  Температура:  {result.T:.2f} К ({result.T - 273.15:.2f} °C)")
    print(f"  Давление:     {result.P:.2f} Па  "
          f"({result.P / 101325:.6f} атм,  {result.P / 1e5:.6f} бар)")
    status = "Да ✓" if result.converged else "Нет ✗"
    print(f"  Сходимость:   {status}  "
          f"({result.iterations} итераций, невязка: {result.residual:.2e})")

    # Элементный состав
    print(f"\n  Элементный состав (моль):")
    for element, moles in sorted(result.elements.items()):
        print(f"    {element:>4s}: {moles:12.6f}")

    # Газовая фаза
    gas_components = result.get_gas_species()
    if gas_components:
        print(f"\n  {'─' * 76}")
        print(f"  ГАЗОВАЯ ФАЗА  (всего молей: {result.total_moles:.6f})")
        print(f"  {'─' * 76}")
        print(f"  {'Компонент':<20s} {'Моль':<16s} {'Мол. доля':<16s} {'Масс. доля':<16s}")
        print(f"  {'─' * 76}")

        # Считаем суммарную массу для расчёта массовых долей
        total_mass = 0.0
        species_mass = {}
        for name, moles, xi in gas_components:
            mw = species_db[name].mol_weight if name in species_db else 28.0
            mass = moles * mw
            species_mass[name] = mass
            total_mass += mass

        for name, moles, xi in gas_components:
            if xi > 1e-10:
                mass_fraction = species_mass[name] / total_mass if total_mass > 0 else 0.0
                print(f"  {name:<20s} {moles:<16.8e} {xi:<16.8e} {mass_fraction:<16.8e}")

        print(f"  {'─' * 76}")
        print(f"  {'Итого':<20s} {result.total_moles:<16.8e} {'1.00000000':<16s}")

        avg_molar_mass = total_mass / result.total_moles if result.total_moles > 0 else 0.0
        print(f"\n  Средняя молярная масса газа: {avg_molar_mass:.4f} г/моль")

    # Конденсированная фаза
    condensed_components = result.get_condensed_species()
    if condensed_components:
        print(f"\n  {'─' * 76}")
        print(f"  КОНДЕНСИРОВАННАЯ ФАЗА")
        print(f"  {'─' * 76}")
        print(f"  {'Компонент':<25s} {'Моль':<16s} {'Фаза':<10s}")
        print(f"  {'─' * 76}")

        phase_names = {0: "газ", 1: "тв.", 2: "жидк."}
        for name, moles in condensed_components:
            idx = result.species_names.index(name)
            phase_str = phase_names.get(result.phase[idx], "?")
            print(f"  {name:<25s} {moles:<16.8e} {phase_str:<10s}")

    print(f"\n{'=' * 80}")


def run_interactive():
    """Запускает программу в интерактивном режиме с диалоговым вводом."""

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    РАСЧЁТ ХИМИЧЕСКОГО РАВНОВЕСИЯ МЕТОДОМ МИНИМИЗАЦИИ ЭНЕРГИИ ГИББСА  ║")
    print("║    База данных NASA 9-коэффициентных полиномов                       ║")
    print("║    Алгоритм Гордона–Макбрайда (NASA CEA)                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Загружаем базу данных
    print("\n  Загрузка термодинамической базы данных...")
    try:
        db_path = find_thermo_db()
        species_db = parse_thermo_file(db_path)
        n_gas = sum(1 for sp in species_db.values() if sp.is_gas)
        n_cond = sum(1 for sp in species_db.values() if sp.is_condensed)
        print(f"  ✓ Загружено {len(species_db)} веществ: {n_gas} газовых, {n_cond} конденсированных")
    except FileNotFoundError as e:
        print(f"  ✗ Ошибка: {e}")
        sys.exit(1)

    while True:
        print("\n" + "─" * 70)

        # Ввод реагентов
        print("\n  Введите реагенты (левая часть уравнения реакции).")
        print("  Примеры:")
        print("    2H2 + O2")
        print("    CH4 + 2O2")
        print("    C2H5OH + 3O2 + 11.28N2")
        print("  ('q' для выхода)\n")

        reactant_input = input("  Реагенты: ").strip()
        if reactant_input.lower() in ('q', 'quit', 'exit', 'выход'):
            print("\n  До свидания!")
            break

        if not reactant_input:
            print("  ✗ Ничего не введено. Попробуйте снова.")
            continue

        # Разбираем введённые реагенты
        try:
            components = parse_reaction_string(reactant_input)
            total_elements = get_total_elements(components)
        except Exception as e:
            print(f"  ✗ Ошибка разбора формулы: {e}")
            continue

        print(f"\n  Реагенты:")
        for coeff, formula, elems in components:
            print(f"    {coeff:.4g} × {formula}  ({elems})")
        print(f"  Суммарный состав: {total_elements}")

        # Ввод температуры
        try:
            T = float(input("\n  Температура (К): ").strip())
            if T <= 0:
                raise ValueError("Температура должна быть положительной.")
        except ValueError as e:
            print(f"  ✗ Ошибка: {e}")
            continue

        # Ввод давления
        print("  Давление (примеры: '1 atm', '1 bar', '101325 Pa', число = атм):")
        try:
            P = parse_pressure(input("  Давление: ").strip())
        except ValueError as e:
            print(f"  ✗ Ошибка: {e}")
            continue

        # Дополнительные параметры
        include_condensed_input = input(
            "\n  Учитывать конденсированную фазу? (y/n, по умолчанию y): "
        ).strip().lower()
        include_condensed = include_condensed_input not in ('n', 'no', 'нет')

        verbose_input = input(
            "  Подробный лог итераций? (y/n, по умолчанию n): "
        ).strip().lower()
        verbose = verbose_input in ('y', 'yes', 'да')

        # Отбор кандидатов
        element_set = set(total_elements.keys())
        print(f"\n  Подбираем вещества-продукты для элементов: {element_set}...")

        candidates = select_candidate_species(
            species_db, element_set, T,
            include_condensed=include_condensed,
            verbose=True,
        )

        if not candidates:
            print("  ✗ Не найдено подходящих веществ в базе данных!")
            continue

        # Решаем задачу равновесия
        print(f"\n  Решаем задачу равновесия при T = {T:.2f} К, P = {P:.2f} Па "
              f"({P / 101325:.4f} атм)...")

        result = solve_equilibrium(
            species_list=candidates,
            element_abundances=total_elements,
            T=T,
            P=P,
            include_condensed=include_condensed,
            verbose=verbose,
        )

        print_result(result, species_db)

        print("\n  Нажмите Enter для нового расчёта или введите 'q' для выхода.")


def run_batch(
    reactants: str,
    T: float,
    P: float,
    thermo_db_path: str = None,
    include_condensed: bool = True,
    verbose: bool = False,
) -> EquilibriumResult:
    """
    Выполняет расчёт равновесия без интерактивного ввода (пакетный режим).

    Удобна для вызова из других скриптов или Jupyter-блокнота.

    Параметры:
        reactants:        Строка реагентов, например "2H2 + O2"
        T:                Температура, К
        P:                Давление, Па
        thermo_db_path:   Путь к thermo.inp (если None — ищем автоматически)
        include_condensed: Учитывать конденсированные фазы
        verbose:          Выводить подробный лог

    Возвращает объект EquilibriumResult.
    """
    if thermo_db_path is None:
        thermo_db_path = find_thermo_db()

    species_db = parse_thermo_file(thermo_db_path)

    components = parse_reaction_string(reactants)
    total_elements = get_total_elements(components)
    element_set = set(total_elements.keys())

    candidates = select_candidate_species(
        species_db, element_set, T,
        include_condensed=include_condensed,
        verbose=verbose,
    )

    result = solve_equilibrium(
        species_list=candidates,
        element_abundances=total_elements,
        T=T,
        P=P,
        include_condensed=include_condensed,
        verbose=verbose,
    )

    if verbose:
        print_result(result, species_db)

    return result


# ---------------------------------------------------------------------------
# Точка входа при запуске скрипта напрямую
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(
        description="Расчёт химического равновесия методом минимизации энергии Гиббса"
    )
    cli.add_argument('--reactants', '-r', type=str, default=None,
                     help='Реагенты, например "2H2 + O2"')
    cli.add_argument('--temperature', '-T', type=float, default=None,
                     help='Температура, К')
    cli.add_argument('--pressure', '-P', type=str, default=None,
                     help='Давление, например "1 atm", "1 bar", "101325 Pa"')
    cli.add_argument('--thermo-db', type=str, default=None,
                     help='Путь к файлу thermo.inp')
    cli.add_argument('--no-condensed', action='store_true',
                     help='Исключить конденсированные фазы')
    cli.add_argument('--verbose', '-v', action='store_true',
                     help='Подробный лог итераций')

    args = cli.parse_args()

    if args.reactants and args.temperature and args.pressure:
        # Пакетный режим: все параметры заданы через флаги
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
        # Интерактивный режим
        run_interactive()
