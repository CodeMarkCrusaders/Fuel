#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPA-подобный интерактивный CLI: ВЫБОР ТОПЛИВНОЙ ПАРЫ + РЕЖИМ ЗАДАНИЯ СООТНОШЕНИЯ.

Пользователь:
    1) выбирает популярную пару из каталога  ИЛИ  вводит окислитель и горючее
       отдельно (с подсказками по аббревиатурам типа LOX, LH2, НДМГ, MMH, NTO, …);
    2) задаёт давления в любых поддерживаемых единицах
       (МПа, Па, атм, кгс/см², бар, фунт/дюйм² (psi), kPa, mmHg);
    3) выбирает способ задания соотношения окислитель/горючее:
       • O/F          — массовое соотношение
       • α (alpha)    — коэффициент избытка окислителя
       • оптимальное  — авто-поиск оптимума по Isp / Isp_vac / C* / T_camera;
    4) получает таблицу параметров по сечениям сопла в стиле РПА.

CLI поддерживает и пакетный режим (через argparse) — для скриптов.

Запуск:
    python rpa_cli.py                       # интерактивный
    python rpa_cli.py --catalog             # показать каталог и выйти
    python rpa_cli.py --batch-csv input.csv out.csv     # из CSV
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from nasa9_parser import parse_thermo_file
from equilibrium import find_thermo_db
from nozzle_flow import Propellant, solve_rocket_nozzle, print_nozzle_table
from iteration_logger import IterationLogger, NullLogger
from units import parse_pressure, SUPPORTED_PRESSURE_UNITS, convert_pressure_pa_to
from propellants_catalog import (
    OXIDIZERS, FUELS, POPULAR_PAIRS,
    list_oxidizers, list_fuels, list_popular_pairs,
    resolve_propellant, print_catalog,
)
from propellant_optimizer import (
    RatioSpec, find_optimal_OF, print_optimization_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Хелперы интерактивного ввода
# ─────────────────────────────────────────────────────────────────────────────

def ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else (default or "")


def ask_pressure(prompt: str, default: str) -> float:
    """Спрашивает давление с поддержкой единиц. По умолчанию — МПа."""
    while True:
        s = ask(prompt, default)
        try:
            return parse_pressure(s, default_unit="MPa")
        except Exception as e:
            print(f"  ❌ {e}.  Поддерживаются: {', '.join(SUPPORTED_PRESSURE_UNITS)}, "
                  f"а также русские (МПа, кПа, Па, бар, атм, кгс/см², фунт/дюйм²).")


def ask_choice(prompt: str, options: list, default_index: int = 1) -> int:
    """Спрашивает номер варианта из списка."""
    while True:
        s = ask(prompt, str(default_index))
        try:
            n = int(s)
            if 1 <= n <= len(options):
                return n - 1
        except ValueError:
            pass
        print(f"  ❌ Введите число от 1 до {len(options)}.")


def ask_float(prompt: str, default: float, positive: bool = True) -> float:
    while True:
        s = ask(prompt, f"{default:g}")
        try:
            v = float(s.replace(",", "."))
            if positive and v <= 0:
                print("  ❌ Значение должно быть > 0.")
                continue
            return v
        except ValueError:
            print("  ❌ Не удалось разобрать число.")


def ask_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    while True:
        s = ask(prompt, str(default))
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
            print(f"  ❌ Должно быть в диапазоне [{lo}, {hi}].")
        except ValueError:
            print("  ❌ Не удалось разобрать число.")


# ─────────────────────────────────────────────────────────────────────────────
# Выбор топливной пары
# ─────────────────────────────────────────────────────────────────────────────

def select_propellant_pair() -> Tuple[str, str, str, str]:
    """Возвращает (oxidizer_name, fuel_name, ox_display, fu_display).

    Имена — как в базе NASA-9; display — человеко-читаемые.
    """
    print()
    print("─" * 70)
    print("  ВЫБОР ТОПЛИВНОЙ ПАРЫ")
    print("─" * 70)
    print("  1) Выбрать ПОПУЛЯРНУЮ комбинацию из списка")
    print("  2) Указать ОКИСЛИТЕЛЬ и ГОРЮЧЕЕ отдельно (можно по аббревиатуре)")
    print("  3) Показать ПОЛНЫЙ КАТАЛОГ окислителей и горючих")

    mode = ask_choice("Способ выбора (1/2/3)", [1, 2, 3], default_index=1)

    if mode == 0:  # популярные комбинации
        print()
        print("  ПОПУЛЯРНЫЕ ТОПЛИВНЫЕ ПАРЫ:")
        for i, p in enumerate(POPULAR_PAIRS, 1):
            print(f"    [{i:2d}]  {p.name:<32s}  — {p.notes}")
        idx = ask_choice("\n  Номер пары", POPULAR_PAIRS, default_index=1)
        pair = POPULAR_PAIRS[idx]

        # вытащим display-названия
        ox_entry, _ = resolve_propellant(pair.oxidizer, kind="oxidizer")
        fu_entry, _ = resolve_propellant(pair.fuel, kind="fuel")
        ox_display = ox_entry.display if ox_entry else pair.oxidizer
        fu_display = fu_entry.display if fu_entry else pair.fuel
        return pair.oxidizer, pair.fuel, ox_display, fu_display

    if mode == 2:  # полный каталог
        print_catalog()

    # mode == 1 или 2 — спрашиваем отдельно
    print("\n  Подсказки: LOX, LH2, LCH4, RP-1, NTO, NDMH/Гептил, MMH, N2H4, H2O2, …")

    while True:
        ox_input = ask("Окислитель (имя или аббревиатура)", "LOX")
        ox_entry, ox_name = resolve_propellant(ox_input, kind="oxidizer")
        if not ox_name:
            print("  ❌ Пустое имя окислителя.")
            continue
        ox_display = ox_entry.display if ox_entry else ox_name
        if ox_entry is None:
            print(f"  ⚠ Не нашёл «{ox_input}» в каталоге; буду пробовать передать в базу NASA-9 как «{ox_name}».")
        else:
            print(f"  ✓ Окислитель распознан: {ox_display}  →  {ox_name}")
        break

    while True:
        fu_input = ask("Горючее (имя или аббревиатура)", "LH2")
        fu_entry, fu_name = resolve_propellant(fu_input, kind="fuel")
        if not fu_name:
            print("  ❌ Пустое имя горючего.")
            continue
        fu_display = fu_entry.display if fu_entry else fu_name
        if fu_entry is None:
            print(f"  ⚠ Не нашёл «{fu_input}» в каталоге; буду пробовать передать в базу NASA-9 как «{fu_name}».")
        else:
            print(f"  ✓ Горючее распознано: {fu_display}  →  {fu_name}")
        break

    return ox_name, fu_name, ox_display, fu_display


# ─────────────────────────────────────────────────────────────────────────────
# Выбор режима задания соотношения
# ─────────────────────────────────────────────────────────────────────────────

def select_ratio_mode() -> RatioSpec:
    print()
    print("─" * 70)
    print("  СПОСОБ ЗАДАНИЯ СООТНОШЕНИЯ КОМПОНЕНТОВ")
    print("─" * 70)
    print("  1) Задать массовое соотношение  O/F  =  m_окислителя / m_горючего")
    print("  2) Задать коэффициент избытка окислителя  α  =  (O/F) / (O/F)_стех.")
    print("  3) АВТО-поиск ОПТИМАЛЬНОГО значения (по Isp / Isp_vac / C* / T_camera)")

    choice = ask_choice("Режим (1/2/3)", [1, 2, 3], default_index=3)

    if choice == 0:  # OF
        of = ask_float("O/F (массовое соотношение)", default=4.5)
        return RatioSpec(mode="OF", value=of)

    if choice == 1:  # alpha
        alpha = ask_float("α  (1.0 = стехиометрия,  0.5 = 50%, 1.5 = +50% окислителя)",
                          default=1.0)
        return RatioSpec(mode="alpha", value=alpha)

    # optimal
    print()
    print("  ЦЕЛЕВАЯ ФУНКЦИЯ для оптимизации:")
    print("    1) Isp        — удельный импульс на срезе (Ve/g0)   [по умолчанию]")
    print("    2) Isp_vac    — вакуумный удельный импульс")
    print("    3) Cstar      — характеристическая скорость C*")
    print("    4) T_chamber  — температура в камере сгорания")
    t_idx = ask_choice("Цель (1..4)", [1, 2, 3, 4], default_index=1)
    target = ["Isp", "Isp_vac", "Cstar", "T_chamber"][t_idx]

    print()
    print("  Диапазон сканирования по α:")
    a_min = ask_float("  α_min", default=0.3)
    a_max = ask_float("  α_max", default=1.6)
    n_grid = ask_int("  число точек сетки", default=13, lo=5, hi=200)

    return RatioSpec(
        mode="optimal", target=target,
        alpha_min=a_min, alpha_max=a_max,
        n_grid=n_grid, refine=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Печать сводки + одна точка
# ─────────────────────────────────────────────────────────────────────────────

def _print_pressure_in_units(P_pa: float, prefix: str) -> None:
    """Выводит давление сразу в нескольких единицах для удобства."""
    parts = []
    for u in ("Pa", "kPa", "MPa", "bar", "atm", "kgf/cm2", "psi"):
        v = convert_pressure_pa_to(P_pa, u)
        # форматируем разумно
        if abs(v) >= 1e5:
            parts.append(f"{v:.2e} {u}")
        elif abs(v) >= 1:
            parts.append(f"{v:.4f} {u}")
        else:
            parts.append(f"{v:.4g} {u}")
    print(f"  {prefix}: {parts[0]}")
    for p in parts[1:]:
        print(f"  {' ' * len(prefix)}  = {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Основной интерактивный сценарий
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive() -> None:
    print()
    print("=" * 70)
    print("   ИНТЕРАКТИВНЫЙ РАСЧЁТ РАКЕТНОГО СОПЛА  (RPA-style)")
    print("=" * 70)
    print("  Минимизация G + равновесное течение по сечениям сопла.")
    print()

    print("Загружаю базу NASA-9 ...", end="", flush=True)
    db = parse_thermo_file(find_thermo_db())
    print(f" OK  ({len(db)} веществ)")

    # 1) Топливная пара
    ox_name, fu_name, ox_display, fu_display = select_propellant_pair()

    # проверка наличия в базе
    if ox_name not in db:
        print(f"\n  ⚠ ВНИМАНИЕ: окислитель «{ox_name}» НЕ найден в базе NASA-9.")
        print(f"     Проверьте имя; возможно, нужен суффикс (L), (cr), и т.п.")
    if fu_name not in db:
        print(f"\n  ⚠ ВНИМАНИЕ: горючее «{fu_name}» НЕ найден в базе NASA-9.")

    # 2) Давления
    print()
    print("─" * 70)
    print("  ДАВЛЕНИЯ  (можно указать с единицами: 10 MPa, 100 atm, 1500 psi, 100 кгс/см² ...)")
    print("─" * 70)
    Pc_pa = ask_pressure("Давление в камере  Pc", default="10 MPa")
    while True:
        Pe_pa = ask_pressure("Давление на срезе  Pe", default="1 atm")
        if Pe_pa < Pc_pa:
            break
        print("  ❌ Должно выполняться Pe < Pc — повторите ввод.")

    print()
    _print_pressure_in_units(Pc_pa, "Pc")
    print()
    _print_pressure_in_units(Pe_pa, "Pe")

    # 3) Соотношение
    spec = select_ratio_mode()

    # 4) Доп. параметры
    print()
    print("─" * 70)
    print("  ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ")
    print("─" * 70)
    n_inter = ask_int("Промежуточных сечений между горловиной и срезом", default=3, lo=0, hi=50)
    log_path = ask("Файл лога итераций (Enter — без лога)", "")

    # 5) Расчёт
    print()
    print("=" * 70)
    print("  РАСЧЁТ...")
    print("=" * 70)
    t0 = time.time()
    logger_cm = IterationLogger(log_path) if log_path else NullLogger()
    try:
        # NullLogger не контекстный менеджер ↓ привести к одному виду:
        if log_path:
            logger_cm.__enter__()
            logger = logger_cm
        else:
            logger = logger_cm

        if spec.mode == "optimal":
            res = find_optimal_OF(
                oxidizer_name=ox_name, fuel_name=fu_name,
                spec=spec,
                P_chamber_Pa=Pc_pa, P_exit_Pa=Pe_pa,
                species_db=db,
                n_intermediate_stations=n_inter,
                logger=logger,
            )
            print_optimization_summary(res)
            print_nozzle_table(res.perf)
            dt = time.time() - t0
            print(f"  Время расчёта: {dt:.1f} с,  расчётов сопла: {res.n_calls}")
        else:
            # один расчёт: используем тот же оптимизатор в режиме OF/alpha
            res = find_optimal_OF(
                oxidizer_name=ox_name, fuel_name=fu_name,
                spec=spec,
                P_chamber_Pa=Pc_pa, P_exit_Pa=Pe_pa,
                species_db=db,
                n_intermediate_stations=n_inter,
                logger=logger,
            )
            print_nozzle_table(res.perf)
            print()
            print(f"  ── Сводка ──────────────────────────────────────────")
            print(f"     Окислитель      : {ox_display}  ({ox_name})")
            print(f"     Горючее         : {fu_display}  ({fu_name})")
            print(f"     O/F (массовое)  : {res.OF:.4f}")
            print(f"     α               : {res.alpha:.4f}")
            print(f"     O/F стехиом.    : {res.OF_stoich:.4f}")
            print(f"     Isp (срез)      : {res.perf.Isp_s:.3f} с")
            print(f"     Isp (вакуум)    : {res.perf.Isp_vac_s:.3f} с")
            print(f"     C*              : {res.perf.Cstar_m_per_s:.2f} м/с")
            print(f"     CF              : {res.perf.CF:.4f}")
            dt = time.time() - t0
            print(f"  Время расчёта: {dt:.1f} с")
    finally:
        if log_path:
            logger_cm.__exit__(None, None, None)

    if log_path:
        print(f"  Журнал итераций: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="RPA-style интерактивный CLI для расчёта ракетного сопла",
    )
    p.add_argument("--catalog", action="store_true",
                   help="Показать каталог топливных пар и выйти.")
    p.add_argument("--batch-csv", nargs=2, metavar=("INPUT_CSV", "OUTPUT_CSV"),
                   help="Запустить batch-расчёт из CSV (см. rocket_csv.py)")
    p.add_argument("--log-dir", default=None,
                   help="Каталог логов для batch-режима.")
    args = p.parse_args()

    if args.catalog:
        print_catalog()
        return 0

    if args.batch_csv:
        from rocket_csv import process_file
        input_csv = Path(args.batch_csv[0])
        output_csv = Path(args.batch_csv[1])
        process_file(
            input_csv=input_csv, output_csv=output_csv,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
        return 0

    try:
        run_interactive()
    except (EOFError, KeyboardInterrupt):
        print("\n  Прервано пользователем.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
