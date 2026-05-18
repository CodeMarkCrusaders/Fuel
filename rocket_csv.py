#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Расчёт параметров газа в ракетном сопле собственным равновесным решателем
(NASA-9 + Gibbs minimization) с поддержкой диапазонов, как в Fuel.py.

Это «свой» аналог Fuel.py — но вместо CEA_Wrap используется собственный
модуль ``nozzle_flow.py`` (HP/SP задачи + поиск горловины).

Что считает:
- Давление в камере / на срезе сопла
- Коэффициент избытка окислителя alpha и phi = 1/alpha
- Стехиометрическое и текущее O/F (по массе)
- Удельный импульс (на срезе) Isp и вакуумный Isp_vac
- Характеристическая скорость C*
- Коэффициент тяги CF
- Относительная площадь среза Ae/At
- ТЕРМОДИНАМИКА ПО СЕЧЕНИЯМ (Injector / Nozzle inlet / Nozzle throat /
  промежуточные / Nozzle exit):
    P, T, H, S, U, Cp_eq, Cv_eq, Gamma_eq, Isentropic_exp, MW, R_spec,
    плотность, скорость звука, скорость, число Маха, Ae/At, mass flux
- ФРАКЦИИ КОМПОНЕНТОВ по сечениям (mole + mass)

Входной CSV (как в Fuel.py — поддерживает одиночные значения И диапазоны):

    oxidizer;fuel;oxidizer_temp_K;fuel_temp_K;
    Pc_MPa[_from/_to/_step];
    Pe_MPa[_from/_to/_step];
    alpha[_from/_to/_step];
    n_intermediate_stations (опц.);
    top_species (опц., 12)

Пример:

    oxidizer;fuel;oxidizer_temp_K;fuel_temp_K;Pc_MPa;Pe_MPa;alpha;n_intermediate_stations;top_species
    O2(L);H2(L);;;10;0.1013;1.0;3;12

(пустые oxidizer_temp_K / fuel_temp_K → берём T_assigned из NASA-базы
для криогенных реагентов типа O2(L), H2(L); либо 298.15 К для газовых.)

Запуск:

    python rocket_csv.py input.csv output.csv

Выходной CSV — semicolon-delimited, UTF-8 with BOM. По одной строке на
комбинацию (PC, Pe, alpha, сечение).  «Сечения» развёрнуты в отдельные
строки: один кейс = N строк (по числу сечений), плюс одна сводная строка
с тяговыми характеристиками — её label = "SUMMARY".
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from decimal import Decimal, InvalidOperation, getcontext
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# наш равновесный решатель
from nasa9_parser import Species, parse_thermo_file
from equilibrium import find_thermo_db
from nozzle_flow import (
    Propellant,
    RocketPerformance,
    StationResult,
    solve_rocket_nozzle,
)
from iteration_logger import IterationLogger, NullLogger

getcontext().prec = 28

R_UNIVERSAL_J_KMOL_K = 8314.46261815324  # Дж/(кмоль·К)


# ─────────────────────────────────────────────────────────────────────────────
# Хелперы парсинга CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_float(value, default: Optional[float] = None) -> float:
    """Гибкий парсер float: понимает запятую как десятичный разделитель."""
    if value is None or str(value).strip() == "":
        if default is not None:
            return default
        raise ValueError("Пустое числовое значение")
    s = str(value).strip().replace(",", ".")
    return float(s)


def parse_int_optional(value, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    s = str(value).strip()
    return int(float(s))  # принимаем '3' и '3.0'


def parse_decimal(value) -> Decimal:
    if value is None or str(value).strip() == "":
        raise ValueError("Пустое числовое значение")
    s = str(value).strip().replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"Некорректное число: {value!r}") from e


def has_value(row: dict, key: str) -> bool:
    return key in row and row[key] is not None and str(row[key]).strip() != ""


def pick(row: dict, *keys: str):
    for key in keys:
        if has_value(row, key):
            return row[key]
    raise KeyError(f"Не найден ни один из столбцов: {keys}")


def decimal_range(start: Decimal, stop: Decimal, step: Decimal) -> List[float]:
    """Аналог range() для Decimal с защитой от плавающей погрешности."""
    if step == 0:
        raise ValueError("Шаг диапазона не может быть 0")

    values: List[float] = []
    x = start
    eps = Decimal("1e-12")

    if start < stop and step < 0:
        raise ValueError("Для возрастающего диапазона шаг должен быть > 0")
    if start > stop and step > 0:
        raise ValueError("Для убывающего диапазона шаг должен быть < 0")

    if step > 0:
        while x <= stop + eps:
            values.append(float(x))
            x += step
    else:
        while x >= stop - eps:
            values.append(float(x))
            x += step

    return values


def get_values_from_row(
    row: dict,
    single_key: str,
    from_key: str,
    to_key: str,
    step_key: str,
) -> List[float]:
    """Достаёт либо единственное значение, либо диапазон from/to/step."""
    if has_value(row, single_key):
        return [parse_float(row[single_key])]

    if has_value(row, from_key) and has_value(row, to_key) and has_value(row, step_key):
        start = parse_decimal(row[from_key])
        stop = parse_decimal(row[to_key])
        step = parse_decimal(row[step_key])
        return decimal_range(start, stop, step)

    raise KeyError(
        f"Для параметра не найдено ни одиночное значение '{single_key}', "
        f"ни полный диапазон '{from_key}/{to_key}/{step_key}'"
    )


def fmt(value) -> str:
    """Аккуратное форматирование значений в CSV."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Базовая работа с CSV-файлами
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_rows(path: Path):
    """Читаем входной CSV, автоопределяя разделитель ';' или ','."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ";"

        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            if not any(str(v).strip() for v in row.values() if v is not None):
                continue
            yield row


def expand_cases_from_input_row(row: dict) -> List[dict]:
    """Развёртывает одну строку входного CSV в список расчётных кейсов."""
    oxidizer = str(pick(row, "oxidizer", "Окислитель")).strip()
    fuel = str(pick(row, "fuel", "Горючее")).strip()

    # температуры — опциональные; None означает «возьми из базы / 298.15»
    def _opt_T(val):
        if val is None or str(val).strip() == "":
            return None
        return parse_float(val)

    oxidizer_temp_k = _opt_T(row.get("oxidizer_temp_K"))
    fuel_temp_k = _opt_T(row.get("fuel_temp_K"))

    pc_values = get_values_from_row(
        row,
        single_key="Pc_MPa",
        from_key="Pc_MPa_from",
        to_key="Pc_MPa_to",
        step_key="Pc_MPa_step",
    )
    pe_values = get_values_from_row(
        row,
        single_key="Pe_MPa",
        from_key="Pe_MPa_from",
        to_key="Pe_MPa_to",
        step_key="Pe_MPa_step",
    )
    alpha_values = get_values_from_row(
        row,
        single_key="alpha",
        from_key="alpha_from",
        to_key="alpha_to",
        step_key="alpha_step",
    )

    n_intermediate = parse_int_optional(row.get("n_intermediate_stations"), default=0)
    top_species = parse_int_optional(row.get("top_species"), default=12)

    cases: List[dict] = []
    for pc_mpa, pe_mpa, alpha in product(pc_values, pe_values, alpha_values):
        cases.append({
            "oxidizer": oxidizer,
            "fuel": fuel,
            "oxidizer_temp_K": oxidizer_temp_k,
            "fuel_temp_K": fuel_temp_k,
            "Pc_MPa": pc_mpa,
            "Pe_MPa": pe_mpa,
            "alpha": alpha,
            "n_intermediate_stations": n_intermediate,
            "top_species": top_species,
        })
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Один кейс → запуск своего решателя
# ─────────────────────────────────────────────────────────────────────────────

def _split_masses_by_alpha(
    oxidizer_name: str,
    fuel_name: str,
    alpha: float,
    species_db: Dict[str, Species],
    total_mass_kg: float = 1.0,
) -> tuple:
    """По alpha и стехиометрическому O/F определяет массы компонентов.

    Возвращает (mass_ox_kg, mass_fu_kg, of_actual, of_stoich).
    Если стехиометрия не определяется — fallback в 1:1.
    """
    from nozzle_flow import stoichiometric_OF
    try:
        ox_sp = species_db[oxidizer_name]
        fu_sp = species_db[fuel_name]
        of_stoich = stoichiometric_OF([ox_sp], [fu_sp])
    except KeyError:
        raise ValueError(
            f"Реагент не найден в базе NASA-9: "
            f"oxidizer={oxidizer_name!r}, fuel={fuel_name!r}"
        )
    if not (of_stoich and math.isfinite(of_stoich) and of_stoich > 0):
        raise ValueError(
            f"Не удалось определить стехиометрическое O/F для "
            f"{oxidizer_name}/{fuel_name}"
        )
    of_actual = alpha * of_stoich
    # mass_ox + mass_fu = total;  mass_ox / mass_fu = of_actual
    mass_fu = total_mass_kg / (1.0 + of_actual)
    mass_ox = total_mass_kg - mass_fu
    return mass_ox, mass_fu, of_actual, of_stoich


def calculate_case(
    case: dict,
    species_db: Dict[str, Species],
    logger: Optional[IterationLogger] = None,
) -> RocketPerformance:
    """Запуск нашего решателя для одного кейса (Pc, Pe, alpha)."""
    pc_mpa = case["Pc_MPa"]
    pe_mpa = case["Pe_MPa"]
    alpha = case["alpha"]

    if pc_mpa <= 0:
        raise ValueError("Pc должно быть > 0")
    if pe_mpa <= 0:
        raise ValueError("Pe должно быть > 0")
    if pe_mpa >= pc_mpa:
        raise ValueError("Должно выполняться Pe < Pc")
    if alpha <= 0:
        raise ValueError("alpha должно быть > 0")

    mass_ox, mass_fu, _of_act, _of_st = _split_masses_by_alpha(
        case["oxidizer"], case["fuel"], alpha, species_db, total_mass_kg=1.0,
    )

    ox = Propellant(name=case["oxidizer"], mass_kg=mass_ox, T_K=case.get("oxidizer_temp_K"))
    fu = Propellant(name=case["fuel"], mass_kg=mass_fu, T_K=case.get("fuel_temp_K"))

    perf = solve_rocket_nozzle(
        oxidizer=ox, fuel=fu,
        P_chamber=pc_mpa * 1e6,
        P_exit=pe_mpa * 1e6,
        species_db=species_db,
        n_intermediate_stations=case.get("n_intermediate_stations", 0),
        include_condensed=True,
        verbose=False,
        logger=logger if logger is not None else NullLogger(),
    )
    return perf


# ─────────────────────────────────────────────────────────────────────────────
# Сбор данных по сечениям в строки CSV
# ─────────────────────────────────────────────────────────────────────────────

# Параметры, описывающие весь кейс (повторяются в каждой строке-сечении).
CASE_HEADER_FIELDS = [
    "Глобальный_ID_варианта",
    "Строка_входного_CSV",
    "Номер_варианта",
    "Окислитель",
    "Горючее",
    "T_окислителя_К",
    "T_горючего_К",
    "Pc_задано_МПа",
    "Pe_задано_МПа",
    "Коэффициент_избытка_окислителя_alpha",
    "phi",
    "Стехиометрическое_соотношение_O_F",
    "Текущее_соотношение_O_F",
    "Удельный_импульс_с",
    "Удельный_импульс_вакуум_с",
    "Характеристическая_скорость_Cstar_м_с",
    "Коэффициент_тяги_CF",
]

# Параметры конкретного сечения (заполняются в каждой строке).
STATION_FIELDS = [
    "Сечение",
    "Метка",
    "Давление_МПа",
    "Температура_К",
    "Энтальпия_кДж_кг",
    "Энтропия_кДж_кгК",
    "Внутренняя_энергия_кДж_кг",
    "Cp_eq_кДж_кгК",
    "Cv_eq_кДж_кгК",
    "Показатель_адиабаты_Gamma_eq",
    "Изэнтропический_показатель",
    "Молекулярная_масса_кг_кмоль",
    "Удельная_газовая_постоянная_Дж_кгК",
    "Плотность_кг_м3",
    "Скорость_звука_м_с",
    "Скорость_потока_м_с",
    "Число_Маха",
    "Относительная_площадь_Ae_At",
    "Удельный_массовый_расход_кг_м2с",
]

ERROR_FIELD = "Ошибка"


def _station_to_row(
    st: StationResult,
    section_index: int,
    species_names: Sequence[str],
    top_species: int,
) -> dict:
    """Превращает StationResult + список топ-N компонентов в dict для CSV."""
    row: Dict[str, object] = {
        "Сечение": section_index,
        "Метка": st.label,
        "Давление_МПа": st.P_Pa / 1e6,
        "Температура_К": st.T_K,
        "Энтальпия_кДж_кг": st.H_J_per_kg / 1000.0,
        "Энтропия_кДж_кгК": st.S_J_per_kgK / 1000.0,
        "Внутренняя_энергия_кДж_кг": st.U_J_per_kg / 1000.0,
        "Cp_eq_кДж_кгК": st.cp_eq_J_per_kgK / 1000.0,
        "Cv_eq_кДж_кгК": st.cv_eq_J_per_kgK / 1000.0,
        "Показатель_адиабаты_Gamma_eq": st.gamma_eq,
        "Изэнтропический_показатель": st.gamma_s,
        "Молекулярная_масса_кг_кмоль": st.mw_g_per_mol,
        "Удельная_газовая_постоянная_Дж_кгК": st.R_specific_J_per_kgK,
        "Плотность_кг_м3": st.rho_kg_per_m3,
        "Скорость_звука_м_с": st.a_m_per_s,
        "Скорость_потока_м_с": st.V_m_per_s,
        "Число_Маха": st.M,
        "Относительная_площадь_Ae_At": st.Ae_At,
        "Удельный_массовый_расход_кг_м2с": st.mass_flux_kg_per_m2_s,
    }
    # фракции компонентов (только перечисленные top_species — те же что
    # пишутся в шапке файла)
    for sp_name in species_names[:top_species]:
        try:
            idx = list(st.species_names).index(sp_name)
        except ValueError:
            row[f"x_{sp_name}"] = ""
            row[f"w_{sp_name}"] = ""
            continue
        x = float(st.mole_fractions[idx]) if st.mole_fractions is not None else 0.0
        w = float(st.mass_fractions[idx]) if st.mass_fractions is not None else 0.0
        row[f"x_{sp_name}"] = x
        row[f"w_{sp_name}"] = w
    return row


def _select_top_species_globally(
    perfs: List[RocketPerformance],
    top_species: int,
) -> List[str]:
    """Выбирает top_species компонентов по сумме мольных долей по всем
    сечениям всех успешных кейсов (так столбцы будут одни и те же)."""
    score: Dict[str, float] = {}
    for perf in perfs:
        for st in perf.stations:
            if st.species_names is None or st.mole_fractions is None:
                continue
            for name, x in zip(st.species_names, st.mole_fractions):
                if x is None:
                    continue
                score[name] = score.get(name, 0.0) + float(x)
    if not score:
        return []
    # сортировка по убыванию суммарной мольной доли
    names_sorted = sorted(score.keys(), key=lambda k: -score[k])
    return names_sorted[:top_species]


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция: читаем CSV, считаем все кейсы, пишем CSV
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    input_csv: Path,
    output_csv: Path,
    thermo_db_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    """Главный батч-обработчик."""
    # 1) загружаем NASA-9 базу один раз
    if not quiet:
        print(f"[1/4] Загружаем базу NASA-9 ...", file=sys.stderr)
    db_path = thermo_db_path if thermo_db_path is not None else Path(find_thermo_db())
    species_db = parse_thermo_file(str(db_path))
    if not quiet:
        print(f"      {len(species_db)} веществ загружено.", file=sys.stderr)

    # 2) читаем входной CSV → плоский список кейсов
    if not quiet:
        print(f"[2/4] Читаем входной CSV: {input_csv}", file=sys.stderr)
    all_cases: List[dict] = []
    case_meta: List[dict] = []  # row_idx, local_idx, global_idx для каждого кейса
    global_idx = 0
    parse_errors: List[dict] = []
    for source_row_index, row in enumerate(read_csv_rows(input_csv), start=1):
        try:
            cases = expand_cases_from_input_row(row)
        except Exception as e:
            parse_errors.append({
                "Глобальный_ID_варианта": "",
                "Строка_входного_CSV": source_row_index,
                "Номер_варианта": "",
                "Окислитель": row.get("oxidizer", row.get("Окислитель", "")),
                "Горючее": row.get("fuel", row.get("Горючее", "")),
                ERROR_FIELD: f"Ошибка разбора диапазона: {e}",
            })
            continue
        for local_idx, case in enumerate(cases, start=1):
            global_idx += 1
            all_cases.append(case)
            case_meta.append({
                "Глобальный_ID_варианта": global_idx,
                "Строка_входного_CSV": source_row_index,
                "Номер_варианта": local_idx,
            })

    if not quiet:
        print(f"      Найдено кейсов: {len(all_cases)}  "
              f"(ошибок разбора: {len(parse_errors)})", file=sys.stderr)

    # 3) считаем все кейсы
    if not quiet:
        print(f"[3/4] Запускаем расчёты ...", file=sys.stderr)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    successes: List[tuple] = []   # (meta, case, perf)
    failures: List[dict] = []     # уже готовые строки ошибок

    t0 = time.time()
    for k, (meta, case) in enumerate(zip(case_meta, all_cases), start=1):
        if not quiet:
            print(f"      [{k}/{len(all_cases)}] "
                  f"{case['oxidizer']}/{case['fuel']}  "
                  f"Pc={case['Pc_MPa']} МПа  Pe={case['Pe_MPa']} МПа  "
                  f"α={case['alpha']}", file=sys.stderr)
        try:
            if log_dir is not None:
                log_path = log_dir / f"case_{meta['Глобальный_ID_варианта']:04d}.log"
                with IterationLogger(str(log_path)) as logger:
                    perf = calculate_case(case, species_db, logger=logger)
            else:
                perf = calculate_case(case, species_db, logger=None)
            successes.append((meta, case, perf))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            failures.append({
                **meta,
                "Окислитель": case["oxidizer"],
                "Горючее": case["fuel"],
                "T_окислителя_К": case.get("oxidizer_temp_K"),
                "T_горючего_К": case.get("fuel_temp_K"),
                "Pc_задано_МПа": case["Pc_MPa"],
                "Pe_задано_МПа": case["Pe_MPa"],
                "Коэффициент_избытка_окислителя_alpha": case["alpha"],
                ERROR_FIELD: f"{e!s}  ::  {tb.splitlines()[-1] if tb else ''}",
            })

    dt = time.time() - t0
    if not quiet:
        print(f"      Готово за {dt:.1f} с.  "
              f"Успешно: {len(successes)},  с ошибками: {len(failures)}",
              file=sys.stderr)

    # 4) выбираем глобальный список «топ-N» веществ для столбцов CSV
    top_species_count = max(
        (c.get("top_species", 12) for c in all_cases), default=12,
    )
    top_species_count = max(int(top_species_count), 0)
    perfs_only = [p for _m, _c, p in successes]
    top_names = _select_top_species_globally(perfs_only, top_species_count)

    species_cols: List[str] = []
    for name in top_names:
        species_cols.append(f"x_{name}")
        species_cols.append(f"w_{name}")

    # Полный список колонок: шапка кейса + параметры сечения + фракции + ошибка
    fieldnames = (
        CASE_HEADER_FIELDS
        + STATION_FIELDS
        + species_cols
        + [ERROR_FIELD]
    )

    # 5) генерируем строки для каждого сечения
    if not quiet:
        print(f"[4/4] Пишем выходной CSV: {output_csv}", file=sys.stderr)

    out_rows: List[dict] = []

    # сначала — ошибки разбора (одна строка каждая)
    for er in parse_errors:
        out_rows.append({fn: er.get(fn, "") for fn in fieldnames})

    # успешные кейсы — по строке на сечение
    for meta, case, perf in successes:
        common = {
            "Глобальный_ID_варианта": meta["Глобальный_ID_варианта"],
            "Строка_входного_CSV": meta["Строка_входного_CSV"],
            "Номер_варианта": meta["Номер_варианта"],
            "Окислитель": case["oxidizer"],
            "Горючее": case["fuel"],
            "T_окислителя_К": case.get("oxidizer_temp_K"),
            "T_горючего_К": case.get("fuel_temp_K"),
            "Pc_задано_МПа": case["Pc_MPa"],
            "Pe_задано_МПа": case["Pe_MPa"],
            "Коэффициент_избытка_окислителя_alpha": perf.alpha,
            "phi": perf.phi,
            "Стехиометрическое_соотношение_O_F": perf.O_F_stoich,
            "Текущее_соотношение_O_F": perf.O_F,
            "Удельный_импульс_с": perf.Isp_s,
            "Удельный_импульс_вакуум_с": perf.Isp_vac_s,
            "Характеристическая_скорость_Cstar_м_с": perf.Cstar_m_per_s,
            "Коэффициент_тяги_CF": perf.CF,
        }
        for i, st in enumerate(perf.stations, start=1):
            station_row = _station_to_row(st, i, top_names, top_species_count)
            full = {**common, **station_row, ERROR_FIELD: ""}
            out_rows.append({fn: full.get(fn, "") for fn in fieldnames})

    # неуспешные — одна строка
    for fr in failures:
        out_rows.append({fn: fr.get(fn, "") for fn in fieldnames})

    # 6) пишем
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: fmt(r.get(k, "")) for k in fieldnames})

    if not quiet:
        print(f"      Записано строк: {len(out_rows)}", file=sys.stderr)
        print(f"\nГотово.  Результат: {output_csv}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Расчёт параметров газа в ракетном сопле собственным "
            "равновесным решателем (NASA-9 + Gibbs).  Поддерживает "
            "диапазоны Pc/Pe/alpha и записывает результаты по сечениям "
            "в CSV (semicolon-delimited)."
        ),
    )
    parser.add_argument("input_csv", help="Путь к входному CSV")
    parser.add_argument("output_csv", help="Путь к выходному CSV")
    parser.add_argument(
        "--thermo-db", default=None,
        help="Путь к файлу NASA-9 thermo (если не указан — ищется автоматически)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Каталог для журнальных файлов решателя (по одному на кейс).",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Не печатать прогресс.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        print(f"Ошибка: входной файл не найден: {input_csv}", file=sys.stderr)
        return 2

    process_file(
        input_csv=input_csv,
        output_csv=output_csv,
        thermo_db_path=Path(args.thermo_db) if args.thermo_db else None,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
