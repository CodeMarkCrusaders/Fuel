"""
Парсер термодинамической базы данных NASA (9-коэффициентный формат).

Читает файл thermo.inp в формате CEA, содержащий полиномиальные данные
NASA-9 для химических веществ. Возвращает словарь объектов Species,
по которым потом считаются Cp, H, S, G.
"""

import re
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TemperatureInterval:
    """
    Один температурный интервал с набором коэффициентов NASA-9.

    Каждое вещество может иметь несколько интервалов, например:
      200–1000 K и 1000–6000 K. Для каждого — свои коэффициенты.
    """
    T_low: float           # нижняя граница интервала, К
    T_high: float          # верхняя граница интервала, К
    n_coeff: int           # количество коэффициентов (обычно 7)
    exponents: List[float] # показатели степеней T (обычно [-2,-1,0,1,2,3,4])
    coeffs: List[float]    # a1..a7 — коэффициенты полинома
    integration: List[float]  # b1, b2 — константы интегрирования (для H и S)


@dataclass
class Species:
    """
    Химическое вещество с термодинамическими данными NASA-9.

    Хранит состав, фазовое состояние, молярную массу,
    теплоту образования и список температурных интервалов.
    """
    name: str
    description: str
    n_intervals: int
    reference: str
    elements: Dict[str, float]  # символ элемента -> количество атомов
    phase: int                  # 0 = газ, 1 = твёрдое, 2 = жидкость
    mol_weight: float           # г/моль
    hf298: float                # Дж/моль, стандартная теплота образования при 298.15 К
    intervals: List[TemperatureInterval] = field(default_factory=list)
    is_reactant_only: bool = False  # True, если вещество только реагент (не продукт)

    @property
    def is_gas(self) -> bool:
        return self.phase == 0

    @property
    def is_condensed(self) -> bool:
        return self.phase != 0

    @property
    def phase_str(self) -> str:
        """Текстовое обозначение фазы."""
        return {0: "газ", 1: "твёрдое", 2: "жидкость"}.get(self.phase, "неизвестно")


# ---------------------------------------------------------------------------
# Вспомогательные функции для разбора формата файла
# ---------------------------------------------------------------------------

def _parse_fortran_float(s: str) -> float:
    """
    Конвертирует строку числа в фортрановском стиле (экспонента 'D') в float.

    Например: '1.23456D+04' -> 12345.6
    """
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace('D', 'E').replace('d', 'e')
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_coefficient_line(line: str) -> List[float]:
    """
    Разбирает строку с 5 коэффициентами NASA-9.

    В формате CEA каждый коэффициент занимает ровно 16 символов.
    """
    coefficients = []
    # Дополняем строку до 80 символов на случай усечения
    padded = line.ljust(80)
    for col in range(5):
        segment = padded[col * 16 : (col + 1) * 16]
        coefficients.append(_parse_fortran_float(segment))
    return coefficients


# ---------------------------------------------------------------------------
# Основная функция разбора файла
# ---------------------------------------------------------------------------

def parse_thermo_file(filepath: str) -> Dict[str, Species]:
    """
    Разбирает файл thermo.inp с данными NASA-9.

    Возвращает словарь: имя вещества -> объект Species.
    Включает только вещества с реальными термодинамическими данными
    (т.е. n_intervals > 0). Вещества из секции «только реагенты»
    помечаются флагом is_reactant_only=True.
    """
    species_db: Dict[str, Species] = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Ищем строку 'thermo' — после неё начинаются данные
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'thermo':
            data_start = i + 2  # пропускаем 'thermo' и строку с глобальным диапазоном T
            break

    in_reactants_section = False
    idx = data_start

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        # Пропускаем пустые строки и комментарии
        if not stripped or stripped.startswith('!'):
            idx += 1
            continue

        # Маркер начала секции «только реагенты»
        if stripped.startswith('END PRODUCTS'):
            in_reactants_section = True
            idx += 1
            continue

        # Маркер конца файла данных
        if stripped.startswith('END REACTANTS'):
            break

        # ---------------------------------------------------------------
        # Строка 1: имя и описание вещества
        # ---------------------------------------------------------------
        species_name = line[:18].strip()
        description = line[18:].strip()

        if not species_name:
            idx += 1
            continue

        idx += 1
        if idx >= len(lines):
            break

        # ---------------------------------------------------------------
        # Строка 2: состав, фаза, молярная масса, теплота образования
        # ---------------------------------------------------------------
        comp_line = lines[idx]
        if len(comp_line) < 50:
            idx += 1
            continue

        padded = comp_line.ljust(80)

        # Количество температурных интервалов
        try:
            n_intervals = int(padded[0:2].strip())
        except (ValueError, IndexError):
            idx += 1
            continue

        reference = padded[3:9].strip()

        # Пять пар «символ элемента + количество атомов» (по 8 символов каждая)
        elements = {}
        for i in range(5):
            pos = 10 + i * 8
            symbol = padded[pos : pos + 2].strip()
            try:
                count = float(padded[pos + 2 : pos + 8].strip())
            except (ValueError, IndexError):
                count = 0.0

            # 'E' — запись для электрона (ионы), пропускаем
            if symbol and abs(count) > 1e-10 and symbol != 'E':
                elements[symbol.upper()] = count

        # Фаза вещества
        try:
            phase = int(padded[50:52].strip())
        except (ValueError, IndexError):
            phase = 0

        # Молярная масса, г/моль
        try:
            mol_weight = float(padded[52:65].strip())
        except (ValueError, IndexError):
            mol_weight = 0.0

        # Стандартная теплота образования при 298.15 К, Дж/моль
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
            is_reactant_only=in_reactants_section,
        )

        idx += 1

        # ---------------------------------------------------------------
        # Блоки температурных интервалов (по 3 строки каждый)
        # ---------------------------------------------------------------
        for _ in range(n_intervals):
            if idx >= len(lines):
                break

            # Строка диапазона T и показателей степеней
            t_line = lines[idx].ljust(80)

            try:
                T_low = float(t_line[0:11].strip())
                T_high = float(t_line[11:22].strip())
                n_coeff = int(t_line[22:23].strip()) if t_line[22:23].strip() else 7
            except (ValueError, IndexError):
                idx += 1
                continue

            # Показатели степеней (8 значений по 5 символов)
            exponents = []
            for i in range(8):
                pos = 24 + i * 5
                try:
                    exponents.append(float(t_line[pos : pos + 5].strip()))
                except (ValueError, IndexError):
                    exponents.append(0.0)

            idx += 1
            if idx >= len(lines):
                break

            # Строка коэффициентов a1..a5
            first_coeffs = _parse_coefficient_line(lines[idx])
            idx += 1

            if idx >= len(lines):
                break

            # Строка коэффициентов a6, a7, [пусто], b1, b2
            second_coeffs = _parse_coefficient_line(lines[idx])
            idx += 1

            # Итоговые массивы коэффициентов
            all_coeffs = first_coeffs[:5] + second_coeffs[:2]   # a1..a7
            integration_consts = [second_coeffs[3], second_coeffs[4]]  # b1, b2

            interval = TemperatureInterval(
                T_low=T_low,
                T_high=T_high,
                n_coeff=n_coeff,
                exponents=exponents,
                coeffs=all_coeffs,
                integration=integration_consts,
            )
            sp.intervals.append(interval)

        # Сохраняем вещество (дубликаты не добавляем, берём первое вхождение)
        if species_name not in species_db and n_intervals > 0:
            species_db[species_name] = sp

    return species_db


def get_products_for_elements(
    species_db: Dict[str, Species],
    element_set: set,
    include_condensed: bool = True,
    T: float = None,
) -> List[Species]:
    """
    Выбирает из базы данных вещества, которые могут быть продуктами реакции.

    Включает только те вещества, все элементы которых присутствуют
    в заданном наборе. Ионы и вещества-только-реагенты исключаются.

    Параметры:
        species_db:        База данных веществ из parse_thermo_file()
        element_set:       Набор символов элементов реагентов (например {'H', 'O'})
        include_condensed: Включать ли конденсированные фазы (тв. и жидк.)
        T:                 Температура (К) для проверки диапазона данных
    """
    candidates = []

    for name, sp in species_db.items():
        # Исключаем вещества, которые нельзя использовать как продукты
        if sp.is_reactant_only:
            continue

        # Исключаем ионы (+ или - в имени) и электрон
        if '+' in name or '-' in name or name == 'e-':
            continue

        # Проверяем, что все элементы вещества есть среди наших реагентов
        if not set(sp.elements.keys()).issubset(element_set):
            continue

        # При необходимости исключаем конденсированные фазы
        if sp.is_condensed and not include_condensed:
            continue

        # Проверяем, что температура попадает в диапазон данных (с допуском ±100 К)
        if T is not None and sp.intervals:
            T_min = min(iv.T_low for iv in sp.intervals)
            T_max = max(iv.T_high for iv in sp.intervals)
            if T < T_min - 100 or T > T_max + 100:
                continue

        candidates.append(sp)

    return candidates


# ---------------------------------------------------------------------------
# Быстрый тест при запуске напрямую
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python nasa9_parser.py <thermo.inp>")
        sys.exit(1)

    db = parse_thermo_file(sys.argv[1])
    print(f"Загружено {len(db)} веществ с термодинамическими данными.")

    # Показываем несколько известных веществ
    for name in ['H2O', 'O2', 'N2', 'CO2', 'CH4', 'H2', 'CO', 'OH']:
        if name in db:
            sp = db[name]
            print(
                f"\n{sp.name}: М = {sp.mol_weight:.4f} г/моль, "
                f"Hf298 = {sp.hf298:.1f} Дж/моль, "
                f"фаза = {sp.phase_str}, элементы = {sp.elements}"
            )
            for iv in sp.intervals:
                print(f"  T: {iv.T_low:.1f} – {iv.T_high:.1f} К, коэффициентов: {iv.n_coeff}")
