# -*- coding: utf-8 -*-
"""
Единицы измерения для модуля ракетных расчётов.

Поддерживаются давления в:
    Pa      — паскаль
    kPa     — килопаскаль
    MPa     — мегапаскаль
    bar     — бар
    atm     — стандартная атмосфера
    at / kgf/cm^2 — техническая атмосфера (кгс/см²)
    psi     — фунты на квадратный дюйм

Парсер принимает строки вида:
    "10 MPa", "1 atm", "1.0133 bar", "100 kgf/cm2",
    "14.7 psi", "5,1 МПа", "100000 Pa", и т.п.
а также эквиваленты на русском (МПа, кПа, Па, атм, бар, кгс/см²).

Все коэффициенты — переводы в Паскали (СИ).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


# Коэффициенты: 1 unit = COEFF Pa
_UNIT_TO_PA = {
    'pa':       1.0,
    'kpa':      1.0e3,
    'mpa':      1.0e6,
    'bar':      1.0e5,
    'mbar':     1.0e2,
    'atm':      101325.0,
    'at':       98066.5,          # техническая атмосфера = 1 кгс/см²
    'kgf/cm2':  98066.5,
    'kgf/cm^2': 98066.5,
    'kg/cm2':   98066.5,
    'kg/cm^2':  98066.5,
    'psi':      6894.757293168361,
    'psia':     6894.757293168361,
    'torr':     133.32236842105263,
    'mmhg':     133.32236842105263,
}

# Русские псевдонимы → латинские
_RU_ALIASES = {
    'мпа':       'mpa',
    'кпа':       'kpa',
    'па':        'pa',
    'бар':       'bar',
    'мбар':      'mbar',
    'атм':       'atm',
    'ат':        'at',
    'кгс/см2':   'kgf/cm2',
    'кгс/см^2':  'kgf/cm^2',
    'кг/см2':    'kg/cm2',
    'кг/см^2':   'kg/cm^2',
    'кгс/см²':   'kgf/cm2',
    'кг/см²':    'kg/cm2',
    'фунт/дюйм2':  'psi',
    'фунт/дюйм^2': 'psi',
    'фунт/дюйм²':  'psi',
    'пси':       'psi',
    'торр':      'torr',
    'мм_рт_ст':  'mmhg',
    'мм.рт.ст':  'mmhg',
}

# Канонические человекочитаемые подписи (для вывода)
PRESSURE_UNITS_DISPLAY = {
    'pa':       'Pa',
    'kpa':      'kPa',
    'mpa':      'MPa',
    'bar':      'bar',
    'mbar':     'mbar',
    'atm':      'atm',
    'at':       'kgf/cm²',
    'kgf/cm2':  'kgf/cm²',
    'kgf/cm^2': 'kgf/cm²',
    'kg/cm2':   'kgf/cm²',
    'kg/cm^2':  'kgf/cm²',
    'psi':      'psi',
    'psia':     'psi',
    'torr':     'Torr',
    'mmhg':     'mmHg',
}

# Список поддерживаемых единиц (для документации/UI), в каноничном виде.
SUPPORTED_PRESSURE_UNITS = (
    'Pa', 'kPa', 'MPa', 'bar', 'atm', 'kgf/cm²', 'psi',
)


_NUM_UNIT_RE = re.compile(
    r"""^\s*
        ([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)   # число
        \s*
        ([A-Za-zА-Яа-я²^./0-9]*)                  # единицы (опц.)
        \s*$""",
    re.VERBOSE,
)


def _normalize_unit(unit: str) -> str:
    """Приводит обозначение единицы к каноничному ключу _UNIT_TO_PA."""
    u = unit.strip().lower().replace(' ', '')
    # русское «²» оставляем как есть — мы его обрабатываем в _RU_ALIASES;
    # для латинских — приводим «^2» / «2» к одной форме «2».
    u = u.replace('²', '2')
    u = u.replace('^2', '2')
    if u in _RU_ALIASES:
        u = _RU_ALIASES[u]
    return u


def parse_pressure(value, default_unit: str = 'Pa') -> float:
    """Преобразует строку или число в давление в Паскалях.

    Примеры:
        parse_pressure("10 MPa")       -> 1e7
        parse_pressure("1 atm")        -> 101325
        parse_pressure("100 kgf/cm²")  -> 9806650
        parse_pressure("14.7 psi")     -> 101352.93...
        parse_pressure("5,1 МПа")      -> 5.1e6
        parse_pressure(101325)         -> 101325.0  (тогда default_unit
                                        используется как обозначение)
        parse_pressure("0.1013")       -> с default_unit (по умолч. Pa)

    Если в строке нет единиц, используется ``default_unit``.
    """
    # численный ввод — считаем что это значение в default_unit
    if isinstance(value, (int, float)):
        s = f"{value} {default_unit}"
    else:
        s = str(value).strip()
        if not s:
            raise ValueError("Пустая строка давления")

    m = _NUM_UNIT_RE.match(s)
    if not m:
        raise ValueError(f"Не удалось разобрать давление: {value!r}")

    num_str, unit_str = m.group(1), m.group(2)
    num = float(num_str.replace(',', '.'))

    if not unit_str:
        unit_str = default_unit

    key = _normalize_unit(unit_str)
    if key not in _UNIT_TO_PA:
        # подсказка
        supported = ', '.join(sorted(set(SUPPORTED_PRESSURE_UNITS)))
        raise ValueError(
            f"Неизвестная единица давления: {unit_str!r}. "
            f"Поддерживаются: {supported} (а также русские эквиваленты: "
            f"МПа, кПа, Па, бар, атм, кгс/см², фунт/дюйм²)."
        )
    return num * _UNIT_TO_PA[key]


def convert_pressure_pa_to(pressure_pa: float, target_unit: str) -> float:
    """Перевод из Паскалей в указанную единицу."""
    key = _normalize_unit(target_unit)
    if key not in _UNIT_TO_PA:
        raise ValueError(f"Неизвестная единица: {target_unit!r}")
    return pressure_pa / _UNIT_TO_PA[key]


def display_unit_label(unit: str) -> str:
    """Получает человекочитаемое название единицы."""
    key = _normalize_unit(unit)
    return PRESSURE_UNITS_DISPLAY.get(key, unit)


# ─────────────────────────────────────────────────────────────────────────────
# Самопроверка
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("10 MPa", 1e7),
        ("10 МПа", 1e7),
        ("1 atm", 101325.0),
        ("1 атм", 101325.0),
        ("1 bar", 1.0e5),
        ("1 бар", 1.0e5),
        ("100 kgf/cm2", 98066.5 * 100),
        ("100 кгс/см²", 98066.5 * 100),
        ("14.7 psi", 14.7 * 6894.757293168361),
        ("14,7 фунт/дюйм²", 14.7 * 6894.757293168361),
        ("101325 Pa", 101325.0),
        ("0.1013 МПа", 1.013e5),
    ]
    print("Тест парсера давления:")
    for s, expected in tests:
        got = parse_pressure(s)
        ok = abs(got - expected) / max(abs(expected), 1.0) < 1e-6
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}]  {s!r:>30s}  →  {got:.4f} Па  (ожидалось {expected:.4f})")
