"""
Расчёт термодинамических свойств по коэффициентам NASA-9.

Вычисляет теплоёмкость (Cp), энтальпию (H), энтропию (S) и энергию Гиббса (G)
для любого вещества из базы данных при заданной температуре.

Используемый справочник:
    NASA TP-2002-211556, McBride, Gordon, Reno
"""

import math
from typing import Optional

from nasa9_parser import Species, TemperatureInterval


# Универсальная газовая постоянная, Дж/(моль·К)
R_UNIVERSAL = 8.314462618


def _get_interval(species: Species, T: float) -> Optional[TemperatureInterval]:
    """
    Возвращает температурный интервал, содержащий T.

    Если T выходит за пределы всех интервалов, берём ближайший
    (экстраполяция). Это лучше, чем полный отказ: для большинства
    веществ данные корректны и вблизи границы.
    """
    # Ищем интервал, в который попадает T
    for interval in species.intervals:
        if interval.T_low <= T <= interval.T_high:
            return interval

    # T ниже всех интервалов — берём первый
    if T < species.intervals[0].T_low:
        return species.intervals[0]

    # T выше всех интервалов — берём последний
    if T > species.intervals[-1].T_high:
        return species.intervals[-1]

    return None


# ---------------------------------------------------------------------------
# Безразмерные термодинамические функции (делённые на R или RT)
# ---------------------------------------------------------------------------

def cp_over_R(species: Species, T: float) -> float:
    """
    Cp/R — безразмерная теплоёмкость при постоянном давлении.

    Полином NASA-9:
        Cp/R = a1·T⁻² + a2·T⁻¹ + a3 + a4·T + a5·T² + a6·T³ + a7·T⁴
    """
    interval = _get_interval(species, T)
    if interval is None:
        return 2.5  # значение по умолчанию для одноатомного газа

    a = interval.coeffs
    return (
        a[0] * T**(-2)
        + a[1] * T**(-1)
        + a[2]
        + a[3] * T
        + a[4] * T**2
        + a[5] * T**3
        + a[6] * T**4
    )


def h_over_RT(species: Species, T: float) -> float:
    """
    H/(RT) — безразмерная энтальпия.

    Получается интегрированием Cp/R по T:
        H/(RT) = -a1·T⁻² + a2·ln(T)/T + a3 + a4·T/2 + a5·T²/3
                 + a6·T³/4 + a7·T⁴/5 + b1/T
    где b1 — константа интегрирования (связана с теплотой образования).
    """
    interval = _get_interval(species, T)
    if interval is None:
        return 2.5

    a = interval.coeffs
    b1 = interval.integration[0]

    return (
        -a[0] * T**(-2)
        + a[1] * math.log(T) / T
        + a[2]
        + a[3] * T / 2.0
        + a[4] * T**2 / 3.0
        + a[5] * T**3 / 4.0
        + a[6] * T**4 / 5.0
        + b1 / T
    )


def s_over_R(species: Species, T: float) -> float:
    """
    S/R — безразмерная энтропия в стандартном состоянии (1 бар).

    Полином NASA-9:
        S/R = -a1·T⁻²/2 - a2·T⁻¹ + a3·ln(T) + a4·T + a5·T²/2
              + a6·T³/3 + a7·T⁴/4 + b2
    где b2 — константа интегрирования (энтропийная).
    """
    interval = _get_interval(species, T)
    if interval is None:
        return 0.0

    a = interval.coeffs
    b2 = interval.integration[1]

    return (
        -a[0] * T**(-2) / 2.0
        - a[1] * T**(-1)
        + a[2] * math.log(T)
        + a[3] * T
        + a[4] * T**2 / 2.0
        + a[5] * T**3 / 3.0
        + a[6] * T**4 / 4.0
        + b2
    )


def g_over_RT(species: Species, T: float) -> float:
    """
    G⁰/(RT) — безразмерная стандартная энергия Гиббса.

    По определению: G = H - T·S, поэтому G/(RT) = H/(RT) - S/R.
    Это и есть стандартный химический потенциал вещества, делённый на RT.
    """
    return h_over_RT(species, T) - s_over_R(species, T)


# ---------------------------------------------------------------------------
# Функции с физическими единицами
# ---------------------------------------------------------------------------

def cp_J(species: Species, T: float) -> float:
    """Теплоёмкость Cp, Дж/(моль·К)."""
    return cp_over_R(species, T) * R_UNIVERSAL


def h_J(species: Species, T: float) -> float:
    """Энтальпия H, Дж/моль."""
    return h_over_RT(species, T) * R_UNIVERSAL * T


def s_J(species: Species, T: float) -> float:
    """Энтропия S в стандартном состоянии (1 бар), Дж/(моль·К)."""
    return s_over_R(species, T) * R_UNIVERSAL


def g_J(species: Species, T: float) -> float:
    """Стандартная энергия Гиббса G⁰ (1 бар), Дж/моль."""
    return g_over_RT(species, T) * R_UNIVERSAL * T


# ---------------------------------------------------------------------------
# Быстрый тест при запуске напрямую
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from nasa9_parser import parse_thermo_file
    import sys

    if len(sys.argv) < 2:
        print("Использование: python thermo_calc.py <thermo.inp> [температура_К]")
        sys.exit(1)

    db = parse_thermo_file(sys.argv[1])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 298.15

    print(f"\nТермодинамические свойства при T = {T:.2f} К:")
    header = (
        f"{'Вещество':<12} {'Cp/R':>10} {'H/RT':>10} {'S/R':>10} {'G/RT':>10} "
        f"{'Cp(Дж/мол·К)':>14} {'H(кДж/мол)':>12} {'S(Дж/мол·К)':>13} {'G(кДж/мол)':>12}"
    )
    print(header)
    print("-" * len(header))

    test_species = ['H2', 'O2', 'N2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'O', 'NO', 'CH4']
    for name in test_species:
        if name in db:
            sp = db[name]
            print(
                f"{name:<12} "
                f"{cp_over_R(sp, T):10.4f} "
                f"{h_over_RT(sp, T):10.4f} "
                f"{s_over_R(sp, T):10.4f} "
                f"{g_over_RT(sp, T):10.4f} "
                f"{cp_J(sp, T):14.4f} "
                f"{h_J(sp, T) / 1000:12.4f} "
                f"{s_J(sp, T):13.4f} "
                f"{g_J(sp, T) / 1000:12.4f}"
            )
