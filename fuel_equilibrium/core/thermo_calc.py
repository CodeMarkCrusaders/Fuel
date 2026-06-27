# расчёт термодинамических свойств по коэффициентам NASA-9
# формулы из NASA TP-2002-211556 (McBride, Gordon, Reno)

import math
from typing import Optional
from .nasa9_parser import Species, TemperatureInterval


R_UNIVERSAL = 8.314462618  # Дж/(моль·К)


def _get_interval(species: Species, T: float) -> Optional[TemperatureInterval]:
    # ищем нужный интервал; если T за границей — берём ближайший
    for iv in species.intervals:
        if iv.T_low <= T <= iv.T_high:
            return iv
    if not species.intervals:
        return None
    if T < species.intervals[0].T_low:
        return species.intervals[0]
    return species.intervals[-1]


def cp_over_R(species: Species, T: float) -> float:
    # Cp/R = a1*T^-2 + a2*T^-1 + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
    iv = _get_interval(species, T)
    if iv is None:
        # реагент с табличной энтальпией — нет полиномов, Cp неизвестно.
        # для практических целей вернём приближённое значение конденсированной фазы.
        if getattr(species, 'is_tabular_only', False):
            return 4.0  # ≈ Cp_v/R для жидкости (порядок величины)
        return 2.5
    a = iv.coeffs
    return (a[0]*T**-2 + a[1]*T**-1 + a[2]
            + a[3]*T + a[4]*T**2 + a[5]*T**3 + a[6]*T**4)


def h_over_RT(species: Species, T: float) -> float:
    # H/RT = -a1*T^-2 + a2*ln(T)/T + a3 + a4*T/2 + ... + b1/T
    iv = _get_interval(species, T)
    if iv is None:
        # «табличный» реагент (например O2(L), H2(L)): полиномов нет,
        # задана только энтальпия hf298 при T_assigned — её и возвращаем.
        if getattr(species, 'is_tabular_only', False):
            return species.hf298 / (R_UNIVERSAL * T)
        return 2.5
    a = iv.coeffs
    b1 = iv.integration[0]
    return (-a[0]*T**-2 + a[1]*math.log(T)/T + a[2]
            + a[3]*T/2 + a[4]*T**2/3 + a[5]*T**3/4 + a[6]*T**4/5 + b1/T)


def s_over_R(species: Species, T: float) -> float:
    # S/R = -a1*T^-2/2 - a2*T^-1 + a3*ln(T) + a4*T + ... + b2
    iv = _get_interval(species, T)
    if iv is None:
        # для табличного реагента энтропия не определена полиномами —
        # вернём 0; реагент в задаче равновесия использоваться не должен.
        return 0.0
    a = iv.coeffs
    b2 = iv.integration[1]
    return (-a[0]*T**-2/2 - a[1]/T + a[2]*math.log(T)
            + a[3]*T + a[4]*T**2/2 + a[5]*T**3/3 + a[6]*T**4/4 + b2)


def g_over_RT(species: Species, T: float) -> float:
    # G/RT = H/RT - S/R
    return h_over_RT(species, T) - s_over_R(species, T)


# варианты с физическими единицами
def cp_J(species, T): return cp_over_R(species, T) * R_UNIVERSAL
def h_J(species, T):  return h_over_RT(species, T) * R_UNIVERSAL * T
def s_J(species, T):  return s_over_R(species, T) * R_UNIVERSAL
def g_J(species, T):  return g_over_RT(species, T) * R_UNIVERSAL * T


if __name__ == "__main__":
    from .nasa9_parser import parse_thermo_file
    import sys

    if len(sys.argv) < 2:
        print("usage: python thermo_calc.py <thermo.inp> [T_K]")
        sys.exit(1)

    db = parse_thermo_file(sys.argv[1])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 298.15

    print(f"\nСвойства при T = {T:.2f} К:")
    print(f"{'Вещество':<10} {'Cp/R':>8} {'H/RT':>8} {'S/R':>8} {'G/RT':>8}")
    print("-" * 42)
    for name in ['H2', 'O2', 'N2', 'H2O', 'CO2', 'CO', 'OH', 'CH4']:
        if name in db:
            sp = db[name]
            print(f"{name:<10} {cp_over_R(sp,T):8.4f} {h_over_RT(sp,T):8.4f} "
                  f"{s_over_R(sp,T):8.4f} {g_over_RT(sp,T):8.4f}")
