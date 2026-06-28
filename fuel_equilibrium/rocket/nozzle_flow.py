# Газодинамика по соплу: равновесное течение, NASA CEA-подобный расчёт.
#
# Что считается:
#   1) Параметры в камере сгорания (Injector):
#      решается HP-задача — энтальпия равна сумме энтальпий реагентов
#      при их собственных температурах; давление = P_chamber; скорость ≈ 0.
#   2) Изэнтропическое расширение по сечениям сопла:
#      для каждого давления P решается SP-задача (S = S_chamber),
#      получаются T, состав, H, далее: V = sqrt(2*(H_ch - H)/m_total),
#      M = V / a, ρ из PV = nRT, Ae/At из сохранения массового расхода.
#   3) Поиск горловины (M=1) одномерной оптимизацией по P.
#
# Модель: «равновесное» (Equilibrium) течение — состав пересчитывается в
# каждой точке. Это эталон CEA для idealized rocket performance.
#
# Замечание о массе:
#   в расчёте равновесия число молей не сохраняется (диссоциация/рекомбинация),
#   3) Газодинамика в остальных сечениях — ПО ИЗВЕСТНОЙ ГЕОМЕТРИИ: для каждой
#      точки контура известна ε = A/A_t; решается SP-задача равновесия при
#      давлении, обращающем в нуль (ρ_t·V_t)/(ρ·V) − ε (дозв./св.-зв. ветвь).
#      V = sqrt(2·(H_кам − H)/m), M = V/a, ρ из PV=nRT, Ae/At = ε.
#
# Модель: «равновесное» (Equilibrium) течение — состав пересчитывается в
# каждой точке. Это эталон CEA для idealized rocket performance.
#
# Замечание о массе:
#   в расчёте равновесия число молей не сохраняется (диссоциация/рекомбинация),
#   а масса сохраняется. Все «удельные» величины пересчитываются на 1 кг смеси.

import math
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from scipy.optimize import brentq

from ..core.nasa9_parser import Species, parse_thermo_file, get_products_for_elements
from ..core.thermo_calc import h_over_RT, s_over_R, cp_over_R, g_over_RT, R_UNIVERSAL
from ..core.gibbs_solver import (
    solve_equilibrium,
    solve_equilibrium_HP,
    solve_equilibrium_SP,
    mixture_enthalpy,
    mixture_entropy,
    EquilibriumResult,
)
from ..io.iteration_logger import IterationLogger, NullLogger
from ..core.formula_parser import parse_formula
from .nozzle_geometry import build_nozzle_geometry, NozzleGeometry


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Propellant:
    """Один компонент топлива (окислитель или горючее).

    name      — имя как в базе данных (например 'O2(L)', 'CH4(L)', 'H2', 'O2').
    mass_kg   — массовый расход, кг.  В батч-режиме мы передаём 1 кг суммы.
    T_K       — температура подачи компонента, К. Если None и имя — это
                «табличный» реагент (O2(L), H2(L) и т.п.), используется
                T_assigned из базы.
    """
    name: str
    mass_kg: float
    T_K: Optional[float] = None


@dataclass
class StationResult:
    """Состояние газа в одном сечении сопла."""
    label: str           # 'Injector' / 'Nozzle inlet' / 'Nozzle throat' / 'Nozzle exit' / ...
    P_Pa: float
    T_K: float
    H_J_per_kg: float    # энтальпия на 1 кг смеси
    S_J_per_kgK: float
    U_J_per_kg: float    # внутренняя энергия = H - P/ρ
    # «замороженные» теплоёмкости (состав фиксирован):
    cp_frozen_J_per_kgK: float
    cv_frozen_J_per_kgK: float
    gamma_frozen: float
    # «равновесные» теплоёмкости (с учётом сдвига равновесия) — как в CEA/RPA:
    cp_eq_J_per_kgK: float
    cv_eq_J_per_kgK: float
    gamma_eq: float
    gamma_s: float              # «эффективный» изэнтропический показатель (Isentropic exp.)
    a_m_per_s: float            # скорость звука (равновесная)
    V_m_per_s: float            # скорость потока
    M: float                    # число Маха
    rho_kg_per_m3: float
    n_moles: float              # моль на 1 кг смеси
    mw_g_per_mol: float         # средняя молярная масса
    R_specific_J_per_kgK: float
    Ae_At: float                # относительная площадь (At = 1)
    mass_flux_kg_per_m2_s: float
    moles: np.ndarray = field(default=None)  # абсолютные моли (на 1 кг смеси)
    mole_fractions: np.ndarray = field(default=None)
    mass_fractions: np.ndarray = field(default=None)
    species_names: List[str] = field(default_factory=list)


@dataclass
class RocketPerformance:
    """Итоговые тяговые характеристики ракетного двигателя."""
    O_F: float                     # массовое отношение окислитель/горючее
    O_F_stoich: float              # стехиометрическое O/F
    alpha: float                   # коэффициент избытка окислителя = O/F_act / O/F_st
    phi: float                     # equivalence ratio (топливная) = 1/alpha
    Cstar_m_per_s: float           # характеристическая скорость
    Isp_s: float                   # удельный импульс на срезе при P_amb=0 -> Ve/g0
    Isp_vac_s: float               # вакуумный удельный импульс
    CF: float                      # коэффициент тяги
    stations: List[StationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NozzleContourPoint:
    """Точка контура сопла в осесимметричной постановке (x, r), м."""
    x_m: float
    r_m: float


@dataclass
class NozzleContour:
    """Геометрия контура сопла (трансзвуковая + сверхзвуковая части)."""
    method: str
    throat_radius_m: float
    exit_radius_m: float
    area_ratio: float
    length_m: float
    theta_exit_deg: float
    theta_max_deg: float
    points: List[NozzleContourPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Стехиометрия и подготовка реагентов
# ─────────────────────────────────────────────────────────────────────────────

# атомные массы для расчёта стехиометрии (приближённо, г/моль)
_ATOMIC_WEIGHTS = {
    'H': 1.00794, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994,
    'F': 18.9984, 'Cl': 35.453, 'S': 32.065, 'AR': 39.948,
    'HE': 4.0026, 'NE': 20.1797,
}


def stoichiometric_OF(oxidizers: List[Species], fuels: List[Species]) -> float:
    """Стехиометрическое массовое O/F для смеси окислитель/горючее.

    Считаем как: 1 моль топлива полностью сгорает в окислителе до
    CO2, H2O, N2 — окисляющий потенциал смешанного окислителя
    распределяется на восстановители (C, H) горючего.
    """
    # «окислительная способность» окислителя (моль атомарного O на моль вещества)
    # и «восстановительная потребность» горючего (требуемые моль O на моль вещества).
    def reduction_demand(sp: Species) -> float:
        # потребность O: C -> CO2 нужно 2 O, H2 -> H2O нужно 0.5 O на H
        # учтём, что внутренний O в горючем уже снижает потребность
        e = sp.elements
        return 2.0 * e.get('C', 0) + 0.5 * e.get('H', 0) - e.get('O', 0)

    def oxidation_capacity(sp: Species) -> float:
        # сколько атомов O даёт окислитель (грубо: считаем весь O свободным)
        return sp.elements.get('O', 0)

    # для простоты возьмём по одному молю каждого
    m_fuel = sum(f.mol_weight for f in fuels) / max(len(fuels), 1)
    m_ox   = sum(o.mol_weight for o in oxidizers) / max(len(oxidizers), 1)

    demand_per_mol_fuel = sum(reduction_demand(f) for f in fuels) / max(len(fuels), 1)
    capacity_per_mol_ox = sum(oxidation_capacity(o) for o in oxidizers) / max(len(oxidizers), 1)

    if capacity_per_mol_ox < 1e-9 or demand_per_mol_fuel < 1e-9:
        return float('nan')

    # сколько молей окислителя нужно на 1 моль горючего
    moles_ox_per_fuel = demand_per_mol_fuel / capacity_per_mol_ox
    return moles_ox_per_fuel * m_ox / m_fuel


def reactant_enthalpy_and_elements(
    components: List[Propellant],
    species_db: Dict[str, Species],
) -> Tuple[float, Dict[str, float], float]:
    """Считает суммарную энтальпию реагентов и их элементный состав.

    Возвращает: (H_react, {элемент: моль}, M_total_g)
    H_react — суммарная физическая энтальпия (Дж) при заданных T_K.
    """
    H_total = 0.0
    elements = {}
    mass_total_g = 0.0

    for comp in components:
        if comp.name not in species_db:
            raise ValueError(
                f"Вещество '{comp.name}' не найдено в базе. "
                f"Проверьте имя (примеры: 'O2(L)', 'CH4(L)', 'H2', 'O2')."
            )
        sp = species_db[comp.name]
        m_g = comp.mass_kg * 1000.0
        n_mol = m_g / sp.mol_weight

        # выбираем рабочую температуру:
        # - для «табличного» реагента (без полиномов) используем T_assigned;
        # - иначе берём пользовательскую T_K или 298.15 по умолчанию.
        if getattr(sp, 'is_tabular_only', False) and sp.T_assigned is not None:
            T_use = sp.T_assigned
        else:
            T_use = comp.T_K if comp.T_K is not None else 298.15

        H_total += n_mol * h_over_RT(sp, T_use) * R_UNIVERSAL * T_use
        mass_total_g += m_g

        for el, cnt in sp.elements.items():
            elements[el] = elements.get(el, 0.0) + n_mol * cnt

    return H_total, elements, mass_total_g


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт термодинамических производных смеси
# ─────────────────────────────────────────────────────────────────────────────

def mixture_cp_frozen(
    species_list: List[Species],
    moles: np.ndarray,
    T: float,
) -> float:
    """«Замороженное» Cp смеси (состав фиксирован), Дж/К.

    Cp = sum_i n_i * Cp_i(T)
    """
    cp = 0.0
    for i, sp in enumerate(species_list):
        if moles[i] > 0:
            cp += moles[i] * cp_over_R(sp, T) * R_UNIVERSAL
    return cp


def mixture_gamma_frozen(
    species_list: List[Species],
    moles: np.ndarray,
    T: float,
    P: float,
) -> Tuple[float, float, float]:
    """Возвращает (Cp_frozen, Cv_frozen, gamma_frozen) для смеси.

    Для смеси идеальных газов:
        Cv = Cp - n_gas * R
    (только газовые моли вносят свой вклад).
    """
    cp = mixture_cp_frozen(species_list, moles, T)
    n_gas = sum(moles[i] for i, sp in enumerate(species_list) if sp.is_gas)
    cv = cp - n_gas * R_UNIVERSAL
    if cv <= 0:
        cv = max(cv, 1e-30)
    return cp, cv, cp / cv


# Точность конечно-разностных под-задач для производных. Главный (и
# единственный реальный) источник «артефактов» на профиле M по соплу —
# это шум численных производных d ln n_gas / d ln T (и по P): при штатном
# допуске SLSQP ftol=1e-6 число молей n_gas сходится лишь до уровня, который,
# будучи поделён на малый шаг 2·ΔT, даёт «зубья» на gamma_s → a → M=V/a.
# Поэтому именно под-задачи для производных решаем с жёстким допуском.
_FD_FTOL = 1e-11

# Профили точности расчёта (оптимизация по скорости/детализации):
#   fast     — грубо, минимум итераций (быстрый предпросмотр);
#   balanced — по умолчанию (ранее зашитые допуски);
#   precise  — жёсткие допуски, максимум итераций (макс. точность, медленнее).
# Под-задачи численных производных всегда решаются с жёстким _FD_FTOL
# независимо от профиля (иначе на профиле M появляются «зубья»).
PRECISION_PROFILES = {
    'fast':     {'ftol': 1e-4,  'tol_H': 1e-3, 'tol_S': 1e-3},
    'balanced': {'ftol': 1e-6,  'tol_H': 1e-5, 'tol_S': 1e-6},
    'precise':  {'ftol': 1e-10, 'tol_H': 1e-7, 'tol_S': 1e-8},
}


def _precision_profile(precision):
    if precision not in PRECISION_PROFILES:
        raise ValueError(
            f"Неизвестный профиль точности '{precision}'. "
            f"Доступные: {sorted(PRECISION_PROFILES)}"
        )
    return PRECISION_PROFILES[precision]


def _solve_fd_point(species_list, element_abundances, T, P, n_warm):
    """Под-задача равновесия для конечной разности (жёсткий допуск, тёплый старт)."""
    return solve_equilibrium(
        species_list, element_abundances, T, P,
        include_condensed=True, verbose=False,
        n0_warm=n_warm, ftol=_FD_FTOL,
    )


def equilibrium_cp_and_sound_speed(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    moles_at_state: np.ndarray,
    delta_T: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """Численная оценка равновесных Cp_eq, Cv_eq, gamma_s, a_eq.

    Использует конечные разности по T и P вблизи текущего состояния
    (по формулам NASA RP-1311, том I, §7).

    Возвращает: (Cp_eq, Cv_eq, gamma_s, a_eq, dlnV_dlnT_P)
        Cp_eq    — равновесная теплоёмкость при P=const, Дж/К (на всю смесь)
        Cv_eq    — равновесная теплоёмкость при V=const, Дж/К
                   Cv_eq = Cp_eq + n_gas R (d lnV/d lnT)_P^2 / (d lnV/d lnP)_T
        gamma_s  — изэнтропический показатель:
                   gamma_s = -1 / [ (d lnV/d lnP)_T
                                    - n_gas R (d lnV/d lnT)_P^2 / Cp_eq ]
        a_eq     — равновесная скорость звука: a^2 = gamma_s * R_spec * T

    Численная устойчивость (исправление артефактов профиля M):
        1) под-задачи производных решаются с жёстким допуском _FD_FTOL —
           иначе шум n_gas (на уровне ftol камеры) делится на малый шаг и
           порождает «зубья» на gamma_s/a/M;
        2) шаги конечных разностей берутся относительными и достаточно
           крупными (ΔT ≈ 0.5%·T, ΔP ≈ 0.5%·P) — это резко снижает
           усиление округления при делении на 2·шаг, оставаясь в области,
           где gamma_s(T,P) гладкая (смещение пренебрежимо);
        3) производные и gamma_s ограничены физически осмысленными рамками;
           при любом сбое возвращается «замороженное» приближение.
    """
    n_warm = np.asarray(moles_at_state, dtype=float)
    gas_idx = [i for i, sp in enumerate(species_list) if sp.is_gas]

    def _n_gas(moles):
        s = 0.0
        for i in gas_idx:
            s += moles[i]
        return max(s, 1e-300)

    # ── относительные шаги конечных разностей ─────────────────────────
    # ΔT: 0.5% от T, но не меньше переданного delta_T и не меньше 2 K.
    dT = max(0.005 * T, delta_T, 2.0)
    # держим точки T±dT строго положительными
    dT = min(dT, 0.45 * T)
    # ΔP: 0.5% от P (центрированно), не меньше 1 Па.
    dP = max(0.005 * P, 1.0)
    dP = min(dP, 0.45 * P)

    # ── 1) производные по T при P = const ─────────────────────────────
    r_plus  = _solve_fd_point(species_list, element_abundances, T + dT, P, n_warm)
    r_minus = _solve_fd_point(species_list, element_abundances, T - dT, P, n_warm)
    cp_eq = (r_plus.enthalpy - r_minus.enthalpy) / (2.0 * dT)

    n_gas_plus  = _n_gas(r_plus.moles)
    n_gas_minus = _n_gas(r_minus.moles)
    # ln V = ln n_gas + ln T - ln P  =>  d lnV/d lnT |_P = (d ln n_gas / d ln T) + 1
    dln_ngas_dlnT = T * (math.log(n_gas_plus) - math.log(n_gas_minus)) / (2.0 * dT)
    dlnV_dlnT_P = dln_ngas_dlnT + 1.0

    # ── 2) производные по P при T = const ────────────────────────────
    r_pP = _solve_fd_point(species_list, element_abundances, T, P + dP, n_warm)
    r_mP = _solve_fd_point(species_list, element_abundances, T, P - dP, n_warm)
    n_gas_pP = _n_gas(r_pP.moles)
    n_gas_mP = _n_gas(r_mP.moles)
    # d lnV/d lnP |_T = (d ln n_gas / d ln P) - 1
    dln_ngas_dlnP = P * (math.log(n_gas_pP) - math.log(n_gas_mP)) / (2.0 * dP)
    dlnV_dlnP_T = dln_ngas_dlnP - 1.0

    # ── 2a) физические рамки производных ──────────────────────────────
    # (d lnV/d lnT)_P ≥ 1 (рекомбинация при охлаждении не уменьшает объём
    #  быстрее идеального газа); шум иногда даёт чуть меньше 1.
    dlnV_dlnT_P = max(dlnV_dlnT_P, 1.0)
    # (d lnV/d lnP)_T ≤ −1 для идеального газа; диссоциация делает его ещё
    #  более отрицательным. Значение > −1 нефизично — ограничиваем сверху.
    dlnV_dlnP_T = min(dlnV_dlnP_T, -1.0)

    # ── 3) равновесные Cv_eq и gamma_s по CEA-формулам ───────────────
    n_gas_now = _n_gas(moles_at_state)
    nR = n_gas_now * R_UNIVERSAL

    # «Замороженное» приближение gamma_s как запасной/опорный ориентир.
    cp_frozen = mixture_cp_frozen(species_list, moles_at_state, T)
    gamma_frozen = (cp_frozen / max(cp_frozen - nR, 1e-30)) if cp_frozen > nR else 1.2

    if abs(dlnV_dlnP_T) > 1e-30:
        # NASA RP-1311 (7.21):  Cv_eq = Cp_eq + nR * (dlnV/dlnT)_P^2 / (dlnV/dlnP)_T
        cv_eq = cp_eq + nR * dlnV_dlnT_P**2 / dlnV_dlnP_T
    else:
        cv_eq = cp_eq * 0.9

    if cp_eq > 1e-30:
        # NASA SP-273 (2.61):  gamma_s = -1 / [(dlnV/dlnP)_T + nR*(dlnV/dlnT)_P^2 / Cp_eq]
        denom = dlnV_dlnP_T + nR * dlnV_dlnT_P**2 / cp_eq
        gamma_s = -1.0 / denom if denom < -1e-30 else gamma_frozen
    else:
        gamma_s = gamma_frozen

    # gamma_s изэнтропический физически в диапазоне ~(1.05 … gamma_frozen].
    # Равновесный показатель всегда ≤ замороженного. Любой выброс за рамки —
    # это остаточный численный шум, обрезаем к опорному значению.
    if not (1.05 <= gamma_s <= gamma_frozen + 1e-6) or not math.isfinite(gamma_s):
        gamma_s = min(max(gamma_s, 1.05), gamma_frozen) if math.isfinite(gamma_s) else gamma_frozen

    # ── 4) равновесная скорость звука ────────────────────────────────
    mass_g = sum(moles_at_state[i] * species_list[i].mol_weight
                 for i in range(len(species_list)))
    n_total = max(moles_at_state.sum(), 1e-30)
    mw = mass_g / n_total
    R_spec = R_UNIVERSAL / (mw / 1000.0)
    a_eq = math.sqrt(max(gamma_s * R_spec * T, 0.0))

    return cp_eq, cv_eq, gamma_s, a_eq, dlnV_dlnT_P


# ─────────────────────────────────────────────────────────────────────────────
# Формирование объекта станции (одно сечение сопла)
# ─────────────────────────────────────────────────────────────────────────────

def _make_station(
    label: str,
    species_list: List[Species],
    element_abundances: Dict[str, float],
    result_eq: EquilibriumResult,
    P: float,
    total_mass_g: float,
    H_chamber_per_kg: float,
    Cstar: Optional[float] = None,
) -> StationResult:
    """Из решения TP/SP-задачи собирает полный набор параметров сечения."""
    T = result_eq.T
    moles = result_eq.moles
    sp_list = species_list

    # масса смеси (должна совпасть с total_mass_g)
    mass_g = sum(moles[i] * sp_list[i].mol_weight for i in range(len(sp_list)))
    mass_kg = mass_g / 1000.0

    n_total = float(moles.sum())
    n_gas = sum(moles[i] for i, sp in enumerate(sp_list) if sp.is_gas)
    mw_g_per_mol = mass_g / max(n_total, 1e-30)

    # удельные величины (на 1 кг)
    H_per_kg = result_eq.enthalpy / mass_kg
    S_per_kg = result_eq.entropy   / mass_kg

    # плотность через идеальный газ для газовой части:
    # rho_gas = P * MW_gas / (R * T)
    # для смеси с конденсатом — приблизительно та же формула
    mw_gas = (sum(moles[i] * sp_list[i].mol_weight
                  for i, sp in enumerate(sp_list) if sp.is_gas)
              / max(n_gas, 1e-30))
    R_spec = R_UNIVERSAL / (mw_g_per_mol / 1000.0)  # Дж/(кг·К)
    rho = P / (R_spec * T)

    # «замороженное» Cp/Cv и gamma_frozen
    cp_f, cv_f, gamma_f = mixture_gamma_frozen(sp_list, moles, T, P)
    cp_f_per_kg = cp_f / mass_kg
    cv_f_per_kg = cv_f / mass_kg

    # «равновесное» Cp, Cv и gamma_s (численные производные)
    try:
        cp_eq, cv_eq, gamma_s, a_eq, _ = equilibrium_cp_and_sound_speed(
            sp_list, element_abundances, T, P, moles
        )
        cp_eq_per_kg = cp_eq / mass_kg
        cv_eq_per_kg = cv_eq / mass_kg
        # NASA RP-1311: Gamma = Cp_eq / Cv_eq (отличается от gamma_s = isentropic exp.)
        gamma_eq = cp_eq / cv_eq if cv_eq > 1e-30 else gamma_f
    except Exception:
        cp_eq_per_kg = cp_f_per_kg
        cv_eq_per_kg = cv_f_per_kg
        gamma_s = gamma_f
        gamma_eq = gamma_f
        a_eq = math.sqrt(gamma_f * R_spec * T)

    # скорость потока из H_chamber - H = V^2/2
    dh = H_chamber_per_kg - H_per_kg
    V = math.sqrt(max(2.0 * dh, 0.0))
    M = V / a_eq if a_eq > 0 else 0.0

    # внутренняя энергия u = h - p/rho
    U_per_kg = H_per_kg - P / rho if rho > 0 else H_per_kg

    # доли компонентов
    xi = np.zeros(len(sp_list))
    if n_gas > 0:
        for i, sp in enumerate(sp_list):
            if sp.is_gas:
                xi[i] = moles[i] / n_gas
    mf = np.zeros(len(sp_list))
    for i, sp in enumerate(sp_list):
        mf[i] = moles[i] * sp_list[i].mol_weight / max(mass_g, 1e-30)

    # массовый поток (только для горловины и далее имеет смысл):
    # rho * V (кг / (м²·с))
    mass_flux = rho * V

    # Ae/At — заполним позднее, после того как найдём горловину
    station = StationResult(
        label=label,
        P_Pa=P, T_K=T,
        H_J_per_kg=H_per_kg, S_J_per_kgK=S_per_kg,
        U_J_per_kg=U_per_kg,
        cp_frozen_J_per_kgK=cp_f_per_kg,
        cv_frozen_J_per_kgK=cv_f_per_kg,
        gamma_frozen=gamma_f,
        cp_eq_J_per_kgK=cp_eq_per_kg,
        cv_eq_J_per_kgK=cv_eq_per_kg,
        gamma_eq=gamma_eq,
        gamma_s=gamma_s,
        a_m_per_s=a_eq,
        V_m_per_s=V,
        M=M,
        rho_kg_per_m3=rho,
        n_moles=n_total / mass_kg,
        mw_g_per_mol=mw_g_per_mol,
        R_specific_J_per_kgK=R_spec,
        Ae_At=float('nan'),
        mass_flux_kg_per_m2_s=mass_flux,
        moles=moles.copy(),
        mole_fractions=xi,
        mass_fractions=mf,
        species_names=[sp.name for sp in sp_list],
    )
    return station


# ─────────────────────────────────────────────────────────────────────────────
# Замороженный (frozen) состав для промежуточных сечений.
#
# Между ОСНОВНЫМИ опорными сечениями (Инжектор / Горловина / Срез) состав НЕ
# пересчитывается равновесно — он «замораживается»:
#   • дозвук (Инжектор → Горловина): состав = состав КАМЕРЫ;
#   • сверхзвук (Горловина → Срез):  состав = состав ГОРЛОВИНЫ.
# Это физически оправдано (химическая «заморозка» при быстром расширении) и
# кратно ускоряет расчёт: вместо SLSQP-минимизации Гиббса в каждой точке
# решается лишь одномерный поиск T по изэнтропе S_frozen(T,P)=S_target.
# ─────────────────────────────────────────────────────────────────────────────

def _frozen_isentropic_temperature(
    species_list, moles, S_target, P, T_init,
):
    """T, при которой замороженный состав даёт энтропию S_target.

    S_frozen(T,P) монотонно растёт по T (dS/dT = Cp/T > 0), поэтому корень
    ищется методом Брента над расширяющимся интервалом вокруг T_init.
    Возвращает None, если не удалось зажать корень.
    """
    moles = np.asarray(moles, dtype=float)

    def dS(T):
        return mixture_entropy(species_list, moles, T, P) - S_target

    T_lo = max(T_init * 0.5, 50.0)
    T_hi = T_init * 1.5
    f_lo = dS(T_lo)
    f_hi = dS(T_hi)
    expand = 0
    while f_lo * f_hi > 0 and expand < 30:
        if f_hi < 0:
            T_hi *= 1.15
            f_hi = dS(T_hi)
        elif f_lo > 0:
            T_lo *= 0.9
            f_lo = dS(T_lo)
        else:
            break
        expand += 1
    if f_lo * f_hi > 0:
        return None
    try:
        return brentq(dS, T_lo, T_hi, xtol=1e-3, rtol=1e-7, maxiter=100)
    except Exception:
        return None


def _make_frozen_station(
    label, species_list, moles, T, P, total_mass_g, H_chamber_per_kg,
):
    """Сечение с ЗАМОРОЖЕННЫМ составом при (T, P).

    Все теплофизические свойства — замороженные (Cp/Cv/gamma_frozen); скорость
    звука a = sqrt(gamma_frozen · R_spec · T). Равновесные поля для
    согласованности интерфейса приравнены к замороженным.
    """
    sp_list = species_list
    n = np.asarray(moles, dtype=float)
    mass_g = sum(n[i] * sp_list[i].mol_weight for i in range(len(sp_list)))
    mass_kg = mass_g / 1000.0
    n_total = float(n.sum())
    n_gas = sum(n[i] for i, sp in enumerate(sp_list) if sp.is_gas)
    mw_g_per_mol = mass_g / max(n_total, 1e-30)

    H = mixture_enthalpy(sp_list, n, T)
    S = mixture_entropy(sp_list, n, T, P)
    H_per_kg = H / mass_kg
    S_per_kg = S / mass_kg

    R_spec = R_UNIVERSAL / (mw_g_per_mol / 1000.0)
    rho = P / (R_spec * T)

    cp_f, cv_f, gamma_f = mixture_gamma_frozen(sp_list, n, T, P)
    cp_f_per_kg = cp_f / mass_kg
    cv_f_per_kg = cv_f / mass_kg

    a_eq = math.sqrt(gamma_f * R_spec * T)
    dh = H_chamber_per_kg - H_per_kg
    V = math.sqrt(max(2.0 * dh, 0.0))
    M = V / a_eq if a_eq > 0 else 0.0
    U_per_kg = H_per_kg - P / rho if rho > 0 else H_per_kg

    xi = np.zeros(len(sp_list))
    if n_gas > 0:
        for i, sp in enumerate(sp_list):
            if sp.is_gas:
                xi[i] = n[i] / n_gas
    mf = np.zeros(len(sp_list))
    for i, sp in enumerate(sp_list):
        mf[i] = n[i] * sp_list[i].mol_weight / max(mass_g, 1e-30)

    return StationResult(
        label=label,
        P_Pa=P, T_K=T,
        H_J_per_kg=H_per_kg, S_J_per_kgK=S_per_kg,
        U_J_per_kg=U_per_kg,
        cp_frozen_J_per_kgK=cp_f_per_kg,
        cv_frozen_J_per_kgK=cv_f_per_kg,
        gamma_frozen=gamma_f,
        cp_eq_J_per_kgK=cp_f_per_kg,
        cv_eq_J_per_kgK=cv_f_per_kg,
        gamma_eq=gamma_f,
        gamma_s=gamma_f,
        a_m_per_s=a_eq,
        V_m_per_s=V,
        M=M,
        rho_kg_per_m3=rho,
        n_moles=n_total / mass_kg,
        mw_g_per_mol=mw_g_per_mol,
        R_specific_J_per_kgK=R_spec,
        Ae_At=float('nan'),
        mass_flux_kg_per_m2_s=rho * V,
        moles=n.copy(),
        mole_fractions=xi,
        mass_fractions=mf,
        species_names=[sp.name for sp in sp_list],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Распределение промежуточных сечений по зонам сопла
# ─────────────────────────────────────────────────────────────────────────────

def _build_segmented_pressure_grid(
    P_chamber: float,
    P_throat: float,
    P_exit: float,
    n_total: int,
    density_subsonic: float = 1.0,
    density_critical: float = 1.0,
    density_supersonic: float = 1.0,
) -> np.ndarray:
    """Строит сетку давлений для промежуточных сечений по трём зонам.

    Зоны:
        - дозвуковая: между камерой и горловиной;
        - критическая: сгущение в окрестности горловины;
        - сверхзвуковая: между горловиной и срезом.

    Возвращает массив давлений (Па), отсортированный по ходу потока
    (от большего к меньшему), без граничных точек Pc/Pt/Pe.
    """
    n_total = int(max(0, min(1048, n_total)))
    if n_total <= 0:
        return np.array([], dtype=float)

    weights = np.array([
        max(0.0, float(density_subsonic)),
        max(0.0, float(density_critical)),
        max(0.0, float(density_supersonic)),
    ], dtype=float)
    if np.all(weights <= 0):
        weights[:] = 1.0

    raw = n_total * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    remainder = n_total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for i in order[:remainder]:
            counts[i] += 1

    n_sub, n_crit, n_sup = int(counts[0]), int(counts[1]), int(counts[2])
    pressures: List[float] = []

    if n_sub > 0 and P_chamber > P_throat:
        P_sub = np.exp(np.linspace(math.log(P_chamber), math.log(P_throat), n_sub + 2))[1:-1]
        pressures.extend(float(p) for p in P_sub)

    if n_sup > 0 and P_throat > P_exit:
        P_sup = np.exp(np.linspace(math.log(P_throat), math.log(P_exit), n_sup + 2))[1:-1]
        pressures.extend(float(p) for p in P_sup)

    if n_crit > 0 and P_chamber > P_throat > P_exit:
        ln_pc, ln_pt, ln_pe = math.log(P_chamber), math.log(P_throat), math.log(P_exit)
        span_sub = max(ln_pc - ln_pt, 1e-9)
        span_sup = max(ln_pt - ln_pe, 1e-9)
        crit_share = 0.18

        n_up = (n_crit + 1) // 2
        n_dn = n_crit - n_up

        for i in range(1, n_up + 1):
            frac = (i / (n_up + 1)) ** 1.4
            ln_p = ln_pt + crit_share * span_sub * frac
            pressures.append(float(math.exp(ln_p)))

        for i in range(1, n_dn + 1):
            frac = (i / (n_dn + 1)) ** 1.4
            ln_p = ln_pt - crit_share * span_sup * frac
            pressures.append(float(math.exp(ln_p)))

    # Фильтрация, сортировка по ходу потока и удаление «почти дублей».
    filtered = [p for p in pressures if (P_exit < p < P_chamber)]
    filtered.sort(reverse=True)

    dedup: List[float] = []
    for p in filtered:
        if not dedup or abs(p - dedup[-1]) > max(1e-6, 1e-8 * abs(p)):
            dedup.append(p)

    return np.array(dedup, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Параллелизм газодинамического расчёта по сечениям
# ─────────────────────────────────────────────────────────────────────────────
#
# Каждое промежуточное сечение считается независимо: своя SP-задача
# (solve_equilibrium_SP) + сборка параметров (_make_station). Зависимостей
# между сечениями нет — это «embarrassingly parallel» задача.
#
# Почему потоки (ThreadPoolExecutor), а не процессы:
#   * тяжёлая часть каждой задачи — это (а) Fortran-ядро SLSQP в SciPy и
#     (б) векторные операции NumPy/Numba; обе освобождают GIL на время счёта,
#     поэтому потоки реально идут параллельно по ядрам;
#   * процессы потребовали бы pickling базы видов и состава на каждую задачу
#     (большой оверхед, ранее измеренный как 0.39x — замедление);
#   * потоки разделяют общий species_list / elements без копирования.
#
# Число воркеров берём из переменной окружения FUEL_NOZZLE_WORKERS, иначе
# по числу доступных CPU (ограничив сверху, чтобы не плодить лишние потоки
# на коротких сетках).

def _resolve_worker_count(n_tasks: int) -> int:
    """Сколько потоков использовать для n_tasks независимых сечений."""
    if n_tasks <= 1:
        return 1
    env = os.environ.get("FUEL_NOZZLE_WORKERS")
    if env:
        try:
            w = int(env)
            if w >= 1:
                return min(w, n_tasks)
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    # один поток на задачу, но не больше числа CPU и не больше 8
    return max(1, min(n_tasks, cpu, 8))


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт сечений по ИЗВЕСТНОЙ ГЕОМЕТРИИ (метод «площадь → состояние»)
# ─────────────────────────────────────────────────────────────────────────────
#
# Для каждой точки контура сопла известна ε = A/A_t (из геометрии). Состояние
# газа определяется из сохранения массового расхода:
#     m_dot = ρ·V·A = ρ_t·V_t·A_t   ⇒   A/A_t = (ρ_t·V_t)/(ρ·V) = ε.
# Для заданного ε решается SP-задача равновесного расширения (S = S_камеры) при
# давлении P, обращающем (ρ_t·V_t)/(ρ(P)·V(P)) − ε в ноль. Уравнение A/A_t=f(P)
# имеет минимум 1 в горловине и две ветви:
#   • дозвуковая (P > P_throat);
#   • сверхзвуковая (P < P_throat).
# Корень ищется методом Брента на нужной ветви.


def _solve_station_for_area_ratio_frozen(
    eps_target, branch,
    species_list,
    S_target_total, H_chamber_per_kg,
    flux_throat, P_throat, P_exit, P_chamber,
    mass_total_g,
    n_ref,            # замороженный состав (моли) для данной ветви
    T_init,
    label='Section',
):
    """Замороженный аналог _solve_station_for_area_ratio.

    Для заданной ε = A/A_t на нужной ветви (sub/sup) ищется давление P, при
    котором массовый поток ρ·V даёт требуемую ε. В каждой пробной точке
    состояние строится по ЗАМОРОЖЕННОМУ составу n_ref (без SLSQP).
    """
    if flux_throat <= 1e-30:
        return None
    eps_target = max(float(eps_target), 1.0 + 1e-4)
    n_ref = np.asarray(n_ref, dtype=float)
    holder = {}

    def _station_at(P):
        T = _frozen_isentropic_temperature(
            species_list, n_ref, S_target_total, float(P), T_init,
        )
        if T is None:
            T = T_init
        st = _make_frozen_station(
            label, species_list, n_ref, T, float(P),
            mass_total_g, H_chamber_per_kg,
        )
        holder['st'] = st
        return st

    def _residual(P):
        st = _station_at(P)
        eps_calc = flux_throat / max(st.mass_flux_kg_per_m2_s, 1e-30)
        return eps_calc - eps_target

    if branch == 'sub':
        P_lo = P_throat * (1.0 + 1e-4)
        P_hi = P_chamber * 0.9999
    else:
        P_lo = max(P_exit * (1.0 + 1e-5), 1.0)
        P_hi = P_throat * (1.0 - 1e-4)
    if P_lo >= P_hi:
        return None

    try:
        f_lo = _residual(P_lo)
        f_hi = _residual(P_hi)
    except Exception:
        return None

    expand = 0
    while f_lo * f_hi > 0 and expand < 6:
        if branch == 'sub':
            P_hi = min(P_hi * 1.05 + 1e3, P_chamber * 0.99999)
        else:
            P_lo = max(P_lo * 0.95, P_exit * 1.0001)
        if P_lo >= P_hi:
            break
        try:
            f_lo = _residual(P_lo)
            f_hi = _residual(P_hi)
        except Exception:
            return None
        expand += 1
    if f_lo * f_hi > 0:
        return None

    try:
        brentq(_residual, P_lo, P_hi,
               xtol=max(P_chamber * 1e-7, 1e-3), rtol=1e-6, maxiter=80)
    except Exception:
        return None

    return holder.get('st') if holder.get('st') is not None else _station_at(
        0.5 * (P_lo + P_hi)
    )


def _sample_interior_indices(n_total, n_pick):
    """Индексы строго между 0 и n_total-1, равномерно выбираемые из контура
    (конечные точки — опорные сечения: горловина, срез/вход)."""
    if n_total <= 2 or n_pick <= 0:
        return []
    n_pick = min(n_pick, n_total - 2)
    return [int(round((i + 1) * (n_total - 1) / (n_pick + 1))) for i in range(n_pick)]


def _solve_station_for_area_ratio(
    eps_target, branch,
    species_list, element_abundances,
    S_target_total, H_chamber_per_kg,
    flux_throat, P_throat, P_exit, P_chamber,
    mass_total_g, include_condensed,
    T_throat, T_chamber, label,
    logger=None,
    tol_S=1e-6,
):
    """Возвращает сечение с относительной площадью A/A_t = eps_target.

    branch = 'sub' — дозвуковая ветвь (P ∈ (P_throat, P_chamber));
    branch = 'sup' — сверхзвуковая ветвь (P ∈ (P_exit, P_throat)).
    Состояние — SP-равновесие (S = S_камеры). При невозможности найти корень
    (ε вне охвата ветви) возвращается None.
    """
    if logger is None:
        logger = NullLogger()
    if flux_throat <= 1e-30:
        return None
    eps_target = max(float(eps_target), 1.0 + 1e-4)
    holder = {}

    def _station_at(P):
        r = solve_equilibrium_SP(
            species_list=species_list,
            element_abundances=element_abundances,
            S_target=S_target_total, P=float(P),
            T_init=(T_chamber * 0.97) if branch == 'sub' else (T_throat * 0.85),
            include_condensed=include_condensed,
            verbose=False, logger=NullLogger(), tol_S=tol_S,
        )
        st = _make_station(label, species_list, element_abundances, r, float(P),
                           mass_total_g, H_chamber_per_kg)
        holder['st'] = st
        return st

    def _residual(P):
        st = _station_at(P)
        eps_calc = flux_throat / max(st.mass_flux_kg_per_m2_s, 1e-30)
        return eps_calc - eps_target

    if branch == 'sub':
        P_lo = P_throat * (1.0 + 1e-4)
        P_hi = P_chamber * 0.9999
    else:
        P_lo = max(P_exit * (1.0 + 1e-5), 1.0)
        P_hi = P_throat * (1.0 - 1e-4)
    if P_lo >= P_hi:
        return None

    try:
        f_lo = _residual(P_lo)
        f_hi = _residual(P_hi)
    except Exception:
        return None

    expand = 0
    while f_lo * f_hi > 0 and expand < 6:
        if branch == 'sub':
            P_hi = min(P_hi * 1.05 + 1e3, P_chamber * 0.99999)
        else:
            P_lo = max(P_lo * 0.95, P_exit * 1.0001)
        if P_lo >= P_hi:
            break
        try:
            f_lo = _residual(P_lo)
            f_hi = _residual(P_hi)
        except Exception:
            return None
        expand += 1
    if f_lo * f_hi > 0:
        return None

    try:
        brentq(_residual, P_lo, P_hi,
               xtol=max(P_chamber * 1e-7, 1e-3), rtol=1e-6, maxiter=80)
    except Exception:
        return None

    st = holder.get('st')
    if st is None:
        return None
    st.Ae_At = float(eps_target)
    return st


def _geometry_stations_from_contour(
    geometry, n_intermediate_stations,
    species_list, element_abundances,
    S_target_total, H_chamber_per_kg,
    flux_throat, P_throat, P_exit, P_chamber,
    mass_total_g, include_condensed,
    T_throat, T_chamber, logger,
    tol_S=1e-6, progress_cb=None,
    n_ref_sub=None, n_ref_sup=None,
):
    """По контуру сопла строит промежуточные сечения «по известной геометрии».

    Возвращает (сечения_до_горловины, сечения_после_горловины, геометрия).
    Каждое сечение соответствует точке контура с известной ε = (r/R_кр)².
    """
    sub_pts = list(getattr(geometry, 'points_subsonic', None) or [])
    sup_pts = list(getattr(geometry, 'points_supersonic', None) or [])
    R_t = geometry.R_throat_m
    n_total = int(max(0, min(1048, n_intermediate_stations)))
    if n_total <= 0 or (not sub_pts and not sup_pts):
        return [], [], geometry

    n_sub_side = len(sub_pts)
    n_sup_side = len(sup_pts)
    n_side_total = max(n_sub_side + n_sup_side, 1)
    n_pick_sub = int(round(n_total * n_sub_side / n_side_total))
    n_pick_sup = n_total - n_pick_sub
    idx_sub = _sample_interior_indices(n_sub_side, n_pick_sub) if n_sub_side > 2 else []
    idx_sup = _sample_interior_indices(n_sup_side, n_pick_sup) if n_sup_side > 2 else []

    tasks = []
    seq = 0
    for i in idx_sub:
        seq += 1
        tasks.append((seq, (sub_pts[i].r_m / R_t) ** 2, 'sub'))
    for i in idx_sup:
        seq += 1
        tasks.append((seq, (sup_pts[i].r_m / R_t) ** 2, 'sup'))

    _done = [0]
    _ntasks = len(tasks)

    # Замороженный состав для ветвей: по умолчанию — состав камеры (дозвук)
    # и состав горловины (сверхзвук). Это главный источник ускорения: ни в одной
    # промежуточной точке не запускается SLSQP-минимизация Гиббса.
    _n_ref = {'sub': n_ref_sub, 'sup': n_ref_sup}
    _T_init = {'sub': T_chamber * 0.97, 'sup': T_throat * 0.85}

    def _compute(args):
        k, eps, branch = args
        st = _solve_station_for_area_ratio_frozen(
            eps_target=eps, branch=branch,
            species_list=species_list,
            S_target_total=S_target_total, H_chamber_per_kg=H_chamber_per_kg,
            flux_throat=flux_throat, P_throat=P_throat,
            P_exit=P_exit, P_chamber=P_chamber,
            mass_total_g=mass_total_g,
            n_ref=_n_ref[branch], T_init=_T_init[branch],
            label=f'Section {k}',
        )
        _done[0] += 1
        if progress_cb:
            try:
                progress_cb(f"Промежуточные сечения · готово {_done[0]}/{_ntasks}")
            except Exception:
                pass
        return k, branch, st

    n_workers = _resolve_worker_count(_ntasks)
    if n_workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_compute, tasks))
    else:
        results = [_compute(t) for t in tasks]

    pre_throat, post_throat = [], []
    for k, branch, st in results:
        if st is None:
            continue
        (pre_throat if branch == 'sub' else post_throat).append(st)

    pre_throat.sort(key=lambda x: x.P_Pa, reverse=True)
    post_throat.sort(key=lambda x: x.P_Pa, reverse=True)

    if logger.enabled:
        logger.log(f'Газодинамика по известной геометрии: '
                   f'{len(pre_throat)} дозв. + {len(post_throat)} св./зв. сечений, '
                   f'{n_workers} поток(ов)')
    return pre_throat, post_throat, geometry


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция: расчёт сопла «от Pc до Pe» с произвольным числом сечений
# ─────────────────────────────────────────────────────────────────────────────

def solve_rocket_nozzle(
    oxidizer: Propellant,
    fuel: Propellant,
    P_chamber: float,
    P_exit: float,
    species_db: Dict[str, Species],
    n_intermediate_stations: int = 0,
    section_density_subsonic: float = 1.0,
    section_density_critical: float = 1.0,
    section_density_supersonic: float = 1.0,
    include_condensed: bool = True,
    injection_velocity: float = 0.0,
    chamber_pressure_drop_frac: float = 0.0,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    max_gas_species: int = 60,
    geometry_method: str = "profiled",
    R_throat_m: float = 1.0,
    R_chamber_factor: float = 2.5,
    geometry_kwargs: Optional[Dict[str, Any]] = None,
    precision: str = "balanced",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> RocketPerformance:
    """Полный расчёт ракетного сопла в равновесном приближении.

    Параметры:
        oxidizer, fuel  — компоненты топлива (mass_kg в сумме обычно = 1 кг).
        P_chamber, P_exit — давления в камере и на срезе, Па.
        n_intermediate_stations — общее число дополнительных сечений
                                  (0..1048) для газодинамического расчёта.
        section_density_*       — относительная плотность сечений по зонам
                                  дозвук/критика/сверхзвук.
        injection_velocity — скорость подачи компонентов на входе (м/с).
                             Полная (тормозная) энтальпия
                             H₀ = h_статич + V_впр²/2 сохраняется по длине,
                             поэтому на сечении инжектора V = V_впр (а не 0).
        chamber_pressure_drop_frac — относительный перепад давления в камере
                             (0…0.3): на входе в сопло P_inlet =
                             P_chamber·(1−Δp). Газ слегка ускоряется, скорость
                             на «Nozzle inlet» становится больше V_впр.
        species_db      — база NASA-9.
        logger          — куда писать журнал итераций.
        precision       — 'fast' | 'balanced' | 'precise': профиль точности
                          (допуски ftol/tol_H/tol_S); грубее → меньше итераций
                          и быстрее расчёт.
        progress_cb     — необязательный колбэк progress_cb(str): вызывается в
                          ключевых точках с сообщением о текущем этапе и номере
                          итерации (для индикации хода расчёта).

    Возвращает RocketPerformance со списком сечений и тяговыми характеристиками.
    """
    if logger is None:
        logger = NullLogger()

    prof = _precision_profile(precision)

    def _emit(msg):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    _n_stages = 5 if n_intermediate_stations > 0 else 4
    _emit(f"Этап 1/{_n_stages}: подготовка реагентов и элементного баланса…")

    n_intermediate_stations = int(max(0, min(1048, n_intermediate_stations)))

    # ── 1) Энтальпия реагентов и элементный баланс ─────────────────────
    components = [oxidizer, fuel]
    H_react, elements, mass_total_g = reactant_enthalpy_and_elements(components, species_db)
    mass_total_kg = mass_total_g / 1000.0

    # стехиометрическое O/F и фактическое
    of_actual = oxidizer.mass_kg / fuel.mass_kg
    try:
        of_stoich = stoichiometric_OF([species_db[oxidizer.name]], [species_db[fuel.name]])
    except Exception:
        of_stoich = float('nan')
    alpha = of_actual / of_stoich if of_stoich and not math.isnan(of_stoich) else float('nan')
    phi = 1.0 / alpha if alpha and not math.isnan(alpha) else float('nan')

    if logger.enabled:
        def _T_str(comp: Propellant) -> str:
            sp = species_db.get(comp.name)
            if sp is not None and getattr(sp, 'is_tabular_only', False) \
                    and sp.T_assigned is not None:
                return f'{sp.T_assigned:.2f} К (assigned)'
            return f'{(comp.T_K if comp.T_K is not None else 298.15):.2f} К'

        logger.section('РАКЕТНОЕ СОПЛО — РАВНОВЕСНЫЙ РАСЧЁТ')
        logger.log(f'Окислитель: {oxidizer.name},  m = {oxidizer.mass_kg:.6f} кг,  '
                   f'T = {_T_str(oxidizer)}')
        logger.log(f'Горючее   : {fuel.name},  m = {fuel.mass_kg:.6f} кг,  '
                   f'T = {_T_str(fuel)}')
        logger.log(f'O/F = {of_actual:.4f}  (стехиометр. ≈ {of_stoich:.4f},  α = {alpha:.4f})')
        logger.log(f'Энтальпия реагентов: H_react = {H_react:.4e} Дж  ({H_react/mass_total_kg/1000:.3f} кДж/кг)')
        logger.log(f'Элементный баланс: {elements}')
        logger.log(f'Pc = {P_chamber:.0f} Па  ({P_chamber/1e6:.4f} МПа)')
        logger.log(f'Pe = {P_exit:.0f} Па  ({P_exit/1e6:.4f} МПа)')

    _emit("Этап 2: отбор веществ-продуктов (кандидатов)…")
    # ── 2) Подбор веществ-продуктов ────────────────────────────────────
    candidates = get_products_for_elements(
        species_db, set(elements.keys()),
        include_condensed=include_condensed, T=2500.0,
    )
    gas_cands = [sp for sp in candidates if sp.is_gas]
    cond_cands = [sp for sp in candidates if sp.is_condensed]

    # отбор: ограничиваем число газов по G0 при «средней» T
    g_sorted = sorted(gas_cands, key=lambda sp: g_over_RT(sp, 2500.0))
    gas_selected = g_sorted[:max_gas_species]
    species_list = gas_selected + cond_cands

    if logger.enabled:
        logger.log(f'Кандидатов: газов = {len(gas_selected)},  конденсата = {len(cond_cands)}')

    # ── 3) Камера (Injector): HP-задача ───────────────────────────────
    if logger.enabled:
        logger.section('Сечение 1/N: КАМЕРА (Injector) — HP-задача')

    _emit(f"Этап 3/{_n_stages}: камера (HP-задача, профиль '{precision}')…")
    chamber = solve_equilibrium_HP(
        species_list=species_list,
        element_abundances=elements,
        H_target=H_react,
        P=P_chamber,
        T_init=2500.0,
        include_condensed=include_condensed,
        verbose=verbose,
        logger=logger,
        tol_H=prof['tol_H'],
        progress_cb=progress_cb,
    )
    H_chamber_total = chamber.enthalpy
    H_chamber_static_per_kg = H_chamber_total / mass_total_kg
    S_chamber_total = chamber.entropy
    T_chamber = chamber.T

    # Скорость подачи компонентов: полная (тормозная) энтальпия
    # H₀ = h_статич + V_впр²/2 сохраняется вдоль сопла, поэтому V на инжекторе
    # равна V_впр, а не нулю. Все сечения строятся от этой H₀.
    V_inj = max(0.0, float(injection_velocity))
    H_chamber_per_kg = H_chamber_static_per_kg + 0.5 * V_inj * V_inj

    # Относительный перепад давления в камере (на входе в сопло).
    dp_frac = min(max(float(chamber_pressure_drop_frac), 0.0), 0.5)
    P_inlet = P_chamber * (1.0 - dp_frac)

    if logger.enabled:
        logger.log(f'T_chamber = {T_chamber:.4f} К')
        logger.log(f'H_chamber(статич) = {H_chamber_static_per_kg/1000:.4f} кДж/кг')
        if V_inj > 0:
            logger.log(f'V_впр = {V_inj:.2f} м/с  →  H₀ = {H_chamber_per_kg/1000:.4f} кДж/кг')
        if dp_frac > 0:
            logger.log(f'Δp камеры = {dp_frac*100:.2f}%  →  P_inlet = {P_inlet/1e6:.5f} МПа')
        logger.log(f'S_chamber = {S_chamber_total/mass_total_kg/1000:.4f} кДж/(кг·К)')

    station_chamber = _make_station(
        'Injector', species_list, elements, chamber, P_chamber,
        mass_total_g, H_chamber_per_kg,
    )

    # ── 4) Поиск горловины (M=1) ───────────────────────────────────────
    if logger.enabled:
        logger.section('Сечение: ГОРЛОВИНА (Throat) — поиск M=1')

    _throat_iter = [0]

    def throat_residual(P_try: float) -> Tuple[float, EquilibriumResult]:
        """Для пробного P в горловине решаем SP-задачу и считаем M-1."""
        _throat_iter[0] += 1
        _emit(f"Горловина (Throat) · поиск M=1: итерация {_throat_iter[0]} (P={P_try/1e6:.4f} МПа)")
        r = solve_equilibrium_SP(
            species_list=species_list,
            element_abundances=elements,
            S_target=S_chamber_total, P=P_try,
            T_init=T_chamber * 0.9,
            include_condensed=include_condensed,
            verbose=False,
            logger=NullLogger(),  # внутрь не пишем — иначе захламит лог
            tol_S=prof['tol_S'],
        )
        st = _make_station('throat?', species_list, elements, r, P_try,
                           mass_total_g, H_chamber_per_kg)
        return st.M - 1.0, r, st

    # начальное приближение: P_throat ≈ P_chamber * (2/(gamma+1))^(gamma/(gamma-1))
    gamma0 = station_chamber.gamma_s
    P_throat_init = P_chamber * (2.0 / (gamma0 + 1.0)) ** (gamma0 / (gamma0 - 1.0))

    # поиск брентом — ищем P, где M=1
    P_lo, P_hi = P_throat_init * 0.5, P_throat_init * 1.5
    P_lo = max(P_lo, P_exit * 1.01)
    P_hi = min(P_hi, P_chamber * 0.999)

    f_lo, _, _ = throat_residual(P_lo)
    f_hi, _, _ = throat_residual(P_hi)
    # расширяем интервал, если знаки совпали
    expand = 0
    while f_lo * f_hi > 0 and expand < 5:
        P_lo = max(P_lo * 0.7, P_exit * 1.001)
        P_hi = min(P_hi * 1.3, P_chamber * 0.9999)
        f_lo, _, _ = throat_residual(P_lo)
        f_hi, _, _ = throat_residual(P_hi)
        expand += 1

    if f_lo * f_hi <= 0:
        P_throat = brentq(lambda P: throat_residual(P)[0], P_lo, P_hi,
                          xtol=P_chamber*1e-7, rtol=1e-6, maxiter=80)
    else:
        # fallback — используем оценку по идеальному газу
        P_throat = P_throat_init
        if logger.enabled:
            logger.log(f'!! не удалось зажать M=1 брентом, используем оценку: '
                       f'P_throat = {P_throat:.0f} Па')

    _, throat_eq, station_throat = throat_residual(P_throat)
    station_throat.label = 'Nozzle throat'
    if logger.enabled:
        logger.log(f'P_throat = {P_throat:.0f} Па  ({P_throat/1e6:.5f} МПа)')
        logger.log(f'T_throat = {station_throat.T_K:.4f} К')
        logger.log(f'M_throat = {station_throat.M:.6f}  (должен быть ≈ 1)')
        logger.log(f'V_throat = {station_throat.V_m_per_s:.4f} м/с')

    # ── 5) Срез сопла (Nozzle exit) ────────────────────────────────────
    if logger.enabled:
        logger.section('Сечение: СРЕЗ СОПЛА (Nozzle exit) — SP-задача')

    _emit(f"Этап 4/{_n_stages}: срез сопла (SP-задача)…")
    exit_eq = solve_equilibrium_SP(
        species_list=species_list,
        element_abundances=elements,
        S_target=S_chamber_total, P=P_exit,
        T_init=T_chamber * 0.6,
        include_condensed=include_condensed,
        verbose=verbose,
        logger=logger,
        tol_S=prof['tol_S'],
        progress_cb=progress_cb,
    )
    station_exit = _make_station(
        'Nozzle exit', species_list, elements, exit_eq, P_exit,
        mass_total_g, H_chamber_per_kg,
    )

    # ── 6) "Nozzle inlet" — вход в сопло.
    # Без перепада давления и без скорости подачи он совпадает с камерой
    # (Injector ≡ Nozzle inlet при stagnation). При наличии перепада давления
    # газ при той же энтропии расширяется до P_inlet, h_статич падает,
    # поэтому скорость на входе V = sqrt(2·(H₀ − h_inlet)) становится больше V_впр.
    if dp_frac > 0.0:
        inlet_eq = solve_equilibrium_SP(
            species_list=species_list,
            element_abundances=elements,
            S_target=S_chamber_total, P=P_inlet,
            T_init=T_chamber * 0.98,
            include_condensed=include_condensed,
            verbose=False,
            logger=NullLogger(),
            tol_S=prof['tol_S'],
        )
        station_inlet = _make_station(
            'Nozzle inlet', species_list, elements, inlet_eq, P_inlet,
            mass_total_g, H_chamber_per_kg,
        )
    else:
        station_inlet = _make_station(
            'Nozzle inlet', species_list, elements, chamber, P_chamber,
            mass_total_g, H_chamber_per_kg,
        )

    # ── 7) Геометрия сопла + газодинамика по ИЗВЕСТНОЙ ГЕОМЕТРИИ ────────
    # Для 4 опорных точек равновесие уже посчитано выше (Инжектор, Вход в
    # сопло, Горловина, Срез). Теперь строится контур сопла по Ae/At среза,
    # а для точек контура с известной ε = A/A_t решается SP-задача равновесия,
    # обращающая (ρ_t·V_t)/(ρ·V) − ε в ноль (метод «площадь → состояние»).
    flux_throat = station_throat.mass_flux_kg_per_m2_s
    intermediate_pre_throat: List[StationResult] = []
    intermediate_post_throat: List[StationResult] = []
    geometry: Optional[NozzleGeometry] = None
    if n_intermediate_stations > 0 and flux_throat > 1e-30:
        eps_exit = flux_throat / max(station_exit.mass_flux_kg_per_m2_s, 1e-30)
        gkw = dict(geometry_kwargs or {})
        gkw.setdefault('R_chamber_m', R_chamber_factor * R_throat_m)
        try:
            geometry = build_nozzle_geometry(
                R_throat_m=R_throat_m, area_ratio=eps_exit,
                method=geometry_method, **gkw,
            )
        except Exception as exc:
            if logger.enabled:
                logger.log(f'Построение геометрии не удалось ({exc!r}); '
                           f'промежуточные сечения пропущены')
            geometry = None
        if geometry is not None:
            if logger.enabled:
                logger.section('ГЕОМЕТРИЯ СОПЛА')
                logger.log(f'метод = {geometry.method},  Ae/At = {eps_exit:.4f},  '
                           f'R_кр = {R_throat_m:.4f} м,  '
                           f'R_к = {geometry.R_chamber_m:.4f} м,  '
                           f'точек контура = {len(geometry.points)}')
            _emit(f"Этап 5/5: промежуточные сечения по контуру ({n_intermediate_stations} шт.)…")
            intermediate_pre_throat, intermediate_post_throat, geometry = \
                _geometry_stations_from_contour(
                    geometry=geometry,
                    n_intermediate_stations=n_intermediate_stations,
                    species_list=species_list,
                    element_abundances=elements,
                    S_target_total=S_chamber_total,
                    H_chamber_per_kg=H_chamber_per_kg,
                    flux_throat=flux_throat, P_throat=P_throat,
                    P_exit=P_exit, P_chamber=P_chamber,
                    mass_total_g=mass_total_g,
                    include_condensed=include_condensed,
                    T_throat=station_throat.T_K,
                    T_chamber=station_chamber.T_K,
                    logger=logger,
                    tol_S=prof['tol_S'],
                    progress_cb=progress_cb,
                    n_ref_sub=chamber.moles,
                    n_ref_sup=station_throat.moles,
                )
    # ── 8) Ae/At — из сохранения массового расхода ────────────────────
    # m_dot = rho * V * A = const => A/At = (rho_t * V_t) / (rho * V)
    # flux_throat вычислен на шаге 7; промежуточные сечения несут Ae/At из геометрии.
    # Ae/At пересчитываем только для ОПОРНЫХ точек; у промежуточных сечений
    # Ae/At уже задан геометрией (ε точки контура).
    all_stations_for_area = [
        station_chamber,
        station_inlet,
        station_throat,
        station_exit,
    ]
    for st in all_stations_for_area:
        if st.mass_flux_kg_per_m2_s > 1e-30:
            st.Ae_At = flux_throat / st.mass_flux_kg_per_m2_s
        else:
            st.Ae_At = float('inf')
    # в камере поток ≈ 0 => Ae/At = infinity, это нормально

    # ── 9) Тяговые характеристики ─────────────────────────────────────
    V_exit = station_exit.V_m_per_s
    g0 = 9.80665
    Isp_s = V_exit / g0
    # вакуумный Isp = (V_e + P_e * A_e / m_dot) / g0
    # m_dot / A_t = flux_throat; m_dot = flux_throat * A_t
    # P_e * A_e / m_dot = P_e * (A_e/A_t) * A_t / (flux_throat * A_t)
    #                  = P_e * (A_e/A_t) / flux_throat
    if flux_throat > 1e-30:
        Isp_vac = (V_exit + P_exit * station_exit.Ae_At / flux_throat) / g0
    else:
        Isp_vac = Isp_s

    # Cstar = P_c * A_t / m_dot = P_chamber / flux_throat
    Cstar = P_chamber / flux_throat if flux_throat > 1e-30 else float('nan')
    CF = V_exit / Cstar if Cstar > 0 else float('nan')

    if logger.enabled:
        logger.section('ТЯГОВЫЕ ХАРАКТЕРИСТИКИ')
        logger.log(f'Isp (на срезе) = {Isp_s:.4f} с')
        logger.log(f'Isp (вакуум)   = {Isp_vac:.4f} с')
        logger.log(f'C* (Cstar)    = {Cstar:.4f} м/с')
        logger.log(f'CF             = {CF:.4f}')
        logger.log(f'V_exit         = {V_exit:.4f} м/с')
        logger.log(f'Ae/At          = {station_exit.Ae_At:.4f}')

    # порядок: Injector, Nozzle inlet, [Section pre...], Nozzle throat,
    #          [Section post...], Nozzle exit
    stations = [
        station_chamber,
        station_inlet,
        *intermediate_pre_throat,
        station_throat,
        *intermediate_post_throat,
        station_exit,
    ]

    return RocketPerformance(
        O_F=of_actual,
        O_F_stoich=of_stoich,
        alpha=alpha,
        phi=phi,
        Cstar_m_per_s=Cstar,
        Isp_s=Isp_s,
        Isp_vac_s=Isp_vac,
        CF=CF,
        stations=stations,
        metadata={'geometry': geometry} if geometry is not None else {},
    )


# Печать таблиц / отчётов вынесена в fuel_equilibrium.io.reporting.
# Здесь — только физика (структуры данных, решатель сопла и построение контуров).


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _throat_arc_points(
    throat_radius_m: float,
    theta_max_deg: float,
    throat_rounding_factor: float,
    n_points: int,
) -> Tuple[List[NozzleContourPoint], float, float]:
    """Строит выходной скруглённый участок горловины AA_n."""
    theta_m = math.radians(theta_max_deg)
    r_skr = throat_rounding_factor * throat_radius_m
    n_points = max(n_points, 8)

    points: List[NozzleContourPoint] = []
    for i in range(n_points):
        phi = theta_m * i / (n_points - 1)
        x = r_skr * math.sin(phi)
        r = throat_radius_m + r_skr * (1.0 - math.cos(phi))
        points.append(NozzleContourPoint(x_m=x, r_m=r))

    p_last = points[-1]
    return points, p_last.x_m, p_last.r_m


def build_profiled_nozzle_contour(
    throat_radius_m: float,
    area_ratio: float,
    theta_exit_deg: float = 12.0,
    theta_max_deg: float = 34.25,
    inlet_rounding_factor: float = 1.5,
    throat_rounding_factor: float = 0.45,
    length_m: Optional[float] = None,
    n_points: int = 160,
) -> NozzleContour:
    """Изменённый метод построения профилированного сопла."""
    if throat_radius_m <= 0:
        raise ValueError("throat_radius_m должен быть > 0")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")

    theta_exit_deg = _clamp(theta_exit_deg, 3.0, 25.0)
    theta_max_deg = _clamp(theta_max_deg, theta_exit_deg + 1e-3, 45.0)

    arc_pts, x_an, r_an = _throat_arc_points(
        throat_radius_m=throat_radius_m,
        theta_max_deg=theta_max_deg,
        throat_rounding_factor=throat_rounding_factor,
        n_points=max(12, n_points // 4),
    )

    r_exit = throat_radius_m * math.sqrt(area_ratio)
    t_m = math.tan(math.radians(theta_max_deg))
    t_a = math.tan(math.radians(theta_exit_deg))

    if length_m is None:
        l_cone = (r_exit - r_an) / max(t_a, 1e-6)
        length_m = max(0.82 * l_cone, 0.05 * throat_radius_m)

    n_div = max(n_points - len(arc_pts), 32)
    div_pts: List[NozzleContourPoint] = []
    for i in range(n_div):
        s = i / (n_div - 1)
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2
        r = (
            h00 * r_an
            + h10 * length_m * t_m
            + h01 * r_exit
            + h11 * length_m * t_a
        )
        x = x_an + s * length_m
        div_pts.append(NozzleContourPoint(x_m=x, r_m=r))

    pts = arc_pts + div_pts[1:]
    return NozzleContour(
        method="profiled_modified",
        throat_radius_m=throat_radius_m,
        exit_radius_m=r_exit,
        area_ratio=area_ratio,
        length_m=pts[-1].x_m,
        theta_exit_deg=theta_exit_deg,
        theta_max_deg=theta_max_deg,
        points=pts,
        metadata={
            "inlet_rounding_factor": inlet_rounding_factor,
            "throat_rounding_factor": throat_rounding_factor,
            "construction": "arc + cubic_hermite",
        },
    )


def build_approximate_optimal_contour_ch26(
    throat_radius_m: float,
    area_ratio: float,
    theta_exit_deg: float = 12.0,
    theta_max_deg: Optional[float] = None,
    inlet_rounding_factor: float = 1.5,
    throat_rounding_factor: float = 0.45,
    length_m: Optional[float] = None,
    n_points: int = 140,
) -> NozzleContour:
    """Приближённый метод построения оптимального контура (гл. 2.6)."""
    if throat_radius_m <= 0:
        raise ValueError("throat_radius_m должен быть > 0")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")

    theta_exit_deg = _clamp(theta_exit_deg, 3.0, 25.0)
    if theta_max_deg is None:
        theta_max_deg = 34.25 - 2.0 * math.log10(max(area_ratio, 1.0001))
    theta_max_deg = _clamp(theta_max_deg, theta_exit_deg + 1e-3, 40.0)

    arc_pts, x_an, r_an = _throat_arc_points(
        throat_radius_m=throat_radius_m,
        theta_max_deg=theta_max_deg,
        throat_rounding_factor=throat_rounding_factor,
        n_points=max(12, n_points // 4),
    )

    r_exit = throat_radius_m * math.sqrt(area_ratio)
    t_m = math.tan(math.radians(theta_max_deg))
    t_a = math.tan(math.radians(theta_exit_deg))

    if length_m is None:
        length_m = 2.0 * (r_exit - r_an) / max(t_m + t_a, 1e-8)
    length_m = max(length_m, 0.05 * throat_radius_m)

    a_par = (t_a - t_m) / (2.0 * length_m)

    n_div = max(n_points - len(arc_pts), 24)
    div_pts: List[NozzleContourPoint] = []
    for i in range(n_div):
        x_local = length_m * i / (n_div - 1)
        r = r_an + t_m * x_local + a_par * x_local * x_local
        x = x_an + x_local
        div_pts.append(NozzleContourPoint(x_m=x, r_m=r))

    pts = arc_pts + div_pts[1:]
    return NozzleContour(
        method="optimal_approx_ch26",
        throat_radius_m=throat_radius_m,
        exit_radius_m=r_exit,
        area_ratio=area_ratio,
        length_m=pts[-1].x_m,
        theta_exit_deg=theta_exit_deg,
        theta_max_deg=theta_max_deg,
        points=pts,
        metadata={
            "R_skr_over_Rkr": inlet_rounding_factor,
            "r_skr_over_Rkr": throat_rounding_factor,
            "construction": "arc + parabola",
            "chapter_ref": "2.6",
        },
    )


def build_optimal_nozzle_contour(
    perf: RocketPerformance,
    throat_radius_m: float,
    p_ambient_Pa: Optional[float] = None,
    theta_exit_deg: Optional[float] = None,
    theta_max_deg: Optional[float] = None,
    n_points: int = 140,
) -> NozzleContour:
    """Построение оптимального сопла по результатам solve_rocket_nozzle."""
    st_exit = next((s for s in perf.stations if s.label == 'Nozzle exit'), perf.stations[-1])
    area_ratio = st_exit.Ae_At
    if not (math.isfinite(area_ratio) and area_ratio > 1.0):
        raise ValueError("Некорректное Ae/At на срезе для построения контура")

    if theta_exit_deg is None:
        theta_calc = None
        if p_ambient_Pa is not None and st_exit.M > 1.0 and st_exit.rho_kg_per_m3 > 0.0:
            q_dyn = 0.5 * st_exit.rho_kg_per_m3 * st_exit.V_m_per_s ** 2
            if q_dyn > 1e-12:
                rhs = ((st_exit.P_Pa - p_ambient_Pa) / q_dyn) * math.sqrt(st_exit.M ** 2 - 1.0)
                rhs = _clamp(rhs, -1.0, 1.0)
                theta_calc = 0.5 * math.degrees(math.asin(rhs))
        theta_exit_deg = theta_calc if theta_calc is not None else 12.0
        theta_exit_deg = _clamp(theta_exit_deg, 10.0, 14.0)

    return build_approximate_optimal_contour_ch26(
        throat_radius_m=throat_radius_m,
        area_ratio=area_ratio,
        theta_exit_deg=float(theta_exit_deg),
        theta_max_deg=theta_max_deg,
        n_points=n_points,
    )


def build_nozzle_contour(
    throat_radius_m: float,
    area_ratio: float,
    method: str = "profiled",
    **kwargs,
) -> NozzleContour:
    """Унифицированный интерфейс построения контуров сопла."""
    key = method.strip().lower()
    if key in ("profiled", "profiled_modified"):
        return build_profiled_nozzle_contour(
            throat_radius_m=throat_radius_m,
            area_ratio=area_ratio,
            **kwargs,
        )
    if key in ("optimal_approx", "optimal_approx_ch26", "ch2.6", "2.6"):
        return build_approximate_optimal_contour_ch26(
            throat_radius_m=throat_radius_m,
            area_ratio=area_ratio,
            **kwargs,
        )
    raise ValueError(
        f"Неизвестный method='{method}'. Допустимо: profiled, optimal_approx"
    )

