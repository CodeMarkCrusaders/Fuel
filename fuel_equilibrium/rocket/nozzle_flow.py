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
#   а масса сохраняется. Все «удельные» величины пересчитываются на 1 кг смеси.

import math
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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


def equilibrium_cp_and_sound_speed(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    moles_at_state: np.ndarray,
    delta_T: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """Численная оценка равновесных Cp_eq, Cv_eq, gamma_s, a_eq.

    Использует малые конечные разности по T и P вблизи текущего состояния
    (по формулам NASA RP-1311, том I, §7).

    Возвращает: (Cp_eq, Cv_eq, gamma_s, a_eq, dlnV_dlnT_P)
        Cp_eq    — равновесная теплоёмкость при P=const, Дж/К (на всю смесь)
        Cv_eq    — равновесная теплоёмкость при V=const, Дж/К
                   Cv_eq = Cp_eq + n_gas R (d lnV/d lnT)_P^2 / (d lnV/d lnP)_T
        gamma_s  — изэнтропический показатель:
                   gamma_s = -1 / [ (d lnV/d lnP)_T
                                    - n_gas R (d lnV/d lnT)_P^2 / Cp_eq ]
        a_eq     — равновесная скорость звука: a^2 = gamma_s * R_spec * T
    """
    # «Тёплый старт»: четыре конечно-разностных решения берутся в малой
    # окрестности уже сошедшегося состояния (T±1 K, P±0.01%), поэтому
    # moles_at_state — отличное начальное приближение. Это резко сокращает
    # число итераций SLSQP (типично в 2–3 раза) без потери точности производных.
    n_warm = np.asarray(moles_at_state, dtype=float)

    # ── 1) производные по T при P = const ─────────────────────────────
    r_plus  = solve_equilibrium(species_list, element_abundances, T + delta_T, P,
                                include_condensed=True, verbose=False,
                                n0_warm=n_warm)
    r_minus = solve_equilibrium(species_list, element_abundances, T - delta_T, P,
                                include_condensed=True, verbose=False,
                                n0_warm=n_warm)
    cp_eq = (r_plus.enthalpy - r_minus.enthalpy) / (2 * delta_T)

    n_gas_plus  = sum(r_plus.moles[i]  for i, sp in enumerate(species_list) if sp.is_gas)
    n_gas_minus = sum(r_minus.moles[i] for i, sp in enumerate(species_list) if sp.is_gas)
    # ln V = ln n_gas + ln T - ln P  =>  d lnV/d lnT |_P = (d ln n_gas / d ln T) + 1
    dlnV_dlnT_P = T * (math.log(n_gas_plus) - math.log(n_gas_minus)) / (2 * delta_T) + 1.0

    # ── 2) производные по P при T = const ────────────────────────────
    dP = max(P * 1e-4, 1.0)
    r_pP = solve_equilibrium(species_list, element_abundances, T, P + dP,
                             include_condensed=True, verbose=False,
                             n0_warm=n_warm)
    r_mP = solve_equilibrium(species_list, element_abundances, T, P - dP,
                             include_condensed=True, verbose=False,
                             n0_warm=n_warm)
    n_gas_pP = sum(r_pP.moles[i] for i, sp in enumerate(species_list) if sp.is_gas)
    n_gas_mP = sum(r_mP.moles[i] for i, sp in enumerate(species_list) if sp.is_gas)
    # d lnV/d lnP |_T = (d ln n_gas / d ln P) - 1
    dlnV_dlnP_T = P * (math.log(n_gas_pP) - math.log(n_gas_mP)) / (2 * dP) - 1.0

    # ── 3) равновесные Cv_eq и gamma_s по CEA-формулам ───────────────
    n_gas_now = sum(moles_at_state[i] for i, sp in enumerate(species_list) if sp.is_gas)
    nR = n_gas_now * R_UNIVERSAL

    if abs(dlnV_dlnP_T) > 1e-30:
        # NASA RP-1311 (7.21):  Cv_eq = Cp_eq + nR * (dlnV/dlnT)_P^2 / (dlnV/dlnP)_T
        # (dlnV/dlnP)_T отрицателен, поэтому формула даёт Cv_eq < Cp_eq.
        cv_eq = cp_eq + nR * dlnV_dlnT_P**2 / dlnV_dlnP_T
    else:
        cv_eq = cp_eq * 0.9

    if cp_eq > 1e-30 and abs(dlnV_dlnP_T) > 1e-30:
        # NASA SP-273 (2.61):  gamma_s = -1 / [(dlnV/dlnP)_T + nR*(dlnV/dlnT)_P^2 / Cp_eq]
        # Знак «+» внутри скобок — даёт gamma_s ~ 1.13 для горячего H2/O2.
        denom = dlnV_dlnP_T + nR * dlnV_dlnT_P**2 / cp_eq
        gamma_s = -1.0 / denom if denom < -1e-30 else 1.4
    else:
        gamma_s = 1.4

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
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    max_gas_species: int = 60,
) -> RocketPerformance:
    """Полный расчёт ракетного сопла в равновесном приближении.

    Параметры:
        oxidizer, fuel  — компоненты топлива (mass_kg в сумме обычно = 1 кг).
        P_chamber, P_exit — давления в камере и на срезе, Па.
        n_intermediate_stations — общее число дополнительных сечений
                                  (0..1048) для газодинамического расчёта.
        section_density_*       — относительная плотность сечений по зонам
                                  дозвук/критика/сверхзвук.
        species_db      — база NASA-9.
        logger          — куда писать журнал итераций.

    Возвращает RocketPerformance со списком сечений и тяговыми характеристиками.
    """
    if logger is None:
        logger = NullLogger()

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

    chamber = solve_equilibrium_HP(
        species_list=species_list,
        element_abundances=elements,
        H_target=H_react,
        P=P_chamber,
        T_init=2500.0,
        include_condensed=include_condensed,
        verbose=verbose,
        logger=logger,
        tol_H=1e-5,
    )
    H_chamber_total = chamber.enthalpy
    H_chamber_per_kg = H_chamber_total / mass_total_kg
    S_chamber_total = chamber.entropy
    T_chamber = chamber.T

    if logger.enabled:
        logger.log(f'T_chamber = {T_chamber:.4f} К')
        logger.log(f'H_chamber = {H_chamber_per_kg/1000:.4f} кДж/кг')
        logger.log(f'S_chamber = {S_chamber_total/mass_total_kg/1000:.4f} кДж/(кг·К)')

    station_chamber = _make_station(
        'Injector', species_list, elements, chamber, P_chamber,
        mass_total_g, H_chamber_per_kg,
    )

    # ── 4) Поиск горловины (M=1) ───────────────────────────────────────
    if logger.enabled:
        logger.section('Сечение: ГОРЛОВИНА (Throat) — поиск M=1')

    def throat_residual(P_try: float) -> Tuple[float, EquilibriumResult]:
        """Для пробного P в горловине решаем SP-задачу и считаем M-1."""
        r = solve_equilibrium_SP(
            species_list=species_list,
            element_abundances=elements,
            S_target=S_chamber_total, P=P_try,
            T_init=T_chamber * 0.9,
            include_condensed=include_condensed,
            verbose=False,
            logger=NullLogger(),  # внутрь не пишем — иначе захламит лог
            tol_S=1e-6,
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

    exit_eq = solve_equilibrium_SP(
        species_list=species_list,
        element_abundances=elements,
        S_target=S_chamber_total, P=P_exit,
        T_init=T_chamber * 0.6,
        include_condensed=include_condensed,
        verbose=verbose,
        logger=logger,
        tol_S=1e-6,
    )
    station_exit = _make_station(
        'Nozzle exit', species_list, elements, exit_eq, P_exit,
        mass_total_g, H_chamber_per_kg,
    )

    # ── 6) "Nozzle inlet" — формально считаем что вход в сопло = камера
    # (CEA так и делает: Injector ≡ Nozzle inlet при stagnation).
    station_inlet = _make_station(
        'Nozzle inlet', species_list, elements, chamber, P_chamber,
        mass_total_g, H_chamber_per_kg,
    )

    # ── 7) Промежуточные сечения: до горловины и после горловины ────────
    intermediate_pre_throat = []
    intermediate_post_throat = []
    if n_intermediate_stations > 0:
        P_grid = _build_segmented_pressure_grid(
            P_chamber=P_chamber,
            P_throat=P_throat,
            P_exit=P_exit,
            n_total=n_intermediate_stations,
            density_subsonic=section_density_subsonic,
            density_critical=section_density_critical,
            density_supersonic=section_density_supersonic,
        )

        eps = max(1e-6, 1e-8 * abs(P_throat))
        P_pre = sorted([float(p) for p in P_grid if p > P_throat + eps], reverse=True)
        P_post = sorted([float(p) for p in P_grid if p < P_throat - eps], reverse=True)

        flow_pressures = [*P_pre, *P_post]

        def _compute_section(args):
            """Считает одно сечение (SP-задача + сборка параметров).

            Полностью независимо от других сечений — пригодно для
            параллельного выполнения в пуле потоков.
            """
            k, P_k = args
            r_k = solve_equilibrium_SP(
                species_list=species_list,
                element_abundances=elements,
                S_target=S_chamber_total, P=float(P_k),
                T_init=station_throat.T_K * 0.8,
                include_condensed=include_condensed,
                verbose=False,
                logger=NullLogger(),
                tol_S=1e-6,
            )
            st = _make_station(
                f'Section {k}', species_list, elements, r_k, float(P_k),
                mass_total_g, H_chamber_per_kg,
            )
            return k, P_k, st

        tasks = list(enumerate(flow_pressures, start=1))
        n_workers = _resolve_worker_count(len(tasks))

        if n_workers > 1:
            # параллельный расчёт сечений в пуле потоков (GIL освобождается
            # внутри SLSQP/NumPy/Numba, поэтому потоки идут параллельно)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_compute_section, tasks))
        else:
            results = [_compute_section(t) for t in tasks]

        # порядок сечений восстанавливаем по давлению (как и раньше),
        # независимо от порядка завершения потоков
        for k, P_k, st in results:
            if P_k > P_throat:
                intermediate_pre_throat.append(st)
            else:
                intermediate_post_throat.append(st)

        intermediate_pre_throat.sort(key=lambda s: s.P_Pa, reverse=True)
        intermediate_post_throat.sort(key=lambda s: s.P_Pa, reverse=True)

        if logger.enabled:
            logger.log(f'Газодинамика по сечениям: {len(tasks)} сечений, '
                       f'{n_workers} поток(ов)')

    # ── 8) Ae/At — из сохранения массового расхода ────────────────────
    # m_dot = rho * V * A = const => A/At = (rho_t * V_t) / (rho * V)
    flux_throat = station_throat.mass_flux_kg_per_m2_s
    all_stations_for_area = [
        station_chamber,
        station_inlet,
        *intermediate_pre_throat,
        station_throat,
        *intermediate_post_throat,
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

