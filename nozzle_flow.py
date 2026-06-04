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
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import brentq, minimize_scalar

from nasa9_parser import Species, parse_thermo_file, get_products_for_elements
from thermo_calc import h_over_RT, s_over_R, cp_over_R, g_over_RT, R_UNIVERSAL
from gibbs_solver import (
    solve_equilibrium,
    solve_equilibrium_HP,
    solve_equilibrium_SP,
    mixture_enthalpy,
    mixture_entropy,
    EquilibriumResult,
)
from iteration_logger import IterationLogger, NullLogger
from formula_parser import parse_formula


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
    # ── 1) производные по T при P = const ─────────────────────────────
    r_plus  = solve_equilibrium(species_list, element_abundances, T + delta_T, P,
                                include_condensed=True, verbose=False)
    r_minus = solve_equilibrium(species_list, element_abundances, T - delta_T, P,
                                include_condensed=True, verbose=False)
    cp_eq = (r_plus.enthalpy - r_minus.enthalpy) / (2 * delta_T)

    n_gas_plus  = sum(r_plus.moles[i]  for i, sp in enumerate(species_list) if sp.is_gas)
    n_gas_minus = sum(r_minus.moles[i] for i, sp in enumerate(species_list) if sp.is_gas)
    # ln V = ln n_gas + ln T - ln P  =>  d lnV/d lnT |_P = (d ln n_gas / d ln T) + 1
    dlnV_dlnT_P = T * (math.log(n_gas_plus) - math.log(n_gas_minus)) / (2 * delta_T) + 1.0

    # ── 2) производные по P при T = const ────────────────────────────
    dP = max(P * 1e-4, 1.0)
    r_pP = solve_equilibrium(species_list, element_abundances, T, P + dP,
                             include_condensed=True, verbose=False)
    r_mP = solve_equilibrium(species_list, element_abundances, T, P - dP,
                             include_condensed=True, verbose=False)
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
    compute_equilibrium_derivatives: bool = True,
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
    # Для быстрого режима можно отключить численные производные —
    # это значительно ускоряет многократные прогоны (поиск оптимального O/F).
    if compute_equilibrium_derivatives:
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
    else:
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
# Основная функция: расчёт сопла «от Pc до Pe» с произвольным числом сечений
# ─────────────────────────────────────────────────────────────────────────────

def solve_rocket_nozzle(
    oxidizer: Propellant,
    fuel: Propellant,
    P_chamber: float,
    P_exit: float,
    species_db: Dict[str, Species],
    n_intermediate_stations: int = 0,
    include_condensed: bool = True,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    max_gas_species: int = 60,
    compute_equilibrium_derivatives: bool = True,
) -> RocketPerformance:
    """Полный расчёт ракетного сопла в равновесном приближении.

    Параметры:
        oxidizer, fuel  — компоненты топлива (mass_kg в сумме обычно = 1 кг).
        P_chamber, P_exit — давления в камере и на срезе, Па.
        n_intermediate_stations — сколько дополнительных сечений между
                                  горловиной и срезом (0 = только injector/throat/exit).
        species_db      — база NASA-9.
        logger          — куда писать журнал итераций.

    Возвращает RocketPerformance со списком сечений и тяговыми характеристиками.
    """
    if logger is None:
        logger = NullLogger()

    # по требованию считаем только 4 сечения:
    # Injector, Nozzle inlet, Nozzle throat, Nozzle exit.
    n_intermediate_stations = 0

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
        compute_equilibrium_derivatives=compute_equilibrium_derivatives,
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
                           mass_total_g, H_chamber_per_kg,
                           compute_equilibrium_derivatives=False)
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
    station_throat = _make_station(
        'Nozzle throat', species_list, elements, throat_eq, P_throat,
        mass_total_g, H_chamber_per_kg,
        compute_equilibrium_derivatives=compute_equilibrium_derivatives,
    )
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
        compute_equilibrium_derivatives=compute_equilibrium_derivatives,
    )

    # ── 6) "Nozzle inlet" — формально считаем что вход в сопло = камера
    # (CEA так и делает: Injector ≡ Nozzle inlet при stagnation).
    station_inlet = _make_station(
        'Nozzle inlet', species_list, elements, chamber, P_chamber,
        mass_total_g, H_chamber_per_kg,
        compute_equilibrium_derivatives=compute_equilibrium_derivatives,
    )

    # ── 7) Промежуточные сечения между throat и exit ───────────────────
    intermediate = []
    if n_intermediate_stations > 0:
        # давления — лог-распределение от P_throat до P_exit
        P_grid = np.exp(np.linspace(
            math.log(P_throat), math.log(P_exit), n_intermediate_stations + 2,
        ))[1:-1]  # без концов
        for k, P_k in enumerate(P_grid, start=1):
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
                compute_equilibrium_derivatives=compute_equilibrium_derivatives,
            )
            intermediate.append(st)

    # ── 8) Ae/At — из сохранения массового расхода ────────────────────
    # m_dot = rho * V * A = const => A/At = (rho_t * V_t) / (rho * V)
    flux_throat = station_throat.mass_flux_kg_per_m2_s
    for st in [station_chamber, station_inlet, station_throat, *intermediate, station_exit]:
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

    # порядок: Injector, Nozzle inlet, Nozzle throat, [Section k...], Nozzle exit
    stations = [station_chamber, station_inlet, station_throat, *intermediate, station_exit]

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


def get_valid_propellant_components(species_db: Dict[str, Species]) -> Tuple[List[str], List[str]]:
    """Возвращает списки допустимых окислителей и горючих для GUI.

    Фильтрация выполняется в два шага:
    1) берём только «чистые» молекулы-реагенты из NASA-базы (без ионов/электронов);
    2) если доступен propellants_catalog.py, дополнительно ограничиваем выбор
       справочными компонентами из каталога (убираем экзотические/нерелевантные
       вещества, которые не должны появляться в обычном выборе топлива).
    """
    def _is_neutral_name(name: str) -> bool:
        lowered = name.lower()
        return ('+' not in name) and ('-' not in name) and lowered not in {'e-', 'electron'}

    def _oxidation_capacity(sp: Species) -> float:
        return sp.elements.get('O', 0.0)

    def _reduction_demand(sp: Species) -> float:
        e = sp.elements
        return 2.0 * e.get('C', 0.0) + 0.5 * e.get('H', 0.0) - e.get('O', 0.0)

    oxidizers: List[str] = []
    fuels: List[str] = []

    for name, sp in species_db.items():
        if not sp.is_reactant_only:
            continue
        if not _is_neutral_name(name):
            continue
        if sp.mol_weight <= 0:
            continue

        ox_cap = _oxidation_capacity(sp)
        red_dem = _reduction_demand(sp)

        if ox_cap > 0 and red_dem <= 0:
            oxidizers.append(name)
        if red_dem > 0:
            fuels.append(name)

    # Дополнительная «санитарная» фильтрация по каталогу RPA-style,
    # чтобы в списках выбора не было неподходящих компонентов.
    try:
        from propellants_catalog import OXIDIZERS, FUELS

        catalog_ox = {entry.name for entry in OXIDIZERS}
        catalog_fu = {entry.name for entry in FUELS}

        oxidizers = [name for name in oxidizers if name in catalog_ox]
        fuels = [name for name in fuels if name in catalog_fu]
    except Exception:
        # fallback: если каталог недоступен, оставляем базовую фильтрацию.
        pass

    return sorted(set(oxidizers)), sorted(set(fuels))


def optimize_lox_lh2_mixture_ratio(
    species_db: Dict[str, Species],
    P_chamber: float = 10e6,
    P_exit: float = 101325.0,
    alpha_min: float = 0.40,
    alpha_max: float = 0.90,
    coarse_points: int = 7,
) -> Dict[str, object]:
    """Быстрый поиск оптимального O/F для LOX/LH2 по максимуму Isp.

    Поиск ускорен:
    1) грубая сетка по alpha;
    2) локальная дооптимизация bounded Brent;
    3) fast-mode (без дорогих численных производных Cp_eq/Cv_eq).
    """
    if "O2(L)" not in species_db or "H2(L)" not in species_db:
        raise ValueError("Для оптимизации нужны O2(L) и H2(L) в NASA-базе.")

    of_stoich = stoichiometric_OF([species_db["O2(L)"]], [species_db["H2(L)"]])
    if not (math.isfinite(of_stoich) and of_stoich > 0):
        raise ValueError("Не удалось вычислить стехиометрическое O/F для O2(L)/H2(L).")

    cache: Dict[float, RocketPerformance] = {}
    eval_count = 0

    def _perf_for_alpha(alpha: float) -> RocketPerformance:
        nonlocal eval_count
        key = round(float(alpha), 6)
        if key in cache:
            return cache[key]

        of_target = key * of_stoich
        mass_fu = 1.0 / (1.0 + of_target)
        mass_ox = 1.0 - mass_fu

        perf = solve_rocket_nozzle(
            oxidizer=Propellant("O2(L)", mass_kg=mass_ox),
            fuel=Propellant("H2(L)", mass_kg=mass_fu),
            P_chamber=P_chamber,
            P_exit=P_exit,
            species_db=species_db,
            n_intermediate_stations=0,
            include_condensed=True,
            verbose=False,
            logger=NullLogger(),
            compute_equilibrium_derivatives=False,
        )
        cache[key] = perf
        eval_count += 1
        return perf

    def _objective(alpha: float) -> float:
        return -_perf_for_alpha(alpha).Isp_s

    coarse_points = max(5, int(coarse_points))
    coarse_grid = np.linspace(alpha_min, alpha_max, coarse_points)
    coarse_vals = [(_objective(a), a) for a in coarse_grid]
    _, alpha_seed = min(coarse_vals, key=lambda t: t[0])

    idx = int(np.argmin([v for v, _ in coarse_vals]))
    lo_idx = max(0, idx - 1)
    hi_idx = min(len(coarse_grid) - 1, idx + 1)
    lo = float(coarse_grid[lo_idx])
    hi = float(coarse_grid[hi_idx])
    if hi <= lo:
        lo, hi = alpha_min, alpha_max

    opt = minimize_scalar(
        _objective,
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': 5e-4, 'maxiter': 24},
    )

    alpha_best = float(opt.x)
    perf_fast = _perf_for_alpha(alpha_best)

    # Финальный точный пересчёт (те же 4 сечения, но с полноценными производными).
    of_target = alpha_best * of_stoich
    mass_fu = 1.0 / (1.0 + of_target)
    mass_ox = 1.0 - mass_fu
    perf_final = solve_rocket_nozzle(
        oxidizer=Propellant("O2(L)", mass_kg=mass_ox),
        fuel=Propellant("H2(L)", mass_kg=mass_fu),
        P_chamber=P_chamber,
        P_exit=P_exit,
        species_db=species_db,
        n_intermediate_stations=0,
        include_condensed=True,
        verbose=False,
        logger=NullLogger(),
        compute_equilibrium_derivatives=True,
    )

    return {
        'alpha_opt': alpha_best,
        'of_stoich': of_stoich,
        'of_opt': perf_final.O_F,
        'isp_opt_s': perf_final.Isp_s,
        'perf': perf_final,
        'perf_fast': perf_fast,
        'evaluations': eval_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Печать таблицы в стиле RPA / CEA
# ─────────────────────────────────────────────────────────────────────────────

def print_nozzle_table(perf: RocketPerformance, top_k_species: int = 12) -> None:
    """Печатает таблицу 'Thermodynamic properties' и 'Fractions of products'
    в стиле RPA: по сечениям, столбец за столбцом."""
    stations = perf.stations
    n = len(stations)

    print()
    print("=" * (28 + 16 * n))
    print(f"  Thermodynamic properties (O/F = {perf.O_F:.4f},  α = {perf.alpha:.4f})")
    print("=" * (28 + 16 * n))

    headers = [s.label for s in stations]
    print(f"  {'Parameter':<28s}" + "".join(f"{h:>16s}" for h in headers) + "   Unit")

    def line(name, fmt_spec, values, unit):
        cells = "".join(format(v, fmt_spec).rjust(16) for v in values)
        print(f"  {name:<28s}{cells}   {unit}")

    line("Pressure",      ".4f", [s.P_Pa/1e6 for s in stations], "MPa")
    line("Temperature",   ".4f", [s.T_K for s in stations],     "K")
    line("Enthalpy",      ".4f", [s.H_J_per_kg/1000 for s in stations], "kJ/kg")
    line("Entropy",       ".4f", [s.S_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Internal energy",".4f",[s.U_J_per_kg/1000 for s in stations], "kJ/kg")
    line("Cp (p=const, eq.)", ".4f", [s.cp_eq_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Cv (V=const, eq.)", ".4f", [s.cv_eq_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Gamma (eq.)",       ".4f", [s.gamma_eq for s in stations], "")
    line("Isentropic exp.",   ".4f", [s.gamma_s for s in stations], "")
    line("Gas constant",  ".4f", [s.R_specific_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Molecular weight",".4f",[s.mw_g_per_mol for s in stations], "kg/kmol")
    line("Density",       ".4f", [s.rho_kg_per_m3 for s in stations], "kg/m^3")
    line("Sonic velocity",".4f", [s.a_m_per_s for s in stations], "m/s")
    line("Velocity",      ".4f", [s.V_m_per_s for s in stations], "m/s")
    line("Mach number",   ".4f", [s.M for s in stations], "")
    # Ae/At — для камеры показываем 'infinity'
    ae_strs = []
    for s in stations:
        if math.isinf(s.Ae_At) or s.Ae_At > 1e6:
            ae_strs.append("infinity".rjust(16))
        else:
            ae_strs.append(f"{s.Ae_At:16.4f}")
    print(f"  {'Area ratio':<28s}" + "".join(ae_strs) + "   ")
    line("Mass flux",     ".4f", [s.mass_flux_kg_per_m2_s for s in stations], "kg/(m^2 s)")

    # фракции
    print()
    print("-" * (28 + 16 * n))
    print(f"  Fractions of the combustion products (top {top_k_species})")
    print("-" * (28 + 16 * n))

    # выбираем top_k веществ по максимуму мольной доли в любом сечении
    sp_names = stations[0].species_names
    max_xi = np.zeros(len(sp_names))
    for s in stations:
        max_xi = np.maximum(max_xi, s.mole_fractions)
    order = np.argsort(-max_xi)[:top_k_species]

    print(f"  {'Species':<28s}" + "".join(f"{h:>16s}" for h in headers))
    for idx in order:
        if max_xi[idx] < 1e-7:
            continue
        vals = "".join(f"{s.mole_fractions[idx]:16.7f}" for s in stations)
        print(f"  {sp_names[idx]:<28s}{vals}")

    print()
    print(f"  Isp (exit)   = {perf.Isp_s:.4f} с")
    print(f"  Isp (vacuum) = {perf.Isp_vac_s:.4f} с")
    print(f"  Cstar         = {perf.Cstar_m_per_s:.4f} м/с")
    print(f"  CF            = {perf.CF:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрация
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from equilibrium import find_thermo_db

    db = parse_thermo_file(find_thermo_db())

    # пример: жидкие H2/O2 (как в RPA: O2(L)@90.17K, H2(L)@20.27K)
    # O/F = 7.937, Pc = 10 МПа, Pe = 0.1013 МПа
    ox = Propellant("O2(L)", mass_kg=7.937)   # T_assigned=90.17 К из базы
    fu = Propellant("H2(L)", mass_kg=1.000)   # T_assigned=20.27 К из базы

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    with IterationLogger(os.path.join(log_dir, 'nozzle_H2_O2.log')) as logger:
        perf = solve_rocket_nozzle(
            oxidizer=ox, fuel=fu,
            P_chamber=10e6,    # 10 МПа
            P_exit=0.1013e6,   # 0.1013 МПа (1 атм)
            species_db=db,
            n_intermediate_stations=0,
            verbose=False,
            logger=logger,
        )

    print_nozzle_table(perf)
