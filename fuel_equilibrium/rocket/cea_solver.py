# -*- coding: utf-8 -*-
"""
CEA-совместимый решатель для ракетного сопла на базе Cantera.

Cantera использует NASA-полиномы (те же, что и в CEA), и встроенная
функция `equilibrate('HP')`/`equilibrate('SP')` решает задачи равновесия
методом минимизации энергии Гиббса — алгоритмически эквивалентно NASA CEA.

Этот модуль предоставляет интерфейс, совместимый с `nozzle_flow.solve_rocket_nozzle`,
поэтому GUI может переключать решатели одним флагом.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy.optimize import brentq

try:
    import cantera as ct
    CANTERA_AVAILABLE = True
except ImportError:
    CANTERA_AVAILABLE = False

# Используем те же dataclass-структуры, что и в nozzle_flow:
from .nozzle_flow import StationResult, RocketPerformance, Propellant


# ─────────────────────────────────────────────────────────────────────────────
# Маппинг ракетных компонентов на состав для Cantera
# ─────────────────────────────────────────────────────────────────────────────

# Имена реагентов в виде "имя_в_базе_NASA -> (имя_в_Cantera, T_assigned_K)"
# Для криогенных топлив используем газовую форму при их температуре кипения,
# как это делает CEA (вычитая теплоту парообразования из энтальпии).
PROPELLANT_MAP = {
    'H2(L)':    ('H2',    20.27),
    'O2(L)':    ('O2',    90.17),
    'CH4(L)':   ('CH4',   111.643),
    'N2H4(L)':  ('N2H4',  298.15),
    'RP-1':     ('C12H26', 298.15),  # приближение
    'H2':       ('H2',    298.15),
    'O2':       ('O2',    298.15),
    'CH4':      ('CH4',   298.15),
    'N2':       ('N2',    298.15),
    'CO':       ('CO',    298.15),
    'CO2':      ('CO2',   298.15),
}

# Энтальпии испарения, Дж/кг — учитываются для криогенных реагентов
# (поправка на то, что в баках находится жидкость, а Cantera работает с газом).
HEAT_OF_VAPORIZATION = {
    'H2(L)':  445_590,   # Дж/кг при 20.27 К
    'O2(L)':  213_058,   # Дж/кг при 90.17 К
    'CH4(L)': 511_000,   # Дж/кг при 111.6 К
}


def _resolve_component(name: str) -> tuple:
    """Возвращает (cantera_name, T_default_K, latent_heat_J_per_kg)."""
    if name in PROPELLANT_MAP:
        ct_name, T = PROPELLANT_MAP[name]
        latent = HEAT_OF_VAPORIZATION.get(name, 0.0)
        return ct_name, T, latent
    # Если имя похоже на стандартное Cantera-имя, пытаемся использовать as-is
    return name, 298.15, 0.0


def _build_mass_fractions(oxidizer: Propellant, fuel: Propellant) -> dict:
    """Возвращает словарь массовых долей для Cantera."""
    ox_name, _, _ = _resolve_component(oxidizer.name)
    fu_name, _, _ = _resolve_component(fuel.name)
    total = oxidizer.mass_kg + fuel.mass_kg
    Y = {ox_name: oxidizer.mass_kg / total, fu_name: fuel.mass_kg / total}
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт станции (одного сечения) — Cantera variant
# ─────────────────────────────────────────────────────────────────────────────

def _make_station_cantera(
    label: str,
    gas: 'ct.Solution',
    P: float,
    H_chamber_per_kg: float,
    species_list: List[str],
) -> StationResult:
    """Собирает StationResult из текущего состояния Cantera-газа."""
    T = gas.T
    P_state = gas.P
    H_per_kg = gas.enthalpy_mass
    S_per_kg = gas.entropy_mass
    rho = gas.density
    mw = gas.mean_molecular_weight  # kg/kmol = g/mol

    # Замороженные Cp/Cv (Cantera по умолчанию даёт frozen)
    cp_f = gas.cp_mass
    cv_f = gas.cv_mass
    gamma_f = cp_f / cv_f if cv_f > 0 else 1.4

    # Равновесные свойства: используем встроенные cp_equilibrium и т.д.
    # Cantera 2.5+ имеет cp_mass с режимом equilibrium через
    # gas.set_equivalence_ratio... Здесь оценим численно
    try:
        cp_eq, cv_eq, gamma_s, a_eq = _eq_properties_cantera(gas)
        gamma_eq = cp_eq / cv_eq if cv_eq > 0 else gamma_f
    except Exception:
        cp_eq = cp_f
        cv_eq = cv_f
        gamma_eq = gamma_f
        gamma_s = gamma_f
        R_specific = ct.gas_constant / (mw if mw > 0 else 28.97)
        a_eq = math.sqrt(gamma_s * R_specific * T)

    R_universal = ct.gas_constant  # 8314.46 Дж/(кмоль·К)
    R_specific = R_universal / mw  # Дж/(кг·К)  (mw в кг/кмоль)

    # Скорость потока из баланса энергии
    dh = H_chamber_per_kg - H_per_kg
    V = math.sqrt(max(2.0 * dh, 0.0))
    M = V / a_eq if a_eq > 0 else 0.0

    U_per_kg = H_per_kg - P_state / rho if rho > 0 else H_per_kg
    mass_flux = rho * V

    # Доли компонентов — берём top-30 по массовой доле
    Y_arr = gas.Y
    X_arr = gas.X
    n_total = 1.0 / mw * 1000.0  # моль/кг (mw в г/моль)

    return StationResult(
        label=label,
        P_Pa=P_state, T_K=T,
        H_J_per_kg=H_per_kg, S_J_per_kgK=S_per_kg,
        U_J_per_kg=U_per_kg,
        cp_frozen_J_per_kgK=cp_f,
        cv_frozen_J_per_kgK=cv_f,
        gamma_frozen=gamma_f,
        cp_eq_J_per_kgK=cp_eq,
        cv_eq_J_per_kgK=cv_eq,
        gamma_eq=gamma_eq,
        gamma_s=gamma_s,
        a_m_per_s=a_eq,
        V_m_per_s=V,
        M=M,
        rho_kg_per_m3=rho,
        n_moles=n_total,
        mw_g_per_mol=mw,
        R_specific_J_per_kgK=R_specific,
        Ae_At=float('nan'),
        mass_flux_kg_per_m2_s=mass_flux,
        moles=np.array([X_arr[i] * n_total for i in range(len(species_list))]),
        mole_fractions=X_arr.copy(),
        mass_fractions=Y_arr.copy(),
        species_names=list(species_list),
    )


def _eq_properties_cantera(gas: 'ct.Solution', dT: float = 1.0, dP_frac: float = 1e-4):
    """Численные равновесные Cp_eq, Cv_eq, gamma_s, a_eq.

    Тот же приём, что и в nozzle_flow.equilibrium_cp_and_sound_speed,
    только реализован через Cantera.
    """
    T0, P0 = gas.T, gas.P
    Y0 = gas.Y.copy()

    # T+
    gas.TPY = T0 + dT, P0, Y0
    gas.equilibrate('TP')
    h_plus = gas.enthalpy_mass
    n_gas_plus = 1.0 / gas.mean_molecular_weight  # моль/г
    # T-
    gas.TPY = T0 - dT, P0, Y0
    gas.equilibrate('TP')
    h_minus = gas.enthalpy_mass
    n_gas_minus = 1.0 / gas.mean_molecular_weight

    cp_eq = (h_plus - h_minus) / (2 * dT)
    dlnV_dlnT_P = T0 * (math.log(n_gas_plus) - math.log(n_gas_minus)) / (2 * dT) + 1.0

    # P-производные
    dP = max(P0 * dP_frac, 1.0)
    gas.TPY = T0, P0 + dP, Y0
    gas.equilibrate('TP')
    n_gas_pP = 1.0 / gas.mean_molecular_weight
    gas.TPY = T0, P0 - dP, Y0
    gas.equilibrate('TP')
    n_gas_mP = 1.0 / gas.mean_molecular_weight
    dlnV_dlnP_T = P0 * (math.log(n_gas_pP) - math.log(n_gas_mP)) / (2 * dP) - 1.0

    # Восстанавливаем исходное состояние
    gas.TPY = T0, P0, Y0
    gas.equilibrate('TP')

    R_univ = ct.gas_constant  # Дж/(кмоль·К)
    mw = gas.mean_molecular_weight  # кг/кмоль
    nR = R_univ / mw  # Дж/(кг·К) — для удельных величин

    if abs(dlnV_dlnP_T) > 1e-30:
        cv_eq = cp_eq + nR * dlnV_dlnT_P**2 / dlnV_dlnP_T
    else:
        cv_eq = cp_eq * 0.9

    if cp_eq > 1e-30 and abs(dlnV_dlnP_T) > 1e-30:
        denom = dlnV_dlnP_T + nR * dlnV_dlnT_P**2 / cp_eq
        gamma_s = -1.0 / denom if denom < -1e-30 else 1.4
    else:
        gamma_s = 1.4

    R_spec = R_univ / mw
    a_eq = math.sqrt(max(gamma_s * R_spec * T0, 0.0))

    return cp_eq, cv_eq, gamma_s, a_eq


# ─────────────────────────────────────────────────────────────────────────────
# Стехиометрия — для совместимости с nozzle_flow
# ─────────────────────────────────────────────────────────────────────────────

def _stoich_OF_cantera(oxidizer: Propellant, fuel: Propellant) -> float:
    """Стехиометрическое массовое O/F через Cantera."""
    if not CANTERA_AVAILABLE:
        return float('nan')
    ox_name, _, _ = _resolve_component(oxidizer.name)
    fu_name, _, _ = _resolve_component(fuel.name)
    try:
        gas = _make_gas([ox_name, fu_name])
        gas.set_equivalence_ratio(1.0, fu_name, ox_name)
        # массовые доли при стехиометрии:
        Y_ox = gas.mass_fraction_dict().get(ox_name, 0.0)
        Y_fu = gas.mass_fraction_dict().get(fu_name, 0.0)
        if Y_fu > 0:
            return Y_ox / Y_fu
    except Exception:
        pass
    return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Создание газа Cantera с подходящим механизмом
# ─────────────────────────────────────────────────────────────────────────────

# Кэш чтобы не создавать gas-объекты заново
_GAS_CACHE = {}

def _pick_mechanism(element_set: set) -> str:
    """Выбирает подходящий yaml-механизм Cantera."""
    # gri30 покрывает C/H/O/N (метан, водород, природный газ)
    # nasa_gas — содержит больше элементов (фтор, хлор и т.п.)
    if element_set <= {'C', 'H', 'O', 'N', 'AR'}:
        return 'gri30.yaml'
    return 'nasa_gas.yaml'


def _make_gas(needed_species: List[str]) -> 'ct.Solution':
    """Создаёт Cantera-объект Solution, гарантируя наличие нужных видов."""
    mech = 'gri30.yaml'
    key = (mech, tuple(sorted(needed_species)))
    if key in _GAS_CACHE:
        gas = _GAS_CACHE[key]
        return gas
    try:
        gas = ct.Solution(mech)
        # Проверяем, что все нужные виды есть
        gas_species = set(gas.species_names)
        missing = [s for s in needed_species if s not in gas_species]
        if missing:
            raise ValueError(f"в {mech} не хватает видов: {missing}")
        _GAS_CACHE[key] = gas
        return gas
    except Exception:
        # Альтернативный механизм
        for mech2 in ('h2o2.yaml', 'air.yaml'):
            try:
                gas = ct.Solution(mech2)
                _GAS_CACHE[(mech2, tuple(sorted(needed_species)))] = gas
                return gas
            except Exception:
                continue
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция: расчёт сопла через Cantera (аналог solve_rocket_nozzle)
# ─────────────────────────────────────────────────────────────────────────────

def solve_rocket_nozzle_cea(
    oxidizer: Propellant,
    fuel: Propellant,
    P_chamber: float,
    P_exit: float,
    n_intermediate_stations: int = 3,
    include_condensed: bool = False,
    verbose: bool = False,
    progress_cb=None,
) -> RocketPerformance:
    """Расчёт ракетного сопла через Cantera (CEA-эквивалент).

    Аналог `solve_rocket_nozzle`, но в качестве решателя используется
    Cantera, чей алгоритм минимизации Гиббса идентичен NASA CEA.
    """
    if not CANTERA_AVAILABLE:
        raise RuntimeError(
            "Решатель CEA (Cantera) недоступен.\n"
            "Установите cantera: pip install cantera"
        )

    ox_name, T_ox_default, ox_latent = _resolve_component(oxidizer.name)
    fu_name, T_fu_default, fu_latent = _resolve_component(fuel.name)

    T_ox = oxidizer.T_K if oxidizer.T_K is not None else T_ox_default
    T_fu = fuel.T_K if fuel.T_K is not None else T_fu_default

    gas = _make_gas([ox_name, fu_name])
    Y_dict = _build_mass_fractions(oxidizer, fuel)

    total_mass = oxidizer.mass_kg + fuel.mass_kg

    # Стехиометрическое O/F и actual
    of_actual = oxidizer.mass_kg / fuel.mass_kg
    of_stoich = _stoich_OF_cantera(oxidizer, fuel)
    alpha = of_actual / of_stoich if of_stoich and not math.isnan(of_stoich) else float('nan')
    phi = 1.0 / alpha if alpha and not math.isnan(alpha) else float('nan')

    # ── 1) Энтальпия реагентов: считаем при их собственных T
    # CEA-подход: gas at T_ref, потом сместить на δH соответствующий нагреву/охлаждению.
    # Для криогенных учитываем теплоту парообразования.
    h_react = 0.0
    # Окислитель
    gas.TPY = T_ox, P_chamber, {ox_name: 1.0}
    h_ox = gas.enthalpy_mass  # энтальпия чистого окислителя при T_ox
    h_ox -= ox_latent  # вычитаем теплоту испарения (жидкость холоднее газа)
    # Горючее
    gas.TPY = T_fu, P_chamber, {fu_name: 1.0}
    h_fu = gas.enthalpy_mass
    h_fu -= fu_latent

    Y_ox = oxidizer.mass_kg / total_mass
    Y_fu = fuel.mass_kg / total_mass
    h_react_per_kg = Y_ox * h_ox + Y_fu * h_fu

    if progress_cb:
        progress_cb("Решение HP-задачи в камере...")

    # ── 2) Камера (Injector): HP — equilibrate('HP')
    # Сначала задаём массовые доли реагентов и удельную энтальпию при разумной T,
    # потом делаем HP-равновесие (состав равновесный, h остаётся равной h_react).
    gas.TPY = 2500.0, P_chamber, Y_dict
    # Используем подход: задать (h, P) через UV-обходной путь
    # h = u + p/rho, для идеального газа удобнее через TP+коррекцию по h
    # Самый надёжный способ — итерационно подобрать T, при которой h(equilibrium)=h_react
    from scipy.optimize import brentq as _brentq
    def _hp_residual(T_try):
        gas.TPY = T_try, P_chamber, Y_dict
        gas.equilibrate('TP')
        return gas.enthalpy_mass - h_react_per_kg

    # Грубо ищем диапазон
    T_lo, T_hi = 1500.0, 5000.0
    try:
        f_lo = _hp_residual(T_lo)
        f_hi = _hp_residual(T_hi)
        if f_lo * f_hi < 0:
            T_eq = _brentq(_hp_residual, T_lo, T_hi, xtol=0.01, rtol=1e-7, maxiter=80)
        else:
            # расширяем
            for T_lo2, T_hi2 in [(800, 6000), (300, 7000)]:
                f_lo = _hp_residual(T_lo2)
                f_hi = _hp_residual(T_hi2)
                if f_lo * f_hi < 0:
                    T_eq = _brentq(_hp_residual, T_lo2, T_hi2,
                                   xtol=0.01, rtol=1e-7, maxiter=80)
                    break
            else:
                # последний шанс — equilibrate('HP') в нативном режиме
                gas.TPY = 3000.0, P_chamber, Y_dict
                gas.equilibrate('HP')
                T_eq = gas.T
        # финальная установка состояния
        gas.TPY = T_eq, P_chamber, Y_dict
        gas.equilibrate('TP')
    except Exception:
        gas.TPY = 3000.0, P_chamber, Y_dict
        gas.equilibrate('HP')

    T_chamber = gas.T
    H_chamber_per_kg = gas.enthalpy_mass
    S_chamber_per_kg = gas.entropy_mass

    species_list = list(gas.species_names)
    station_chamber = _make_station_cantera(
        'Injector', gas, P_chamber, H_chamber_per_kg, species_list
    )
    station_inlet = _make_station_cantera(
        'Nozzle inlet', gas, P_chamber, H_chamber_per_kg, species_list
    )

    # ── 3) Горловина (M=1): ищем брентом по P
    if progress_cb:
        progress_cb("Поиск горловины (M=1)...")

    def _sp_solve(S_target, P_target, T_guess=None):
        """SP-equilibrium через итерации по T (более надёжно, чем gas.SPY)."""
        if T_guess is None:
            T_guess = T_chamber * 0.9
        def _resid(T_try):
            gas.TPY = T_try, P_target, Y_dict
            gas.equilibrate('TP')
            return gas.entropy_mass - S_target
        try:
            T_lo, T_hi = 200.0, max(T_chamber * 1.5, 5000.0)
            f_lo = _resid(T_lo)
            f_hi = _resid(T_hi)
            if f_lo * f_hi < 0:
                T_eq = brentq(_resid, T_lo, T_hi, xtol=0.01, rtol=1e-7, maxiter=80)
            else:
                # последний шанс — нативный equilibrate
                gas.TPY = T_guess, P_target, Y_dict
                gas.equilibrate('SP')
                T_eq = gas.T
            gas.TPY = T_eq, P_target, Y_dict
            gas.equilibrate('TP')
        except Exception:
            gas.TPY = T_guess, P_target, Y_dict
            gas.equilibrate('SP')

    def throat_residual(P_try):
        _sp_solve(S_chamber_per_kg, P_try, T_guess=T_chamber * 0.9)
        st = _make_station_cantera(
            'throat?', gas, P_try, H_chamber_per_kg, species_list
        )
        return st.M - 1.0, st

    gamma0 = station_chamber.gamma_s
    P_throat_init = P_chamber * (2.0 / (gamma0 + 1.0)) ** (gamma0 / (gamma0 - 1.0))
    P_lo = max(P_throat_init * 0.5, P_exit * 1.01)
    P_hi = min(P_throat_init * 1.5, P_chamber * 0.999)

    f_lo, _ = throat_residual(P_lo)
    f_hi, _ = throat_residual(P_hi)
    expand = 0
    while f_lo * f_hi > 0 and expand < 5:
        P_lo = max(P_lo * 0.7, P_exit * 1.001)
        P_hi = min(P_hi * 1.3, P_chamber * 0.9999)
        f_lo, _ = throat_residual(P_lo)
        f_hi, _ = throat_residual(P_hi)
        expand += 1

    if f_lo * f_hi <= 0:
        P_throat = brentq(lambda P: throat_residual(P)[0], P_lo, P_hi,
                          xtol=P_chamber*1e-7, rtol=1e-6, maxiter=80)
    else:
        P_throat = P_throat_init

    _, station_throat = throat_residual(P_throat)
    station_throat.label = 'Nozzle throat'

    # ── 4) Срез сопла
    if progress_cb:
        progress_cb("Расчёт среза сопла...")
    _sp_solve(S_chamber_per_kg, P_exit, T_guess=T_chamber * 0.6)
    station_exit = _make_station_cantera(
        'Nozzle exit', gas, P_exit, H_chamber_per_kg, species_list
    )

    # ── 5) Промежуточные сечения
    intermediate = []
    if n_intermediate_stations > 0:
        if progress_cb:
            progress_cb(f"Расчёт {n_intermediate_stations} промежуточных сечений...")
        P_grid = np.exp(np.linspace(
            math.log(P_throat), math.log(P_exit), n_intermediate_stations + 2,
        ))[1:-1]
        for k, P_k in enumerate(P_grid, start=1):
            _sp_solve(S_chamber_per_kg, float(P_k),
                      T_guess=station_throat.T_K * 0.8)
            st = _make_station_cantera(
                f'Section {k}', gas, float(P_k), H_chamber_per_kg, species_list
            )
            intermediate.append(st)

    # ── 6) Ae/At
    flux_throat = station_throat.mass_flux_kg_per_m2_s
    for st in [station_chamber, station_inlet, station_throat, *intermediate, station_exit]:
        if st.mass_flux_kg_per_m2_s > 1e-30:
            st.Ae_At = flux_throat / st.mass_flux_kg_per_m2_s
        else:
            st.Ae_At = float('inf')

    # ── 7) Тяговые характеристики
    V_exit = station_exit.V_m_per_s
    g0 = 9.80665
    Isp_s = V_exit / g0
    if flux_throat > 1e-30:
        Isp_vac = (V_exit + P_exit * station_exit.Ae_At / flux_throat) / g0
    else:
        Isp_vac = Isp_s
    Cstar = P_chamber / flux_throat if flux_throat > 1e-30 else float('nan')
    CF = V_exit / Cstar if Cstar > 0 else float('nan')

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


# ─────────────────────────────────────────────────────────────────────────────
# Геометрия конического сопла — для построения профиля по длине
# ─────────────────────────────────────────────────────────────────────────────

def build_nozzle_geometry(
    stations: List[StationResult],
    L_chamber: float = 0.10,    # м — длина камеры
    L_conv: float = 0.05,       # м — длина конфузора
    L_div: float = 0.20,        # м — длина дивергента
    R_throat: float = 1.0,      # м — радиус горловины (нормировка)
) -> np.ndarray:
    """Строит координату x вдоль сопла для каждой станции.

    Возвращает массив (N,) с x-координатами в метрах.
    Геометрия — простая коническая модель:
      - камера 0…L_chamber  (Injector, Nozzle inlet)
      - конфузор L_ch…L_ch+L_conv  (от nozzle inlet до throat)
      - дивергент L_ch+L_conv…L_ch+L_conv+L_div (от throat до exit)
    """
    n = len(stations)
    x = np.zeros(n)
    L_total = L_chamber + L_conv + L_div

    for i, st in enumerate(stations):
        lab = st.label.lower()
        if lab in ('injector', 'камера'):
            x[i] = 0.0
        elif lab in ('nozzle inlet', 'вход в сопло'):
            x[i] = L_chamber
        elif lab in ('nozzle throat', 'горловина'):
            x[i] = L_chamber + L_conv
        elif lab in ('nozzle exit', 'срез сопла'):
            x[i] = L_total
        else:
            # Промежуточные сечения — распределяем равномерно между throat и exit
            # на основе их Ae/At (логарифмическая интерполяция)
            try:
                ratio = math.log(max(st.Ae_At, 1.0)) / math.log(max(stations[-1].Ae_At, 1.001))
                x[i] = L_chamber + L_conv + L_div * ratio
            except Exception:
                # fallback — равномерно
                section_idx = i - 2  # после injector, inlet, throat
                total_intermediate = n - 4
                if total_intermediate > 0:
                    x[i] = L_chamber + L_conv + L_div * section_idx / (total_intermediate + 1)
                else:
                    x[i] = L_chamber + L_conv

    return x


def nozzle_radius(stations: List[StationResult], R_chamber_rel: float = 3.0) -> np.ndarray:
    """Радиус сопла в каждом сечении, нормированный на R_throat = 1.

    R(x) / R_throat = sqrt(A(x) / A_throat) = sqrt(Ae/At)

    Для injector/inlet (где V≈0 ⇒ Ae/At формально → ∞) берём фиксированный
    радиус камеры R_chamber_rel = 3 (типичное значение для ЖРД).
    """
    n = len(stations)
    r = np.ones(n)
    # Найдём максимальный реальный (не-injector/inlet) радиус для масштаба
    real_radii = []
    for st in stations:
        lab = st.label.lower()
        if lab in ('injector', 'nozzle inlet', 'камера', 'вход в сопло'):
            continue
        if math.isfinite(st.Ae_At) and st.Ae_At > 0 and st.Ae_At < 1e4:
            real_radii.append(math.sqrt(st.Ae_At))
    R_chamber = max(R_chamber_rel, max(real_radii) * 0.9 if real_radii else R_chamber_rel)

    for i, st in enumerate(stations):
        lab = st.label.lower()
        if lab in ('injector', 'nozzle inlet', 'камера', 'вход в сопло'):
            r[i] = R_chamber
        elif math.isfinite(st.Ae_At) and st.Ae_At > 0 and st.Ae_At < 1e4:
            r[i] = math.sqrt(st.Ae_At)
        else:
            r[i] = R_chamber
    return r
