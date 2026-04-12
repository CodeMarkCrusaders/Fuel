"""
Решатель задачи химического равновесия методом минимизации энергии Гиббса.

Алгоритм:
  1. Основной метод — прямая минимизация G/RT через scipy SLSQP
     с ограничениями-равенствами на сохранение элементного состава.
  2. Резервный метод — trust-constr, если SLSQP не сходится.

Справочник: NASA RP-1311, Gordon & McBride, 1994.
"""

import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nasa9_parser import Species
from thermo_calc import g_over_RT, R_UNIVERSAL


# Опорное давление (стандартное состояние): 1 бар
P_REFERENCE = 1e5  # Па

# Порог «следовой» концентрации: вещества с n_i < TRACE_CUTOFF считаются отсутствующими
TRACE_CUTOFF = 1e-30


@dataclass
class EquilibriumResult:
    """
    Результат расчёта химического равновесия.

    Содержит мольные количества всех веществ, доли, условия расчёта
    и диагностическую информацию об итерационном процессе.
    """
    converged: bool           # True, если оптимизатор сошёлся
    iterations: int           # число итераций
    T: float                  # температура, К
    P: float                  # давление, Па
    species_names: List[str]  # имена всех веществ
    mole_fractions: np.ndarray  # мольные доли (только газ)
    moles: np.ndarray           # мольные количества (газ + конденсат)
    total_moles: float          # суммарное число молей газа
    elements: Dict[str, float]  # заданный элементный состав
    phase: List[int]            # фаза каждого вещества (0=газ, 1=тв., 2=жидк.)
    g_over_rt: np.ndarray       # стандартные G⁰/(RT) для всех веществ
    residual: float             # невязка баланса элементов

    def get_gas_species(self) -> List[Tuple[str, float, float]]:
        """
        Возвращает газовые компоненты с концентрацией > TRACE_CUTOFF.

        Результат: список кортежей (имя, моли, мольная_доля),
        отсортированный по убыванию мольной доли.
        """
        total_gas = sum(
            self.moles[i]
            for i in range(len(self.species_names))
            if self.phase[i] == 0
        )
        gas_components = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] == 0 and self.moles[i] > TRACE_CUTOFF:
                mole_fraction = self.moles[i] / total_gas if total_gas > 0 else 0.0
                gas_components.append((name, self.moles[i], mole_fraction))
        gas_components.sort(key=lambda x: -x[2])
        return gas_components

    def get_condensed_species(self) -> List[Tuple[str, float]]:
        """
        Возвращает конденсированные компоненты с количеством > TRACE_CUTOFF.

        Результат: список кортежей (имя, моли), отсортированный по убыванию молей.
        """
        condensed = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] != 0 and self.moles[i] > TRACE_CUTOFF:
                condensed.append((name, self.moles[i]))
        condensed.sort(key=lambda x: -x[1])
        return condensed


def solve_equilibrium(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    include_condensed: bool = True,
    verbose: bool = False,
) -> EquilibriumResult:
    """
    Находит равновесный состав смеси минимизацией полной энергии Гиббса.

    Задача оптимизации:
        минимизировать  G/RT = Σᵢ nᵢ·[g⁰ᵢ/(RT) + ln(nᵢ/n_total) + ln(P/P⁰)]
        при условии     Σᵢ aₖᵢ·nᵢ = bₖ  для каждого элемента k

    где:
        nᵢ         — мольное количество вещества i
        g⁰ᵢ/(RT)   — безразмерная стандартная энергия Гиббса
        aₖᵢ        — стехиометрический коэффициент элемента k в веществе i
        bₖ         — заданное количество молей элемента k

    Для конденсированных фаз слагаемые с ln отсутствуют (активность = 1).

    Параметры:
        species_list:       Список веществ-кандидатов (газы + конденсат)
        element_abundances: Словарь {элемент: количество_молей}
        T:                  Температура, К
        P:                  Давление, Па
        include_condensed:  Учитывать конденсированные фазы
        verbose:            Выводить ли подробный лог итераций
    """
    # Разделяем вещества на газовые и конденсированные
    gas_species = [sp for sp in species_list if sp.is_gas]
    cond_species = [sp for sp in species_list if sp.is_condensed] if include_condensed else []
    all_species = gas_species + cond_species

    N_gas = len(gas_species)
    N_cond = len(cond_species)
    N_total = N_gas + N_cond

    if N_total == 0:
        raise ValueError("Список веществ для расчёта равновесия пуст.")

    # Составляем упорядоченный список элементов и их количества (вектор b)
    elem_list = sorted(element_abundances.keys())
    N_elem = len(elem_list)
    elem_index = {element: i for i, element in enumerate(elem_list)}
    b = np.array([element_abundances[e] for e in elem_list], dtype=float)

    # Стехиометрическая матрица a[k, i]: сколько атомов элемента k в веществе i
    a = np.zeros((N_elem, N_total))
    for i, sp in enumerate(all_species):
        for element, count in sp.elements.items():
            if element in elem_index:
                a[elem_index[element], i] = count

    # Стандартные безразмерные энергии Гиббса при температуре T
    g0_RT = np.array([g_over_RT(sp, T) for sp in all_species])

    # Логарифм отношения давления к опорному
    ln_pressure_ratio = math.log(P / P_REFERENCE)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Минимизация Гиббса: T = {T:.2f} К, P = {P / 1e5:.4f} бар")
        print(f"Газов: {N_gas}, конденсата: {N_cond}, элементов: {N_elem}")
        print(f"Элементы: {elem_list}")
        print(f"b = {b}")
        print(f"{'=' * 70}")

    # Минимальное число молей для газа (предотвращает ln(0))
    n_min = 1e-20

    # ------------------------------------------------------------------
    # Целевая функция: G/RT как функция вектора n
    # ------------------------------------------------------------------
    def gibbs_total(n_vec: np.ndarray) -> float:
        """Полная безразмерная энергия Гиббса смеси."""
        G = 0.0
        n_gas_total = max(np.sum(n_vec[:N_gas]), n_min)
        ln_n_total = math.log(n_gas_total)

        # Вклад газовых компонентов: μᵢ/RT = g⁰ᵢ/RT + ln(nᵢ/n_total) + ln(P/P⁰)
        for i in range(N_gas):
            ni = max(n_vec[i], n_min)
            G += ni * (g0_RT[i] + math.log(ni) - ln_n_total + ln_pressure_ratio)

        # Вклад конденсированных фаз: μⱼ/RT = g⁰ⱼ/RT (активность = 1)
        for j in range(N_cond):
            nj = max(n_vec[N_gas + j], 0.0)
            G += nj * g0_RT[N_gas + j]

        return G

    # ------------------------------------------------------------------
    # Градиент: ∂(G/RT)/∂nᵢ = химический потенциал вещества i
    # ------------------------------------------------------------------
    def gibbs_gradient(n_vec: np.ndarray) -> np.ndarray:
        """
        Аналитический градиент G/RT по всем мольным количествам.

        Для газа:       ∂G/∂nᵢ = g⁰ᵢ/RT + ln(nᵢ/n_total) + ln(P/P⁰)
        Для конденсата: ∂G/∂nⱼ = g⁰ⱼ/RT
        """
        grad = np.zeros(N_total)
        n_gas_total = max(np.sum(n_vec[:N_gas]), n_min)
        ln_n_total = math.log(n_gas_total)

        for i in range(N_gas):
            ni = max(n_vec[i], n_min)
            grad[i] = g0_RT[i] + math.log(ni) - ln_n_total + ln_pressure_ratio

        for j in range(N_cond):
            grad[N_gas + j] = g0_RT[N_gas + j]

        return grad

    # ------------------------------------------------------------------
    # Ограничения: баланс элементов a @ n = b
    # ------------------------------------------------------------------
    constraints = []
    for k in range(N_elem):
        constraints.append({
            'type': 'eq',
            'fun': lambda n, kk=k: np.dot(a[kk, :], n) - b[kk],
            'jac': lambda n, kk=k: a[kk, :],
        })

    # Границы переменных: газ >= n_min, конденсат >= 0
    bounds = [(n_min, None)] * N_gas + [(0.0, None)] * N_cond

    # ------------------------------------------------------------------
    # Начальное приближение: метод NNLS для распределения элементов
    # ------------------------------------------------------------------
    n0 = np.ones(N_total) * n_min
    try:
        from scipy.optimize import nnls
        n_init, _ = nnls(a, b)
        for i in range(N_total):
            n0[i] = max(n_init[i], n_min)
    except Exception:
        # Запасной вариант: равномерное распределение
        total_atoms = float(np.sum(b))
        n0[:N_gas] = total_atoms / max(N_gas, 1)

    if verbose:
        G_init = gibbs_total(n0)
        print(f"Начальное G/RT = {G_init:.6f}")
        print(f"Начальная сумма молей газа = {np.sum(n0[:N_gas]):.6f}")

    # ------------------------------------------------------------------
    # Решение SLSQP (основной метод)
    # ------------------------------------------------------------------
    iteration_counter = [0]

    def log_iteration(xk):
        iteration_counter[0] += 1
        if verbose and iteration_counter[0] % 50 == 0:
            print(f"  Итерация {iteration_counter[0]}: G/RT = {gibbs_total(xk):.8f}")

    opt_result = minimize(
        gibbs_total,
        n0,
        method='SLSQP',
        jac=gibbs_gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False},
        callback=log_iteration if verbose else None,
    )

    n_solution = opt_result.x

    # ------------------------------------------------------------------
    # Резервный метод: trust-constr, если SLSQP не сошёлся
    # ------------------------------------------------------------------
    if not opt_result.success:
        if verbose:
            print(f"  SLSQP: {opt_result.message}. Пробуем trust-constr...")

        lin_constraint = LinearConstraint(a, b, b)
        bounds_tc = [(n_min if i < N_gas else 0.0, None) for i in range(N_total)]

        opt_result2 = minimize(
            gibbs_total,
            n0,
            method='trust-constr',
            jac=gibbs_gradient,
            bounds=bounds_tc,
            constraints=lin_constraint,
            options={'maxiter': 5000, 'verbose': 0},
        )

        # Принимаем trust-constr, только если он дал меньшее значение G
        if opt_result2.fun < opt_result.fun:
            opt_result = opt_result2
            n_solution = opt_result.x

    # ------------------------------------------------------------------
    # Постобработка: проверяем баланс элементов
    # ------------------------------------------------------------------
    element_residual = 0.0
    for k in range(N_elem):
        computed = np.dot(a[k, :], n_solution)
        element_residual += abs(computed - b[k]) / max(b[k], 1e-30)
    element_residual /= max(N_elem, 1)

    n_gas_total_final = max(np.sum(n_solution[:N_gas]), n_min)

    # Мольные доли газовой фазы
    mole_fractions = np.zeros(N_total)
    mole_fractions[:N_gas] = n_solution[:N_gas] / n_gas_total_final

    # Считаем сходившимся, если оптимизатор доволен или невязка мала
    converged = opt_result.success or element_residual < 1e-6

    if verbose:
        print(f"\n  Результат оптимизатора: {opt_result.message}")
        print(f"  G/RT = {opt_result.fun:.8f}")
        print(f"  Невязка баланса элементов = {element_residual:.2e}")
        print(f"  Суммарные моли газа = {n_gas_total_final:.6f}")
        print(f"  Итераций: {opt_result.nit}")
        if converged:
            print("  ✓ Сошлось!")
        else:
            print(f"  ✗ Не сошлось (невязка элементов: {element_residual:.2e})")

    return EquilibriumResult(
        converged=converged,
        iterations=opt_result.nit,
        T=T,
        P=P,
        species_names=[sp.name for sp in all_species],
        mole_fractions=mole_fractions,
        moles=n_solution.copy(),
        total_moles=n_gas_total_final,
        elements=element_abundances,
        phase=[sp.phase for sp in all_species],
        g_over_rt=g0_RT,
        residual=element_residual,
    )
