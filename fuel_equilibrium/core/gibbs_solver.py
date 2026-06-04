# минимизация энергии Гиббса для поиска равновесного состава
# метод: SLSQP с ограничениями на сохранение элементов
# резерв: trust-constr если SLSQP не сошёлся
# справочник: NASA RP-1311, Gordon & McBride, 1994

import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .nasa9_parser import Species
from .thermo_calc import g_over_RT, h_over_RT, s_over_R, R_UNIVERSAL
from ..io.iteration_logger import IterationLogger, NullLogger


P_REF = 1e5      # опорное давление, 1 бар
TRACE = 1e-30    # порог "следовой" концентрации


@dataclass
class EquilibriumResult:
    converged: bool
    iterations: int
    T: float
    P: float
    species_names: List[str]
    mole_fractions: np.ndarray
    moles: np.ndarray
    total_moles: float
    elements: Dict[str, float]
    phase: List[int]
    g_over_rt: np.ndarray
    residual: float
    # энтальпия и энтропия смеси при найденном составе и T,P, Дж и Дж/К
    enthalpy: float = 0.0
    entropy: float = 0.0
    # тип задачи: 'TP', 'HP', 'SP'
    problem_type: str = 'TP'

    def get_gas_species(self) -> List[Tuple[str, float, float]]:
        """Газовые компоненты выше порога, отсортированные по доле."""
        total_gas = sum(self.moles[i] for i in range(len(self.species_names)) if self.phase[i] == 0)
        result = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] == 0 and self.moles[i] > TRACE:
                xi = self.moles[i] / total_gas if total_gas > 0 else 0.0
                result.append((name, self.moles[i], xi))
        result.sort(key=lambda x: -x[2])
        return result

    def get_condensed_species(self) -> List[Tuple[str, float]]:
        result = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] != 0 and self.moles[i] > TRACE:
                result.append((name, self.moles[i]))
        result.sort(key=lambda x: -x[1])
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для термодинамических свойств смеси
# ──────────────────────────────────────────────────────────────────────────────

def mixture_enthalpy(species_list: List[Species], n_vec: np.ndarray, T: float) -> float:
    """Полная энтальпия смеси, Дж.

    H = sum_i n_i * h_i(T),  h_i(T) = (H/RT)_i * R * T
    Энтальпия не зависит от давления для идеальных газов и (приближённо) для конденсата.
    """
    H = 0.0
    RT = R_UNIVERSAL * T
    for i, sp in enumerate(species_list):
        if n_vec[i] > 0:
            H += n_vec[i] * h_over_RT(sp, T) * RT
    return H


def mixture_entropy(
    species_list: List[Species],
    n_vec: np.ndarray,
    T: float,
    P: float,
) -> float:
    """Полная энтропия смеси при давлении P, Дж/К.

    Для газа: S_i = S0_i(T) - R*ln(P_i / P_ref) = S0_i(T) - R*ln((n_i/n_tot)*P/P_ref)
    Для конденсата: S_i = S0_i(T)
    """
    S = 0.0
    n_gas_total = sum(n_vec[i] for i, sp in enumerate(species_list) if sp.is_gas)
    n_gas_total = max(n_gas_total, 1e-30)
    ln_P_over_Pref = math.log(P / P_REF)

    for i, sp in enumerate(species_list):
        if n_vec[i] <= 0:
            continue
        S0 = s_over_R(sp, T) * R_UNIVERSAL  # Дж/(моль·К)
        if sp.is_gas:
            x_i = n_vec[i] / n_gas_total
            # член смешения: -R*ln(x_i) и сдвиг по давлению: -R*ln(P/P_ref)
            S_i = S0 - R_UNIVERSAL * (math.log(max(x_i, 1e-300)) + ln_P_over_Pref)
        else:
            S_i = S0
        S += n_vec[i] * S_i
    return S


# ──────────────────────────────────────────────────────────────────────────────
# Базовая TP-задача: минимизация G при фиксированных T и P
# ──────────────────────────────────────────────────────────────────────────────

def solve_equilibrium(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    include_condensed: bool = True,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    outer_index: int = -1,
    n0_warm: Optional[np.ndarray] = None,
) -> EquilibriumResult:
    """
    Находит равновесный состав минимизацией G/RT при заданных T и P.

    Задача: min G/RT = sum_i n_i*(g0_i/RT + ln(n_i/n_total) + ln(P/Pref))
    При условии: sum_i a_ki * n_i = b_k  для каждого элемента k

    Параметры:
        n0_warm — «тёплый старт», начальное приближение (для внешних итераций HP/SP).
        outer_index — номер внешнего шага для лога (если -1 — TP-задача без внешнего цикла).
    """
    if logger is None:
        logger = NullLogger()

    gas = [sp for sp in species_list if sp.is_gas]
    cond = [sp for sp in species_list if sp.is_condensed] if include_condensed else []
    all_sp = gas + cond

    Ng = len(gas)
    Nc = len(cond)
    N = Ng + Nc

    if N == 0:
        raise ValueError("Список веществ пуст")

    elem_list = sorted(element_abundances.keys())
    Ne = len(elem_list)
    elem_idx = {e: i for i, e in enumerate(elem_list)}
    b = np.array([element_abundances[e] for e in elem_list], dtype=float)

    # стехиометрическая матрица
    a = np.zeros((Ne, N))
    for i, sp in enumerate(all_sp):
        for el, cnt in sp.elements.items():
            if el in elem_idx:
                a[elem_idx[el], i] = cnt

    g0 = np.array([g_over_RT(sp, T) for sp in all_sp])
    ln_P = math.log(P / P_REF)

    if verbose:
        print(f"\nT={T:.0f} К, P={P/1e5:.3f} бар, газов={Ng}, конденсата={Nc}")

    if logger.enabled and outer_index < 0:
        logger.section(f'TP-задача:  T = {T:.4f} К,  P = {P:.4f} Па')
        logger.log(f'веществ всего: {N}  (газов {Ng}, конденсата {Nc})')

    n_min = 1e-20  # минимум молей (чтобы не было ln(0))

    def gibbs(n):
        G = 0.0
        ntot = max(n[:Ng].sum(), n_min)
        ln_nt = math.log(ntot)
        for i in range(Ng):
            ni = max(n[i], n_min)
            G += ni * (g0[i] + math.log(ni) - ln_nt + ln_P)
        for j in range(Nc):
            G += max(n[Ng+j], 0.0) * g0[Ng+j]
        return G

    def grad(n):
        gr = np.zeros(N)
        ntot = max(n[:Ng].sum(), n_min)
        ln_nt = math.log(ntot)
        for i in range(Ng):
            gr[i] = g0[i] + math.log(max(n[i], n_min)) - ln_nt + ln_P
        for j in range(Nc):
            gr[Ng+j] = g0[Ng+j]
        return gr

    constraints = [
        {'type': 'eq',
         'fun': lambda n, k=k: np.dot(a[k], n) - b[k],
         'jac': lambda n, k=k: a[k]}
        for k in range(Ne)
    ]
    bounds = [(n_min, None)]*Ng + [(0.0, None)]*Nc

    # начальное приближение
    if n0_warm is not None and len(n0_warm) == N:
        n0 = np.maximum(n0_warm.copy(), n_min)
    else:
        n0 = np.ones(N) * n_min
        try:
            from scipy.optimize import nnls
            n_init, _ = nnls(a, b)
            n0 = np.maximum(n_init, n_min)
        except Exception:
            n0[:Ng] = b.sum() / max(Ng, 1)

    if verbose:
        print(f"Начальное G/RT = {gibbs(n0):.4f}")

    if logger.enabled:
        logger.log(f'начальное G/RT = {gibbs(n0):.6e}')

    # основной решатель
    iter_count = [0]
    sp_names = [sp.name for sp in all_sp]

    def cb(xk):
        iter_count[0] += 1
        if verbose and iter_count[0] % 50 == 0:
            print(f"  iter {iter_count[0]}: G/RT={gibbs(xk):.6f}")
        if logger.enabled:
            # невязка ограничений
            res_eq = sum(abs(np.dot(a[k], xk) - b[k]) / max(abs(b[k]), 1e-30)
                         for k in range(Ne)) / max(Ne, 1)
            logger.inner_iter(outer_index, iter_count[0], gibbs(xk),
                              xk, sp_names, residual=res_eq, top_k=8)

    res = minimize(gibbs, n0, method='SLSQP', jac=grad,
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False},
                   callback=cb)

    n_sol = res.x

    # если не сошлось — пробуем trust-constr
    if not res.success:
        if verbose:
            print(f"  SLSQP: {res.message}, пробуем trust-constr...")
        if logger.enabled:
            logger.log(f'SLSQP не сошёлся: {res.message}. Пробуем trust-constr...')
        res2 = minimize(gibbs, n0, method='trust-constr', jac=grad,
                        bounds=[(n_min if i < Ng else 0.0, None) for i in range(N)],
                        constraints=LinearConstraint(a, b, b),
                        options={'maxiter': 5000, 'verbose': 0})
        if res2.fun < res.fun:
            res, n_sol = res2, res2.x
            if logger.enabled:
                logger.log(f'trust-constr: G/RT = {res.fun:.6e}')

    # невязка баланса элементов
    residual = sum(abs(np.dot(a[k], n_sol) - b[k]) / max(b[k], 1e-30)
                   for k in range(Ne)) / max(Ne, 1)

    ntot_final = max(n_sol[:Ng].sum(), n_min)
    xi = np.zeros(N)
    xi[:Ng] = n_sol[:Ng] / ntot_final

    converged = res.success or residual < 1e-6

    if verbose:
        print(f"  G/RT={res.fun:.6f}, невязка={residual:.2e}, итераций={res.nit}")
        print(f"  {'сошлось ✓' if converged else 'не сошлось ✗'}")

    if logger.enabled:
        logger.log(
            f'TP-решение: G/RT={res.fun:.6e}, невязка={residual:.3e}, '
            f'итераций={res.nit}, сходимость={"ДА" if converged else "НЕТ"}'
        )

    # доп. термодинамические свойства смеси
    H_mix = mixture_enthalpy(all_sp, n_sol, T)
    S_mix = mixture_entropy(all_sp, n_sol, T, P)

    return EquilibriumResult(
        converged=converged,
        iterations=res.nit,
        T=T, P=P,
        species_names=sp_names,
        mole_fractions=xi,
        moles=n_sol.copy(),
        total_moles=ntot_final,
        elements=element_abundances,
        phase=[sp.phase for sp in all_sp],
        g_over_rt=g0,
        residual=residual,
        enthalpy=H_mix,
        entropy=S_mix,
        problem_type='TP',
    )


# ──────────────────────────────────────────────────────────────────────────────
# HP-задача: фиксированы H и P (адиабатическое горение)
# Внешний цикл — поиск T такого, что H(состав(T), T) = H_target
# ──────────────────────────────────────────────────────────────────────────────

def solve_equilibrium_HP(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    H_target: float,
    P: float,
    T_init: float = 2000.0,
    T_bounds: Tuple[float, float] = (200.0, 6000.0),
    include_condensed: bool = True,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    tol_H: float = 1e-3,        # отн. невязка по энтальпии
    max_outer: int = 60,
) -> EquilibriumResult:
    """
    Находит равновесие при заданных H и P.

    Алгоритм:
      1) Задаём T.
      2) Решаем TP-задачу — получаем равновесный состав n*(T).
      3) Считаем H(n*, T) и сравниваем с H_target.
      4) Корректируем T методом секущей и повторяем.

    Производная dH/dT при фиксированном P и равновесном составе
    приближённо равна Cp смеси. Это используется для шага по T.
    """
    if logger is None:
        logger = NullLogger()

    if logger.enabled:
        logger.section(f'HP-задача:  H = {H_target:.4f} Дж,  P = {P:.4f} Па')
        logger.log(f'T_init = {T_init:.2f} К,  T_bounds = {T_bounds},  tol = {tol_H:.2e}')

    T = float(T_init)
    T_min, T_max = T_bounds

    last_result = None
    n_warm = None
    iter_total = 0
    H_prev = None
    T_prev = None

    converged = False
    for outer in range(max_outer):
        # клипируем T в допустимый диапазон
        T = min(max(T, T_min), T_max)

        result = solve_equilibrium(
            species_list=species_list,
            element_abundances=element_abundances,
            T=T, P=P,
            include_condensed=include_condensed,
            verbose=False,
            logger=logger,
            outer_index=outer,
            n0_warm=n_warm,
        )
        iter_total += result.iterations
        H_now = result.enthalpy
        last_result = result
        # тёплый старт — следующий запуск стартует из текущего состава
        n_warm = result.moles.copy()

        # относительная невязка по H
        denom = max(abs(H_target), 1.0)
        rel_dH = (H_now - H_target) / denom

        if logger.enabled:
            logger.outer_iter(
                outer, T, 'H', H_target, H_now, rel_dH,
            )
        if verbose:
            print(f"  HP outer {outer:3d}: T={T:.3f} К, H={H_now:.4e}, "
                  f"target={H_target:.4e}, rel.dH={rel_dH:+.3e}")

        if abs(rel_dH) < tol_H:
            converged = True
            break

        # шаг по T
        if H_prev is not None and abs(H_now - H_prev) > 1e-10 * denom and (T - T_prev) != 0:
            # метод секущей
            slope = (H_now - H_prev) / (T - T_prev)  # ≈ Cp_mix
            if abs(slope) < 1e-30:
                slope = 1e3  # запасное значение Cp
            T_new = T - (H_now - H_target) / slope
        else:
            # первый шаг — оценка через приблизительное Cp
            # типичное Cp_mix ≈ 35 Дж/(моль·К) * число молей
            n_total = max(result.moles.sum(), 1e-6)
            cp_est = 35.0 * n_total
            T_new = T - (H_now - H_target) / cp_est

        # ограничиваем шаг, чтобы не «улететь»
        max_step = 500.0
        if T_new - T > max_step:   T_new = T + max_step
        if T - T_new > max_step:   T_new = T - max_step
        T_new = min(max(T_new, T_min), T_max)

        T_prev, H_prev = T, H_now
        T = T_new

    # финальный результат
    last_result.problem_type = 'HP'
    last_result.iterations = iter_total
    last_result.converged = converged and last_result.converged

    if logger.enabled:
        logger.section('HP — итоговый результат')
        logger.log(
            f'T_final = {last_result.T:.4f} К,  H = {last_result.enthalpy:.4e} Дж '
            f'(target {H_target:.4e}),  S = {last_result.entropy:.4e} Дж/К'
        )
        logger.log(f'внешних шагов: {outer + 1},  суммарно внутренних итераций: {iter_total}')

    if verbose:
        status = "сошлось ✓" if converged else "НЕ сошлось ✗"
        print(f"  HP {status}: T_final = {last_result.T:.2f} К, "
              f"H = {last_result.enthalpy:.4e} Дж (цель {H_target:.4e})")

    return last_result


# ──────────────────────────────────────────────────────────────────────────────
# SP-задача: фиксированы S и P (изэнтропическое расширение)
# Внешний цикл — поиск T такого, что S(состав(T), T, P) = S_target
# ──────────────────────────────────────────────────────────────────────────────

def solve_equilibrium_SP(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    S_target: float,
    P: float,
    T_init: float = 2000.0,
    T_bounds: Tuple[float, float] = (200.0, 6000.0),
    include_condensed: bool = True,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
    tol_S: float = 1e-4,
    max_outer: int = 60,
) -> EquilibriumResult:
    """
    Находит равновесие при заданных S и P.

    Аналогично HP-задаче, но внешний цикл идёт по энтропии.
    Производная dS/dT |_{P, equil} ≈ Cp_mix / T (для идеальных газов и
    приближённо — для смесей с конденсатом).
    """
    if logger is None:
        logger = NullLogger()

    if logger.enabled:
        logger.section(f'SP-задача:  S = {S_target:.4f} Дж/К,  P = {P:.4f} Па')
        logger.log(f'T_init = {T_init:.2f} К,  T_bounds = {T_bounds},  tol = {tol_S:.2e}')

    T = float(T_init)
    T_min, T_max = T_bounds

    last_result = None
    n_warm = None
    iter_total = 0
    S_prev = None
    T_prev = None

    converged = False
    for outer in range(max_outer):
        T = min(max(T, T_min), T_max)

        result = solve_equilibrium(
            species_list=species_list,
            element_abundances=element_abundances,
            T=T, P=P,
            include_condensed=include_condensed,
            verbose=False,
            logger=logger,
            outer_index=outer,
            n0_warm=n_warm,
        )
        iter_total += result.iterations
        S_now = result.entropy
        last_result = result
        n_warm = result.moles.copy()

        denom = max(abs(S_target), 1.0)
        rel_dS = (S_now - S_target) / denom

        if logger.enabled:
            logger.outer_iter(outer, T, 'S', S_target, S_now, rel_dS)
        if verbose:
            print(f"  SP outer {outer:3d}: T={T:.3f} К, S={S_now:.4e}, "
                  f"target={S_target:.4e}, rel.dS={rel_dS:+.3e}")

        if abs(rel_dS) < tol_S:
            converged = True
            break

        # шаг по T
        if S_prev is not None and abs(S_now - S_prev) > 1e-12 * denom and (T - T_prev) != 0:
            slope = (S_now - S_prev) / (T - T_prev)  # ≈ Cp/T
            if abs(slope) < 1e-30:
                slope = 1.0
            T_new = T - (S_now - S_target) / slope
        else:
            # первый шаг: dS/dT ≈ Cp_mix / T,  Cp_mix ≈ 35 Дж/(моль·К) * n_total
            n_total = max(result.moles.sum(), 1e-6)
            cp_est = 35.0 * n_total
            slope_est = cp_est / max(T, 1.0)
            T_new = T - (S_now - S_target) / slope_est

        max_step = 500.0
        if T_new - T > max_step:   T_new = T + max_step
        if T - T_new > max_step:   T_new = T - max_step
        T_new = min(max(T_new, T_min), T_max)

        T_prev, S_prev = T, S_now
        T = T_new

    last_result.problem_type = 'SP'
    last_result.iterations = iter_total
    last_result.converged = converged and last_result.converged

    if logger.enabled:
        logger.section('SP — итоговый результат')
        logger.log(
            f'T_final = {last_result.T:.4f} К,  S = {last_result.entropy:.4e} Дж/К '
            f'(target {S_target:.4e}),  H = {last_result.enthalpy:.4e} Дж'
        )
        logger.log(f'внешних шагов: {outer + 1},  суммарно внутренних итераций: {iter_total}')

    if verbose:
        status = "сошлось ✓" if converged else "НЕ сошлось ✗"
        print(f"  SP {status}: T_final = {last_result.T:.2f} К, "
              f"S = {last_result.entropy:.4e} Дж/К (цель {S_target:.4e})")

    return last_result
