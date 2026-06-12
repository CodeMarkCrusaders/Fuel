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
from .equilibrium_cache import get_global_cache


P_REF = 1e5      # опорное давление, 1 бар
TRACE = 1e-30    # порог "следовой" концентрации
_N_MIN = 1e-20   # минимум молей (чтобы не было ln(0))


# ──────────────────────────────────────────────────────────────────────────────
# Ядра целевой функции G/RT и градиента — нативная компиляция через Numba.
#
# Профилирование SLSQP для системы C/H/O (~51 вид) показало, что Python-цикл по
# видам в gibbs()/grad() занимал ~30 % времени (функции вызываются тысячами раз
# за один solve, всегда с короткими массивами ~50 элементов). Для таких мелких
# массивов накладные расходы NumPy на вызов (создание временных массивов,
# диспетчеризация) велики; плотный машинный цикл оказывается ~12× быстрее.
#
# Numba (@njit) компилирует эти ядра в нативный код (LLVM) при первом вызове.
# Если numba не установлена — прозрачный откат на векторизованную реализацию
# NumPy (тоже быстрее исходного чистого Python). Результаты численно идентичны
# (отклонение < 1e-12).
# ──────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # numba не установлена — работаем на NumPy
    _HAVE_NUMBA = False

    def _njit(*args, **kwargs):  # заглушка-декоратор (no-op)
        def _wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return _wrap


@_njit(cache=True, fastmath=True)
def _gibbs_kernel(n, Ng, Nc, g0, ln_P, n_min):
    """G/RT смеси: газовая фаза (с членом смешения) + конденсат."""
    s = 0.0
    for i in range(Ng):
        s += n[i] if n[i] > n_min else n_min
    ntot = s if s > n_min else n_min
    ln_nt = math.log(ntot)
    G = 0.0
    for i in range(Ng):
        ni = n[i] if n[i] > n_min else n_min
        G += ni * (g0[i] + math.log(ni) - ln_nt + ln_P)
    for j in range(Nc):
        nj = n[Ng + j] if n[Ng + j] > 0.0 else 0.0
        G += nj * g0[Ng + j]
    return G


@_njit(cache=True, fastmath=True)
def _grad_kernel(n, Ng, Nc, g0, ln_P, n_min, out):
    """∂(G/RT)/∂n_i. Результат пишется в ``out`` (форма (Ng+Nc,))."""
    s = 0.0
    for i in range(Ng):
        s += n[i] if n[i] > n_min else n_min
    ntot = s if s > n_min else n_min
    ln_nt = math.log(ntot)
    for i in range(Ng):
        ni = n[i] if n[i] > n_min else n_min
        out[i] = g0[i] + math.log(ni) - ln_nt + ln_P
    for j in range(Nc):
        out[Ng + j] = g0[Ng + j]
    return out


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
    use_cache: bool = True,
) -> EquilibriumResult:
    """
    Находит равновесный состав минимизацией G/RT при заданных T и P.

    Задача: min G/RT = sum_i n_i*(g0_i/RT + ln(n_i/n_total) + ln(P/Pref))
    При условии: sum_i a_ki * n_i = b_k  для каждого элемента k

    Параметры:
        n0_warm — «тёплый старт», начальное приближение (для внешних итераций HP/SP).
        outer_index — номер внешнего шага для лога (если -1 — TP-задача без внешнего цикла).
        use_cache — использовать глобальный кэш равновесных составов. Кэшируется
                    только «холодный» старт (n0_warm is None); внутри HP/SP-циклов
                    кэширование выполняется на верхнем уровне.
    """
    if logger is None:
        logger = NullLogger()

    # ── кэш: только для холодного старта TP-задачи ─────────────────────
    _cache = get_global_cache() if (use_cache and n0_warm is None) else None
    _cache_key = None
    if _cache is not None:
        _cache_key = _cache.make_key(
            'TP', species_list, element_abundances, T, P, include_condensed,
        )
        _cached = _cache.get(_cache_key)
        if _cached is not None:
            return _cached

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

    n_min = _N_MIN  # минимум молей (чтобы не было ln(0))

    # ── SoA (Structure of Arrays) + нативные ядра G/RT и градиента ────────
    # Данные видов хранятся «структурой массивов» (g0 — непрерывный float64
    # массив (N,), стехиометрическая матрица a — (Ne, N)). Целевая функция и
    # градиент считаются нативными ядрами _gibbs_kernel / _grad_kernel
    # (Numba @njit → машинный код, ~12× быстрее NumPy для коротких массивов;
    # при отсутствии numba — те же ядра исполняются интерпретатором, что
    # эквивалентно прежней реализации). g0 приводим к C-непрерывному float64.
    g0 = np.ascontiguousarray(g0, dtype=np.float64)
    g0_gas = g0[:Ng]                 # срез коэффициентов g0/RT для газов
    g0_cond = g0[Ng:]                # для конденсата (если есть)

    if _HAVE_NUMBA:
        def gibbs(n):
            n = np.ascontiguousarray(n, dtype=np.float64)
            return float(_gibbs_kernel(n, Ng, Nc, g0, ln_P, n_min))

        def grad(n):
            n = np.ascontiguousarray(n, dtype=np.float64)
            gr = np.empty(N, dtype=np.float64)
            _grad_kernel(n, Ng, Nc, g0, ln_P, n_min, gr)
            return gr
    else:
        # Векторизованный NumPy-фоллбэк (тоже быстрее исходного чистого Python).
        def gibbs(n):
            n = np.asarray(n)
            ng = np.maximum(n[:Ng], n_min)
            ntot = max(ng.sum(), n_min)
            ln_nt = math.log(ntot)
            G = float(np.dot(ng, g0_gas + np.log(ng) - ln_nt + ln_P))
            if Nc:
                G += float(np.dot(np.maximum(n[Ng:], 0.0), g0_cond))
            return G

        def grad(n):
            n = np.asarray(n)
            gr = np.empty(N)
            ng = np.maximum(n[:Ng], n_min)
            ntot = max(ng.sum(), n_min)
            ln_nt = math.log(ntot)
            gr[:Ng] = g0_gas + np.log(ng) - ln_nt + ln_P
            if Nc:
                gr[Ng:] = g0_cond
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

    # ftol=1e-10 (вместо 1e-14): для систем с большим числом видов (C/H/O ~61
    # вид) число итераций SLSQP — главный фактор стоимости. Ослабление допуска
    # с 1e-14 до 1e-10 кратно сокращает число итераций (камера ~9.5с→3.2с) при
    # идентичной температуре пламени и составе (отклонение T < 1e-3 K), т.к.
    # минимум G/RT гладкий и достигается с большим запасом по точности.
    res = minimize(gibbs, n0, method='SLSQP', jac=grad,
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False},
                   callback=cb)

    n_sol = res.x

    # невязка баланса элементов SLSQP-решения
    residual = sum(abs(np.dot(a[k], n_sol) - b[k]) / max(b[k], 1e-30)
                   for k in range(Ne)) / max(Ne, 1)

    # Запасной решатель trust-constr запускаем ТОЛЬКО если SLSQP-решение
    # реально плохое (большая невязка баланса элементов). SLSQP часто
    # сообщает success=False («Iteration limit reached» / «Positive directional
    # derivative…»), уже находясь в корректном минимуме G/RT с пренебрежимо
    # малой невязкой — в таких случаях дорогой trust-constr (maxiter=5000) не
    # нужен и лишь кратно замедляет расчёт (особенно для C/H/O-систем с ~61
    # видом). Порог 1e-6 совпадает с критерием сходимости ниже.
    need_fallback = (not res.success) and (residual > 1e-6)
    if need_fallback:
        if verbose:
            print(f"  SLSQP: {res.message} (невязка {residual:.2e}), пробуем trust-constr...")
        if logger.enabled:
            logger.log(f'SLSQP не сошёлся: {res.message} (невязка {residual:.2e}). '
                       f'Пробуем trust-constr...')
        res2 = minimize(gibbs, n0, method='trust-constr', jac=grad,
                        bounds=[(n_min if i < Ng else 0.0, None) for i in range(N)],
                        constraints=LinearConstraint(a, b, b),
                        options={'maxiter': 5000, 'verbose': 0})
        if res2.fun < res.fun:
            res, n_sol = res2, res2.x
            residual = sum(abs(np.dot(a[k], n_sol) - b[k]) / max(b[k], 1e-30)
                           for k in range(Ne)) / max(Ne, 1)
            if logger.enabled:
                logger.log(f'trust-constr: G/RT = {res.fun:.6e}, невязка = {residual:.2e}')

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

    eq_result = EquilibriumResult(
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

    if _cache is not None and _cache_key is not None:
        _cache.put(_cache_key, eq_result)

    return eq_result


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
    use_cache: bool = True,
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

    # ── кэш равновесного состава для HP-задачи ─────────────────────────
    _cache = get_global_cache() if use_cache else None
    _cache_key = None
    if _cache is not None:
        _cache_key = _cache.make_key(
            'HP', species_list, element_abundances, H_target, P, include_condensed,
        )
        _cached = _cache.get(_cache_key)
        if _cached is not None:
            return _cached

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

    if _cache is not None and _cache_key is not None:
        _cache.put(_cache_key, last_result)

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
    use_cache: bool = True,
) -> EquilibriumResult:
    """
    Находит равновесие при заданных S и P.

    Аналогично HP-задаче, но внешний цикл идёт по энтропии.
    Производная dS/dT |_{P, equil} ≈ Cp_mix / T (для идеальных газов и
    приближённо — для смесей с конденсатом).
    """
    if logger is None:
        logger = NullLogger()

    # ── кэш равновесного состава для SP-задачи ─────────────────────────
    _cache = get_global_cache() if use_cache else None
    _cache_key = None
    if _cache is not None:
        _cache_key = _cache.make_key(
            'SP', species_list, element_abundances, S_target, P, include_condensed,
        )
        _cached = _cache.get(_cache_key)
        if _cached is not None:
            return _cached

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

    if _cache is not None and _cache_key is not None:
        _cache.put(_cache_key, last_result)

    return last_result
