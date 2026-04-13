# минимизация энергии Гиббса для поиска равновесного состава
# метод: SLSQP с ограничениями на сохранение элементов
# резерв: trust-constr если SLSQP не сошёлся
# справочник: NASA RP-1311, Gordon & McBride, 1994

import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple
from dataclasses import dataclass

from nasa9_parser import Species
from thermo_calc import g_over_RT, R_UNIVERSAL


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


def solve_equilibrium(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    include_condensed: bool = True,
    verbose: bool = False,
) -> EquilibriumResult:
    """
    Находит равновесный состав минимизацией G/RT.

    Задача: min G/RT = sum_i n_i*(g0_i/RT + ln(n_i/n_total) + ln(P/Pref))
    При условии: sum_i a_ki * n_i = b_k  для каждого элемента k
    """
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

    # начальное приближение через NNLS
    n0 = np.ones(N) * n_min
    try:
        from scipy.optimize import nnls
        n_init, _ = nnls(a, b)
        n0 = np.maximum(n_init, n_min)
    except Exception:
        n0[:Ng] = b.sum() / max(Ng, 1)

    if verbose:
        print(f"Начальное G/RT = {gibbs(n0):.4f}")

    # основной решатель
    iter_count = [0]
    def cb(xk):
        iter_count[0] += 1
        if verbose and iter_count[0] % 50 == 0:
            print(f"  iter {iter_count[0]}: G/RT={gibbs(xk):.6f}")

    res = minimize(gibbs, n0, method='SLSQP', jac=grad,
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False},
                   callback=cb if verbose else None)

    n_sol = res.x

    # если не сошлось — пробуем trust-constr
    if not res.success:
        if verbose:
            print(f"  SLSQP: {res.message}, пробуем trust-constr...")
        res2 = minimize(gibbs, n0, method='trust-constr', jac=grad,
                        bounds=[(n_min if i < Ng else 0.0, None) for i in range(N)],
                        constraints=LinearConstraint(a, b, b),
                        options={'maxiter': 5000, 'verbose': 0})
        if res2.fun < res.fun:
            res, n_sol = res2, res2.x

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

    return EquilibriumResult(
        converged=converged,
        iterations=res.nit,
        T=T, P=P,
        species_names=[sp.name for sp in all_sp],
        mole_fractions=xi,
        moles=n_sol.copy(),
        total_moles=ntot_final,
        elements=element_abundances,
        phase=[sp.phase for sp in all_sp],
        g_over_rt=g0,
        residual=residual,
    )
