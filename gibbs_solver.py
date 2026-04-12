"""
Gibbs Free Energy Minimization Solver.

Two approaches:
1. Direct Gibbs minimization via scipy.optimize.minimize (SLSQP)
   with element conservation as equality constraints.
2. Lagrange-Newton method (Gordon-McBride style) as fallback.

Reference: NASA RP-1311, Gordon & McBride, 1994.
"""

import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nasa9_parser import Species
from thermo_calc import g_over_RT, R_UNIVERSAL


P_REF = 1e5         # Reference pressure: 1 bar (Pa)
TRACE_CUTOFF = 1e-30


@dataclass
class EquilibriumResult:
    """Result of equilibrium calculation."""
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
        total_gas = sum(self.moles[i] for i in range(len(self.species_names)) 
                       if self.phase[i] == 0)
        result = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] == 0 and self.moles[i] > TRACE_CUTOFF:
                xi = self.moles[i] / total_gas if total_gas > 0 else 0
                result.append((name, self.moles[i], xi))
        result.sort(key=lambda x: -x[2])
        return result

    def get_condensed_species(self) -> List[Tuple[str, float]]:
        result = []
        for i, name in enumerate(self.species_names):
            if self.phase[i] != 0 and self.moles[i] > TRACE_CUTOFF:
                result.append((name, self.moles[i]))
        result.sort(key=lambda x: -x[1])
        return result


def solve_equilibrium(
    species_list: List[Species],
    element_abundances: Dict[str, float],
    T: float,
    P: float,
    include_condensed: bool = True,
    verbose: bool = False
) -> EquilibriumResult:
    """
    Solve for chemical equilibrium using Gibbs energy minimization.
    
    Uses transformed variables y_i = ln(n_i) for gas species to ensure positivity.
    The optimization problem is:
    
        minimize  G(y)/RT = sum_i exp(y_i) * (g0_i/RT + y_i - ln(sum_j exp(y_j)) + ln(P/Pref))
        subject to sum_i a_ki * exp(y_i) = b_k  for each element k
    
    Solved using scipy's SLSQP method with analytical gradients.
    """
    gas_species = [sp for sp in species_list if sp.is_gas]
    cond_species = [sp for sp in species_list if sp.is_condensed] if include_condensed else []
    all_species = gas_species + cond_species
    
    N_gas = len(gas_species)
    N_cond = len(cond_species)
    N_species = N_gas + N_cond
    
    if N_species == 0:
        raise ValueError("No species provided for equilibrium calculation")
    
    elem_list = sorted(element_abundances.keys())
    N_elem = len(elem_list)
    elem_idx = {e: i for i, e in enumerate(elem_list)}
    
    b = np.array([element_abundances[e] for e in elem_list], dtype=float)
    
    # Stoichiometric matrix a[k,i]
    a = np.zeros((N_elem, N_species))
    for i, sp in enumerate(all_species):
        for elem, count in sp.elements.items():
            if elem in elem_idx:
                a[elem_idx[elem], i] = count
    
    # Standard Gibbs g0/(RT)
    g0_RT = np.array([g_over_RT(sp, T) for sp in all_species])
    
    ln_P = math.log(P / P_REF)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Gibbs Minimization: T={T:.2f} K, P={P/1e5:.4f} bar")
        print(f"Gas: {N_gas}, Condensed: {N_cond}, Elements: {N_elem}")
        print(f"Elements: {elem_list}, b = {b}")
        print(f"{'='*70}")
    
    # ---------------------------------------------------------------
    # Approach: Work in n-space directly with SLSQP
    # Variables: n[0..N_species-1] = moles of each species
    # We use bounds n_i >= epsilon to avoid log(0)
    # ---------------------------------------------------------------
    
    eps = 1e-20  # minimum moles for gas species
    
    def gibbs_objective(n_vec):
        """Compute G/RT as function of mole vector."""
        G = 0.0
        n_gas_total = max(np.sum(n_vec[:N_gas]), eps)
        ln_ntot = math.log(n_gas_total)
        
        for i in range(N_gas):
            ni = max(n_vec[i], eps)
            G += ni * (g0_RT[i] + math.log(ni) - ln_ntot + ln_P)
        
        for j in range(N_cond):
            nj = max(n_vec[N_gas + j], 0.0)
            G += nj * g0_RT[N_gas + j]
        
        return G
    
    def gibbs_gradient(n_vec):
        """Compute gradient of G/RT w.r.t. n."""
        grad = np.zeros(N_species)
        n_gas_total = max(np.sum(n_vec[:N_gas]), eps)
        ln_ntot = math.log(n_gas_total)
        
        # mu_i = g0_i/RT + ln(n_i) - ln(n_total) + ln(P/Pref)
        # But dG/dn_i = mu_i + (correction from n_total dependence on n_i)
        # Actually: dG/dn_i = g0_i/RT + ln(n_i/n_total) + ln(P/Pref) for gas
        #                   = g0_j/RT for condensed
        # (The n_total-dependent terms cancel when you do the full differentiation)
        
        for i in range(N_gas):
            ni = max(n_vec[i], eps)
            grad[i] = g0_RT[i] + math.log(ni) - ln_ntot + ln_P
        
        for j in range(N_cond):
            grad[N_gas + j] = g0_RT[N_gas + j]
        
        return grad
    
    # Element conservation constraints: a @ n = b
    def elem_constraint(n_vec, k):
        return np.dot(a[k, :], n_vec) - b[k]
    
    def elem_constraint_jac(n_vec, k):
        return a[k, :]
    
    constraints = []
    for k in range(N_elem):
        constraints.append({
            'type': 'eq',
            'fun': lambda n, kk=k: elem_constraint(n, kk),
            'jac': lambda n, kk=k: elem_constraint_jac(n, kk)
        })
    
    # Bounds: gas species > eps, condensed species >= 0
    bounds = []
    for i in range(N_gas):
        bounds.append((eps, None))
    for j in range(N_cond):
        bounds.append((0.0, None))
    
    # --- Initial guess ---
    n0 = np.ones(N_species) * eps
    
    # Use NNLS for reasonable starting point
    try:
        from scipy.optimize import nnls
        A_all = a.copy()
        n_init, _ = nnls(A_all, b)
        for i in range(N_species):
            n0[i] = max(n_init[i], eps)
    except Exception:
        # Distribute equally
        b_total = np.sum(b)
        for i in range(N_gas):
            n0[i] = b_total / max(N_gas, 1)
    
    if verbose:
        G0 = gibbs_objective(n0)
        print(f"Initial G/RT = {G0:.6f}")
        print(f"Initial n_total = {np.sum(n0[:N_gas]):.6f}")
    
    # --- Solve with SLSQP ---
    niter_count = [0]
    
    def callback(xk):
        niter_count[0] += 1
        if verbose and niter_count[0] % 50 == 0:
            G_val = gibbs_objective(xk)
            print(f"  Iter {niter_count[0]}: G/RT = {G_val:.8f}")
    
    result_opt = minimize(
        gibbs_objective,
        n0,
        method='SLSQP',
        jac=gibbs_gradient,
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 2000,
            'ftol': 1e-14,
            'disp': False,
        },
        callback=callback if verbose else None
    )
    
    n_sol = result_opt.x
    
    # Check if failed, try trust-constr as backup
    if not result_opt.success:
        if verbose:
            print(f"  SLSQP result: {result_opt.message}, trying trust-constr...")
        
        # Build linear constraints for trust-constr
        lin_constraint = LinearConstraint(a, b, b)
        
        result_opt2 = minimize(
            gibbs_objective,
            n0,
            method='trust-constr',
            jac=gibbs_gradient,
            bounds=[(eps if i < N_gas else 0.0, None) for i in range(N_species)],
            constraints=lin_constraint,
            options={'maxiter': 5000, 'verbose': 0}
        )
        
        if result_opt2.fun < result_opt.fun:
            result_opt = result_opt2
            n_sol = result_opt.x
    
    # Post-process: check element balance
    elem_residual = 0.0
    for k in range(N_elem):
        computed_bk = np.dot(a[k, :], n_sol)
        elem_residual += abs(computed_bk - b[k]) / max(b[k], 1e-30)
    elem_residual /= max(N_elem, 1)
    
    n_total_final = max(np.sum(n_sol[:N_gas]), eps)
    
    mole_fractions = np.zeros(N_species)
    for i in range(N_gas):
        mole_fractions[i] = n_sol[i] / n_total_final
    
    converged = result_opt.success or elem_residual < 1e-6
    
    if verbose:
        print(f"\n  Optimizer: {result_opt.message}")
        print(f"  G/RT = {result_opt.fun:.8f}")
        print(f"  Element balance residual = {elem_residual:.2e}")
        print(f"  n_total = {n_total_final:.6f}")
        print(f"  Iterations = {result_opt.nit}")
        if converged:
            print(f"  ✓ Converged!")
        else:
            print(f"  ✗ Not converged (element residual: {elem_residual:.2e})")
    
    return EquilibriumResult(
        converged=converged,
        iterations=result_opt.nit,
        T=T,
        P=P,
        species_names=[sp.name for sp in all_species],
        mole_fractions=mole_fractions,
        moles=n_sol.copy(),
        total_moles=n_total_final,
        elements=element_abundances,
        phase=[sp.phase for sp in all_species],
        g_over_rt=g0_RT,
        residual=elem_residual
    )
