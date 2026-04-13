"""
fuel_equilibrium — расчёт химического равновесия газовых смесей.

Метод: минимизация энергии Гиббса, база данных NASA-9.

Пример:
    from equilibrium import run_batch
    result = run_batch("2H2 + O2", T=3000, P=101325)
    for name, moles, xi in result.get_gas_species():
        print(f"{name}: {xi:.4%}")
"""

__version__ = "1.0.0"

from .nasa9_parser import parse_thermo_file, Species, get_products_for_elements
from .thermo_calc import cp_over_R, h_over_RT, s_over_R, g_over_RT, R_UNIVERSAL
from .formula_parser import parse_reaction_string, get_total_elements, parse_formula
from .gibbs_solver import solve_equilibrium, EquilibriumResult
from .equilibrium import run_batch, run_interactive
