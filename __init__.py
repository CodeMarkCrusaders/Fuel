"""
fuel_equilibrium — расчёт химического равновесия газовых смесей.

Метод: минимизация энергии Гиббса, база данных NASA-9.
Поддерживаются три типа задач:
    TP — заданы температура и давление
    HP — заданы энтальпия и давление (адиабатическое горение)
    SP — заданы энтропия  и давление (изэнтропическое расширение)

Пример (TP):
    from equilibrium import run_batch
    result = run_batch("2H2 + O2", T=3000, P=101325, problem_type='TP')

Пример (HP):
    result = run_batch("2H2 + O2", H=-200000, P=101325, problem_type='HP')

Пример (SP) с логом итераций:
    result = run_batch("2H2 + O2", S=700, P=101325,
                       problem_type='SP', log_path='run.log')
"""

__version__ = "1.2.0"

from .nasa9_parser import parse_thermo_file, Species, get_products_for_elements
from .thermo_calc import cp_over_R, h_over_RT, s_over_R, g_over_RT, R_UNIVERSAL
from .formula_parser import parse_reaction_string, get_total_elements, parse_formula
from .gibbs_solver import (
    solve_equilibrium,
    solve_equilibrium_HP,
    solve_equilibrium_SP,
    EquilibriumResult,
    mixture_enthalpy,
    mixture_entropy,
)
from .iteration_logger import IterationLogger, NullLogger
from .equilibrium import run_batch, run_interactive
from .nozzle_flow import (
    Propellant,
    StationResult,
    RocketPerformance,
    solve_rocket_nozzle,
    print_nozzle_table,
    stoichiometric_OF,
)
