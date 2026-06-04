"""
fuel_equilibrium.core — чистая физическая логика (без I/O и GUI).

Содержит:
    * парсинг базы NASA-9                    — nasa9_parser
    * термодинамические полиномы (cp, h, s)  — thermo_calc
    * парсинг химических формул и реакций    — formula_parser
    * минимизация энергии Гиббса (TP/HP/SP)  — gibbs_solver
    * высокоуровневый run_batch и find_thermo_db — equilibrium
"""

from .nasa9_parser import (
    parse_thermo_file,
    Species,
    TemperatureInterval,
    get_products_for_elements,
)
from .thermo_calc import (
    cp_over_R,
    h_over_RT,
    s_over_R,
    g_over_RT,
    R_UNIVERSAL,
)
from .formula_parser import (
    parse_reaction_string,
    get_total_elements,
    parse_formula,
)
from .gibbs_solver import (
    solve_equilibrium,
    solve_equilibrium_HP,
    solve_equilibrium_SP,
    EquilibriumResult,
    mixture_enthalpy,
    mixture_entropy,
)
from .equilibrium import (
    run_batch,
    run_interactive,
    find_thermo_db,
    parse_pressure,
    select_candidate_species,
)

__all__ = [
    # nasa9_parser
    "parse_thermo_file", "Species", "TemperatureInterval", "get_products_for_elements",
    # thermo_calc
    "cp_over_R", "h_over_RT", "s_over_R", "g_over_RT", "R_UNIVERSAL",
    # formula_parser
    "parse_reaction_string", "get_total_elements", "parse_formula",
    # gibbs_solver
    "solve_equilibrium", "solve_equilibrium_HP", "solve_equilibrium_SP",
    "EquilibriumResult", "mixture_enthalpy", "mixture_entropy",
    # equilibrium
    "run_batch", "run_interactive", "find_thermo_db",
    "parse_pressure", "select_candidate_species",
]
