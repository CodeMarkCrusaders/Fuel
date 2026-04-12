"""
Fuel_Equilibrium - Chemical Equilibrium Calculator

Determines equilibrium composition of gas mixtures at specified 
temperature and pressure using Gibbs free energy minimization 
with NASA 9-coefficient thermodynamic database.
"""

__version__ = "1.0.0"
__author__ = "Fuel_Equilibrium Project"

from .nasa9_parser import parse_thermo_file, Species, get_products_for_elements
from .thermo_calc import cp_over_R, h_over_RT, s_over_R, g_over_RT, R_UNIVERSAL
from .formula_parser import parse_reaction_string, get_total_elements, parse_formula
from .gibbs_solver import solve_equilibrium, EquilibriumResult
from .equilibrium import run_batch, run_interactive
