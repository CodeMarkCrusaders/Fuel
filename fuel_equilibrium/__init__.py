"""
fuel_equilibrium — расчёт химического равновесия газовых смесей
и газодинамики ракетного сопла.

Слои пакета:
    core    — чистая физика (термодинамика NASA-9 + минимизация Гиббса)
    rocket  — прикладной слой: расчёт сопла, ракетные характеристики
    io      — логирование, batch-CSV, форматирование отчётов
    gui     — PyQt5-интерфейс (опционально, требует PyQt5 + matplotlib)

Метод: минимизация энергии Гиббса, база данных NASA-9.
Поддерживаются три типа задач:
    TP — заданы температура и давление
    HP — заданы энтальпия и давление (адиабатическое горение)
    SP — заданы энтропия  и давление (изэнтропическое расширение)

Быстрый старт (TP)::

    from fuel_equilibrium.core import run_batch
    result = run_batch("2H2 + O2", T=3000, P=101325, problem_type='TP')

Адиабатическое пламя (HP)::

    result = run_batch("2H2 + O2", H=-200000, P=101325, problem_type='HP')

Расширение в сопле (SP) с логом итераций::

    result = run_batch("2H2 + O2", S=700, P=101325,
                       problem_type='SP', log_path='run.log')

Расчёт сопла::

    from fuel_equilibrium.core import parse_thermo_file, find_thermo_db
    from fuel_equilibrium.rocket import Propellant, solve_rocket_nozzle
    from fuel_equilibrium.io import print_nozzle_table

    db = parse_thermo_file(find_thermo_db())
    ox = Propellant("O2(L)", mass_kg=7.937)
    fu = Propellant("H2(L)", mass_kg=1.000)
    perf = solve_rocket_nozzle(oxidizer=ox, fuel=fu,
                               P_chamber=10e6, P_exit=0.1013e6,
                               species_db=db)
    print_nozzle_table(perf)
"""

__version__ = "1.2.0"

# Реэкспорт самых часто используемых имён — для удобства пользователей,
# которым не хочется помнить, в каком из подмодулей что лежит.
from .core import (
    parse_thermo_file,
    Species,
    get_products_for_elements,
    cp_over_R, h_over_RT, s_over_R, g_over_RT, R_UNIVERSAL,
    parse_reaction_string, get_total_elements, parse_formula,
    solve_equilibrium, solve_equilibrium_HP, solve_equilibrium_SP,
    EquilibriumResult, mixture_enthalpy, mixture_entropy,
    run_batch, run_interactive, find_thermo_db,
)
from .io import IterationLogger, NullLogger
from .rocket import (
    Propellant, StationResult, RocketPerformance,
    solve_rocket_nozzle, stoichiometric_OF,
)

__all__ = [
    "__version__",
    # core
    "parse_thermo_file", "Species", "get_products_for_elements",
    "cp_over_R", "h_over_RT", "s_over_R", "g_over_RT", "R_UNIVERSAL",
    "parse_reaction_string", "get_total_elements", "parse_formula",
    "solve_equilibrium", "solve_equilibrium_HP", "solve_equilibrium_SP",
    "EquilibriumResult", "mixture_enthalpy", "mixture_entropy",
    "run_batch", "run_interactive", "find_thermo_db",
    # io
    "IterationLogger", "NullLogger",
    # rocket
    "Propellant", "StationResult", "RocketPerformance",
    "solve_rocket_nozzle", "stoichiometric_OF",
]
