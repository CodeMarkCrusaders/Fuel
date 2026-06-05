"""
fuel_equilibrium.rocket — прикладной слой: газодинамика ракетного сопла.

Содержит:
    * структуры Propellant / StationResult / RocketPerformance и решатель
      ``solve_rocket_nozzle`` на собственном Gibbs-движке     — nozzle_flow
    * альтернативный CEA-эквивалентный решатель на Cantera   — cea_solver
"""

from .nozzle_flow import (
    Propellant,
    StationResult,
    RocketPerformance,
    NozzleContourPoint,
    NozzleContour,
    solve_rocket_nozzle,
    stoichiometric_OF,
    build_profiled_nozzle_contour,
    build_approximate_optimal_contour_ch26,
    build_optimal_nozzle_contour,
    build_nozzle_contour,
)

# CEA-решатель опционален: требует cantera
try:
    from .cea_solver import (
        solve_rocket_nozzle_cea,
        build_nozzle_geometry,
        nozzle_radius,
        CANTERA_AVAILABLE,
    )
except ImportError:
    CANTERA_AVAILABLE = False

__all__ = [
    "Propellant",
    "StationResult",
    "RocketPerformance",
    "NozzleContourPoint",
    "NozzleContour",
    "solve_rocket_nozzle",
    "stoichiometric_OF",
    "build_profiled_nozzle_contour",
    "build_approximate_optimal_contour_ch26",
    "build_optimal_nozzle_contour",
    "build_nozzle_contour",
    "CANTERA_AVAILABLE",
]

if CANTERA_AVAILABLE:
    __all__ += ["solve_rocket_nozzle_cea", "build_nozzle_geometry", "nozzle_radius"]
