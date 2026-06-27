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

# Построение геометрии сопла по учебнику Добровольского (гл. 2):
# коническое (§2.3) и профилированное оптимальное (§2.6) сопла.
from .nozzle_geometry import (
    ContourPoint,
    NozzleGeometry,
    dispersion_loss_coeff,
    exit_angle_from_dispersion,
    exit_angle_from_underexpansion,
    optimal_angles_from_area_ratio,
    set_optimal_grid,
    build_conical_nozzle,
    build_profiled_nozzle,
    build_rpa_parabolic_nozzle,
    rao_reference_length_15deg,
    estimate_bell_angles,
    build_geometry_from_performance,
    build_nozzle_geometry,
)

# Двумерный (осесимметричный) расчёт сопла — заготовка (1D/2D выбор в GUI).
from .nozzle_flow_2d import (
    Nozzle2DField,
    Nozzle2DResult,
    solve_nozzle_2d,
    build_axisymmetric_grid,
)

# Развёртка характеристик по соотношению компонентов O/F с поиском оптимума
# (классическая функция RPA / NASA CEA «Isp vs O/F»).
from .of_sweep import (
    OFSweepPoint,
    OFSweepResult,
    sweep_of_ratio,
)

# CEA-решатель опционален: требует cantera
try:
    from .cea_solver import (
        solve_rocket_nozzle_cea,
        build_axial_coordinates,
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
    # геометрия сопла по Добровольскому (гл. 2)
    "ContourPoint",
    "NozzleGeometry",
    "dispersion_loss_coeff",
    "exit_angle_from_dispersion",
    "exit_angle_from_underexpansion",
    "optimal_angles_from_area_ratio",
    "set_optimal_grid",
    "build_conical_nozzle",
    "build_profiled_nozzle",
    "build_rpa_parabolic_nozzle",
    "rao_reference_length_15deg",
    "estimate_bell_angles",
    "build_nozzle_geometry",
    "build_geometry_from_performance",
    # двумерный (осесимметричный) расчёт — заготовка
    "Nozzle2DField",
    "Nozzle2DResult",
    "solve_nozzle_2d",
    "build_axisymmetric_grid",
    # развёртка по O/F и поиск оптимума
    "OFSweepPoint",
    "OFSweepResult",
    "sweep_of_ratio",
    "CANTERA_AVAILABLE",
]

if CANTERA_AVAILABLE:
    __all__ += ["solve_rocket_nozzle_cea", "build_axial_coordinates", "nozzle_radius"]
