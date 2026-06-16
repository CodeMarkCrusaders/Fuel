"""
fuel_equilibrium.rocket.of_sweep
=================================

Развёртка характеристик ракетного двигателя по соотношению компонентов
**O/F** (oxidizer/fuel mass ratio) с поиском оптимального соотношения —
классическая функция RPA / NASA CEA «Isp vs O/F».

Идея: при фиксированных давлениях в камере и на срезе мы многократно
запускаем равновесный расчёт сопла :func:`solve_rocket_nozzle` для серии
значений O/F и собираем кривые удельного импульса, характеристической
скорости, температуры в камере и т.п. Затем по этим точкам находим O/F,
при котором ``Isp`` (или ``Isp_vac``) максимален — параболической
интерполяцией по трём точкам вокруг максимума на сетке.

Модуль «чистый» (слой ``rocket/``): только физика поверх ``core`` и
``nozzle_flow``, без зависимостей от GUI и без побочного ввода-вывода
(кроме опционального ``IterationLogger``). Форматирование таблиц вынесено
в слой ``io`` (см. ``io.reporting.print_of_sweep_table``).

Пример::

    from fuel_equilibrium.core import parse_thermo_file, find_thermo_db
    from fuel_equilibrium.rocket import sweep_of_ratio

    db = parse_thermo_file(find_thermo_db())
    sweep = sweep_of_ratio(
        oxidizer_name="O2(L)", fuel_name="H2(L)",
        P_chamber=10e6, P_exit=0.1013e6,
        species_db=db, of_min=3.0, of_max=8.0, n_points=11,
    )
    print(f"Оптимум по Isp: O/F = {sweep.best_of:.3f}, "
          f"Isp = {sweep.best_Isp_s:.2f} с")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.nasa9_parser import Species
from ..io.iteration_logger import IterationLogger, NullLogger
from .nozzle_flow import Propellant, RocketPerformance, solve_rocket_nozzle


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OFSweepPoint:
    """Одна точка развёртки по O/F."""
    of: float                       # массовое соотношение O/F
    alpha: float                    # коэффициент избытка окислителя
    phi: float                      # equivalence ratio = 1/alpha
    T_chamber_K: float              # температура в камере (Injector)
    Cstar_m_per_s: float            # характеристическая скорость
    Isp_s: float                    # удельный импульс на срезе (P_amb = 0)
    Isp_vac_s: float                # вакуумный удельный импульс
    CF: float                       # коэффициент тяги
    performance: Optional[RocketPerformance] = field(default=None, repr=False)
    error: Optional[str] = None     # текст ошибки, если расчёт точки не удался

    @property
    def ok(self) -> bool:
        """True, если точка посчитана без ошибок и Isp конечен."""
        return self.error is None and math.isfinite(self.Isp_s)


@dataclass
class OFSweepResult:
    """Результат развёртки по O/F: серия точек + найденный оптимум."""
    oxidizer_name: str
    fuel_name: str
    P_chamber_Pa: float
    P_exit_Pa: float
    points: List[OFSweepPoint]

    # Оптимум по удельному импульсу (на срезе при P_amb = 0)
    best_of: float                  # O/F, дающий максимальный Isp
    best_Isp_s: float               # сам максимальный Isp, с
    best_point_index: int           # индекс ближайшей узловой точки сетки

    # Оптимум по вакуумному удельному импульсу
    best_of_vac: float
    best_Isp_vac_s: float

    optimize_for: str = "Isp"       # критерий оптимизации: 'Isp' | 'Isp_vac'
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def ok_points(self) -> List[OFSweepPoint]:
        """Только успешно посчитанные точки."""
        return [p for p in self.points if p.ok]


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная математика: параболический оптимум
# ─────────────────────────────────────────────────────────────────────────────

def _parabolic_vertex(x0: float, x1: float, x2: float,
                      y0: float, y1: float, y2: float) -> Optional[float]:
    """Абсцисса вершины параболы через 3 точки (x1 — середина, y1 — максимум).

    Возвращает x вершины, если он лежит в ``[x0, x2]``; иначе ``None``
    (вырожденный случай / минимум / экстремум вне интервала).
    """
    # Парабола y = a·x² + b·x + c через 3 точки. Вершина: x* = -b/(2a).
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-30:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    if abs(a) < 1e-30 or a >= 0.0:
        # a >= 0 → ветви вверх (минимум), нам нужен максимум (a < 0)
        return None
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    x_star = -b / (2.0 * a)
    lo, hi = min(x0, x2), max(x0, x2)
    if not (lo <= x_star <= hi):
        return None
    return x_star


def _refine_optimum(ofs: List[float], ys: List[float]) -> tuple:
    """По дискретным (O/F, y) находит уточнённый оптимум (of*, y*).

    Сначала берётся узловая точка с максимальным y, затем —
    если у неё есть оба соседа — выполняется параболическое уточнение.
    Возвращает (of_opt, y_opt, index_of_best_node).
    """
    if not ofs:
        return float("nan"), float("nan"), -1
    i_best = max(range(len(ys)), key=lambda i: ys[i])
    of_best, y_best = ofs[i_best], ys[i_best]
    # Параболическое уточнение по тройке вокруг узла-максимума
    if 0 < i_best < len(ofs) - 1:
        x_star = _parabolic_vertex(
            ofs[i_best - 1], ofs[i_best], ofs[i_best + 1],
            ys[i_best - 1], ys[i_best], ys[i_best + 1],
        )
        if x_star is not None:
            # значение y в вершине параболы
            y_star = _parabolic_value(
                ofs[i_best - 1], ofs[i_best], ofs[i_best + 1],
                ys[i_best - 1], ys[i_best], ys[i_best + 1],
                x_star,
            )
            if math.isfinite(y_star) and y_star >= y_best:
                return x_star, y_star, i_best
    return of_best, y_best, i_best


def _parabolic_value(x0, x1, x2, y0, y1, y2, x) -> float:
    """Значение параболы (через 3 точки) в точке x."""
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-30:
        return float("nan")
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1
         + x0 * x1 * (x0 - x1) * y2) / denom
    return a * x * x + b * x + c


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция: развёртка по O/F
# ─────────────────────────────────────────────────────────────────────────────

def sweep_of_ratio(
    oxidizer_name: str,
    fuel_name: str,
    P_chamber: float,
    P_exit: float,
    species_db: Dict[str, Species],
    of_min: float,
    of_max: float,
    n_points: int = 11,
    of_values: Optional[List[float]] = None,
    fuel_mass_kg: float = 1.0,
    oxidizer_T_K: Optional[float] = None,
    fuel_T_K: Optional[float] = None,
    optimize_for: str = "Isp",
    n_intermediate_stations: int = 0,
    include_condensed: bool = True,
    verbose: bool = False,
    logger: Optional[IterationLogger] = None,
) -> OFSweepResult:
    """Развёртка тяговых характеристик по соотношению компонентов O/F.

    Для серии значений O/F (равномерная сетка ``of_min..of_max`` из
    ``n_points`` точек, либо явный список ``of_values``) запускается
    равновесный расчёт сопла и собираются кривые ``Isp``, ``Isp_vac``,
    ``C*``, ``T_chamber``, ``CF``. Затем находится O/F, максимизирующий
    выбранный критерий (``optimize_for`` = ``'Isp'`` или ``'Isp_vac'``),
    с параболическим уточнением между узлами сетки.

    Параметры:
        oxidizer_name, fuel_name — имена компонентов как в базе NASA-9
                                   (например ``'O2(L)'`` и ``'H2(L)'``).
        P_chamber, P_exit        — давления в камере и на срезе, Па.
        species_db               — база NASA-9.
        of_min, of_max, n_points — равномерная сетка по O/F.
        of_values                — явный список O/F (приоритетнее сетки).
        fuel_mass_kg             — базовая масса горючего (масса окислителя
                                   = O/F · fuel_mass_kg).
        oxidizer_T_K, fuel_T_K   — температуры подачи (None → из базы).
        optimize_for             — критерий оптимума: ``'Isp'`` | ``'Isp_vac'``.
        n_intermediate_stations  — число доп. сечений для каждого расчёта.
        include_condensed        — учитывать конденсированные фазы.
        logger                   — журнал итераций (опционально).

    Возвращает :class:`OFSweepResult`. Точки, где расчёт упал, помечаются
    полем ``error`` и не участвуют в поиске оптимума.
    """
    if logger is None:
        logger = NullLogger()

    if optimize_for not in ("Isp", "Isp_vac"):
        raise ValueError("optimize_for должен быть 'Isp' или 'Isp_vac'")

    # ── сформировать список O/F ──────────────────────────────────────────
    if of_values is not None:
        ofs = [float(v) for v in of_values if float(v) > 0.0]
        if not ofs:
            raise ValueError("of_values не содержит положительных значений")
        ofs = sorted(ofs)
    else:
        if not (math.isfinite(of_min) and math.isfinite(of_max)):
            raise ValueError("of_min / of_max должны быть конечными числами")
        if of_min <= 0.0 or of_max <= 0.0:
            raise ValueError("O/F должно быть положительным")
        if of_max < of_min:
            of_min, of_max = of_max, of_min
        n_points = int(max(2, n_points))
        if abs(of_max - of_min) < 1e-12:
            ofs = [of_min]
        else:
            step = (of_max - of_min) / (n_points - 1)
            ofs = [of_min + i * step for i in range(n_points)]

    if fuel_mass_kg <= 0.0:
        raise ValueError("fuel_mass_kg должно быть положительным")

    if logger.enabled:
        logger.section("РАЗВЁРТКА ПО O/F")
        logger.log(f"Окислитель: {oxidizer_name},  горючее: {fuel_name}")
        logger.log(f"Pc = {P_chamber:.0f} Па,  Pe = {P_exit:.0f} Па")
        logger.log(f"O/F-сетка: {', '.join(f'{v:.4f}' for v in ofs)}")
        logger.log(f"Критерий оптимума: {optimize_for}")

    # ── пройти по всем O/F ───────────────────────────────────────────────
    points: List[OFSweepPoint] = []
    for of in ofs:
        ox = Propellant(oxidizer_name, mass_kg=of * fuel_mass_kg, T_K=oxidizer_T_K)
        fu = Propellant(fuel_name, mass_kg=fuel_mass_kg, T_K=fuel_T_K)
        try:
            perf = solve_rocket_nozzle(
                oxidizer=ox, fuel=fu,
                P_chamber=P_chamber, P_exit=P_exit,
                species_db=species_db,
                n_intermediate_stations=n_intermediate_stations,
                include_condensed=include_condensed,
                verbose=verbose,
                logger=logger,
            )
            T_chamber = perf.stations[0].T_K if perf.stations else float("nan")
            pt = OFSweepPoint(
                of=of,
                alpha=perf.alpha,
                phi=perf.phi,
                T_chamber_K=T_chamber,
                Cstar_m_per_s=perf.Cstar_m_per_s,
                Isp_s=perf.Isp_s,
                Isp_vac_s=perf.Isp_vac_s,
                CF=perf.CF,
                performance=perf,
            )
            if logger.enabled:
                logger.log(f"  O/F = {of:7.4f}  →  Isp = {perf.Isp_s:8.3f} с,  "
                           f"Tк = {T_chamber:7.1f} К,  C* = {perf.Cstar_m_per_s:7.1f} м/с")
        except Exception as exc:  # noqa: BLE001 — точка не должна валить всю развёртку
            pt = OFSweepPoint(
                of=of, alpha=float("nan"), phi=float("nan"),
                T_chamber_K=float("nan"), Cstar_m_per_s=float("nan"),
                Isp_s=float("nan"), Isp_vac_s=float("nan"), CF=float("nan"),
                performance=None, error=str(exc),
            )
            if logger.enabled:
                logger.log(f"  O/F = {of:7.4f}  →  ОШИБКА: {exc}")
        points.append(pt)

    # ── поиск оптимума по обоим критериям ────────────────────────────────
    ok = [p for p in points if p.ok]
    if not ok:
        raise RuntimeError("Ни одна точка развёртки не посчитана успешно")

    ofs_ok = [p.of for p in ok]
    isp_ok = [p.Isp_s for p in ok]
    ispv_ok = [p.Isp_vac_s for p in ok]

    best_of, best_isp, i_best = _refine_optimum(ofs_ok, isp_ok)
    best_of_vac, best_ispv, _ = _refine_optimum(ofs_ok, ispv_ok)

    # индекс лучшего узла в полном списке points (по совпадению O/F)
    best_node_of = ofs_ok[i_best]
    best_point_index = next(
        (i for i, p in enumerate(points) if p.of == best_node_of), -1
    )

    if logger.enabled:
        logger.section("ОПТИМУМ РАЗВЁРТКИ")
        logger.log(f"max Isp     : O/F = {best_of:.4f},  Isp = {best_isp:.3f} с")
        logger.log(f"max Isp_vac : O/F = {best_of_vac:.4f},  Isp_vac = {best_ispv:.3f} с")

    return OFSweepResult(
        oxidizer_name=oxidizer_name,
        fuel_name=fuel_name,
        P_chamber_Pa=P_chamber,
        P_exit_Pa=P_exit,
        points=points,
        best_of=best_of if optimize_for == "Isp" else best_of_vac,
        best_Isp_s=best_isp,
        best_point_index=best_point_index,
        best_of_vac=best_of_vac,
        best_Isp_vac_s=best_ispv,
        optimize_for=optimize_for,
        notes={
            "method": "Развёртка по O/F поверх solve_rocket_nozzle",
            "optimum": "узел сетки + параболическое уточнение по 3 точкам",
        },
    )
