# -*- coding: utf-8 -*-
"""
Авто-определение оптимального соотношения компонентов топлива.

Поддерживает 3 способа задания соотношения «окислитель/горючее»:

    mode='OF'       — задано массовое соотношение O/F (m_ox / m_fuel)
    mode='alpha'    — задан коэффициент избытка окислителя α = (O/F) / (O/F)_стех.
    mode='optimal'  — авто-поиск оптимума (по Isp или иной целевой функции)

В режиме 'optimal' выполняется 1D-сканирование по α (или O/F) с последующим
уточнением максимума параболической интерполяцией ± несколько шагов
квадратичного поиска вблизи экстремума.

Целевые функции:
    'Isp'      — оптимум по удельному импульсу на срезе (V_e/g0)
    'Isp_vac'  — оптимум по вакуумному удельному импульсу
    'Cstar'    — оптимум по характеристической скорости C*
    'T_chamber'— оптимум по температуре в камере (для прикидок)

Пример использования:

    from propellant_optimizer import find_optimal_OF, RatioSpec, OptimizationResult

    # 1) Фиксированный O/F = 6.0
    spec = RatioSpec(mode='OF', value=6.0)

    # 2) α = 0.8
    spec = RatioSpec(mode='alpha', value=0.8)

    # 3) Авто-оптимум по Isp на срезе
    spec = RatioSpec(mode='optimal', target='Isp',
                     alpha_min=0.3, alpha_max=1.5, n_grid=21)

    result = find_optimal_OF(
        oxidizer_name="O2(L)", fuel_name="H2(L)",
        spec=spec,
        P_chamber_Pa=10e6, P_exit_Pa=0.1013e6,
        species_db=db,
    )
    # result.OF, result.alpha, result.Isp, result.scan_table

Скан-таблица сохраняется в result.scan_table — это список словарей с полями
{alpha, OF, Isp_s, Isp_vac_s, Cstar, T_chamber}, что позволяет легко строить
кривые Isp(α) и сохранять их в CSV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nasa9_parser import Species
from nozzle_flow import (
    Propellant,
    RocketPerformance,
    solve_rocket_nozzle,
    stoichiometric_OF,
)
from iteration_logger import IterationLogger, NullLogger


# ─────────────────────────────────────────────────────────────────────────────
# Спецификация режима задания соотношения
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RatioSpec:
    """Как задано (или ищется) соотношение «окислитель/горючее».

    mode:
        'OF'      — задано массовое O/F  (value = OF)
        'alpha'   — задан коэффициент избытка окислителя (value = α)
        'optimal' — авто-поиск оптимума (value игнорируется)

    target (только для mode='optimal'):
        'Isp', 'Isp_vac', 'Cstar', 'T_chamber'

    Параметры сканирования:
        alpha_min, alpha_max — диапазон α для сканирования
        n_grid               — число точек на грубом скане
        refine               — выполнить ли уточнение макс. параболой
        tol                  — относительная точность поиска
    """
    mode: str = "OF"
    value: float = 1.0
    target: str = "Isp"
    alpha_min: float = 0.3
    alpha_max: float = 1.6
    n_grid: int = 13
    refine: bool = True
    tol: float = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Результат оптимизации
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanPoint:
    alpha: float
    OF: float
    Isp_s: float
    Isp_vac_s: float
    Cstar_m_per_s: float
    T_chamber_K: float
    CF: float


@dataclass
class OptimizationResult:
    OF: float
    alpha: float
    OF_stoich: float
    target: str
    target_value: float
    mode: str
    perf: RocketPerformance              # полный расчёт в оптимальной точке
    scan_table: List[ScanPoint] = field(default_factory=list)
    refined: bool = False
    n_calls: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Хелперы
# ─────────────────────────────────────────────────────────────────────────────

def _target_value(perf: RocketPerformance, target: str) -> float:
    """Извлекает целевую функцию из RocketPerformance."""
    if target == "Isp":
        return perf.Isp_s
    if target == "Isp_vac":
        return perf.Isp_vac_s
    if target == "Cstar":
        return perf.Cstar_m_per_s
    if target == "T_chamber":
        # T в камере = первое сечение (Injector)
        return perf.stations[0].T_K if perf.stations else float('nan')
    raise ValueError(f"Неизвестная цель оптимизации: {target!r}.  "
                     f"Допустимые: Isp, Isp_vac, Cstar, T_chamber")


def _evaluate(
    alpha: float,
    OF_stoich: float,
    oxidizer_name: str, fuel_name: str,
    oxidizer_T_K: Optional[float], fuel_T_K: Optional[float],
    P_chamber_Pa: float, P_exit_Pa: float,
    species_db: Dict[str, Species],
    n_intermediate_stations: int = 0,
    include_condensed: bool = True,
    logger: Optional[IterationLogger] = None,
) -> RocketPerformance:
    """Один расчёт сопла при заданном α."""
    of_actual = alpha * OF_stoich
    if of_actual <= 0:
        raise ValueError(f"O/F должно быть > 0  (α={alpha}, OF_st={OF_stoich})")
    # 1 кг суммарной массы: m_ox + m_fu = 1, m_ox/m_fu = of_actual
    m_fu = 1.0 / (1.0 + of_actual)
    m_ox = 1.0 - m_fu
    ox = Propellant(name=oxidizer_name, mass_kg=m_ox, T_K=oxidizer_T_K)
    fu = Propellant(name=fuel_name,     mass_kg=m_fu, T_K=fuel_T_K)
    return solve_rocket_nozzle(
        oxidizer=ox, fuel=fu,
        P_chamber=P_chamber_Pa, P_exit=P_exit_Pa,
        species_db=species_db,
        n_intermediate_stations=n_intermediate_stations,
        include_condensed=include_condensed,
        verbose=False,
        logger=logger if logger is not None else NullLogger(),
    )


def _make_scan_point(alpha: float, OF_stoich: float, perf: RocketPerformance) -> ScanPoint:
    return ScanPoint(
        alpha=alpha,
        OF=alpha * OF_stoich,
        Isp_s=perf.Isp_s,
        Isp_vac_s=perf.Isp_vac_s,
        Cstar_m_per_s=perf.Cstar_m_per_s,
        T_chamber_K=perf.stations[0].T_K if perf.stations else float('nan'),
        CF=perf.CF,
    )


def _parabolic_vertex(
    xs: Tuple[float, float, float],
    ys: Tuple[float, float, float],
) -> Optional[float]:
    """Возвращает x-координату вершины параболы через 3 точки (или None)."""
    x1, x2, x3 = xs
    y1, y2, y3 = ys
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-30:
        return None
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    if abs(a) < 1e-30 or a >= 0:   # не парабола вниз — нет максимума
        return None
    return -b / (2.0 * a)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_OF(
    oxidizer_name: str,
    fuel_name: str,
    spec: RatioSpec,
    P_chamber_Pa: float,
    P_exit_Pa: float,
    species_db: Dict[str, Species],
    oxidizer_T_K: Optional[float] = None,
    fuel_T_K: Optional[float] = None,
    n_intermediate_stations: int = 0,
    include_condensed: bool = True,
    logger: Optional[IterationLogger] = None,
    progress_cb: Optional[Callable[[int, int, float, RocketPerformance], None]] = None,
) -> OptimizationResult:
    """Считает оптимальное (или заданное) соотношение и возвращает полный
    расчёт сопла в этой точке.

    Параметры:
        spec      — RatioSpec
        progress_cb — необязательный колбэк (i, n, alpha, perf) для прогресса.
    """
    if logger is None:
        logger = NullLogger()

    # стехиометрия — пробуем посчитать; если не получится (для смешанных
    # окислителей типа IRFNA с дробными элементами), оставляем NaN и работаем
    # в режиме «OF/α» через alpha=of_actual/OF_stoich = of/OF_stoich.
    try:
        OF_stoich = stoichiometric_OF([species_db[oxidizer_name]],
                                       [species_db[fuel_name]])
    except Exception:
        OF_stoich = float('nan')

    if not (math.isfinite(OF_stoich) and OF_stoich > 0):
        # для смесей вроде IRFNA, Air — не определяем α, но всё равно
        # можем работать в режиме 'OF'. Для 'alpha' и 'optimal' выдаём ошибку,
        # если нет реального стехиометрического значения.
        if spec.mode in ("alpha", "optimal"):
            raise ValueError(
                f"Не удалось определить стехиометрическое O/F для "
                f"{oxidizer_name}/{fuel_name}.  Используйте mode='OF' и "
                f"задайте соотношение явно."
            )

    if logger.enabled:
        logger.section("ОПТИМИЗАЦИЯ СООТНОШЕНИЯ КОМПОНЕНТОВ")
        logger.log(f"Окислитель / Горючее : {oxidizer_name} / {fuel_name}")
        logger.log(f"Стехиометрическое O/F = {OF_stoich:.6f}")
        logger.log(f"Pc = {P_chamber_Pa/1e6:.4f} МПа,  Pe = {P_exit_Pa/1e6:.4f} МПа")
        logger.log(f"Режим: {spec.mode}, целевая функция: {spec.target}")

    # ── режим 'OF' ────────────────────────────────────────────────────────
    if spec.mode == "OF":
        of_actual = float(spec.value)
        if of_actual <= 0:
            raise ValueError("O/F должно быть > 0")
        alpha = (of_actual / OF_stoich) if (OF_stoich and math.isfinite(OF_stoich) and OF_stoich > 0) else float('nan')

        perf = _evaluate(
            alpha if (alpha and math.isfinite(alpha)) else of_actual / max(OF_stoich, 1e-9),
            OF_stoich if (OF_stoich and math.isfinite(OF_stoich) and OF_stoich > 0) else 1.0,
            # NB: если OF_stoich неизвестен, используем «фиктивный» 1.0,
            # чтобы _evaluate увидел of_actual = alpha * 1.0 = of_actual.
            oxidizer_name, fuel_name, oxidizer_T_K, fuel_T_K,
            P_chamber_Pa, P_exit_Pa, species_db,
            n_intermediate_stations, include_condensed, logger,
        )
        tv = _target_value(perf, spec.target)
        return OptimizationResult(
            OF=of_actual,
            alpha=alpha,
            OF_stoich=OF_stoich,
            target=spec.target,
            target_value=tv,
            mode="OF",
            perf=perf,
            scan_table=[],
            refined=False,
            n_calls=1,
        )

    # ── режим 'alpha' ─────────────────────────────────────────────────────
    if spec.mode == "alpha":
        alpha = float(spec.value)
        if alpha <= 0:
            raise ValueError("α должно быть > 0")
        perf = _evaluate(
            alpha, OF_stoich,
            oxidizer_name, fuel_name, oxidizer_T_K, fuel_T_K,
            P_chamber_Pa, P_exit_Pa, species_db,
            n_intermediate_stations, include_condensed, logger,
        )
        tv = _target_value(perf, spec.target)
        return OptimizationResult(
            OF=alpha * OF_stoich,
            alpha=alpha,
            OF_stoich=OF_stoich,
            target=spec.target,
            target_value=tv,
            mode="alpha",
            perf=perf,
            scan_table=[],
            refined=False,
            n_calls=1,
        )

    # ── режим 'optimal' ───────────────────────────────────────────────────
    if spec.mode != "optimal":
        raise ValueError(f"Неизвестный режим: {spec.mode!r}.  "
                         f"Допустимые: OF, alpha, optimal")

    a_lo, a_hi = float(spec.alpha_min), float(spec.alpha_max)
    n_grid = max(int(spec.n_grid), 5)

    if a_hi <= a_lo:
        raise ValueError(f"alpha_max ({a_hi}) должно быть > alpha_min ({a_lo})")

    if logger.enabled:
        logger.log(f"Скан α ∈ [{a_lo:.4f}, {a_hi:.4f}],  {n_grid} точек")

    grid = np.linspace(a_lo, a_hi, n_grid)
    scan: List[ScanPoint] = []
    n_calls = 0
    best_idx = -1
    best_val = -math.inf

    # На грубой сетке возможны точки, где SP-решатель не сойдётся (очень
    # бедная или богатая смесь). Их пропускаем, но если все точки упали —
    # выбрасываем ошибку.
    for i, a in enumerate(grid):
        a = float(a)
        try:
            perf = _evaluate(
                a, OF_stoich,
                oxidizer_name, fuel_name, oxidizer_T_K, fuel_T_K,
                P_chamber_Pa, P_exit_Pa, species_db,
                n_intermediate_stations, include_condensed,
                NullLogger(),  # внутрь не пишем — иначе захламит
            )
            n_calls += 1
            pt = _make_scan_point(a, OF_stoich, perf)
            scan.append(pt)
            tv = _target_value(perf, spec.target)
            if math.isfinite(tv) and tv > best_val:
                best_val = tv
                best_idx = len(scan) - 1
            if logger.enabled:
                logger.log(
                    f"  α = {a:.4f}  O/F = {pt.OF:.4f}  "
                    f"Isp = {pt.Isp_s:.2f} с  Isp_vac = {pt.Isp_vac_s:.2f} с  "
                    f"C* = {pt.Cstar_m_per_s:.1f}  T_c = {pt.T_chamber_K:.1f} К"
                )
            if progress_cb is not None:
                progress_cb(i + 1, n_grid, a, perf)
        except Exception as e:
            if logger.enabled:
                logger.log(f"  α = {a:.4f}: ошибка решателя ({e!s}) — пропускаю")

    if best_idx < 0 or not scan:
        raise RuntimeError(
            "Скан по α не дал ни одной успешной точки. "
            "Попробуйте сузить диапазон [alpha_min, alpha_max]."
        )

    # ── уточнение оптимума параболической интерполяцией ──────────────────
    refined = False
    alpha_best = scan[best_idx].alpha
    if spec.refine and 1 <= best_idx <= len(scan) - 2:
        a1, a2, a3 = scan[best_idx-1].alpha, scan[best_idx].alpha, scan[best_idx+1].alpha
        y1 = _target_value_from_point(scan[best_idx-1], spec.target)
        y2 = _target_value_from_point(scan[best_idx],   spec.target)
        y3 = _target_value_from_point(scan[best_idx+1], spec.target)
        a_vert = _parabolic_vertex((a1, a2, a3), (y1, y2, y3))
        if a_vert is not None and (a1 < a_vert < a3):
            # ещё несколько шагов уточнения «золотым» сужением вокруг a_vert
            a_left, a_right = a1, a3
            a_curr = a_vert
            curr_y = -math.inf
            for it in range(8):
                try:
                    perf_v = _evaluate(
                        a_curr, OF_stoich,
                        oxidizer_name, fuel_name, oxidizer_T_K, fuel_T_K,
                        P_chamber_Pa, P_exit_Pa, species_db,
                        n_intermediate_stations, include_condensed,
                        NullLogger(),
                    )
                    n_calls += 1
                    pt = _make_scan_point(a_curr, OF_stoich, perf_v)
                    scan.append(pt)
                    y = _target_value(perf_v, spec.target)
                    if y > best_val:
                        best_val = y
                        alpha_best = a_curr
                        best_idx = len(scan) - 1
                        refined = True
                    if abs(y - curr_y) / max(abs(y), 1.0) < spec.tol:
                        break
                    curr_y = y
                    # шаг к вершине: смещаем a_curr к точке с большим Isp
                    # (грубо: симметричное уточнение)
                    a_curr_new = (a_left + a_right) / 2.0 \
                        if abs(a_curr - (a_left + a_right) / 2) > spec.tol \
                        else a_curr * (1.0 + spec.tol)
                    a_curr = a_curr_new
                except Exception:
                    break

    # отсортируем скан-таблицу по α для красивой выдачи
    scan.sort(key=lambda p: p.alpha)

    # пересчитаем «оптимум» точно (с полным набором сечений), чтобы вернуть
    # пользователю полный perf c n_intermediate_stations
    perf_best = _evaluate(
        alpha_best, OF_stoich,
        oxidizer_name, fuel_name, oxidizer_T_K, fuel_T_K,
        P_chamber_Pa, P_exit_Pa, species_db,
        n_intermediate_stations, include_condensed, logger,
    )
    n_calls += 1
    best_tv = _target_value(perf_best, spec.target)

    if logger.enabled:
        logger.section("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
        logger.log(f"α_opt        = {alpha_best:.6f}")
        logger.log(f"O/F_opt      = {alpha_best * OF_stoich:.6f}")
        logger.log(f"{spec.target:<12s} = {best_tv:.6f}")
        logger.log(f"Расчётов сопла: {n_calls}")

    return OptimizationResult(
        OF=alpha_best * OF_stoich,
        alpha=alpha_best,
        OF_stoich=OF_stoich,
        target=spec.target,
        target_value=best_tv,
        mode="optimal",
        perf=perf_best,
        scan_table=scan,
        refined=refined,
        n_calls=n_calls,
    )


def _target_value_from_point(pt: ScanPoint, target: str) -> float:
    if target == "Isp":      return pt.Isp_s
    if target == "Isp_vac":  return pt.Isp_vac_s
    if target == "Cstar":    return pt.Cstar_m_per_s
    if target == "T_chamber":return pt.T_chamber_K
    raise ValueError(f"Неизвестная цель: {target}")


# ─────────────────────────────────────────────────────────────────────────────
# Печать сводки оптимизации
# ─────────────────────────────────────────────────────────────────────────────

def print_optimization_summary(result: OptimizationResult) -> None:
    """Печатает таблицу скана и итог в стиле РПА."""
    print()
    print("=" * 78)
    print(f"  АВТО-ПОДБОР СООТНОШЕНИЯ КОМПОНЕНТОВ  "
          f"(цель: максимум {result.target})")
    print("=" * 78)
    print(f"  Стехиометрическое O/F = {result.OF_stoich:.4f}")
    print()
    print(f"  {'α':>8s}  {'O/F':>8s}  {'Isp_s':>10s}  {'Isp_vac':>10s}  "
          f"{'C*':>10s}  {'T_camera':>10s}  {'CF':>8s}")
    print("  " + "-" * 70)
    for pt in result.scan_table:
        marker = "  ←  opt." if abs(pt.alpha - result.alpha) < 1e-6 else ""
        print(f"  {pt.alpha:>8.4f}  {pt.OF:>8.4f}  "
              f"{pt.Isp_s:>10.3f}  {pt.Isp_vac_s:>10.3f}  "
              f"{pt.Cstar_m_per_s:>10.2f}  {pt.T_chamber_K:>10.2f}  "
              f"{pt.CF:>8.4f}{marker}")
    print()
    print(f"  >> ОПТИМУМ:  α = {result.alpha:.6f},  "
          f"O/F = {result.OF:.6f},  {result.target} = {result.target_value:.4f}")
    print(f"  Всего расчётов сопла: {result.n_calls}"
          + ("  (с уточнением параболой)" if result.refined else ""))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрация
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from nasa9_parser import parse_thermo_file
    from equilibrium import find_thermo_db

    db = parse_thermo_file(find_thermo_db())

    # Пример: оптимум по Isp для H2/O2 при Pc=10 МПа, Pe=1 атм
    spec = RatioSpec(
        mode="optimal", target="Isp",
        alpha_min=0.4, alpha_max=1.2, n_grid=9, refine=True,
    )

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    with IterationLogger(os.path.join(log_dir, 'optimize_H2_O2.log')) as logger:
        res = find_optimal_OF(
            oxidizer_name="O2(L)", fuel_name="H2(L)",
            spec=spec,
            P_chamber_Pa=10e6, P_exit_Pa=0.1013e6,
            species_db=db,
            logger=logger,
        )
    print_optimization_summary(res)

    # Печатаем таблицу сечений для оптимальной точки
    from nozzle_flow import print_nozzle_table
    print_nozzle_table(res.perf)
