"""
Двумерный (осесимметричный) газодинамический расчёт сопла — ЗАГОТОВКА.

Назначение
----------
Модуль задаёт каркас (scaffolding) для перехода от одномерного (квази-1D)
газодинамического расчёта к двумерному осесимметричному. Сейчас реализована
структура данных, единый интерфейс ``solve_nozzle_2d`` и приближённая
заглушка, которая «оборачивает» одномерный профиль в 2D-поле, добавляя
поправку на угол потока вдоль стенки (дисперсию). Это позволяет уже сейчас:

    * выбрать в GUI режим расчёта 1D/2D;
    * получить 2D-структуры результатов (сетка x, r и поля параметров);
    * визуализировать контур и поля без реализации полного метода
      характеристик (MOC).

Полноценная 2D-постановка (метод характеристик для сверхзвуковой части,
осесимметричные уравнения Эйлера) реализуется поэтапно в местах, помеченных
``TODO(2D)`` ниже.

Ссылки
------
* М.В. Добровольский. «Жидкостные ракетные двигатели. Основы проектирования»,
  2016 — гл. 2 (сопла ЖРД), §2.6 — профилирование, дисперсионные потери.
* Anderson J.D. «Modern Compressible Flow» — метод характеристик (MOC) для
  осесимметричных сопел.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict

import numpy as np

from .nozzle_geometry import NozzleGeometry


__all__ = [
    "Nozzle2DField",
    "Nozzle2DResult",
    "solve_nozzle_2d",
    "build_axisymmetric_grid",
]


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных 2D-результата
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Nozzle2DField:
    """Скалярное поле параметра на 2D-сетке (x, r).

    ``values`` имеет форму (n_r, n_x): строки — по радиусу, столбцы — по оси x.
    """
    name: str
    unit: str
    values: np.ndarray  # shape (n_r, n_x)


@dataclass
class Nozzle2DResult:
    """Результат двумерного (осесимметричного) расчёта сопла.

    Поля заданы на структурированной сетке (x_grid, r_grid) формы (n_r, n_x),
    где строки — радиальные узлы (0 — ось, последняя — стенка).
    """
    method: str                 # 'quasi2d_stub' | 'moc' | ...
    x_grid: np.ndarray          # (n_r, n_x) координаты по оси, м
    r_grid: np.ndarray          # (n_r, n_x) координаты по радиусу, м
    wall_x: np.ndarray          # (n_x,) координата стенки по оси, м
    wall_r: np.ndarray          # (n_x,) радиус стенки, м
    fields: Dict[str, Nozzle2DField] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.x_grid.shape

    def field_values(self, name: str) -> Optional[np.ndarray]:
        f = self.fields.get(name)
        return None if f is None else f.values


# ─────────────────────────────────────────────────────────────────────────────
# Построение осесимметричной сетки по контуру
# ─────────────────────────────────────────────────────────────────────────────

def build_axisymmetric_grid(
    geom: NozzleGeometry,
    n_radial: int = 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Строит структурированную сетку (x, r) внутри контура сопла.

    Возвращает (x_grid, r_grid, wall_x, wall_r) форм (n_r, n_x) и (n_x,).
    Радиальные узлы распределяются от оси (r=0) до стенки (r=wall_r(x)).
    """
    xw, rw = geom.as_xy_arrays()
    xw = np.asarray(xw, dtype=float)
    rw = np.asarray(rw, dtype=float)
    n_r = max(3, int(n_radial))

    # нормированный радиус 0..1 (равномерно; при желании — сгущать у стенки)
    eta = np.linspace(0.0, 1.0, n_r)
    x_grid = np.tile(xw, (n_r, 1))                      # (n_r, n_x)
    r_grid = np.outer(eta, rw)                          # (n_r, n_x)
    return x_grid, r_grid, xw, rw


# ─────────────────────────────────────────────────────────────────────────────
# Главный интерфейс 2D-решателя
# ─────────────────────────────────────────────────────────────────────────────

def solve_nozzle_2d(
    perf: Any,
    geom: NozzleGeometry,
    *,
    n_radial: int = 21,
    method: str = "quasi2d_stub",
) -> Nozzle2DResult:
    """Двумерный (осесимметричный) расчёт сопла — ЗАГОТОВКА.

    Параметры
    ---------
    perf : RocketPerformance
        Результат 1D-расчёта (``solve_rocket_nozzle``) — источник
        осреднённых параметров вдоль оси (P, T, M, V, ρ).
    geom : NozzleGeometry
        Геометрия контура (стенка) для построения сетки.
    n_radial : int
        Число радиальных узлов сетки.
    method : str
        'quasi2d_stub' — приближённое «развёртывание» 1D-профиля в 2D-поле
        с поправкой на угол потока у стенки. Иные значения зарезервированы
        под полноценный MOC (``TODO(2D)``).

    Возвращает
    ----------
    Nozzle2DResult с полями давления, температуры, числа Маха, скорости и
    угла потока на осесимметричной сетке.
    """
    x_grid, r_grid, wall_x, wall_r = build_axisymmetric_grid(geom, n_radial)

    # ── осевые (1D) распределения, интерполированные на узлы стенки по x ──
    axial = _axial_profiles_from_perf(perf, geom, wall_x)

    if method in ("quasi2d_stub", "quasi2d", "source_flow"):
        fields, meta = _quasi2d_fields(axial, x_grid, r_grid, wall_r)
    else:
        # TODO(2D): здесь подключается полноценный метод характеристик (MOC)
        #   1. Трансзвуковая стартовая линия у горловины (Sauer / Hall).
        #   2. Сетка характеристик в сверхзвуковой части до стенки.
        #   3. Согласование с профилем стенки (обратная задача профилирования).
        raise NotImplementedError(
            f"2D-метод '{method}' ещё не реализован. Доступно: 'quasi2d_stub'."
        )

    return Nozzle2DResult(
        method=method,
        x_grid=x_grid,
        r_grid=r_grid,
        wall_x=wall_x,
        wall_r=wall_r,
        fields=fields,
        metadata=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Внутренние помощники (заглушки приближённого 2D-поля)
# ─────────────────────────────────────────────────────────────────────────────

def _axial_profiles_from_perf(
    perf: Any,
    geom: NozzleGeometry,
    wall_x: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Готовит осевые (1D) распределения P, T, M, V, ρ, интерполированные
    на координаты ``wall_x`` (узлы контура по оси).

    Привязка 1D-сечений к координате x выполняется по относительной площади
    Ae/At ↔ радиусу контура: сечение с заданным Ae/At размещается в той точке
    контура, где локальная площадь совпадает.
    """
    stations = list(getattr(perf, "stations", []) or [])
    R_throat = geom.R_throat_m

    xs_st: List[float] = []
    P = []; T = []; M = []; V = []; RHO = []; GAM = []; RSPEC = []
    for st in stations:
        ar = float(getattr(st, "Ae_At", 1.0))
        r_target = R_throat * math.sqrt(max(ar, 1e-9))
        x_st = _x_for_radius(geom, r_target, st)
        xs_st.append(x_st)
        P.append(float(getattr(st, "P_Pa", 0.0)))
        T.append(float(getattr(st, "T_K", 0.0)))
        M.append(float(getattr(st, "M", 0.0)))
        V.append(float(getattr(st, "V_m_per_s", 0.0)))
        RHO.append(float(getattr(st, "rho_kg_per_m3", 0.0)))
        GAM.append(float(getattr(st, "gamma_s", 0.0) or 1.2))
        RSPEC.append(float(getattr(st, "R_specific_J_per_kgK", 0.0) or 0.0))

    xs_st = np.asarray(xs_st, dtype=float)
    order = np.argsort(xs_st)
    xs_sorted = xs_st[order]

    def _interp(arr, default=0.0):
        a = np.asarray(arr, dtype=float)[order]
        xs_u, idx = np.unique(xs_sorted, return_index=True)
        if xs_u.size < 2:
            return np.full_like(wall_x, a[0] if a.size else default)
        return np.interp(wall_x, xs_u, a[idx])

    return {
        "P_Pa": _interp(P),
        "T_K": _interp(T),
        "M": _interp(M),
        "V_m_per_s": _interp(V),
        "rho_kg_per_m3": _interp(RHO),
        "gamma": _interp(GAM, default=1.2),
        "R_specific": _interp(RSPEC),
    }


def _x_for_radius(geom: NozzleGeometry, r_target: float, station: Any) -> float:
    """Грубая привязка сечения к координате x по радиусу контура.

    Для дозвука (M<1) ищем точку в дозвуковой части, для сверхзвука — в
    сверхзвуковой; на горловине — x_throat.
    """
    x_throat = geom.length_subsonic_m
    M = float(getattr(station, "M", 1.0))
    pts = geom.points
    if abs(M - 1.0) < 1e-3:
        return x_throat
    candidates = [p for p in pts if (p.x_m <= x_throat) == (M < 1.0)]
    if not candidates:
        candidates = pts
    best = min(candidates, key=lambda p: abs(p.r_m - r_target))
    return best.x_m


def _quasi2d_fields(
    axial: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    r_grid: np.ndarray,
    wall_r: np.ndarray,
) -> Tuple[Dict[str, Nozzle2DField], Dict[str, Any]]:
    """Квази-2D поле течения с реальным изменением параметров по ДВУМ
    координатам (осевой x и радиальной r).

    Физическая модель (source-flow / коническое приближение)
    --------------------------------------------------------
    В расширяющейся (сверхзвуковой) части сопла линии тока расходятся, и
    течение приближается к источниковому (radial / source flow): у стенки,
    где угол расхождения максимален, газ успевает расшириться сильнее, чем
    на оси. Это даёт радиальный градиент числа Маха.

    Алгоритм:
      1. Берём осевое (1D) распределение M0(x), gamma(x) как «среднемассовое».
      2. Строим радиальное распределение числа Маха
             M(x, r) = M0(x) * [1 + k(x) * (r/R)^2],
         где коэффициент k(x) растёт с локальным углом стенки θ(x):
             k(x) = c_div * sin(θ_wall(x))     (в расширении, θ>0),
         а в сужении (θ<0) знак меняется (на оси Маха больше). Параметр
         нормируется так, чтобы среднемассовое M по сечению ≈ M0(x).
      3. По M(x,r) и изэнтропическим соотношениям пересчитываются T, P, V:
             T0/T = 1 + (γ-1)/2 · M²,
             P/P0_tot = (T/T0)^(γ/(γ-1)),
         где локальные «полные» параметры берутся из осевого решения
         (по M0, T0_axis, P0_axis). Скорость V = M · a, a = sqrt(γ R T).
      4. Угол потока меняется радиально (0 на оси → θ_wall у стенки).

    Это НЕ полное решение 2D-уравнений Эйлера (для него — MOC, ``TODO(2D)``),
    но параметры действительно зависят и от x, и от r.
    """
    n_r, n_x = x_grid.shape
    eta = np.linspace(0.0, 1.0, n_r)             # нормированный радиус r/R
    ETA = np.tile(eta.reshape(-1, 1), (1, n_x))  # (n_r, n_x)
    ETA2 = ETA ** 2

    # осевые (1D) распределения, растиражированные по сечению как «опорные»
    M0 = np.tile(axial["M"].reshape(1, -1), (n_r, 1))
    T0_axis = np.tile(axial["T_K"].reshape(1, -1), (n_r, 1))
    P0_axis = np.tile(axial["P_Pa"].reshape(1, -1), (n_r, 1))
    gamma = np.tile(axial["gamma"].reshape(1, -1), (n_r, 1))
    gamma = np.clip(gamma, 1.05, 1.67)
    Rg = np.tile(axial["R_specific"].reshape(1, -1), (n_r, 1))

    # локальный угол стенки θ(x) = atan(dr/dx)
    dx = np.gradient(x_grid[-1, :])
    dr = np.gradient(wall_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_wall = np.arctan2(dr, dx)          # рад, (n_x,)
    theta_wall = np.nan_to_num(theta_wall)
    theta_row = theta_wall.reshape(1, -1)

    # ── 1) Радиальное распределение числа Маха (source-flow приближение) ──
    # коэффициент радиальной неравномерности растёт с углом расхождения
    c_div = 0.35                                  # масштаб эффекта расхождения
    k = c_div * np.sin(theta_row)                 # >0 в расширении, <0 в сужении
    k = np.tile(k, (n_r, 1))
    # радиальный профиль (квадратичный): на оси (η=0) множитель 1
    radial_shape = 1.0 + k * ETA2
    # нормировка: среднее по площади (∝ 2·∫ η·shape dη) приводим к 1, чтобы
    # M0(x) оставался среднемассовым числом Маха сечения.
    #   ∫_0^1 2η(1+kη²)dη = 1 + k/2  → делим на (1 + k/2)
    norm = 1.0 + 0.5 * k
    radial_shape = radial_shape / np.where(np.abs(norm) > 1e-9, norm, 1.0)
    M_field = np.clip(M0 * radial_shape, 0.0, None)

    # ── 2) Изэнтропический пересчёт T, P, V из локального M(x,r) ──
    gm1 = gamma - 1.0
    # «полная» температура по осевому решению: T_tot = T0_axis·(1+gm1/2·M0²)
    T_tot = T0_axis * (1.0 + 0.5 * gm1 * M0 ** 2)
    T_field = T_tot / (1.0 + 0.5 * gm1 * M_field ** 2)

    # «полное» давление по осевому решению (изэнтропа): P_tot = P0_axis·(T_tot/T0_axis)^(γ/(γ-1))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exp_g = gamma / np.where(np.abs(gm1) > 1e-9, gm1, 1e-9)
        P_tot = P0_axis * np.power(
            np.clip(T_tot / np.where(T0_axis > 1e-9, T0_axis, 1e-9), 1e-9, None),
            exp_g,
        )
        P_field = P_tot * np.power(
            np.clip(T_field / np.where(T_tot > 1e-9, T_tot, 1e-9), 1e-9, None),
            exp_g,
        )
    P_field = np.nan_to_num(P_field, nan=0.0, posinf=0.0, neginf=0.0)

    # скорость звука и скорость потока по локальным (T, M)
    with np.errstate(invalid="ignore"):
        a_field = np.sqrt(np.clip(gamma * Rg * T_field, 0.0, None))
    # если R_specific недоступен (0), откатываемся к V = M/M0 · V0_axis
    V0_axis = np.tile(axial["V_m_per_s"].reshape(1, -1), (n_r, 1))
    V_from_a = M_field * a_field
    V_field = np.where(a_field > 1e-6, V_from_a,
                       V0_axis * np.where(M0 > 1e-9, M_field / M0, 1.0))

    # ── 3) Угол потока: 0 на оси, θ_wall у стенки (линейно по η) ──
    flow_angle = ETA * np.tile(theta_row, (n_r, 1))

    fields = {
        "P_Pa": Nozzle2DField("Давление", "Па", P_field),
        "T_K": Nozzle2DField("Температура", "К", T_field),
        "M": Nozzle2DField("Число Маха", "-", M_field),
        "V_m_per_s": Nozzle2DField("Скорость", "м/с", V_field),
        "flow_angle_deg": Nozzle2DField(
            "Угол потока", "град", np.degrees(flow_angle)
        ),
    }
    meta = {
        "note": "quasi-2D: source-flow радиальное распределение M + "
                "изэнтропический пересчёт T, P, V; параметры зависят от (x, r). "
                "Не является полным решением 2D-уравнений Эйлера (MOC — TODO).",
        "is_stub": False,
        "model": "source_flow_isentropic",
        "n_radial": n_r,
        "n_axial": n_x,
        "c_div": c_div,
        "theta_wall_max_deg": float(np.degrees(np.max(np.abs(theta_wall)))),
        "M_radial_spread_max": float(
            np.max(np.abs(M_field[-1, :] - M_field[0, :]))
        ),
    }
    return fields, meta


# Обратная совместимость: старое имя-заглушка делегирует новому решателю.
def _quasi2d_stub_fields(
    axial: Dict[str, np.ndarray],
    x_grid: np.ndarray,
    r_grid: np.ndarray,
    wall_r: np.ndarray,
) -> Tuple[Dict[str, Nozzle2DField], Dict[str, Any]]:
    return _quasi2d_fields(axial, x_grid, r_grid, wall_r)
