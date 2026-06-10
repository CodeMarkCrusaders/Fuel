"""
fuel_equilibrium.rocket.nozzle_geometry
========================================

Построение геометрии (контура) сопла ЖРД по методике учебника

    М. В. Добровольский. «Жидкостные ракетные двигатели. Основы
    проектирования», 2-е изд., 2016. Глава 2 «Сопла ЖРД».

Реализованы два типа сопел:

*  **Коническое сопло** (§2.3). Сверхзвуковая часть — прямой конус с
   полуу­глом раствора ``θ_a`` (2θ_a = 25…30°), скруглённый у горловины
   радиусом ``r_скр``. Дозвуковая (входная) часть — конус с полу­углом
   ``θ_вх`` (2θ_вх = 45…80°), сопряжённый со скруглением ``R_скр`` у
   горловины и со скруглением ``R_1`` у входа из камеры.

*  **Профилированное (укороченное оптимальное) сопло** (§2.6). Дозвуковая
   часть строится так же; сверхзвуковая часть представляет собой параболу,
   касающуюся в начальной точке ``A_n`` направления под углом ``θ_m`` (угол
   в начале сверхзвуковой части), а в конечной точке ``C`` (срез) —
   направления под углом ``θ_a``. Контур аппроксимируется квадратичной
   кривой Безье через точку пересечения касательных ``f``.

Дополнительно реализованы:

*  φ_рас — коэффициент рассеяния потока на срезе (§2.2):
        φ_рас = (1 + cos θ_a) / 2.
*  Угол среза θ_a из условия безотрывного течения при недорасширении
   (ур. 2.23): sin 2θ_a = (p_a − p_н)/(½ ρ_a w_a²) · ctg μ_a,
   где μ_a — угол Маха на срезе, sin μ_a = 1/M_a (ур. 2.24).
*  Семейство оптимальных контуров (Рис. 2.14): по отношению площадей
   ``R_a/R_кр`` (для γ = 1.23) восстанавливаются углы ``θ_m``, ``θ_a`` и
   относительная длина ``x̄_a = L / R_кр``. Аппроксимация табличная с
   возможностью переопределения пользователем (``set_optimal_grid``).

Модуль «чистый»: только геометрия, без зависимостей от GUI/ввода-вывода.
Все размеры — в метрах, углы во входных параметрах — в градусах.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContourPoint:
    """Точка контура сопла в осесимметричной постановке (x, r), м.

    x отсчитывается вдоль оси сопла (от входа в дозвуковую часть к срезу),
    r — радиус (расстояние от оси).
    """
    x_m: float
    r_m: float


@dataclass
class NozzleGeometry:
    """Полная геометрия контура сопла по Добровольскому (гл. 2).

    Атрибуты длины — в метрах, углы — в градусах. Контур ``points`` упорядочен
    по возрастанию ``x_m``: от входа в дозвуковую часть к срезу.
    """
    method: str                          # 'conical' | 'profiled'
    R_throat_m: float                    # R_кр — радиус критического сечения
    R_exit_m: float                      # R_a  — радиус среза
    R_chamber_m: float                   # R_к  — радиус камеры (вход)
    area_ratio: float                    # F_a / F_кр = (R_a / R_кр)^2

    theta_in_deg: float                  # θ_вх — полуугол дозвукового конуса
    theta_max_deg: float                 # θ_m  — угол в начале св/зв части
    theta_exit_deg: float                # θ_a  — полуугол на срезе

    R1_inlet_m: float                    # R_1  — скругление на входе из камеры
    R_round_sub_m: float                 # R_скр — скругление перед горловиной
    r_round_sup_m: float                 # r_скр — скругление за горловиной

    length_subsonic_m: float             # длина дозвуковой части
    length_supersonic_m: float           # длина сверхзвуковой части
    length_total_m: float                # полная длина

    phi_dispersion: float                # φ_рас — коэффициент рассеяния

    points: List[ContourPoint] = field(default_factory=list)
    points_subsonic: List[ContourPoint] = field(default_factory=list)
    points_supersonic: List[ContourPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── удобные представления ────────────────────────────────────────────────
    @property
    def throat_index(self) -> int:
        """Индекс точки горловины (минимальный радиус) в ``points``."""
        if not self.points:
            return -1
        return min(range(len(self.points)), key=lambda i: self.points[i].r_m)

    def as_xy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (x, r) как numpy-массивы по всему контуру."""
        x = np.fromiter((p.x_m for p in self.points), dtype=float, count=len(self.points))
        r = np.fromiter((p.r_m for p in self.points), dtype=float, count=len(self.points))
        return x, r


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _radius_from_area_ratio(R_throat_m: float, area_ratio: float) -> float:
    """R_a = R_кр · sqrt(F_a/F_кр)."""
    return R_throat_m * math.sqrt(area_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# §2.2 Коэффициент рассеяния и углы потока на срезе
# ─────────────────────────────────────────────────────────────────────────────

def dispersion_loss_coeff(theta_exit_deg: float) -> float:
    """Коэффициент рассеяния потока на срезе φ_рас (§2.2).

        φ_рас = (1 + cos θ_a) / 2.

    Для конического сопла θ_a — полуугол раствора конуса; для профилированного —
    угол наклона контура на срезе. Чем меньше θ_a, тем ближе φ_рас к 1.
    """
    theta = math.radians(theta_exit_deg)
    return 0.5 * (1.0 + math.cos(theta))


def exit_angle_from_dispersion(phi: float) -> float:
    """Обратная зависимость к :func:`dispersion_loss_coeff`.

        θ_a = arccos(2 φ_рас − 1)   [градусы].
    """
    c = _clamp(2.0 * phi - 1.0, -1.0, 1.0)
    return math.degrees(math.acos(c))


def exit_angle_from_underexpansion(
    p_a: float,
    p_amb: float,
    rho_a: float,
    w_a: float,
    M_a: float,
) -> float:
    """Полуугол потока на срезе θ_a из условия безотрывного течения при
    недорасширении (ур. 2.23 и 2.24, §2.3):

        sin 2θ_a = (p_a − p_н) / (½ ρ_a w_a²) · ctg μ_a,
        sin μ_a  = 1 / M_a.

    Параметры:
        p_a   — статическое давление на срезе, Па;
        p_amb — давление окружающей среды p_н, Па;
        rho_a — плотность газа на срезе, кг/м³;
        w_a   — скорость потока на срезе, м/с;
        M_a   — число Маха на срезе (> 1).

    Возвращает θ_a в градусах (≥ 0). При M_a ≤ 1 или нулевом скоростном напоре
    возвращает 0.
    """
    if M_a <= 1.0:
        return 0.0
    q_dyn = 0.5 * rho_a * w_a * w_a
    if q_dyn <= 1e-12:
        return 0.0
    # μ_a — угол Маха, ctg μ_a = sqrt(M^2 - 1)
    cot_mu = math.sqrt(M_a * M_a - 1.0)
    rhs = ((p_a - p_amb) / q_dyn) * cot_mu
    rhs = _clamp(rhs, -1.0, 1.0)
    two_theta = math.asin(rhs)
    return max(0.0, 0.5 * math.degrees(two_theta))


# ─────────────────────────────────────────────────────────────────────────────
# §2.6 / Рис. 2.14 — семейство оптимальных контуров (γ = 1.23)
# ─────────────────────────────────────────────────────────────────────────────
#
# Таблица аппроксимирует графики Рис. 2.14: по отношению R_a/R_кр восстанавливаются
#   θ_m  — угол наклона контура в начальной точке сверхзвуковой части A_n,
#   θ_a  — угол наклона контура на срезе,
#   x̄_a = L_сверхзв / R_кр — относительная длина сверхзвуковой части.
#
# Значения калиброваны по примерам учебника (короткое оптимальное сопло
# даёт ~80 % длины конического 15° при близком φ_рас). Пользователь может
# заменить сетку через set_optimal_grid().
#
#                 R_a/R_кр   θ_m,°   θ_a,°   x̄_a
_OPTIMAL_GRID: List[Tuple[float, float, float, float]] = [
    (2.0,   22.0, 14.0,  2.6),
    (3.0,   27.0, 13.0,  4.8),
    (4.0,   30.0, 11.5,  7.1),
    (5.0,   32.0, 10.5,  9.5),
    (6.0,   33.5,  9.8, 12.0),
    (8.0,   35.0,  8.8, 17.0),
    (10.0,  36.0,  8.0, 22.5),
    (15.0,  37.5,  6.8, 36.0),
    (20.0,  38.5,  6.0, 50.0),
    (30.0,  39.5,  5.0, 80.0),
]


def set_optimal_grid(grid: Sequence[Tuple[float, float, float, float]]) -> None:
    """Заменяет встроенную аппроксимацию Рис. 2.14 пользовательской.

    ``grid`` — последовательность кортежей ``(R_a/R_кр, θ_m, θ_a, x̄_a)``,
    отсортированная по возрастанию ``R_a/R_кр``. Используйте, если есть более
    точные данные семейства оптимальных контуров (для своего γ).
    """
    global _OPTIMAL_GRID
    cleaned = sorted((tuple(map(float, row)) for row in grid), key=lambda r: r[0])
    if len(cleaned) < 2:
        raise ValueError("grid должна содержать не менее двух точек")
    _OPTIMAL_GRID = list(cleaned)  # type: ignore[assignment]


def optimal_angles_from_area_ratio(
    area_ratio: float,
    theta_exit_deg: Optional[float] = None,
    theta_max_deg: Optional[float] = None,
    length_ratio: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Восстанавливает (θ_m, θ_a, x̄_a) по отношению площадей (Рис. 2.14).

    Любой из углов/длину можно задать явно — он подменит интерполированное
    значение. Возвращает кортеж ``(theta_max_deg, theta_exit_deg, length_ratio)``.
    """
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")
    radius_ratio = math.sqrt(area_ratio)  # R_a/R_кр

    grid = _OPTIMAL_GRID
    xs = [row[0] for row in grid]

    if radius_ratio <= xs[0]:
        _, tm, ta, xa = grid[0]
    elif radius_ratio >= xs[-1]:
        _, tm, ta, xa = grid[-1]
    else:
        # линейная интерполяция по соседним узлам
        tm = ta = xa = 0.0
        for (x0, m0, a0, l0), (x1, m1, a1, l1) in zip(grid[:-1], grid[1:]):
            if x0 <= radius_ratio <= x1:
                t = (radius_ratio - x0) / (x1 - x0)
                tm = m0 + t * (m1 - m0)
                ta = a0 + t * (a1 - a0)
                xa = l0 + t * (l1 - l0)
                break

    if theta_max_deg is not None:
        tm = float(theta_max_deg)
    if theta_exit_deg is not None:
        ta = float(theta_exit_deg)
    if length_ratio is not None:
        xa = float(length_ratio)

    # физические ограничения
    ta = _clamp(ta, 1.0, 25.0)
    tm = _clamp(tm, ta + 1e-3, 50.0)
    xa = max(xa, 0.1)
    return tm, ta, xa


# ─────────────────────────────────────────────────────────────────────────────
# Построение дозвуковой части (камера → R_1 → конус θ_вх → R_скр → горловина)
# ─────────────────────────────────────────────────────────────────────────────

def _build_subsonic_contour(
    R_throat_m: float,
    R_chamber_m: float,
    theta_in_deg: float,
    R_round_sub_m: float,
    R1_inlet_m: float,
    n_points: int,
) -> Tuple[List[ContourPoint], float]:
    """Строит дозвуковую часть СПРАВА НАЛЕВО от горловины и возвращает
    точки, упорядоченные по возрастанию x (вход … горловина), а также
    суммарную длину дозвуковой части.

    Геометрия (от горловины к камере):
        1. Дуга скругления R_скр (от вертикали у горловины до угла θ_вх).
        2. Прямой конус с полу­углом θ_вх.
        3. Дуга скругления R_1 у входа из камеры (до горизонтали при r = R_к).
    """
    theta = math.radians(theta_in_deg)
    n_arc = max(8, n_points // 4)
    n_cone = max(8, n_points // 2)

    # ── В системе, где горловина в (0,0), ось x направлена «назад» (в камеру)
    # Точки накапливаем в массивах (x_back, r); затем отразим x → −x_back.
    xb: List[float] = []
    rr: List[float] = []

    # 1) Дуга R_скр (перед горловиной). Центр дуги в (0, R_кр + R_скр).
    #    Угол φ от 0 (у горловины) до θ_вх.
    for i in range(n_arc):
        phi = theta * i / (n_arc - 1)
        x_b = R_round_sub_m * math.sin(phi)
        r = R_throat_m + R_round_sub_m * (1.0 - math.cos(phi))
        xb.append(x_b)
        rr.append(r)

    x_tan = xb[-1]      # точка касания конуса со скруглением R_скр
    r_tan = rr[-1]

    # 3) Дуга R_1 у входа: центр в (x_c1, R_chamber − R_1), касательная
    #    к конусу θ_вх с одной стороны и горизонтальна при r = R_к с другой.
    #    На стыке конуса и дуги R_1 радиус:
    r_join = R_chamber_m - R1_inlet_m * (1.0 - math.cos(theta))
    if r_join <= r_tan + 1e-9:
        # Камера слишком близка к горловине — конуса нет, дугу R_1 сводим к стыку
        r_join = r_tan + 1e-6

    # 2) Прямой конус θ_вх между точкой касания R_скр (r_tan) и r_join.
    dr_cone = r_join - r_tan
    dx_cone = dr_cone / math.tan(theta) if math.tan(theta) > 1e-9 else 0.0
    x_join = x_tan + dx_cone
    for i in range(1, n_cone):
        t = i / (n_cone - 1)
        r = r_tan + t * dr_cone
        x_b = x_tan + t * dx_cone
        xb.append(x_b)
        rr.append(r)

    # 3) Дуга R_1: угол ψ от θ_вх (стык с конусом) до 0 (горизонталь у камеры).
    #    Центр дуги: x_center = x_join + R_1 · sin θ_вх; r_center = R_chamber − R_1.
    x_center = x_join + R1_inlet_m * math.sin(theta)
    for i in range(1, n_arc):
        psi = theta * (1.0 - i / (n_arc - 1))   # от θ_вх к 0
        x_b = x_center - R1_inlet_m * math.sin(psi)
        r = (R_chamber_m - R1_inlet_m) + R1_inlet_m * math.cos(psi)
        xb.append(x_b)
        rr.append(r)

    length_sub = max(xb)

    # Отражаем: вход слева (x = 0), горловина справа (x = length_sub).
    pts = [ContourPoint(x_m=length_sub - x_b, r_m=r) for x_b, r in zip(xb, rr)]
    pts.sort(key=lambda p: p.x_m)

    # Гарантируем строгую монотонность по x (чистка дубликатов/инверсий)
    cleaned: List[ContourPoint] = []
    last_x = -math.inf
    for p in pts:
        if p.x_m > last_x + 1e-12:
            cleaned.append(p)
            last_x = p.x_m
    # последняя точка должна быть горловиной (r = R_кр)
    if cleaned:
        cleaned[-1] = ContourPoint(x_m=length_sub, r_m=R_throat_m)
    return cleaned, length_sub


# ─────────────────────────────────────────────────────────────────────────────
# Сверхзвуковой конус (§2.3) и парабола (§2.6)
# ─────────────────────────────────────────────────────────────────────────────

def _build_supersonic_cone(
    x0: float,
    R_throat_m: float,
    R_exit_m: float,
    theta_exit_deg: float,
    r_round_sup_m: float,
    n_points: int,
) -> Tuple[List[ContourPoint], float]:
    """Сверхзвуковая часть конического сопла: скругление r_скр за горловиной +
    прямой конус с полу­углом θ_a. Возвращает точки и длину части.

    ``x0`` — координата горловины по оси x.
    """
    theta = math.radians(theta_exit_deg)
    n_arc = max(6, n_points // 5)
    n_cone = max(8, n_points - n_arc)

    pts: List[ContourPoint] = []

    # Скругление r_скр сразу за горловиной (центр в (x0, R_кр + r_скр)).
    for i in range(n_arc):
        phi = theta * i / (n_arc - 1)
        x = x0 + r_round_sup_m * math.sin(phi)
        r = R_throat_m + r_round_sup_m * (1.0 - math.cos(phi))
        pts.append(ContourPoint(x_m=x, r_m=r))

    x_tan = pts[-1].x_m
    r_tan = pts[-1].r_m

    # Прямой конус до радиуса среза.
    dr = R_exit_m - r_tan
    dx = dr / math.tan(theta) if math.tan(theta) > 1e-9 else 0.0
    for i in range(1, n_cone):
        t = i / (n_cone - 1)
        x = x_tan + t * dx
        r = r_tan + t * dr
        pts.append(ContourPoint(x_m=x, r_m=r))

    length_sup = pts[-1].x_m - x0
    return pts, length_sup


def _parabola_envelope_AnC(
    x_an: float,
    r_an: float,
    theta_m_deg: float,
    x_c: float,
    r_c: float,
    theta_a_deg: float,
    n_points: int = 120,
) -> List[ContourPoint]:
    """Параболический контур сверхзвуковой части (§2.6) как квадратичная кривая
    Безье через точку пересечения касательных ``f``.

    A_n — начальная точка (стык со скруглением r_скр), касательная под θ_m.
    C   — срез, касательная под θ_a. Точка управления f — пересечение
    прямых, проведённых из A_n под θ_m и из C под θ_a.
    """
    tm = math.tan(math.radians(theta_m_deg))
    ta = math.tan(math.radians(theta_a_deg))

    # Прямая 1: r = r_an + tm (x − x_an)
    # Прямая 2: r = r_c  + ta (x − x_c)
    denom = (tm - ta)
    if abs(denom) < 1e-9:
        # касательные почти параллельны — линейная интерполяция
        f_x = 0.5 * (x_an + x_c)
        f_r = 0.5 * (r_an + r_c)
    else:
        f_x = (r_c - r_an + tm * x_an - ta * x_c) / denom
        f_r = r_an + tm * (f_x - x_an)

    pts: List[ContourPoint] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        mt = 1.0 - t
        # квадратичная Безье: B = mt^2 A + 2 mt t f + t^2 C
        x = mt * mt * x_an + 2 * mt * t * f_x + t * t * x_c
        r = mt * mt * r_an + 2 * mt * t * f_r + t * t * r_c
        pts.append(ContourPoint(x_m=x, r_m=r))
    return pts, (f_x, f_r)


def _build_supersonic_parabola(
    x0: float,
    R_throat_m: float,
    R_exit_m: float,
    theta_max_deg: float,
    theta_exit_deg: float,
    length_supersonic_m: float,
    r_round_sup_m: float,
    n_points: int,
) -> Tuple[List[ContourPoint], float, Dict[str, Any]]:
    """Профилированная сверхзвуковая часть (§2.6): скругление r_скр + парабола
    A_n → C. ``x0`` — координата горловины."""
    theta_m = math.radians(theta_max_deg)
    n_arc = max(6, n_points // 6)

    pts: List[ContourPoint] = []
    # Скругление r_скр за горловиной до угла θ_m (начало параболы A_n).
    for i in range(n_arc):
        phi = theta_m * i / (n_arc - 1)
        x = x0 + r_round_sup_m * math.sin(phi)
        r = R_throat_m + r_round_sup_m * (1.0 - math.cos(phi))
        pts.append(ContourPoint(x_m=x, r_m=r))

    x_an = pts[-1].x_m
    r_an = pts[-1].r_m

    # Конечная точка C: срез. Длина параболической части задаётся
    # length_supersonic_m (полная сверхзвуковая длина), отсчитываемая от x0.
    x_c = x0 + length_supersonic_m
    if x_c <= x_an:
        x_c = x_an + max(length_supersonic_m, R_throat_m)
    r_c = R_exit_m

    n_par = max(16, n_points - n_arc)
    par_pts, f_pt = _parabola_envelope_AnC(
        x_an, r_an, theta_max_deg, x_c, r_c, theta_exit_deg, n_points=n_par
    )
    pts.extend(par_pts[1:])

    length_sup = pts[-1].x_m - x0
    meta = {
        "A_n": (x_an, r_an),
        "C": (x_c, r_c),
        "f_tangent_intersection": f_pt,
        "construction": "arc(r_скр) + quadratic_Bezier(A_n→f→C)",
    }
    return pts, length_sup, meta


# ─────────────────────────────────────────────────────────────────────────────
# Публичные построители
# ─────────────────────────────────────────────────────────────────────────────

def _join_contours(
    subsonic: List[ContourPoint],
    supersonic: List[ContourPoint],
) -> List[ContourPoint]:
    """Сшивает дозвуковую и сверхзвуковую части в один монотонный контур.

    Сверхзвуковая часть строится в собственной системе (горловина в x = 0),
    её нужно сдвинуть к концу дозвуковой (горловина в x = length_subsonic).
    """
    if not subsonic:
        return list(supersonic)
    x_throat = subsonic[-1].x_m
    shifted = [ContourPoint(x_m=x_throat + p.x_m, r_m=p.r_m) for p in supersonic]
    # стыкуем без дублирования точки горловины
    return subsonic + shifted[1:]


def build_conical_nozzle(
    R_throat_m: float,
    area_ratio: float,
    *,
    R_chamber_m: Optional[float] = None,
    theta_exit_deg: float = 15.0,
    theta_in_deg: float = 30.0,
    R_round_sub_factor: float = 1.0,
    R1_inlet_factor: float = 3.0,
    r_round_sup_factor: float = 0.45,
    n_points: int = 200,
) -> NozzleGeometry:
    """Коническое сопло по §2.3 Добровольского.

    Параметры (все настраиваемые):
        R_throat_m         — радиус критического сечения R_кр, м;
        area_ratio         — F_a/F_кр = (R_a/R_кр)²;
        R_chamber_m        — радиус камеры R_к (по умолч. 2.5·R_кр);
        theta_exit_deg     — полу­угол раствора сверхзв. конуса θ_a
                             (2θ_a = 25…30° ⇒ θ_a = 12.5…15°);
        theta_in_deg       — полу­угол дозвукового конуса θ_вх
                             (2θ_вх = 45…80° ⇒ θ_вх = 22.5…40°);
        R_round_sub_factor — R_скр = factor·D_кр перед горловиной
                             (рекоменд. 0.65…1.5 от D_кр; здесь от D_кр=2R_кр);
        R1_inlet_factor    — R_1 = factor·D_кр на входе из камеры (2…4·D_кр);
        r_round_sup_factor — r_скр = factor·R_кр за горловиной (≈0.45·R_кр);
        n_points           — число точек контура.
    """
    if R_throat_m <= 0:
        raise ValueError("R_throat_m должен быть > 0")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")

    D_throat = 2.0 * R_throat_m
    if R_chamber_m is None:
        R_chamber_m = 2.5 * R_throat_m

    theta_exit_deg = _clamp(theta_exit_deg, 3.0, 25.0)
    theta_in_deg = _clamp(theta_in_deg, 10.0, 45.0)

    R_round_sub_m = R_round_sub_factor * D_throat        # R_скр от D_кр
    R1_inlet_m = R1_inlet_factor * D_throat              # R_1   от D_кр
    r_round_sup_m = r_round_sup_factor * R_throat_m      # r_скр от R_кр

    R_exit_m = _radius_from_area_ratio(R_throat_m, area_ratio)

    sub_pts, len_sub = _build_subsonic_contour(
        R_throat_m=R_throat_m,
        R_chamber_m=R_chamber_m,
        theta_in_deg=theta_in_deg,
        R_round_sub_m=R_round_sub_m,
        R1_inlet_m=R1_inlet_m,
        n_points=n_points,
    )
    sup_pts, len_sup = _build_supersonic_cone(
        x0=0.0,
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        theta_exit_deg=theta_exit_deg,
        r_round_sup_m=r_round_sup_m,
        n_points=n_points,
    )

    full = _join_contours(sub_pts, sup_pts)
    phi = dispersion_loss_coeff(theta_exit_deg)

    return NozzleGeometry(
        method="conical",
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        R_chamber_m=R_chamber_m,
        area_ratio=area_ratio,
        theta_in_deg=theta_in_deg,
        theta_max_deg=theta_exit_deg,    # у конуса θ_m == θ_a
        theta_exit_deg=theta_exit_deg,
        R1_inlet_m=R1_inlet_m,
        R_round_sub_m=R_round_sub_m,
        r_round_sup_m=r_round_sup_m,
        length_subsonic_m=len_sub,
        length_supersonic_m=len_sup,
        length_total_m=len_sub + len_sup,
        phi_dispersion=phi,
        points=full,
        points_subsonic=sub_pts,
        points_supersonic=[ContourPoint(len_sub + p.x_m, p.r_m) for p in sup_pts],
        metadata={
            "reference": "Добровольский, §2.3 (коническое сопло)",
            "R_round_sub_factor_x_Dkr": R_round_sub_factor,
            "R1_inlet_factor_x_Dkr": R1_inlet_factor,
            "r_round_sup_factor_x_Rkr": r_round_sup_factor,
            "two_theta_exit_deg": 2.0 * theta_exit_deg,
            "two_theta_in_deg": 2.0 * theta_in_deg,
        },
    )


def build_profiled_nozzle(
    R_throat_m: float,
    area_ratio: float,
    *,
    R_chamber_m: Optional[float] = None,
    theta_exit_deg: Optional[float] = None,
    theta_max_deg: Optional[float] = None,
    length_ratio: Optional[float] = None,
    theta_in_deg: float = 30.0,
    R_round_sub_factor: float = 1.5,
    r_round_sup_factor: float = 0.45,
    R1_inlet_factor: float = 3.0,
    n_points: int = 240,
) -> NozzleGeometry:
    """Профилированное (укороченное оптимальное) сопло по §2.6 Добровольского.

    Дозвуковая часть строится как у конического сопла. Сверхзвуковая часть —
    парабола A_n → C: скругление r_скр (≈0.45·R_кр), затем парабола с углом θ_m
    в начале и θ_a на срезе. По умолчанию углы и длина берутся из семейства
    оптимальных контуров (Рис. 2.14) по area_ratio, но могут быть заданы явно.

    Параметры:
        theta_max_deg  — угол контура в начале св/зв части θ_m (по умолч. из Рис.2.14);
        theta_exit_deg — угол контура на срезе θ_a (по умолч. из Рис.2.14);
        length_ratio   — относительная длина x̄_a = L/R_кр (по умолч. из Рис.2.14);
        R_round_sub_factor — R_скр = factor·R_кр перед горловиной (≈1.5·R_кр);
        прочие параметры — как в build_conical_nozzle.
    """
    if R_throat_m <= 0:
        raise ValueError("R_throat_m должен быть > 0")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")

    if R_chamber_m is None:
        R_chamber_m = 2.5 * R_throat_m
    theta_in_deg = _clamp(theta_in_deg, 10.0, 45.0)

    theta_m, theta_a, x_bar_a = optimal_angles_from_area_ratio(
        area_ratio,
        theta_exit_deg=theta_exit_deg,
        theta_max_deg=theta_max_deg,
        length_ratio=length_ratio,
    )

    D_throat = 2.0 * R_throat_m
    R_round_sub_m = R_round_sub_factor * R_throat_m       # R_скр = 1.5·R_кр (§2.6)
    r_round_sup_m = r_round_sup_factor * R_throat_m       # r_скр = 0.45·R_кр
    R1_inlet_m = R1_inlet_factor * D_throat               # R_1 от D_кр

    R_exit_m = _radius_from_area_ratio(R_throat_m, area_ratio)
    length_supersonic_m = x_bar_a * R_throat_m            # L = x̄_a · R_кр

    sub_pts, len_sub = _build_subsonic_contour(
        R_throat_m=R_throat_m,
        R_chamber_m=R_chamber_m,
        theta_in_deg=theta_in_deg,
        R_round_sub_m=R_round_sub_m,
        R1_inlet_m=R1_inlet_m,
        n_points=n_points,
    )
    sup_pts, len_sup, sup_meta = _build_supersonic_parabola(
        x0=0.0,
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        theta_max_deg=theta_m,
        theta_exit_deg=theta_a,
        length_supersonic_m=length_supersonic_m,
        r_round_sup_m=r_round_sup_m,
        n_points=n_points,
    )

    full = _join_contours(sub_pts, sup_pts)
    phi = dispersion_loss_coeff(theta_a)

    meta = {
        "reference": "Добровольский, §2.6 (профилированное оптимальное сопло)",
        "Ra_over_Rkr": math.sqrt(area_ratio),
        "x_bar_a_L_over_Rkr": x_bar_a,
        "R_round_sub_factor_x_Rkr": R_round_sub_factor,
        "r_round_sup_factor_x_Rkr": r_round_sup_factor,
        "R1_inlet_factor_x_Dkr": R1_inlet_factor,
        "gamma_assumed": 1.23,
    }
    meta.update(sup_meta)

    return NozzleGeometry(
        method="profiled",
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        R_chamber_m=R_chamber_m,
        area_ratio=area_ratio,
        theta_in_deg=theta_in_deg,
        theta_max_deg=theta_m,
        theta_exit_deg=theta_a,
        R1_inlet_m=R1_inlet_m,
        R_round_sub_m=R_round_sub_m,
        r_round_sup_m=r_round_sup_m,
        length_subsonic_m=len_sub,
        length_supersonic_m=len_sup,
        length_total_m=len_sub + len_sup,
        phi_dispersion=phi,
        points=full,
        points_subsonic=sub_pts,
        points_supersonic=[ContourPoint(len_sub + p.x_m, p.r_m) for p in sup_pts],
        metadata=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RPA-стиль: параболическая аппроксимация bell-контура (Rao)
# ─────────────────────────────────────────────────────────────────────────────

def rao_reference_length_15deg(R_throat_m: float, area_ratio: float) -> float:
    """Длина эталонного конического сопла с полу­углом 15° (Le15) — база для
    относительной длины Le/Le15 в RPA.

        Le15 = R_кр·(sqrt(ε) − 1) / tan(15°),
    где ε = Ae/At = (R_a/R_кр)².
    """
    return R_throat_m * (math.sqrt(area_ratio) - 1.0) / math.tan(math.radians(15.0))


# Аппроксимация графиков Rao (углы оптимального bell-контура) по степени
# расширения ε для относительной длины 80 % (как в RPA по умолчанию).
#            ε      Tn,°   Te,°
_RAO_BELL_GRID: List[Tuple[float, float, float]] = [
    (4.0,   20.5, 14.5),
    (5.0,   21.5, 13.5),
    (8.0,   23.5, 11.8),
    (10.0,  24.5, 11.0),
    (15.0,  26.5,  9.8),
    (20.0,  28.0,  9.0),
    (25.0,  29.0,  8.5),
    (30.0,  30.0,  8.0),
    (40.0,  31.5,  7.2),
    (50.0,  32.5,  6.5),
    (100.0, 35.0,  5.0),
]


def estimate_bell_angles(area_ratio: float, length_fraction: float = 80.0) -> Tuple[float, float]:
    """Оценка углов параболы (Tn, Te) по степени расширения для bell-контура
    (как делает RPA, когда углы «не заданы»). length_fraction — Le/Le15, %.

    Возвращает (Tn, Te) в градусах. Зависимость от длины — мягкая поправка:
    более короткое сопло (меньше %) → больше Tn и больше Te.
    """
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")
    grid = _RAO_BELL_GRID
    xs = [g[0] for g in grid]
    if area_ratio <= xs[0]:
        tn, te = grid[0][1], grid[0][2]
    elif area_ratio >= xs[-1]:
        tn, te = grid[-1][1], grid[-1][2]
    else:
        tn = te = 0.0
        for (e0, n0, x0), (e1, n1, x1) in zip(grid[:-1], grid[1:]):
            if e0 <= area_ratio <= e1:
                t = (area_ratio - e0) / (e1 - e0)
                tn = n0 + t * (n1 - n0)
                te = x0 + t * (x1 - x0)
                break
    # поправка на относительную длину (80 % — опорная)
    k = (80.0 - float(length_fraction)) / 80.0
    tn += 8.0 * k
    te += 6.0 * k
    tn = _clamp(tn, 5.0, 55.0)
    te = _clamp(te, 1.0, 24.0)
    if te >= tn:
        te = tn - 0.5
    return tn, te


def build_rpa_parabolic_nozzle(
    R_throat_m: float,
    area_ratio: float,
    *,
    R_chamber_m: Optional[float] = None,
    contraction_angle_deg: float = 30.0,
    R1_over_Rt: float = 1.5,
    Rn_over_Rt: float = 0.382,
    R2_over_R2max: float = 0.5,
    theta_n_deg: Optional[float] = None,
    theta_e_deg: Optional[float] = None,
    length_fraction_pct: Optional[float] = 80.0,
    n_points: int = 260,
) -> NozzleGeometry:
    """Параболическая аппроксимация bell-контура в стиле RPA.

    Параметры (нотация RPA):
        R_throat_m            — радиус горловины Rt, м;
        area_ratio            — Ae/At = ε;
        R_chamber_m           — радиус камеры (по умолч. 2.5·Rt);
        contraction_angle_deg — угол сжатия конфузора b (град);
        R1_over_Rt            — R1/Rt (скругление сходящейся стороны горловины);
        Rn_over_Rt            — Rn/Rt (скругление расходящейся стороны, RPA=0.382);
        R2_over_R2max         — относительный радиус входа в конфузор (0..1);
        theta_n_deg           — начальный угол параболы Tn (если None — оценка по ε);
        theta_e_deg           — конечный угол параболы Te (если None — оценка по ε);
        length_fraction_pct   — Le/Le15, % (по умолч. 80 %; RPA-default).
    """
    if R_throat_m <= 0:
        raise ValueError("R_throat_m должен быть > 0")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio должен быть > 1")
    R2_over_R2max = _clamp(R2_over_R2max, 0.0, 1.0)
    if length_fraction_pct is None:
        length_fraction_pct = 80.0
    length_fraction_pct = _clamp(float(length_fraction_pct), 50.0, 120.0)

    if R_chamber_m is None:
        R_chamber_m = 2.5 * R_throat_m
    contraction_angle_deg = _clamp(contraction_angle_deg, 10.0, 60.0)

    # Углы параболы: если не заданы — оценка по ε и длине (как в RPA).
    tn_est, te_est = estimate_bell_angles(area_ratio, length_fraction_pct)
    theta_n = float(theta_n_deg) if (theta_n_deg and theta_n_deg > 0) else tn_est
    theta_e = float(theta_e_deg) if (theta_e_deg and theta_e_deg > 0) else te_est
    theta_e = _clamp(theta_e, 1.0, 24.0)
    theta_n = _clamp(theta_n, theta_e + 0.5, 55.0)

    D_throat = 2.0 * R_throat_m
    R_round_sub_m = R1_over_Rt * R_throat_m       # R1 — скругление сходящейся стороны
    r_round_sup_m = Rn_over_Rt * R_throat_m       # Rn — скругление расходящейся стороны
    # R2 = доля от R2max. За R2max принимаем «характерный» радиус входа из камеры
    # порядка диаметра камеры; масштабируем к D_кр для совместимости с контуром.
    R2max_m = max(1.5 * D_throat, 1.5 * R_chamber_m)
    R1_inlet_m = max(R2_over_R2max * R2max_m, 0.3 * D_throat)

    R_exit_m = _radius_from_area_ratio(R_throat_m, area_ratio)
    # Длина сверхзвуковой части = Le/Le15 · Le15.
    length_supersonic_m = (length_fraction_pct / 100.0) * \
        rao_reference_length_15deg(R_throat_m, area_ratio)

    sub_pts, len_sub = _build_subsonic_contour(
        R_throat_m=R_throat_m,
        R_chamber_m=R_chamber_m,
        theta_in_deg=contraction_angle_deg,
        R_round_sub_m=R_round_sub_m,
        R1_inlet_m=R1_inlet_m,
        n_points=n_points,
    )
    sup_pts, len_sup, sup_meta = _build_supersonic_parabola(
        x0=0.0,
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        theta_max_deg=theta_n,
        theta_exit_deg=theta_e,
        length_supersonic_m=length_supersonic_m,
        r_round_sup_m=r_round_sup_m,
        n_points=n_points,
    )

    full = _join_contours(sub_pts, sup_pts)
    phi = dispersion_loss_coeff(theta_e)

    meta = {
        "reference": "RPA-style parabolic bell (Rao approximation)",
        "contour_type": "parabolic_bell",
        "R1_over_Rt": R1_over_Rt,
        "Rn_over_Rt": Rn_over_Rt,
        "R2_over_R2max": R2_over_R2max,
        "contraction_angle_deg": contraction_angle_deg,
        "Le_over_Le15_pct": length_fraction_pct,
        "Le15_m": rao_reference_length_15deg(R_throat_m, area_ratio),
        "theta_n_deg": theta_n,
        "theta_e_deg": theta_e,
        "Ra_over_Rkr": math.sqrt(area_ratio),
    }
    meta.update(sup_meta)

    return NozzleGeometry(
        method="rpa_parabolic",
        R_throat_m=R_throat_m,
        R_exit_m=R_exit_m,
        R_chamber_m=R_chamber_m,
        area_ratio=area_ratio,
        theta_in_deg=contraction_angle_deg,
        theta_max_deg=theta_n,
        theta_exit_deg=theta_e,
        R1_inlet_m=R1_inlet_m,
        R_round_sub_m=R_round_sub_m,
        r_round_sup_m=r_round_sup_m,
        length_subsonic_m=len_sub,
        length_supersonic_m=len_sup,
        length_total_m=len_sub + len_sup,
        phi_dispersion=phi,
        points=full,
        points_subsonic=sub_pts,
        points_supersonic=[ContourPoint(len_sub + p.x_m, p.r_m) for p in sup_pts],
        metadata=meta,
    )


def build_nozzle_geometry(
    R_throat_m: float,
    area_ratio: float,
    method: str = "profiled",
    **kwargs,
) -> NozzleGeometry:
    """Унифицированный диспетчер построения геометрии сопла.

    method:
        'conical'       — коническое сопло (§2.3);
        'profiled'      — профилированное оптимальное сопло (§2.6);
        'rpa_parabolic' — параболическая аппроксимация bell-контура (RPA-стиль).
    Лишние именованные аргументы фильтруются под выбранный метод.
    """
    key = method.strip().lower()
    if key in ("conical", "cone", "2.3"):
        # коническое не принимает theta_max_deg / length_ratio
        for drop in ("theta_max_deg", "length_ratio"):
            kwargs.pop(drop, None)
        return build_conical_nozzle(R_throat_m, area_ratio, **kwargs)
    if key in ("profiled", "optimal", "2.6", "bell"):
        return build_profiled_nozzle(R_throat_m, area_ratio, **kwargs)
    if key in ("rpa_parabolic", "rpa", "parabolic", "parabolic_bell"):
        return build_rpa_parabolic_nozzle(R_throat_m, area_ratio, **kwargs)
    raise ValueError(
        f"Неизвестный method='{method}'. Допустимо: 'conical', 'profiled', "
        "'rpa_parabolic'."
    )


def build_geometry_from_performance(
    perf,
    R_throat_m: float,
    method: str = "profiled",
    *,
    p_ambient_Pa: Optional[float] = None,
    theta_exit_deg: Optional[float] = None,
    theta_max_deg: Optional[float] = None,
    theta_in_deg: float = 30.0,
    use_eq_2_23: bool = True,
    **kwargs,
) -> NozzleGeometry:
    """Строит геометрию сопла по результату ``solve_rocket_nozzle`` (RocketPerformance).

    area_ratio берётся из сечения 'Nozzle exit' (Ae/At). Если θ_a не задан явно
    и ``use_eq_2_23`` истинно — он вычисляется по условию недорасширения
    (ур. 2.23) из давления окружающей среды ``p_ambient_Pa`` и параметров среза.
    """
    # найти сечение среза
    st_exit = None
    for s in getattr(perf, "stations", []):
        if getattr(s, "label", "") == "Nozzle exit":
            st_exit = s
    if st_exit is None and getattr(perf, "stations", None):
        st_exit = perf.stations[-1]
    if st_exit is None:
        raise ValueError("В perf нет сечений для построения геометрии")

    area_ratio = float(st_exit.Ae_At)
    if not (math.isfinite(area_ratio) and area_ratio > 1.0):
        raise ValueError("Некорректное Ae/At на срезе")

    # вычислить θ_a из ур. 2.23, если не задан
    if theta_exit_deg is None and use_eq_2_23 and p_ambient_Pa is not None:
        theta_calc = exit_angle_from_underexpansion(
            p_a=float(st_exit.P_Pa),
            p_amb=float(p_ambient_Pa),
            rho_a=float(st_exit.rho_kg_per_m3),
            w_a=float(st_exit.V_m_per_s),
            M_a=float(st_exit.M),
        )
        if theta_calc > 0.0:
            theta_exit_deg = _clamp(theta_calc, 5.0, 20.0)

    return build_nozzle_geometry(
        R_throat_m=R_throat_m,
        area_ratio=area_ratio,
        method=method,
        theta_exit_deg=theta_exit_deg,
        theta_max_deg=theta_max_deg,
        theta_in_deg=theta_in_deg,
        **kwargs,
    )


__all__ = [
    "ContourPoint",
    "NozzleGeometry",
    "dispersion_loss_coeff",
    "exit_angle_from_dispersion",
    "exit_angle_from_underexpansion",
    "set_optimal_grid",
    "optimal_angles_from_area_ratio",
    "build_conical_nozzle",
    "build_profiled_nozzle",
    "build_rpa_parabolic_nozzle",
    "rao_reference_length_15deg",
    "estimate_bell_angles",
    "build_nozzle_geometry",
    "build_geometry_from_performance",
]
