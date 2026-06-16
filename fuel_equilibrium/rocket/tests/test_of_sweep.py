"""Тесты развёртки характеристик по соотношению компонентов O/F.

Делятся на две группы:

* «Чистая» математика поиска оптимума (параболическое уточнение) — быстрая,
  без обращения к термодинамическому решателю.
* Интеграционный тест на реальном топливе H2/O2 — медленнее, проверяет, что
  развёртка собирает кривые и находит физически разумный оптимум O/F.
"""
import math

import pytest

from fuel_equilibrium.rocket.of_sweep import (
    OFSweepPoint,
    OFSweepResult,
    sweep_of_ratio,
    _parabolic_vertex,
    _parabolic_value,
    _refine_optimum,
)


# ─────────────────────────────────────────────────────────────────────────────
# Чистая математика: параболический оптимум
# ─────────────────────────────────────────────────────────────────────────────

def test_parabolic_vertex_exact_max():
    """y = -(x-5)^2 + 100 — вершина ровно в x=5."""
    f = lambda x: -(x - 5.0) ** 2 + 100.0
    xs = [4.0, 5.0, 6.0]
    ys = [f(x) for x in xs]
    x_star = _parabolic_vertex(*xs, *ys)
    assert x_star is not None
    assert x_star == pytest.approx(5.0, abs=1e-9)


def test_parabolic_vertex_offset():
    """Вершина между узлами (x=5.37) восстанавливается точно."""
    f = lambda x: -3.0 * (x - 5.37) ** 2 + 42.0
    xs = [5.0, 5.5, 6.0]
    ys = [f(x) for x in xs]
    x_star = _parabolic_vertex(*xs, *ys)
    assert x_star == pytest.approx(5.37, abs=1e-6)


def test_parabolic_vertex_returns_none_for_minimum():
    """Для ветвей вверх (минимум) функция не должна давать «оптимум-максимум»."""
    f = lambda x: (x - 5.0) ** 2  # минимум, a > 0
    xs = [4.0, 5.0, 6.0]
    ys = [f(x) for x in xs]
    assert _parabolic_vertex(*xs, *ys) is None


def test_parabolic_vertex_outside_interval():
    """Если вершина вне [x0, x2], возвращается None."""
    # вершина при x=10, а интервал [4,6]
    f = lambda x: -(x - 10.0) ** 2
    xs = [4.0, 5.0, 6.0]
    ys = [f(x) for x in xs]
    assert _parabolic_vertex(*xs, *ys) is None


def test_parabolic_value_reproduces_points():
    """_parabolic_value в узлах совпадает с исходными значениями."""
    xs = [3.0, 4.5, 7.0]
    ys = [2.0, 9.0, -1.0]
    for x, y in zip(xs, ys):
        assert _parabolic_value(*xs, *ys, x) == pytest.approx(y, abs=1e-9)


def test_refine_optimum_node_is_max():
    """Если максимум — крайний узел, уточнение не выходит за сетку."""
    ofs = [2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 8.0, 6.0, 4.0]  # монотонно убывает — макс. на левом крае
    of_opt, y_opt, idx = _refine_optimum(ofs, ys)
    assert idx == 0
    assert of_opt == pytest.approx(2.0)
    assert y_opt == pytest.approx(10.0)


def test_refine_optimum_interior_parabolic():
    """Гладкий максимум внутри сетки уточняется параболой."""
    f = lambda x: -2.0 * (x - 4.3) ** 2 + 500.0
    ofs = [3.0, 4.0, 5.0, 6.0]
    ys = [f(x) for x in ofs]
    of_opt, y_opt, idx = _refine_optimum(ofs, ys)
    assert ofs[idx] == pytest.approx(4.0)  # ближайший узел
    assert of_opt == pytest.approx(4.3, abs=0.02)
    assert y_opt == pytest.approx(500.0, abs=0.5)


def test_refine_optimum_empty():
    of_opt, y_opt, idx = _refine_optimum([], [])
    assert math.isnan(of_opt)
    assert idx == -1


# ─────────────────────────────────────────────────────────────────────────────
# Валидация входных параметров
# ─────────────────────────────────────────────────────────────────────────────

def test_sweep_rejects_bad_optimize_for():
    with pytest.raises(ValueError):
        sweep_of_ratio(
            "O2", "H2", 1e6, 1e5, species_db={},
            of_min=1.0, of_max=2.0, optimize_for="bogus",
        )


def test_sweep_rejects_nonpositive_of():
    with pytest.raises(ValueError):
        sweep_of_ratio(
            "O2", "H2", 1e6, 1e5, species_db={},
            of_min=-1.0, of_max=2.0,
        )


def test_sweep_rejects_empty_of_values():
    with pytest.raises(ValueError):
        sweep_of_ratio(
            "O2", "H2", 1e6, 1e5, species_db={},
            of_min=1.0, of_max=2.0, of_values=[-1.0, 0.0],
        )


def test_sweep_rejects_bad_fuel_mass():
    with pytest.raises(ValueError):
        sweep_of_ratio(
            "O2", "H2", 1e6, 1e5, species_db={},
            of_min=1.0, of_max=2.0, fuel_mass_kg=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OFSweepPoint.ok
# ─────────────────────────────────────────────────────────────────────────────

def test_point_ok_flag():
    good = OFSweepPoint(of=4.0, alpha=0.5, phi=2.0, T_chamber_K=3000.0,
                        Cstar_m_per_s=2400.0, Isp_s=400.0, Isp_vac_s=420.0, CF=1.5)
    assert good.ok is True

    bad = OFSweepPoint(of=4.0, alpha=float("nan"), phi=float("nan"),
                       T_chamber_K=float("nan"), Cstar_m_per_s=float("nan"),
                       Isp_s=float("nan"), Isp_vac_s=float("nan"), CF=float("nan"),
                       error="boom")
    assert bad.ok is False

    nan_isp = OFSweepPoint(of=4.0, alpha=0.5, phi=2.0, T_chamber_K=3000.0,
                           Cstar_m_per_s=2400.0, Isp_s=float("nan"),
                           Isp_vac_s=420.0, CF=1.5)
    assert nan_isp.ok is False


# ─────────────────────────────────────────────────────────────────────────────
# Интеграционный тест на реальном топливе H2/O2
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def thermo_db():
    from fuel_equilibrium.core import parse_thermo_file, find_thermo_db
    return parse_thermo_file(find_thermo_db())


def test_sweep_h2_o2_finds_physical_optimum(thermo_db):
    """Развёртка H2/O2 при Pc=10 МПа: оптимум Isp по O/F в районе 4–5."""
    sweep = sweep_of_ratio(
        oxidizer_name="O2(L)", fuel_name="H2(L)",
        P_chamber=10e6, P_exit=0.1013e6,
        species_db=thermo_db, of_min=3.0, of_max=8.0, n_points=6,
    )
    # все точки посчитались
    assert len(sweep.points) == 6
    assert len(sweep.ok_points) == 6

    # оптимальное O/F по импульсу на срезе — в физичном диапазоне 3.5..5.5
    assert 3.5 < sweep.best_of < 5.5, sweep.best_of
    # вакуумный оптимум смещён в сторону больших O/F (богаче окислителем)
    assert sweep.best_of_vac >= sweep.best_of - 0.5

    # Isp в разумных пределах для H2/O2 (порядка 380..410 с на срезе)
    assert 380.0 < sweep.best_Isp_s < 410.0, sweep.best_Isp_s
    # вакуумный Isp выше, чем на срезе
    assert sweep.best_Isp_vac_s > sweep.best_Isp_s

    # температура в камере монотонно растёт с O/F (богаче окислителем — горячее)
    temps = [p.T_chamber_K for p in sweep.points]
    assert temps[0] < temps[-1]


def test_sweep_explicit_of_values(thermo_db):
    """Явный список of_values принимается и сортируется."""
    sweep = sweep_of_ratio(
        oxidizer_name="O2(L)", fuel_name="H2(L)",
        P_chamber=5e6, P_exit=0.1013e6,
        species_db=thermo_db, of_min=0, of_max=0,
        of_values=[6.0, 4.0, 5.0],
    )
    assert [p.of for p in sweep.points] == [4.0, 5.0, 6.0]
    assert sweep.best_point_index >= 0


def test_sweep_optimize_for_vac_changes_best_of(thermo_db):
    """При optimize_for='Isp_vac' поле best_of берётся по вакуумному импульсу."""
    sweep = sweep_of_ratio(
        oxidizer_name="O2(L)", fuel_name="H2(L)",
        P_chamber=10e6, P_exit=0.1013e6,
        species_db=thermo_db, of_min=3.0, of_max=8.0, n_points=6,
        optimize_for="Isp_vac",
    )
    assert sweep.best_of == pytest.approx(sweep.best_of_vac)
