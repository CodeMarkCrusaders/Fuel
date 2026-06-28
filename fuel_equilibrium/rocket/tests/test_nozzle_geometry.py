"""
Тесты построения геометрии сопла по методике Добровольского (гл. 2).

Проверяются:
    * базовые формулы (φ_рас, θ_a из ур. 2.23/2.24, инверсия φ);
    * восстановление углов оптимального контура (Рис. 2.14);
    * инварианты контура (монотонность по x, горловина — минимум радиуса,
      радиус среза = R_кр·sqrt(F_a/F_кр));
    * диспетчер и обработка некорректного ввода.
"""

import math

import numpy as np
import pytest

from fuel_equilibrium.rocket.nozzle_geometry import (
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
    build_nozzle_geometry,
)


# ── формулы §2.2 ──────────────────────────────────────────────────────────────

def test_dispersion_loss_coeff_known_values():
    # θ_a = 0 → φ = 1
    assert dispersion_loss_coeff(0.0) == pytest.approx(1.0)
    # θ_a = 15° → φ = (1 + cos15)/2
    assert dispersion_loss_coeff(15.0) == pytest.approx(0.5 * (1 + math.cos(math.radians(15))))
    # монотонно убывает с ростом угла
    assert dispersion_loss_coeff(10.0) > dispersion_loss_coeff(20.0)


def test_dispersion_inverse_roundtrip():
    for theta in (5.0, 12.0, 15.0, 25.0):
        phi = dispersion_loss_coeff(theta)
        assert exit_angle_from_dispersion(phi) == pytest.approx(theta, abs=1e-6)


def test_exit_angle_underexpansion_eq223():
    # При p_a == p_amb (расчётный режим) угол = 0
    assert exit_angle_from_underexpansion(1e5, 1e5, 0.3, 2500.0, 3.0) == pytest.approx(0.0)
    # При недорасширении (p_a > p_amb) угол > 0
    theta = exit_angle_from_underexpansion(1.5e5, 1.0e5, 0.3, 2500.0, 3.0)
    assert theta > 0.0
    # Дозвук → 0
    assert exit_angle_from_underexpansion(1.5e5, 1.0e5, 0.3, 2500.0, 0.9) == 0.0


# ── Рис. 2.14 ─────────────────────────────────────────────────────────────────

def test_optimal_angles_monotonic_trends():
    tm10, ta10, xa10 = optimal_angles_from_area_ratio(10.0)
    tm25, ta25, xa25 = optimal_angles_from_area_ratio(25.0)
    # с ростом степени расширения θ_m растёт, θ_a падает, длина растёт
    assert tm25 > tm10
    assert ta25 < ta10
    assert xa25 > xa10
    # углы физически осмысленны
    assert 1.0 <= ta10 <= 25.0
    assert ta10 < tm10 < 50.0


def test_optimal_angles_override():
    tm, ta, xa = optimal_angles_from_area_ratio(
        16.0, theta_exit_deg=8.0, theta_max_deg=33.0, length_ratio=20.0
    )
    assert ta == pytest.approx(8.0)
    assert tm == pytest.approx(33.0)
    assert xa == pytest.approx(20.0)


def test_set_optimal_grid_override_and_restore():
    import fuel_equilibrium.rocket.nozzle_geometry as ng
    original = list(ng._OPTIMAL_GRID)
    try:
        set_optimal_grid([
            (2.0, 20.0, 15.0, 2.0),
            (10.0, 40.0, 5.0, 25.0),
        ])
        tm, ta, xa = optimal_angles_from_area_ratio((6.0) ** 2)  # Ra/Rkr=6
        assert 20.0 <= tm <= 40.0
    finally:
        ng._OPTIMAL_GRID = original


# ── инварианты контура ────────────────────────────────────────────────────────

def _check_invariants(geom: NozzleGeometry, area_ratio: float, R_throat: float):
    x, r = geom.as_xy_arrays()
    assert len(x) == len(r) > 10
    # x строго возрастает
    assert np.all(np.diff(x) > 0.0), "x должна строго возрастать"
    # все радиусы положительны
    assert np.all(r > 0.0)
    # минимум радиуса == R_кр (горловина)
    assert r.min() == pytest.approx(R_throat, rel=1e-6)
    # радиус среза = R_кр·sqrt(area_ratio)
    R_exit = R_throat * math.sqrt(area_ratio)
    assert r[-1] == pytest.approx(R_exit, rel=1e-3)
    assert geom.R_exit_m == pytest.approx(R_exit, rel=1e-9)
    # длины складываются
    assert geom.length_total_m == pytest.approx(
        geom.length_subsonic_m + geom.length_supersonic_m, rel=1e-6
    )
    # горловина — минимум по индексу
    ti = geom.throat_index
    assert r[ti] == pytest.approx(R_throat, rel=1e-6)


def test_conical_nozzle_invariants():
    R_throat = 0.05
    ar = 16.0
    geom = build_conical_nozzle(R_throat, ar, theta_exit_deg=15.0, theta_in_deg=30.0)
    assert geom.method == "conical"
    _check_invariants(geom, ar, R_throat)
    # у конуса θ_m == θ_a
    assert geom.theta_max_deg == pytest.approx(geom.theta_exit_deg)


def test_profiled_nozzle_invariants():
    R_throat = 0.05
    ar = 16.0
    geom = build_profiled_nozzle(R_throat, ar, theta_in_deg=30.0)
    assert geom.method == "profiled"
    _check_invariants(geom, ar, R_throat)
    # профилированное: θ_m > θ_a (поджатие к срезу)
    assert geom.theta_max_deg > geom.theta_exit_deg


def test_profiled_shorter_than_conical():
    # Укороченное оптимальное сопло короче конического 15° при тех же F_a/F_кр.
    R_throat = 0.05
    ar = 16.0
    cone = build_conical_nozzle(R_throat, ar, theta_exit_deg=15.0)
    prof = build_profiled_nozzle(R_throat, ar)
    assert prof.length_supersonic_m < cone.length_supersonic_m


def test_explicit_angles_profiled():
    geom = build_profiled_nozzle(
        0.04, 20.0, theta_max_deg=34.0, theta_exit_deg=8.0
    )
    assert geom.theta_max_deg == pytest.approx(34.0)
    assert geom.theta_exit_deg == pytest.approx(8.0)


# ── диспетчер ─────────────────────────────────────────────────────────────────

def test_dispatch_conical_filters_kwargs():
    # theta_max_deg / length_ratio не должны ломать конический путь
    geom = build_nozzle_geometry(
        0.05, 16.0, method="conical",
        theta_max_deg=34.0, length_ratio=20.0, theta_exit_deg=15.0,
    )
    assert geom.method == "conical"


def test_dispatch_profiled():
    geom = build_nozzle_geometry(0.05, 16.0, method="profiled")
    assert geom.method == "profiled"


def test_dispatch_unknown_method():
    with pytest.raises(ValueError):
        build_nozzle_geometry(0.05, 16.0, method="banana")


# ── обработка некорректного ввода ─────────────────────────────────────────────

def test_invalid_inputs():
    with pytest.raises(ValueError):
        build_conical_nozzle(0.0, 16.0)
    with pytest.raises(ValueError):
        build_conical_nozzle(0.05, 1.0)
    with pytest.raises(ValueError):
        build_profiled_nozzle(0.05, 0.5)


# ── RPA-стиль: параболический bell (Rao) ──────────────────────────────────────

def test_rao_reference_length_formula():
    R, eps = 0.05, 16.0
    expected = R * (math.sqrt(eps) - 1.0) / math.tan(math.radians(15.0))
    assert rao_reference_length_15deg(R, eps) == pytest.approx(expected)
    assert rao_reference_length_15deg(R, 25.0) > rao_reference_length_15deg(R, 16.0)


def test_estimate_bell_angles_ranges_and_order():
    tn, te = estimate_bell_angles(16.0)
    assert 5.0 <= tn <= 55.0
    assert 1.0 <= te <= 24.0
    assert te < tn
    tn2, _ = estimate_bell_angles(50.0)
    assert tn2 > tn


def test_estimate_bell_angles_length_correction():
    tn80, te80 = estimate_bell_angles(16.0, 80.0)
    tn60, te60 = estimate_bell_angles(16.0, 60.0)
    assert tn60 > tn80
    assert te60 > te80


def test_estimate_bell_angles_invalid():
    with pytest.raises(ValueError):
        estimate_bell_angles(1.0)


def test_rpa_parabolic_basic():
    geom = build_rpa_parabolic_nozzle(0.05, 16.0)
    assert geom.method == "rpa_parabolic"
    assert geom.R_exit_m == pytest.approx(0.05 * math.sqrt(16.0))
    rmin = min(p.r_m for p in geom.points)
    assert geom.R_throat_m == pytest.approx(rmin, abs=1e-6)
    xs = [p.x_m for p in geom.points]
    assert all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1))
    assert geom.phi_dispersion == pytest.approx(
        0.5 * (1 + math.cos(math.radians(geom.theta_exit_deg)))
    )


def test_rpa_parabolic_metadata_and_notation():
    geom = build_rpa_parabolic_nozzle(
        0.05, 16.0, R1_over_Rt=1.5, Rn_over_Rt=0.382, R2_over_R2max=0.5,
        contraction_angle_deg=30.0, length_fraction_pct=80.0,
    )
    md = geom.metadata
    assert md["contour_type"] == "parabolic_bell"
    assert md["R1_over_Rt"] == pytest.approx(1.5)
    assert md["Rn_over_Rt"] == pytest.approx(0.382)
    assert md["Le_over_Le15_pct"] == pytest.approx(80.0)
    assert geom.R_round_sub_m == pytest.approx(1.5 * 0.05)
    assert geom.r_round_sup_m == pytest.approx(0.382 * 0.05)
    assert geom.theta_in_deg == pytest.approx(30.0)


def test_rpa_parabolic_length_fraction_scaling():
    g80 = build_rpa_parabolic_nozzle(0.05, 16.0, length_fraction_pct=80.0)
    g100 = build_rpa_parabolic_nozzle(0.05, 16.0, length_fraction_pct=100.0)
    assert g100.length_supersonic_m > g80.length_supersonic_m


def test_rpa_parabolic_manual_angles():
    geom = build_rpa_parabolic_nozzle(0.05, 16.0, theta_n_deg=34.0, theta_e_deg=8.0)
    assert geom.theta_max_deg == pytest.approx(34.0)
    assert geom.theta_exit_deg == pytest.approx(8.0)


def test_rpa_parabolic_invalid_inputs():
    with pytest.raises(ValueError):
        build_rpa_parabolic_nozzle(0.0, 16.0)
    with pytest.raises(ValueError):
        build_rpa_parabolic_nozzle(0.05, 1.0)


def test_dispatch_rpa_aliases():
    for alias in ("rpa_parabolic", "rpa", "parabolic", "parabolic_bell"):
        geom = build_nozzle_geometry(0.05, 16.0, method=alias)
        assert geom.method == "rpa_parabolic"


# ── map_area_ratios: согласование профиля с реальным контуром ──────────────────

def test_map_area_ratios_matches_contour_endpoints():
    """Профиль по отношениям площадей должен совпадать с реальным контуром."""
    for build in (build_profiled_nozzle, build_conical_nozzle):
        g = build(0.05, 16.0, R_chamber_m=0.125)
        ar_cham = (g.R_chamber_m / g.R_throat_m) ** 2
        ars = np.array([ar_cham, 1.0, 16.0])
        sup = np.array([False, False, True])
        x, r = g.map_area_ratios(ars, supersonic_flags=sup)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(r))
        assert r[1] == pytest.approx(g.R_throat_m, rel=0.05)   # горловина
        assert r[2] == pytest.approx(g.R_exit_m, rel=0.05)     # срез
        assert r[0] == pytest.approx(g.R_chamber_m, rel=0.1)   # камера
        assert x[0] <= x[1] <= x[2]


def test_map_area_ratios_branch_selection():
    """Одинаковое A/A_кр на дозвуковой и сверхзвуковой ветви даёт разный x."""
    g = build_profiled_nozzle(0.05, 16.0, R_chamber_m=0.125)
    x, r = g.map_area_ratios(np.array([4.0, 4.0]),
                             supersonic_flags=np.array([False, True]))
    assert x[0] < x[1]
