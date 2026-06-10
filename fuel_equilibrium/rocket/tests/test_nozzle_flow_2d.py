"""Тесты заготовки двумерного (осесимметричного) расчёта сопла."""

import math

import numpy as np
import pytest

from fuel_equilibrium.rocket.nozzle_geometry import build_rpa_parabolic_nozzle
from fuel_equilibrium.rocket.nozzle_flow_2d import (
    solve_nozzle_2d,
    build_axisymmetric_grid,
    Nozzle2DResult,
)


class _St:
    def __init__(self, P, T, M, V, rho, ar, gamma=1.2, Rspec=350.0):
        self.P_Pa = P
        self.T_K = T
        self.M = M
        self.V_m_per_s = V
        self.rho_kg_per_m3 = rho
        self.Ae_At = ar
        self.gamma_s = gamma
        self.R_specific_J_per_kgK = Rspec


class _Perf:
    def __init__(self):
        self.stations = [
            _St(1.0e7, 3500.0, 0.10, 200.0, 5.00, 1.5),
            _St(5.6e6, 3300.0, 1.00, 1100.0, 1.80, 1.0),
            _St(5.0e4, 1500.0, 3.50, 2800.0, 0.05, 16.0),
        ]


def _geom():
    return build_rpa_parabolic_nozzle(0.05, 16.0)


def test_build_axisymmetric_grid_shapes():
    geom = _geom()
    x, r, wx, wr = build_axisymmetric_grid(geom, n_radial=15)
    assert x.shape == r.shape
    assert x.shape[0] == 15
    assert wx.shape[0] == x.shape[1]
    assert wr.shape[0] == x.shape[1]
    # ось: r=0; стенка: r=wall_r
    assert np.allclose(r[0, :], 0.0)
    assert np.allclose(r[-1, :], wr)


def test_solve_nozzle_2d_fields_shapes():
    geom = _geom()
    res = solve_nozzle_2d(_Perf(), geom, n_radial=11)
    assert isinstance(res, Nozzle2DResult)
    assert res.method == "quasi2d_stub"
    n_r, n_x = res.shape
    assert n_r == 11
    for key in ("P_Pa", "T_K", "M", "V_m_per_s", "flow_angle_deg"):
        f = res.fields[key]
        assert f.values.shape == (n_r, n_x)
    # угол потока на оси близок к нулю
    fa = res.field_values("flow_angle_deg")
    assert np.allclose(fa[0, :], 0.0, atol=1e-6)
    # модель — реальный квази-2D (не заглушка)
    assert res.metadata.get("is_stub") is False
    assert res.metadata.get("model") == "source_flow_isentropic"


def test_solve_nozzle_2d_varies_in_two_coordinates():
    """Параметры должны меняться И по оси x, И по радиусу r."""
    geom = _geom()
    res = solve_nozzle_2d(_Perf(), geom, n_radial=21)
    n_r, n_x = res.shape

    M = res.field_values("M")
    P = res.field_values("P_Pa")
    T = res.field_values("T_K")
    V = res.field_values("V_m_per_s")

    # 1) Осевая (x) изменчивость: число Маха растёт вдоль сопла на оси.
    assert M[0, -1] > M[0, 0] + 0.5

    # 2) Радиальная (r) изменчивость в сверхзвуковой части: в расширении
    #    у стенки Маха выше, чем на оси (source-flow), значит P/T ниже.
    #    Берём столбец вблизи среза.
    j = n_x - 2
    assert abs(M[-1, j] - M[0, j]) > 1e-3, "M не меняется по радиусу"
    assert M[-1, j] > M[0, j], "у стенки Маха должно быть больше на расширении"
    assert P[-1, j] < P[0, j], "давление у стенки должно быть ниже на расширении"
    assert T[-1, j] < T[0, j], "температура у стенки должна быть ниже"
    assert abs(V[-1, j] - V[0, j]) > 1e-3, "скорость не меняется по радиусу"

    # 3) Поля конечны (без NaN/inf)
    for key in ("M", "P_Pa", "T_K", "V_m_per_s"):
        arr = res.field_values(key)
        assert np.all(np.isfinite(arr))


def test_solve_nozzle_2d_unknown_method():
    geom = _geom()
    with pytest.raises(NotImplementedError):
        solve_nozzle_2d(_Perf(), geom, method="moc")
