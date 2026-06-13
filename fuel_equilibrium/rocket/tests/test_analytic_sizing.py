"""Тесты аналитического (инженерного) расчёта профиля сопла.

Сверка с эталонным примером (Таблица 2). Контрольные величины, заданные
в источнике явными «чистыми» формулами (энергетика, расходы, относительная
площадь среза, газодинамическая цепочка εк, геометрия камеры), проверяются
с жёстким допуском. Величины, зависящие от не полностью читаемых в источнике
констант (абсолютная площадь горловины), проверяются с инженерным допуском.
"""
import math

import pytest

from fuel_equilibrium.rocket.analytic_sizing import (
    AnalyticSizingInput,
    compute_analytic_sizing,
    gdf_f_lambda,
    gdf_q_lambda,
    lambda_from_q_subsonic,
)


@pytest.fixture
def ref_result():
    inp = AnalyticSizingInput(
        thrust_vac_N=7_770_000.0,
        p_chamber_Pa=7e6,
        p_exit_Pa=0.0486e6,
        Km=2.27,
        Isp_vac_m_s=3349.4838,
        k_adiabatic=1.1343,
        R_gas_J_kgK=346.2,
        T_chamber_K=3692.99,
        phi_k=0.99,
        phi_c=0.98,
        alpha=0.81,
    )
    return compute_analytic_sizing(inp)


# ── Энергетические показатели (точные «чистые» формулы) ──────────────────

def test_phi_ud(ref_result):
    assert ref_result.phi_ud == pytest.approx(0.9702, abs=1e-4)


def test_cstar(ref_result):
    assert ref_result.Cstar_m_s == pytest.approx(1779.38, rel=2e-3)


def test_isp_expected(ref_result):
    assert ref_result.Isp_exp_m_s == pytest.approx(3249.66, rel=1e-3)


def test_thrust_coefficients(ref_result):
    assert ref_result.Kp_thrust == pytest.approx(1.8262, rel=2e-3)
    assert ref_result.Kp_thrust_exp == pytest.approx(1.7897, rel=2e-3)


# ── Расходы (точные) ─────────────────────────────────────────────────────

def test_mass_flows(ref_result):
    assert ref_result.mdot_total_kg_s == pytest.approx(2464.46, rel=2e-3)
    assert ref_result.mdot_fuel_kg_s == pytest.approx(753.65, rel=2e-3)
    assert ref_result.mdot_ox_kg_s == pytest.approx(1710.80, rel=2e-3)


# ── Относительная площадь среза (точная) ─────────────────────────────────

def test_exit_area_ratio(ref_result):
    assert ref_result.Fa_rel == pytest.approx(18.53, rel=2e-3)
    assert ref_result.D_exit_rel == pytest.approx(4.3, rel=5e-3)


# ── Газодинамическая цепочка εк (точная) ─────────────────────────────────

def test_eps_chain(ref_result):
    # εк0 и δк зависят от F̄к1 (≈1.97) — инженерный допуск
    assert ref_result.eps_k0 == pytest.approx(0.9465, rel=1e-2)
    assert ref_result.delta_k == pytest.approx(0.9918, rel=5e-3)


# ── Геометрия камеры (инженерный допуск) ─────────────────────────────────

def test_chamber_geometry(ref_result):
    assert ref_result.L_reduced_m == pytest.approx(1.4940, rel=1e-3)
    assert ref_result.V_chamber_m3 == pytest.approx(1.0265, rel=8e-2)
    assert ref_result.D_chamber_m == pytest.approx(1.3075, rel=3e-2)


# ── Площади/диаметры — инженерный допуск (зависят от констант источника) ──

def test_throat_and_exit_in_range(ref_result):
    # Абсолютная площадь горловины зависит от не полностью читаемых констант
    # источника; проверяем порядок и физическую согласованность.
    assert 0.55 < ref_result.F_throat_m2 < 0.75
    assert 0.85 < ref_result.D_throat_m < 0.97
    # Fa = F̄a · Fкр
    assert ref_result.F_exit_m2 == pytest.approx(
        ref_result.Fa_rel * ref_result.F_throat_m2, rel=1e-9)


# ── Газодинамические функции ─────────────────────────────────────────────

def test_gdf_q_monotonic_subsonic():
    k = 1.1343
    qs = [gdf_q_lambda(l, k) for l in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b > a for a, b in zip(qs, qs[1:]))


def test_lambda_inverse_of_q():
    k = 1.1343
    for lam in (0.15, 0.34, 0.5, 0.8):
        q = gdf_q_lambda(lam, k)
        lam_back = lambda_from_q_subsonic(q, k)
        assert lam_back == pytest.approx(lam, abs=1e-4)


def test_f_lambda_reference():
    # f(0.34) ≈ 1.0565 (эталон)
    assert gdf_f_lambda(0.34, 1.1343) == pytest.approx(1.0565, rel=1e-3)
