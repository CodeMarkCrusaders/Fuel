"""Тесты параллельного газодинамического расчёта по сечениям.

Проверяем:
  * выбор числа потоков (_resolve_worker_count) при разных условиях;
  * эквивалентность результата при последовательном (1 поток) и
    параллельном (несколько потоков) расчёте сечений — порядок сечений
    и значения должны совпадать с точностью до численного шума.
"""
import os
import pytest

from fuel_equilibrium.rocket.nozzle_flow import _resolve_worker_count


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_worker_count
# ─────────────────────────────────────────────────────────────────────────────

def test_worker_count_single_task_is_one():
    os.environ.pop("FUEL_NOZZLE_WORKERS", None)
    assert _resolve_worker_count(1) == 1
    assert _resolve_worker_count(0) == 1


def test_worker_count_clamped_to_cpu():
    os.environ.pop("FUEL_NOZZLE_WORKERS", None)
    cpu = os.cpu_count() or 1
    w = _resolve_worker_count(1000)
    assert 1 <= w <= min(cpu, 8)


def test_worker_count_env_override():
    try:
        os.environ["FUEL_NOZZLE_WORKERS"] = "3"
        assert _resolve_worker_count(10) == 3
        # не больше числа задач
        assert _resolve_worker_count(2) == 2
    finally:
        os.environ.pop("FUEL_NOZZLE_WORKERS", None)


def test_worker_count_bad_env_falls_back():
    try:
        os.environ["FUEL_NOZZLE_WORKERS"] = "не-число"
        w = _resolve_worker_count(8)
        assert w >= 1
    finally:
        os.environ.pop("FUEL_NOZZLE_WORKERS", None)


# ─────────────────────────────────────────────────────────────────────────────
# Эквивалентность serial / parallel на реальном расчёте сопла
# ─────────────────────────────────────────────────────────────────────────────

def _load_db():
    try:
        from fuel_equilibrium.core.nasa9_parser import parse_thermo_file
        from fuel_equilibrium.core.equilibrium import find_thermo_db
        return parse_thermo_file(find_thermo_db())
    except Exception:
        return None


def test_parallel_matches_serial():
    db = _load_db()
    if db is None:
        pytest.skip("База NASA-9 недоступна")

    from fuel_equilibrium.rocket.nozzle_flow import (
        solve_rocket_nozzle, Propellant,
    )

    # лёгкий кейс H2/O2, немного сечений — быстро в CI
    ox = Propellant("O2", 0.85)
    fu = Propellant("H2", 0.15)
    Pc, Pe = 5.0e6, 0.1e6

    def run():
        return solve_rocket_nozzle(
            ox, fu, Pc, Pe, db,
            n_intermediate_stations=6,
            include_condensed=False,
            max_gas_species=40,
        )

    try:
        os.environ["FUEL_NOZZLE_WORKERS"] = "1"
        perf_serial = run()
        os.environ["FUEL_NOZZLE_WORKERS"] = "4"
        perf_par = run()
    finally:
        os.environ.pop("FUEL_NOZZLE_WORKERS", None)

    # одинаковое число сечений и их порядок (по давлению, убывающий ход потока)
    assert len(perf_serial.stations) == len(perf_par.stations)
    p_serial = [s.P_Pa for s in perf_serial.stations]
    p_par = [s.P_Pa for s in perf_par.stations]
    for a, b in zip(p_serial, p_par):
        assert a == pytest.approx(b, rel=1e-9)

    # ключевые интегральные характеристики совпадают
    assert perf_par.Cstar_m_per_s == pytest.approx(
        perf_serial.Cstar_m_per_s, rel=1e-6)
    assert perf_par.Isp_s == pytest.approx(perf_serial.Isp_s, rel=1e-6)

    # температура в каждом сечении совпадает
    for s_ser, s_par in zip(perf_serial.stations, perf_par.stations):
        assert s_par.T_K == pytest.approx(s_ser.T_K, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Эквивалентность serial / parallel для CEA-решателя (Cantera)
# ─────────────────────────────────────────────────────────────────────────────

def test_cea_parallel_matches_serial():
    """Параллельный расчёт сечений CEA-решателя совпадает с последовательным.

    Каждый поток использует собственную копию Cantera-газа, поэтому результат
    должен быть идентичен последовательному (с точностью до численного шума).
    """
    pytest.importorskip("cantera")
    from fuel_equilibrium.rocket.cea_solver import solve_rocket_nozzle_cea
    from fuel_equilibrium.rocket.nozzle_flow import Propellant

    ox = Propellant("O2", 0.85)
    fu = Propellant("H2", 0.15)
    Pc, Pe = 5.0e6, 0.1e6

    def run():
        return solve_rocket_nozzle_cea(
            ox, fu, Pc, Pe, n_intermediate_stations=10,
        )

    try:
        os.environ.pop("FUEL_NOZZLE_WORKERS", None)
        perf_serial = run()           # по умолчанию — последовательно
        os.environ["FUEL_NOZZLE_WORKERS"] = "4"
        perf_par = run()              # явный параллельный режим
    finally:
        os.environ.pop("FUEL_NOZZLE_WORKERS", None)

    assert len(perf_serial.stations) == len(perf_par.stations)
    p_serial = [s.P_Pa for s in perf_serial.stations]
    p_par = [s.P_Pa for s in perf_par.stations]
    for a, b in zip(p_serial, p_par):
        assert a == pytest.approx(b, rel=1e-9)

    assert perf_par.Cstar_m_per_s == pytest.approx(
        perf_serial.Cstar_m_per_s, rel=1e-9)
    assert perf_par.Isp_s == pytest.approx(perf_serial.Isp_s, rel=1e-9)

    for s_ser, s_par in zip(perf_serial.stations, perf_par.stations):
        assert s_par.T_K == pytest.approx(s_ser.T_K, rel=1e-9)
