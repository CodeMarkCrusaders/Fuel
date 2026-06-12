"""
Тесты кэша равновесных составов.

Проверяется:
  * корректность ключа (одинаковые входы -> попадание, разные -> промах);
  * результат из кэша — независимая копия (мутация не портит кэш);
  * кэш не меняет физический результат решателя (TP/HP/SP);
  * включение/выключение и очистка кэша.
"""

import numpy as np
import pytest

from fuel_equilibrium.core.equilibrium_cache import (
    EquilibriumCache,
    get_global_cache,
    clear_cache,
    set_cache_enabled,
)
from fuel_equilibrium.core.gibbs_solver import (
    solve_equilibrium,
    solve_equilibrium_HP,
    solve_equilibrium_SP,
)
from fuel_equilibrium.core.nasa9_parser import parse_thermo_file
from fuel_equilibrium.core.equilibrium import find_thermo_db, select_candidate_species


# ─────────────────────────────────────────────────────────────────────────────
# Юнит-тесты самого кэша
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSp:
    def __init__(self, name):
        self.name = name


def test_key_stability_and_collision():
    c = EquilibriumCache()
    sp = [_FakeSp("H2"), _FakeSp("O2"), _FakeSp("H2O")]
    k1 = c.make_key("TP", sp, {"H": 2.0, "O": 1.0}, 3000.0, 1e5, True)
    # тот же вход (порядок веществ/элементов не важен) -> тот же ключ
    sp2 = [_FakeSp("O2"), _FakeSp("H2O"), _FakeSp("H2")]
    k2 = c.make_key("TP", sp2, {"O": 1.0, "H": 2.0}, 3000.0, 1e5, True)
    assert k1 == k2
    # другой target -> другой ключ
    k3 = c.make_key("TP", sp, {"H": 2.0, "O": 1.0}, 3001.0, 1e5, True)
    assert k1 != k3
    # другой тип задачи -> другой ключ
    k4 = c.make_key("SP", sp, {"H": 2.0, "O": 1.0}, 3000.0, 1e5, True)
    assert k1 != k4


def test_put_get_returns_copy():
    c = EquilibriumCache()
    sp = [_FakeSp("H2")]
    key = c.make_key("TP", sp, {"H": 2.0}, 1000.0, 1e5, True)

    class R:
        def __init__(self):
            self.moles = np.array([1.0, 2.0])
            self.T = 1000.0

    c.put(key, R())
    got1 = c.get(key)
    got1.moles[0] = 999.0  # мутируем копию
    got2 = c.get(key)
    assert got2.moles[0] == 1.0, "кэш должен возвращать независимую копию"
    assert c.hits == 2 and c.misses == 0


def test_disable_and_clear():
    c = EquilibriumCache()
    sp = [_FakeSp("H2")]
    key = c.make_key("TP", sp, {"H": 2.0}, 1000.0, 1e5, True)
    c.put(key, object())
    c.set_enabled(False)
    assert c.get(key) is None  # выключен — ничего не отдаёт
    c.set_enabled(True)
    assert c.get(key) is not None
    c.clear()
    assert c.get(key) is None
    assert c.stats()["entries"] == 0


def test_fifo_eviction():
    c = EquilibriumCache(max_entries=3)
    keys = []
    for i in range(5):
        sp = [_FakeSp(f"S{i}")]
        k = c.make_key("TP", sp, {"H": float(i + 1)}, 1000.0, 1e5, True)
        keys.append(k)
        c.put(k, i)
    # должно остаться не больше 3 записей
    assert c.stats()["entries"] <= 3
    # последние три ключа должны быть на месте
    assert c.get(keys[-1]) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Интеграция с решателем — кэш не меняет результат
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def setup():
    path = find_thermo_db()
    db = parse_thermo_file(path)
    elements = {"H": 2.0, "O": 1.0}
    cands = select_candidate_species(db, set(elements), 3000.0,
                                     include_condensed=False, max_gas=25)
    return cands, elements


def test_tp_cache_matches_nocache(setup):
    cands, elements = setup
    set_cache_enabled(False)
    clear_cache()
    r0 = solve_equilibrium(cands, elements, T=3000.0, P=1e6,
                           include_condensed=False)
    set_cache_enabled(True)
    clear_cache()
    r1 = solve_equilibrium(cands, elements, T=3000.0, P=1e6,
                           include_condensed=False)
    # второй вызов — попадание в кэш
    r2 = solve_equilibrium(cands, elements, T=3000.0, P=1e6,
                           include_condensed=False)
    assert np.allclose(r0.moles, r1.moles, rtol=1e-9, atol=1e-12)
    assert np.allclose(r1.moles, r2.moles, rtol=0, atol=0)  # из кэша — точная копия
    assert get_global_cache().hits >= 1


def test_sp_cache_matches_nocache(setup):
    cands, elements = setup
    set_cache_enabled(False)
    clear_cache()
    r0 = solve_equilibrium_SP(cands, elements, S_target=1500.0, P=1e6,
                              T_init=3000.0, include_condensed=False)
    set_cache_enabled(True)
    clear_cache()
    r1 = solve_equilibrium_SP(cands, elements, S_target=1500.0, P=1e6,
                              T_init=3000.0, include_condensed=False)
    r2 = solve_equilibrium_SP(cands, elements, S_target=1500.0, P=1e6,
                              T_init=3000.0, include_condensed=False)
    assert abs(r0.T - r1.T) < 1e-6
    assert np.allclose(r1.moles, r2.moles, rtol=0, atol=0)


def test_cache_isolated_by_pressure(setup):
    cands, elements = setup
    set_cache_enabled(True)
    clear_cache()
    ra = solve_equilibrium(cands, elements, T=2800.0, P=1e6,
                           include_condensed=False)
    rb = solve_equilibrium(cands, elements, T=2800.0, P=5e6,
                           include_condensed=False)
    # разное давление -> разный состав (мольные доли не совпадают полностью)
    assert not np.allclose(ra.mole_fractions, rb.mole_fractions, rtol=1e-3)
    # оба должны быть промахами (разные ключи)
    assert get_global_cache().stats()["entries"] == 2


@pytest.fixture(autouse=True)
def _reset_cache():
    # каждый тест начинает с включённого пустого кэша
    set_cache_enabled(True)
    clear_cache()
    yield
    set_cache_enabled(True)
    clear_cache()
