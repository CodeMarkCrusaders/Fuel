"""
Тесты определения агрегатного состояния веществ (Species.aggregate_state).

Проверяется, что состояние (газ / жидкость / твёрдое) определяется по
СУФФИКСУ имени вещества, а не по индексу фазы NASA-9 (который лишь
различает газ=0 и конденсат>=1, но не отличает твёрдое от жидкого).
"""

import os

import pytest

from fuel_equilibrium.core.nasa9_parser import Species, parse_thermo_file
from fuel_equilibrium.core.equilibrium import find_thermo_db


def _mk(name, phase):
    """Минимальный Species для проверки только логики состояния."""
    return Species(
        name=name,
        description="",
        n_intervals=0,
        reference="",
        elements={},
        phase=phase,
        mol_weight=0.0,
        hf298=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Юнит-тесты логики (без реальной БД)
# ─────────────────────────────────────────────────────────────────────────────

def test_gas_by_phase_zero():
    sp = _mk("O2", 0)
    assert sp.aggregate_state == "gas"
    assert sp.is_gas and not sp.is_condensed
    assert sp.aggregate_state_ru == "газ"
    assert sp.phase_str == "газ"


def test_liquid_by_suffix_L():
    # (L) => жидкость, независимо от индекса фазы (1..5)
    for phase in (1, 2, 3, 4, 5):
        sp = _mk("AL2O3(L)", phase)
        assert sp.aggregate_state == "liquid", f"phase={phase}"
        assert sp.is_liquid and sp.is_condensed
        assert sp.aggregate_state_ru == "жидкость"


def test_solid_crystal_suffixes():
    # кристаллические/аллотропные суффиксы => твёрдое
    for name in ["C(gr)", "AL2O3(a)", "FE(b)", "SI(cr)", "S(c)", "P(III)"]:
        sp = _mk(name, 1)
        assert sp.aggregate_state == "solid", name
        assert sp.is_solid and sp.is_condensed
        assert sp.phase_str == "тв."


def test_phase1_is_not_always_solid():
    """Ключевой регресс: phase==1 может быть и жидкостью (например H2(L))."""
    liquid = _mk("H2(L)", 1)
    solid = _mk("AL2O3(a)", 1)
    assert liquid.aggregate_state == "liquid"
    assert solid.aggregate_state == "solid"
    # оба конденсированные
    assert liquid.is_condensed and solid.is_condensed


def test_high_phase_index_no_crash():
    """phase 3/4/5 не должны падать и не давать '?'."""
    for phase in (3, 4, 5):
        sp = _mk("SOMEOXIDE(L)", phase)
        assert sp.aggregate_state in ("gas", "liquid", "solid")
        assert sp.aggregate_state_ru in ("газ", "жидкость", "твёрдое")


# ─────────────────────────────────────────────────────────────────────────────
# Интеграционные проверки против реальной БД NASA-9
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db():
    path = find_thermo_db()
    if not path or not os.path.exists(path):
        pytest.skip("База NASA-9 thermo.inp не найдена")
    return parse_thermo_file(path)


def test_all_species_classified(db):
    """Все вещества классифицируются без исключений и без '?'."""
    for sp in db.values():
        st = sp.aggregate_state
        assert st in ("gas", "liquid", "solid"), f"{sp.name}: {st}"
        assert sp.aggregate_state_ru in ("газ", "жидкость", "твёрдое")
        assert sp.phase_str in ("газ", "жидк.", "тв.")


def test_gas_matches_phase_zero(db):
    """Газ <=> индекс фазы 0."""
    for sp in db.values():
        assert (sp.aggregate_state == "gas") == (sp.phase == 0), sp.name


def test_known_species_states(db):
    expected = {
        "H2O(L)": "liquid",
        "H2O": "gas",
        "O2": "gas",
        "AL2O3(a)": "solid",
        "AL2O3(L)": "liquid",
        "C(gr)": "solid",
        "H2(L)": "liquid",
    }
    for name, st in expected.items():
        if name in db:
            assert db[name].aggregate_state == st, f"{name} -> {db[name].aggregate_state}"


def test_liquids_exist_across_multiple_phase_indices(db):
    """Жидкости (L) встречаются при разных индексах фазы — подтверждает,
    что индекс фазы НЕ кодирует состояние."""
    phases_with_liquid = {
        sp.phase for sp in db.values() if sp.aggregate_state == "liquid"
    }
    # жидкости должны встречаться при phase==1 и при phase>=2
    assert 1 in phases_with_liquid
    assert any(p >= 2 for p in phases_with_liquid)
