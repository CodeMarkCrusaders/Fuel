"""
fuel_equilibrium.io.reporting — форматирование результатов расчётов.

Здесь лежат «чистые» print-функции, которые раньше были разбросаны по
core/equilibrium.py и rocket/nozzle_flow.py:

    * print_result        — таблица равновесия (TP/HP/SP)
    * print_nozzle_table  — таблица параметров по сечениям сопла (RPA-style)
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..core.nasa9_parser import Species
from ..core.gibbs_solver import EquilibriumResult
from ..rocket.nozzle_flow import RocketPerformance


# ─────────────────────────────────────────────────────────────────────────────
# Печать результата равновесия (TP / HP / SP)
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: EquilibriumResult, species_db: Dict[str, Species]) -> None:
    """Печатает результаты расчёта химического равновесия в человекочитаемом
    табличном виде (T, P, H, S, элементы, газовая фаза, конденсат)."""
    print("\n" + "=" * 70)
    print(f"  РЕЗУЛЬТАТЫ РАСЧЁТА ХИМИЧЕСКОГО РАВНОВЕСИЯ  (тип: {result.problem_type})")
    print("=" * 70)

    print(f"\n  T = {result.T:.2f} К  ({result.T-273.15:.2f} °C)")
    print(f"  P = {result.P:.0f} Па  ({result.P/101325:.5f} атм)")
    print(f"  H = {result.enthalpy:.4e} Дж")
    print(f"  S = {result.entropy:.4e} Дж/К")
    conv = "сошлось ✓" if result.converged else "не сошлось ✗"
    print(f"  {conv}  (итераций: {result.iterations}, невязка: {result.residual:.2e})")

    print(f"\n  Элементный состав (моль):")
    for el, n in sorted(result.elements.items()):
        print(f"    {el}: {n:.6f}")

    gas = result.get_gas_species()
    if gas:
        # массовые доли
        total_mass = 0.0
        masses = {}
        for name, moles, xi in gas:
            mw = species_db[name].mol_weight if name in species_db else 28.0
            masses[name] = moles * mw
            total_mass += masses[name]

        print(f"\n  ГАЗОВАЯ ФАЗА  (молей: {result.total_moles:.6f})")
        print(f"  {'Компонент':<20} {'моль':>14} {'мол.доля':>14} {'масс.доля':>14}")
        print("  " + "-" * 64)
        for name, moles, xi in gas:
            if xi > 1e-10:
                wf = masses[name] / total_mass if total_mass > 0 else 0.0
                print(f"  {name:<20} {moles:>14.6e} {xi:>14.6e} {wf:>14.6e}")
        print("  " + "-" * 64)
        print(f"  {'Итого':<20} {result.total_moles:>14.6e}")

        avg_mw = total_mass / result.total_moles if result.total_moles > 0 else 0.0
        print(f"\n  Средняя М газа: {avg_mw:.3f} г/моль")

    cond = result.get_condensed_species()
    if cond:
        print(f"\n  КОНДЕНСАТ:")
        for name, moles in cond:
            # агрегатное состояние определяем по суффиксу имени вещества
            state = "жидк." if name.strip().endswith("(L)") else "тв."
            print(f"  {name:<25} {moles:.6e}  {state}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Печать таблицы по сечениям сопла (RPA / CEA style)
# ─────────────────────────────────────────────────────────────────────────────

def print_nozzle_table(perf: RocketPerformance, top_k_species: int = 12) -> None:
    """Печатает таблицы 'Thermodynamic properties' и 'Fractions of products'
    в стиле RPA: по сечениям, столбец за столбцом."""
    stations = perf.stations
    n = len(stations)

    print()
    print("=" * (28 + 16 * n))
    print(f"  Thermodynamic properties (O/F = {perf.O_F:.4f},  α = {perf.alpha:.4f})")
    print("=" * (28 + 16 * n))

    headers = [s.label for s in stations]
    print(f"  {'Parameter':<28s}" + "".join(f"{h:>16s}" for h in headers) + "   Unit")

    def line(name, fmt_spec, values, unit):
        cells = "".join(format(v, fmt_spec).rjust(16) for v in values)
        print(f"  {name:<28s}{cells}   {unit}")

    line("Pressure",          ".4f", [s.P_Pa/1e6 for s in stations],            "MPa")
    line("Temperature",       ".4f", [s.T_K for s in stations],                  "K")
    line("Enthalpy",          ".4f", [s.H_J_per_kg/1000 for s in stations],      "kJ/kg")
    line("Entropy",           ".4f", [s.S_J_per_kgK/1000 for s in stations],     "kJ/(kg-K)")
    line("Internal energy",   ".4f", [s.U_J_per_kg/1000 for s in stations],      "kJ/kg")
    line("Cp (p=const, eq.)", ".4f", [s.cp_eq_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Cv (V=const, eq.)", ".4f", [s.cv_eq_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Gamma (eq.)",       ".4f", [s.gamma_eq for s in stations],             "")
    line("Isentropic exp.",   ".4f", [s.gamma_s for s in stations],              "")
    line("Gas constant",      ".4f", [s.R_specific_J_per_kgK/1000 for s in stations], "kJ/(kg-K)")
    line("Molecular weight",  ".4f", [s.mw_g_per_mol for s in stations],         "kg/kmol")
    line("Density",           ".4f", [s.rho_kg_per_m3 for s in stations],        "kg/m^3")
    line("Sonic velocity",    ".4f", [s.a_m_per_s for s in stations],            "m/s")
    line("Velocity",          ".4f", [s.V_m_per_s for s in stations],            "m/s")
    line("Mach number",       ".4f", [s.M for s in stations],                    "")

    # Ae/At — для камеры показываем 'infinity'
    ae_strs = []
    for s in stations:
        if math.isinf(s.Ae_At) or s.Ae_At > 1e6:
            ae_strs.append("infinity".rjust(16))
        else:
            ae_strs.append(f"{s.Ae_At:16.4f}")
    print(f"  {'Area ratio':<28s}" + "".join(ae_strs) + "   ")
    line("Mass flux", ".4f", [s.mass_flux_kg_per_m2_s for s in stations], "kg/(m^2 s)")

    # фракции
    print()
    print("-" * (28 + 16 * n))
    print(f"  Fractions of the combustion products (top {top_k_species})")
    print("-" * (28 + 16 * n))

    sp_names = stations[0].species_names
    max_xi = np.zeros(len(sp_names))
    for s in stations:
        max_xi = np.maximum(max_xi, s.mole_fractions)
    order = np.argsort(-max_xi)[:top_k_species]

    print(f"  {'Species':<28s}" + "".join(f"{h:>16s}" for h in headers))
    for idx in order:
        if max_xi[idx] < 1e-7:
            continue
        vals = "".join(f"{s.mole_fractions[idx]:16.7f}" for s in stations)
        print(f"  {sp_names[idx]:<28s}{vals}")

    print()
    print(f"  Isp (exit)   = {perf.Isp_s:.4f} с")
    print(f"  Isp (vacuum) = {perf.Isp_vac_s:.4f} с")
    print(f"  Cstar         = {perf.Cstar_m_per_s:.4f} м/с")
    print(f"  CF            = {perf.CF:.4f}")
    print()


__all__ = ["print_result", "print_nozzle_table"]
