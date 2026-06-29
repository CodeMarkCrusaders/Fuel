#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPA-Style Rocket Nozzle Calculator — GUI на Dear PyGui.

Переписано с PyQt5 на Dear PyGui (GPU-рендеринг, нативные графики,
сплиттеры вместо слайдеров — без лагов при ресайзе).

Возможности:
  • Расчёт параметров по длине сопла (P, T, V, M, ρ, гамма, состав)
  • Два решателя: собственный (Gibbs minimisation) и CEA (Cantera)
  • Тёмная тема в стиле Claude.ai
  • Экспорт точек в CSV (формат, совместимый с Amesim)
  • Графики на нативном DPG plot (без WebEngine/Plotly)
  • Сплиттеры вместо слайдеров для настройки ширины панелей
  • Сохранение/загрузка конфигурации
  • Аналитический (инженерный) расчёт профиля по тяге
"""

import sys
import os
import json
import csv
import math
import threading
import queue
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

import dearpygui.dearpygui as dpg

# Импорт решателей
from ..rocket.nozzle_flow import (
    Propellant, StationResult, RocketPerformance,
    solve_rocket_nozzle, stoichiometric_OF,
)
from ..rocket.nozzle_geometry import (
    build_conical_nozzle, build_profiled_nozzle,
    build_geometry_from_performance, optimal_angles_from_area_ratio,
    NozzleGeometry,
    build_rpa_parabolic_nozzle,
)
from ..rocket.analytic_sizing import (
    AnalyticSizingInput, AnalyticSizingResult, compute_analytic_sizing,
)
from ..io.reporting import print_nozzle_table
from ..core.nasa9_parser import parse_thermo_file
from ..core.equilibrium import find_thermo_db
from ..core.equilibrium_cache import clear_cache as clear_equilibrium_cache
from ..io.iteration_logger import IterationLogger, NullLogger
from ..io.action_logger import ActionLogger
from .component_selector_dpg import (
    MixturePropellantWidgetDPG,
    is_ion, classify_role, allowed_for_mode,
)

try:
    from ..rocket.cea_solver import (
        solve_rocket_nozzle_cea, build_axial_coordinates,
        nozzle_radius, CANTERA_AVAILABLE,
    )
except ImportError:
    CANTERA_AVAILABLE = False
    def build_axial_coordinates(stations, **kwargs):  # type: ignore[no-redef]
        return np.linspace(0, 1, len(stations))
    def nozzle_radius(stations):  # type: ignore[no-redef]
        return np.ones(len(stations))


APP_NAME = "Rocket Nozzle Calculator"
APP_VERSION = "2.0"

# Отображаемое имя профиля точности -> внутренний ключ решателя.
PRECISION_MAP = {
    "Быстро (грубо)": "fast",
    "Сбалансировано": "balanced",
    "Точно (медленно)": "precise",
}


# ═══════════════════════════════════════════════════════════════════════════
# Тема (Claude.ai dark)
# ═══════════════════════════════════════════════════════════════════════════

# Палитра Claude.ai
C_BG = (38, 38, 36)        # #262624
C_BG_DARK = (30, 30, 28)   # #1e1e1c
C_BG_PANEL = (48, 48, 46)  # #30302e
C_FG = (250, 250, 249)     # #fafaf9
C_MUTED = (168, 162, 158)  # #a8a29e
C_ACCENT = (204, 120, 92)  # #cc785c
C_ACCENT_DARK = (184, 107, 80)
C_BORDER = (58, 58, 55)    # #3a3a37


def apply_dark_theme(theme_id: Optional[int] = None) -> int:
    """Создаёт и возвращает тёмную тему в стиле Claude.ai."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, C_BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, C_BG_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_Border, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Text, C_FG)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, C_MUTED)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (74, 74, 71))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_Header, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, C_BG_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, C_BG)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, C_BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, C_MUTED)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, C_ACCENT_DARK)

            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 3)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 5)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14)

        # Специальный компонент для primary-кнопок (акцентный цвет)
        with dpg.theme_component(dpg.mvButton, parent=t) as tc_btn:
            pass  # наследует общую тему

    if theme_id is not None:
        dpg.bind_theme(t)
    return t


def make_primary_button_theme() -> int:
    """Тема для акцентных (primary) кнопок."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_ACCENT_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 90, 64))
            dpg.add_theme_color(dpg.mvThemeCol_Text, C_BG_DARK)
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Worker для асинхронного расчёта (threading.Thread вместо QThread)
# ═══════════════════════════════════════════════════════════════════════════

class NozzleSolverWorker:
    """Асинхронный расчёт сопла в отдельном потоке.

    Вместо QThread/сигналов использует threading.Thread + queue.Queue
    для потокобезопасной передачи результатов в главный цикл DPG.
    """

    def __init__(self, params: dict, solver: str, species_db=None):
        self.params = params
        self.solver = solver
        self.species_db = species_db
        self._result_q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll(self) -> Optional[dict]:
        """Опросить очередь результатов (вызывать из главного цикла).

        Возвращает dict с одним из ключей:
          {'type': 'progress', 'msg': str}
          {'type': 'ok', 'perf': RocketPerformance}
          {'type': 'error', 'msg': str}
        или None, если результатов нет.
        """
        try:
            return self._result_q.get_nowait()
        except queue.Empty:
            return None

    def _emit(self, msg_dict):
        self._result_q.put(msg_dict)

    def _solve_for_of(self, of_ratio: float):
        p = self.params
        ox_components = p["ox_components"]
        fu_components = p["fuel_components"]
        if not ox_components or not fu_components:
            raise ValueError("Не заданы компоненты окислителя и/или горючего.")
        of_ratio = max(float(of_ratio), 1e-9)
        fuel_mass_kg = 1.0 / (1.0 + of_ratio)
        oxidizer_mass_kg = of_ratio / (1.0 + of_ratio)
        ox_comp = ox_components[0]
        ox_T = ox_comp["T"] if ox_comp["T"] > 0 else None
        ox = Propellant(name=ox_comp["name"], mass_kg=oxidizer_mass_kg, T_K=ox_T)
        fu_comp = fu_components[0]
        fu_T = fu_comp["T"] if fu_comp["T"] > 0 else None
        fu = Propellant(name=fu_comp["name"], mass_kg=fuel_mass_kg, T_K=fu_T)

        if self.solver == "cea":
            perf = solve_rocket_nozzle_cea(
                oxidizer=ox, fuel=fu,
                P_chamber=p["P_chamber"], P_exit=p["P_exit"],
                n_intermediate_stations=p.get("n_inter", 5),
                include_condensed=p.get("include_condensed", False),
                injection_velocity=p.get("injection_velocity", 0.0),
                chamber_pressure_drop_frac=p.get("chamber_pressure_drop_frac", 0.0),
                verbose=False,
                progress_cb=lambda s: self._emit({"type": "progress", "msg": s}),
            )
        else:
            perf = solve_rocket_nozzle(
                oxidizer=ox, fuel=fu,
                P_chamber=p["P_chamber"], P_exit=p["P_exit"],
                species_db=self.species_db,
                n_intermediate_stations=p.get("n_inter", 5),
                include_condensed=p.get("include_condensed", True),
                injection_velocity=p.get("injection_velocity", 0.0),
                chamber_pressure_drop_frac=p.get("chamber_pressure_drop_frac", 0.0),
                verbose=False,
                logger=NullLogger(),
                precision=p.get("precision", "balanced"),
                progress_cb=lambda s: self._emit({"type": "progress", "msg": s}),
            )
        return perf

    def _find_optimum_of(self):
        """Поиск оптимального Km (массовое O/F), максимизирующего Isp."""
        p = self.params

        def isp_of(of_ratio):
            perf = self._solve_for_of(of_ratio)
            return perf, (perf.Isp_s if perf and perf.Isp_s is not None
                          and math.isfinite(perf.Isp_s) else -1.0)

        km0 = p.get("of_stoich", float("nan"))
        if km0 is not None and math.isfinite(km0) and km0 > 0:
            of_lo = 0.2 * km0
            of_hi = 2.2 * km0
        else:
            of_lo, of_hi = 0.3, 20.0

        n_grid = 9
        grid = [of_lo * (of_hi / of_lo) ** (i / (n_grid - 1)) for i in range(n_grid)]
        best_perf = None
        best_of = None
        best_isp = -1.0
        cache = {}
        for k, of in enumerate(grid):
            self._emit({"type": "progress",
                        "msg": f"Поиск оптимума Km: сетка {k+1}/{n_grid} (Km={of:.3f})..."})
            try:
                perf, isp = isp_of(of)
            except Exception:
                continue
            cache[of] = (perf, isp)
            if isp > best_isp:
                best_isp, best_perf, best_of = isp, perf, of

        if best_of is None:
            fallback_of = km0 if (km0 and math.isfinite(km0) and km0 > 0) else 1.0
            return self._solve_for_of(fallback_of), fallback_of

        idx = grid.index(best_of)
        a = grid[max(0, idx - 1)]
        b = grid[min(n_grid - 1, idx + 1)]
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        def eval_of(of):
            if of in cache:
                return cache[of]
            try:
                perf, isp = isp_of(of)
            except Exception:
                perf, isp = None, -1.0
            cache[of] = (perf, isp)
            return perf, isp

        pc, fc = eval_of(c)
        pd, fd = eval_of(d)
        for it in range(6):
            self._emit({"type": "progress", "msg": f"Уточнение оптимума Km: итерация {it+1}/6..."})
            if fc >= fd:
                b, d, fd, pd = d, c, fc, pc
                c = b - gr * (b - a)
                pc, fc = eval_of(c)
            else:
                a, c, fc, pc = c, d, fd, pd
                d = a + gr * (b - a)
                pd, fd = eval_of(d)

        for of, (perf, isp) in cache.items():
            if perf is not None and isp > best_isp:
                best_isp, best_perf, best_of = isp, perf, of

        self._emit({"type": "progress", "msg": f"Оптимум найден: Km = {best_of:.4f} (Isp = {best_isp:.2f} с)."})
        return best_perf, best_of

    def _run(self):
        try:
            p = self.params
            if not p.get("ox_components") or not p.get("fuel_components"):
                raise ValueError("Не заданы компоненты окислителя и/или горючего.")
            if p.get("optimize_of"):
                self._emit({"type": "progress", "msg": "Поиск оптимального соотношения компонентов (max Isp)..."})
                perf, _ = self._find_optimum_of()
            else:
                of_ratio = float(p.get("of_ratio", 1.0))
                if self.solver == "cea":
                    self._emit({"type": "progress", "msg": "Запуск CEA-решателя (Cantera)..."})
                else:
                    self._emit({"type": "progress", "msg": "Запуск собственного решателя (Gibbs)..."})
                perf = self._solve_for_of(of_ratio)
            self._emit({"type": "ok", "perf": perf})
        except Exception as e:
            tb = traceback.format_exc()
            self._emit({"type": "error", "msg": f"{e}\n\n{tb}"})


# ═══════════════════════════════════════════════════════════════════════════
# Данные графиков (перенесено из _section_series)
# ═══════════════════════════════════════════════════════════════════════════

PLOT_PARAM_DEFS = [
    ("P",     "Давление P",            "МПа",     (204, 120, 92)),
    ("T",     "Температура T",         "К",       (106, 176, 255)),
    ("V",     "Скорость потока V",     "м/с",     (130, 210, 122)),
    ("M",     "Число Маха M",          "",        (230, 184, 0)),
    ("rho",   "Плотность ρ",           "кг/м³",   (204, 120, 92)),
    ("gs",    "Изэнтр. показатель γs", "",        (192, 132, 252)),
    ("a",     "Скорость звука a",      "м/с",     (77, 208, 225)),
    ("S",     "Энтропия S",            "Дж/(кг·К)", (244, 114, 182)),
    ("H",     "Энтальпия H",           "МДж/кг",  (251, 146, 60)),
    ("q_dyn", "Динам. давление q",     "МПа",     (52, 211, 153)),
    ("tau",   "τ(λ) = T/T₀",           "",        (106, 176, 255)),
    ("pi",    "π(λ) = P/P₀",           "",        (204, 120, 92)),
    ("eps",   "ε(λ) = ρ/ρ₀",           "",        (192, 132, 252)),
    ("lam",   "λ(x) — скор. коэф.",    "",        (230, 184, 0)),
    ("q_gd",  "q(λ) — прив. расход",   "",        (130, 210, 122)),
    ("y_gd",  "y(λ) — функ. имп.",     "",        (77, 208, 225)),
]
PLOT_DEFAULT_KEYS = ["P", "T", "V", "M", "rho", "gs"]


def hampel_filter(y, window=2, n_sigma=3.0):
    """Медианный фильтр Хампеля — убирает одиночные выбросы."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 * window + 1:
        return y.copy()
    out = y.copy()
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        seg = y[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sigma = 1.4826 * mad
        if sigma > 0 and abs(y[i] - med) > n_sigma * sigma:
            out[i] = med
    return out


def section_series(perf, chamber_length_m, conv_div_lengths,
                   geometry=None) -> dict:
    """Единый расчёт параметров газа по сечениям (очищенных от шума).

    Если передан ``geometry`` (NozzleGeometry выбранного типа), ось X и
    нормированный профиль ``r_rel`` берутся С РЕАЛЬНОГО контура этой
    геометрии (через NozzleGeometry.map_area_ratios). Тогда профиль на
    графиках газодинамики и кривые параметров соответствуют тому же
    контуру, что и на вкладке «Геометрия». Без geometry — прежняя
    обобщённая разбивка (build_axial_coordinates + sqrt(Ae/At)).
    """
    if perf is None or not getattr(perf, "stations", None):
        return {}
    stations = list(perf.stations)
    L_chamber = chamber_length_m
    L_conv, L_div = conv_div_lengths
    x = np.asarray(build_axial_coordinates(
        stations, L_chamber=L_chamber, L_conv=L_conv, L_div=L_div),
        dtype=float)
    P = np.array([float(s.P_Pa) for s in stations])
    T = np.array([float(s.T_K) for s in stations])
    rho = np.array([float(getattr(s, "rho_kg_per_m3", 0.0)) for s in stations])
    V = np.array([float(getattr(s, "V_m_per_s", 0.0)) for s in stations])
    a = np.array([float(getattr(s, "a_m_per_s", 0.0)) for s in stations])
    gs = np.array([float(getattr(s, "gamma_s", 0.0)) for s in stations])
    Ae = np.array([float(getattr(s, "Ae_At", float("inf"))) for s in stations])
    labels = [getattr(s, "label", "") for s in stations]

    # Профиль/ось X по РЕАЛЬНОЙ геометрии выбранного типа: каждому сечению
    # сопоставляем точку (x, r) на фактическом контуре по Ae/At; ветвь
    # выбираем по числу Маха станции (V/a > 1 → сверхзвук).
    geom_r_rel = None
    if geometry is not None:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                M_raw = np.where(a > 0, V / a, 0.0)
            sup_flags = M_raw > 1.0
            gx, gr = geometry.map_area_ratios(Ae, supersonic_flags=sup_flags)
            R_thr = float(getattr(geometry, "R_throat_m", 0.0)) or 1.0
            gx = np.asarray(gx, dtype=float)
            gr = np.asarray(gr, dtype=float)
            if np.all(np.isfinite(gx)) and np.all(np.isfinite(gr)):
                x = gx
                geom_r_rel = gr / R_thr
        except Exception:
            geom_r_rel = None

    order = np.argsort(x, kind="stable")
    x = x[order]; P = P[order]; T = T[order]; rho = rho[order]
    V = V[order]; a = a[order]; gs = gs[order]; Ae = Ae[order]
    labels = [labels[i] for i in order]
    if geom_r_rel is not None:
        geom_r_rel = geom_r_rel[order]
    _, iu = np.unique(np.round(x, 9), return_index=True)
    iu = np.sort(iu)
    x = x[iu]; P = P[iu]; T = T[iu]; rho = rho[iu]
    V = V[iu]; a = a[iu]; gs = gs[iu]; Ae = Ae[iu]
    labels = [labels[i] for i in iu]
    if geom_r_rel is not None:
        geom_r_rel = geom_r_rel[iu]

    # Решатель газодинамики (rocket/nozzle_flow.py) теперь выдаёт гладкие
    # профили gamma_s/a/M (производные считаются с жёстким допуском и
    # относительными шагами, см. equilibrium_cp_and_sound_speed). Поэтому
    # агрессивное медианное сглаживание больше не нужно: оставляем лишь
    # мягкий «предохранитель» от единичных грубых выбросов (n_sigma высокий),
    # чтобы показывать пользователю настоящий результат расчёта, а не маску.
    gs = hampel_filter(gs, window=2, n_sigma=5.0)
    a = hampel_filter(a, window=2, n_sigma=5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(a > 0, V / a, 0.0)
    M = hampel_filter(M, window=2, n_sigma=5.0)
    try:
        i_throat = int(np.nanargmin(np.abs(M - 1.0)))
    except Exception:
        i_throat = 0

    if geom_r_rel is not None and geom_r_rel.size == Ae.size:
        # Профиль с реального контура выбранной геометрии (r/R_кр).
        r_rel = np.asarray(geom_r_rel, dtype=float)
        finite = r_rel[np.isfinite(r_rel)]
        r_cap = float(np.nanmax(finite)) if finite.size else 1.0
        r_rel = np.where(np.isfinite(r_rel), r_rel, r_cap)
    else:
        with np.errstate(invalid="ignore"):
            r_rel = np.sqrt(np.clip(Ae, 0.0, None))
        finite = r_rel[np.isfinite(r_rel)]
        r_cap = float(np.nanmax(finite)) if finite.size else 1.0
        r_rel = np.where(np.isfinite(r_rel), r_rel, r_cap)

    S = np.array([float(getattr(s, "S_J_per_kgK", float("nan"))) for s in stations])
    H = np.array([float(getattr(s, "H_J_per_kg", float("nan"))) for s in stations])
    if S.size == order.size:
        S = S[order][iu]
    else:
        S = np.full_like(x, float("nan"))
    if H.size == order.size:
        H = H[order][iu]
    else:
        H = np.full_like(x, float("nan"))

    # Газодинамические функции
    try:
        i0 = int(np.argmin(x)) if x.size else 0
    except Exception:
        i0 = 0
    T0 = float(T[i0]) if T.size and T[i0] > 0 else (float(np.nanmax(T)) if T.size else 1.0)
    P0 = float(P[i0]) if P.size and P[i0] > 0 else (float(np.nanmax(P)) if P.size else 1.0)
    rho0 = float(rho[i0]) if rho.size and rho[i0] > 0 else (float(np.nanmax(rho)) if rho.size else 1.0)

    finite_gs = gs[np.isfinite(gs) & (gs > 1.0)]
    k_ref = float(np.median(finite_gs)) if finite_gs.size else 1.2
    k_ref = min(max(k_ref, 1.05), 1.67)

    with np.errstate(divide="ignore", invalid="ignore"):
        lam2 = ((k_ref + 1.0) / 2.0 * M * M) / (1.0 + (k_ref - 1.0) / 2.0 * M * M)
    lam = np.sqrt(np.clip(lam2, 0.0, None))
    lam = hampel_filter(lam, window=2, n_sigma=5.0)
    lam_max = math.sqrt((k_ref + 1.0) / (k_ref - 1.0))
    lam = np.clip(lam, 0.0, lam_max - 1e-6)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 1.0 - (k_ref - 1.0) / (k_ref + 1.0) * lam * lam
        tau = np.clip(tau, 0.0, 1.0)
        pi = tau ** (k_ref / (k_ref - 1.0))
        eps = tau ** (1.0 / (k_ref - 1.0))
        q_gd = (lam * ((k_ref + 1.0) / 2.0) ** (1.0 / (k_ref - 1.0)) * tau ** (1.0 / (k_ref - 1.0)))
        y_gd = np.where(lam > 1e-9, (1.0 + lam * lam) / (2.0 * lam) * q_gd, 0.0)
    q_dyn = 0.5 * rho * V * V

    return {
        "x_m": x, "P_Pa": P, "T_K": T, "rho": rho,
        "V": V, "a": a, "M": M, "gamma_s": gs, "Ae_At": Ae,
        "r_rel": r_rel, "S": S, "H": H,
        "tau": tau, "pi": pi, "eps": eps,
        "lam": lam, "q_gd": q_gd, "y_gd": y_gd, "q_dyn": q_dyn,
        "label": labels, "i_throat": i_throat,
        "x_throat_m": float(x[i_throat]) if x.size else 0.0,
    }


def plot_param_value(key: str, ser: dict):
    """Возвращает массив значений величины key из единого источника."""
    if key == "P":
        return ser["P_Pa"] / 1e6
    if key == "T":
        return ser["T_K"]
    if key == "V":
        return ser["V"]
    if key == "M":
        return ser["M"]
    if key == "rho":
        return ser["rho"]
    if key == "gs":
        return ser["gamma_s"]
    if key == "a":
        return ser["a"]
    if key == "S":
        return ser.get("S")
    if key == "H":
        v = ser.get("H")
        return v / 1e6 if v is not None else None
    if key == "q_dyn":
        v = ser.get("q_dyn")
        return v / 1e6 if v is not None else None
    if key in ("tau", "pi", "eps", "lam", "q_gd", "y_gd"):
        return ser.get(key)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Главное окно (Dear PyGui)
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow:
    """Главное окно приложения на Dear PyGui.

    Сплиттер: левая панель ввода | правая панель результатов.
    Вместо QSlider для ширины боковой панели используется перетаскиваемая
    граница между двумя child_window (сплиттер). Это устраняет лаги при
    ресайзе — DPG перерисовывает GPU-сцену мгновенно, без пересборки
    Plotly-фигур на каждый пиксель.
    """

    def __init__(self):
        self.perf: Optional[RocketPerformance] = None
        self.species_db = None
        self.worker: Optional[NozzleSolverWorker] = None
        self.mixture_widget: Optional[MixturePropellantWidgetDPG] = None
        self._side_width = 280
        self._left_width = 460
        # Высота одного графика газодинамических параметров (px). Раньше
        # задавалась числовым полем, теперь регулируется перетаскиваемым
        # сплиттером под областью графиков (self._on_plot_height_splitter_drag).
        self._plot_row_h = 280
        # Ширина одного графика (px). Регулируется вертикальным сплиттером
        # между колонками. 0 — делить строку поровну (старое поведение);
        # как только потянут сплиттер ширины, значение фиксируется в px.
        self._plot_col_w = 0
        # Темы графиков создаются в корне (Dear PyGui не делает их детьми
        # plot-виджета), поэтому при удалении графиков они «утекают».
        # Накапливаем их id здесь и чистим в начале каждой перерисовки —
        # иначе после нескольких добавлений/переключений рендер деградирует.
        self._plot_theme_ids: List[int] = []
        self._plot_keys = list(PLOT_DEFAULT_KEYS)
        self._show_profile_1d = False
        self._last_geometry: Optional[NozzleGeometry] = None
        self._last_geometries: Dict[str, NozzleGeometry] = {}
        self._last_analytic_result = None
        # Выбранный тип сопла в панели расчёта
        self._calc_geom_type = "profiled"
        self._calc_use_rpa = False
        self._solver = "own"

        self._build()

        ActionLogger.info("Приложение запущено")
    # ─── Построение UI ───────────────────────────────────────────────────

    def _build(self):
        # Тема акцентных (primary) кнопок создаётся заранее — на неё
        # ссылаются как кнопка расчёта, так и кнопки во вкладках результатов.
        self._primary_theme = make_primary_button_theme()
        # Главное окно на весь вьюпорт
        with dpg.window(tag="main_window", label=APP_NAME,
                        no_title_bar=True, no_resize=True,
                        no_move=True, no_close=True):
            self._build_menu_bar()
            self._build_splitter_layout()
        # Диалоги выбора файла (проводник) для сохранения/загрузки
        # конфигурации создаём один раз, скрытыми; показываем по требованию.
        self._build_config_file_dialogs()

    def _build_config_file_dialogs(self):
        """Нативные диалоги выбора файла (проводник) для конфигурации JSON."""
        default_dir = os.path.expanduser("~")
        # ── Диалог сохранения ──
        with dpg.file_dialog(tag="dlg_save_config", show=False, modal=True,
                             directory_selector=False, width=720, height=480,
                             default_path=default_dir,
                             default_filename="rpa_config.json",
                             callback=self._on_save_config_selected,
                             cancel_callback=lambda *a: None):
            dpg.add_file_extension(".json", color=(204, 120, 92, 255),
                                   custom_text="[JSON]")
            dpg.add_file_extension(".*")
        # ── Диалог загрузки ──
        with dpg.file_dialog(tag="dlg_load_config", show=False, modal=True,
                             directory_selector=False, width=720, height=480,
                             default_path=default_dir,
                             callback=self._on_load_config_selected,
                             cancel_callback=lambda *a: None):
            dpg.add_file_extension(".json", color=(204, 120, 92, 255),
                                   custom_text="[JSON]")
            dpg.add_file_extension(".*")

    def _build_menu_bar(self):
        with dpg.menu_bar(parent="main_window"):
            with dpg.menu(label="Файл"):
                dpg.add_menu_item(label="Экспорт CSV…",
                                  callback=self.on_export_csv)
                dpg.add_menu_item(label="Экспорт Amesim (.data)…",
                                  callback=self.on_export_amesim)
                dpg.add_separator()
                dpg.add_menu_item(label="Сохранить конфигурацию…",
                                  callback=self.on_save_config)
                dpg.add_menu_item(label="Загрузить конфигурацию…",
                                  callback=self.on_load_config)
                dpg.add_separator()
                dpg.add_menu_item(label="Выход",
                                  callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Справка"):
                dpg.add_menu_item(label="О программе…",
                                  callback=self._about)

    def _build_splitter_layout(self):
        """Двухколоночный сплиттер: ввод | результаты.

        DPG не имеет готового QSplitter, но сплиттер реализуется через
        два child_window с обработчиком перетаскивания (drag) на границе.
        Ширина левой панели хранится в self._left_width и обновляется
        при перетаскивании. Благодаря GPU-рендерингу DPG перерисовка
        мгновенна — никакого лага при ресайзе (в отличие от PyQt+Plotly).
        """
        with dpg.group(parent="main_window", horizontal=True, tag="split_root"):
            # Левая колонка — панель ввода (сплиттер: перетаскиваемая граница)
            dpg.add_child_window(tag="left_panel",
                                 width=self._left_width,
                                 border=True, autosize_x=False,
                                 autosize_y=True,
                                 horizontal_scrollbar=False)
            self._build_input_panel()

            # Сплиттер (перетаскиваемая граница). Кнопка-ручка: пока удерживается
            # ЛКМ (is_item_active), курсор задаёт новую ширину левой панели.
            dpg.add_button(tag="vsplit", label="|", width=8, height=-1)
            with dpg.theme() as split_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, C_BORDER)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_ACCENT)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C_ACCENT_DARK)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, C_MUTED)
            dpg.bind_item_theme("vsplit", split_theme)

            # Правая колонка — результаты (вкладки) + кнопка расчёта внизу.
            with dpg.child_window(tag="right_panel", border=False,
                                  autosize_x=True, autosize_y=True):
                # Контейнер вкладок занимает всё пространство, кроме нижней
                # полосы с кнопкой «Рассчитать сопло» (высота отрицательная —
                # «всё, кроме зарезервированных снизу пикселей»).
                with dpg.child_window(tag="results_container", border=False,
                                      autosize_x=True, height=-92):
                    self._build_results_tabs()
                # Нижняя панель действий правой колонки
                self._build_action_bar()

        # Глобальные обработчики перетаскивания сплиттеров.
        # Используем is_item_active (кнопка «активна», пока удерживается ЛКМ —
        # даже если курсор ушёл с самой кнопки), что делает drag устойчивым.
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       threshold=0,
                                       callback=self._on_splitter_drag)

    def _on_splitter_drag(self):
        """Перетаскивание сплиттеров мышью.

        Вертикальный (vsplit): ширина левой панели = X-координата курсора
        относительно левого края корневой группы.
        Горизонтальный (hsplit): ширина боковой панели стиля графиков =
        правый край контейнера графиков − X-координата курсора.
        """
        # ── Вертикальный сплиттер: ширина левой панели ──
        if dpg.does_item_exist("vsplit") and dpg.is_item_active("vsplit"):
            mx = dpg.get_mouse_pos(local=False)[0]
            try:
                root_x = dpg.get_item_rect_min("split_root")[0]
            except Exception:
                root_x = 0
            new_w = int(mx - root_x)
            new_w = max(300, min(new_w, 1100))
            if new_w != self._left_width:
                self._left_width = new_w
                dpg.set_item_width("left_panel", self._left_width)
            return

        # ── Сплиттер высоты графиков: тянем полосу вверх/вниз ──
        if dpg.does_item_exist("plot_h_split") and dpg.is_item_active("plot_h_split"):
            my = dpg.get_mouse_pos(local=False)[1]
            last = getattr(self, "_plot_h_last_y", None)
            if last is None:
                self._plot_h_last_y = my
                return
            dy = my - last
            if abs(dy) >= 1.0:
                self._plot_h_last_y = my
                new_h = int(self._plot_row_h + dy)
                new_h = max(140, min(new_h, 900))
                if new_h != self._plot_row_h:
                    self._plot_row_h = new_h
                    # Высота графиков и сплиттеров ширины «на лету».
                    self._apply_plot_row_height(self._plot_row_h)
            return
        else:
            # Сбрасываем якорь, когда полоса отпущена.
            if getattr(self, "_plot_h_last_y", None) is not None:
                self._plot_h_last_y = None

        # ── Сплиттер(ы) ширины графиков: граница между колонками ──
        active_wsplit = self._active_width_splitter()
        if active_wsplit is not None:
            mx = dpg.get_mouse_pos(local=False)[0]
            last = getattr(self, "_plot_w_last_x", None)
            if last is None:
                self._plot_w_last_x = mx
                return
            dx = mx - last
            if abs(dx) >= 1.0:
                self._plot_w_last_x = mx
                base = int(self._plot_col_w or 0)
                if base <= 0:
                    base = self._current_plot_pixel_width()
                new_w = int(base + dx)
                new_w = max(220, min(new_w, 1400))
                if new_w != self._plot_col_w:
                    self._plot_col_w = new_w
                    self._apply_plot_col_width(self._plot_col_w)
            return
        else:
            if getattr(self, "_plot_w_last_x", None) is not None:
                self._plot_w_last_x = None

        # ── Горизонтальный сплиттер: ширина панели стиля графиков ──
        if dpg.does_item_exist("hsplit") and dpg.is_item_active("hsplit"):
            mx = dpg.get_mouse_pos(local=False)[0]
            try:
                cont_min = dpg.get_item_rect_min("plots_row")[0]
                cont_w = dpg.get_item_rect_size("plots_row")[0]
                right_edge = cont_min + cont_w
            except Exception:
                return
            new_w = int(right_edge - mx)
            new_w = max(180, min(new_w, 600))
            if new_w != self._side_width:
                self._side_width = new_w
                if dpg.does_item_exist("style_panel"):
                    dpg.set_item_width("style_panel", self._side_width)
                if dpg.does_item_exist("plots_container"):
                    dpg.set_item_width("plots_container",
                                       -(self._side_width + 14))
            return

    # ─── Хелперы сплиттеров графиков ─────────────────────────────────
    def _iter_plot_items(self):
        """Генератор всех plot-виджетов в контейнере графиков."""
        grp = "plots_group"
        if not dpg.does_item_exist(grp):
            return
        info = dpg.get_item_info(grp).get("children", {})
        if not isinstance(info, dict):
            return
        for child_list in info.values():
            for row in child_list:
                try:
                    if dpg.get_item_type(row) == "mvAppItemType::mvPlot":
                        yield row
                        continue
                except Exception:
                    pass
                row_info = dpg.get_item_info(row).get("children", {})
                if isinstance(row_info, dict):
                    for sub_l in row_info.values():
                        for item in sub_l:
                            try:
                                if dpg.get_item_type(item) == "mvAppItemType::mvPlot":
                                    yield item
                            except Exception:
                                pass

    def _iter_width_splitters(self):
        """Генератор тегов вертикальных сплиттеров ширины графиков."""
        grp = "plots_group"
        if not dpg.does_item_exist(grp):
            return
        info = dpg.get_item_info(grp).get("children", {})
        if not isinstance(info, dict):
            return
        for child_list in info.values():
            for row in child_list:
                row_info = dpg.get_item_info(row).get("children", {})
                if isinstance(row_info, dict):
                    for sub_l in row_info.values():
                        for item in sub_l:
                            try:
                                alias = dpg.get_item_alias(item)
                            except Exception:
                                alias = None
                            if alias and str(alias).startswith("plot_wsplit_"):
                                yield alias

    def _active_width_splitter(self):
        """Тег активного (зажатого ЛКМ) сплиттера ширины или None."""
        for tag in self._iter_width_splitters():
            try:
                if dpg.does_item_exist(tag) and dpg.is_item_active(tag):
                    return tag
            except Exception:
                pass
        return None

    def _current_plot_pixel_width(self):
        """Фактическая ширина первого графика в px (старт драга)."""
        for item in self._iter_plot_items():
            try:
                w = int(dpg.get_item_rect_size(item)[0])
                if w > 0:
                    return w
            except Exception:
                pass
        return 480

    def _even_split_plot_width(self, ncols):
        """Ширина одного графика (px) при равномерном делении строки.

        В горизонтальной группе Dear PyGui не делит ширину автоматически:
        ``width=0`` даёт график нулевой ширины (графики «исчезают» при
        переходе в 2-колоночную раскладку). Поэтому вычисляем ширину явно
        из доступной ширины контейнера графиков с учётом разделителей.
        """
        ncols = max(1, int(ncols))
        avail = 0
        for tag in ("plots_container", "plots_group"):
            if dpg.does_item_exist(tag):
                try:
                    w = int(dpg.get_item_rect_size(tag)[0])
                except Exception:
                    w = 0
                if w > 0:
                    avail = w
                    break
        if avail <= 0:
            # Контейнер ещё не отрисован (первый расчёт) — оценка по окну.
            try:
                avail = int(dpg.get_viewport_client_width()) - self._side_width - 60
            except Exception:
                avail = 900
        # Вычитаем ширину вертикальных сплиттеров (≈10 px) между колонками.
        splitters = (ncols - 1) * 10
        per = int((avail - splitters - 12) / ncols)
        return max(180, per)

    def _apply_plot_row_height(self, h):
        """Высота всех графиков и сплиттеров ширины на лету."""
        for item in self._iter_plot_items():
            try:
                dpg.set_item_height(item, int(h))
            except Exception:
                pass
        for tag in self._iter_width_splitters():
            try:
                dpg.set_item_height(tag, int(h))
            except Exception:
                pass

    def _apply_plot_col_width(self, w):
        """Фиксированная ширина всех графиков на лету."""
        for item in self._iter_plot_items():
            try:
                dpg.set_item_width(item, int(w))
            except Exception:
                pass

    def _reset_plot_width(self):
        """Сброс ширины графиков к авто-доле строки."""
        self._plot_col_w = 0
        self._redraw_plots()

    def _build_action_bar(self):
        """Нижняя панель правой колонки: кнопка расчёта + статус/прогресс."""
        dpg.add_separator()
        dpg.add_button(tag="btn_calc", label="▶  Рассчитать сопло",
                       width=-1, height=40,
                       callback=self.on_calculate)
        dpg.bind_item_theme("btn_calc", self._primary_theme)
        with dpg.group(horizontal=True):
            dpg.add_text("Готово. Введите параметры и нажмите «Рассчитать».",
                         tag="status_text", color=C_MUTED, wrap=0)
            dpg.add_text("", tag="progress_text", color=C_ACCENT)
            dpg.add_text("", tag="iter_text", color=C_MUTED)

    # ─── Панель ввода ────────────────────────────────────────────────────

    def _build_input_panel(self):
        with dpg.collapsing_header(parent="left_panel",
                                   label="Топливо (RPA-style)",
                                   default_open=True, closable=False):
            mix_tag = "mix_widget_container"
            with dpg.group(tag=mix_tag):
                pass
            self.mixture_widget = MixturePropellantWidgetDPG(
                mix_tag, self.species_db,
                on_change=lambda m: self._update_of_from_mixture())

            # Соотношение компонентов
            with dpg.group(horizontal=True):
                dpg.add_text("Соотношение:")
                dpg.add_combo(
                    ["Km (массовое O/F)", "α (Km/Km0)", "Оптимум (max Isp)"],
                    tag="cb_mix_mode", default_value="Km (массовое O/F)",
                    width=-40, callback=lambda s, a: self._on_mix_mode_changed())
            dpg.add_input_text(label="Значение", hint="Km (O/F)",
                               tag="ed_mix_value", width=-1,
                               callback=lambda s: self._update_of_from_mixture())
            dpg.add_text("Km0 = —", tag="lbl_of", color=C_ACCENT)

        with dpg.collapsing_header(parent="left_panel",
                                   label="Параметры расчёта",
                                   default_open=True, closable=False):
            # Исходные данные
            with dpg.collapsing_header(label="Исходные данные",
                                        default_open=True, closable=False,
                                        leaf=True):
                dpg.add_input_text(label="P камеры", hint="давление",
                                   tag="ed_Pc", width=-1)
                dpg.add_combo(["Па", "кПа", "МПа", "бар", "атм"],
                              tag="cb_Pc_unit", default_value="МПа", width=-1)
                dpg.add_input_text(label="P среза", hint="давление",
                                   tag="ed_Pe", width=-1)
                dpg.add_combo(["Па", "кПа", "МПа", "бар", "атм"],
                              tag="cb_Pe_unit", default_value="МПа", width=-1)
                dpg.add_slider_float(label="Скорость подачи (м/с)",
                                     tag="sp_inj_velocity",
                                     default_value=0.0, min_value=0.0,
                                     max_value=500.0, format="%.1f")
                dpg.add_slider_float(label="Перепад давления (%)",
                                     tag="sp_chamber_dp",
                                     default_value=0.0, min_value=0.0,
                                     max_value=30.0, format="%.2f")
                dpg.add_checkbox(label="Учитывать конденсат",
                                 tag="chk_condensed", default_value=True)
            # Газодинамика
            with dpg.collapsing_header(label="Газодинамика (1D)",
                                        default_open=False, closable=False,
                                        leaf=True):
                dpg.add_slider_int(label="Промежут. сечений",
                                    tag="sp_n_inter", default_value=8,
                                    min_value=0, max_value=1048)
                dpg.add_text("Промежуточные сечения распределяются равномерно\n"
                             "по длине сопла (дозвук → горловина → сверхзвук).",
                             color=C_MUTED, wrap=380)
                dpg.add_text("Точность расчёта:")
                dpg.add_combo(
                    ["Быстро (грубо)", "Сбалансировано", "Точно (медленно)"],
                    tag="cb_precision", default_value="Сбалансировано", width=-1)
                dpg.add_text("Грубее точность → меньше итераций и быстрее расчёт.\n"
                             "«Точно» — максимум итераций, максимальная точность.",
                             color=C_MUTED, wrap=380)
            # Геометрия (для оси X)
            with dpg.collapsing_header(label="Геометрия (Size & Geometry)",
                                        default_open=False, closable=False,
                                        leaf=True):
                self._build_geometry_input()

        with dpg.collapsing_header(parent="left_panel",
                                   label="Решатель", default_open=False,
                                   closable=False):
            dpg.add_checkbox(label="Собственный (NASA-9, минимизация G)",
                             tag="rb_own", default_value=True,
                             callback=self._on_solver_changed)
            cea_label = "CEA (Cantera)"
            if not CANTERA_AVAILABLE:
                cea_label += "  ⛔ не установлен"
            dpg.add_checkbox(label=cea_label, tag="rb_cea", default_value=False,
                             callback=self._on_solver_changed)
            dpg.add_text("Собственный решатель использует NASA-9 полиномы и SLSQP.\n"
                         "CEA-решатель (Cantera) даёт идентичные результаты NASA CEA.",
                         color=C_MUTED, wrap=380)

        with dpg.collapsing_header(parent="left_panel",
                                   label="Потери (реализуемые КПД)",
                                   default_open=False, closable=False):
            dpg.add_slider_float(label="КПД реакции ηр",
                                 tag="sp_eff_reaction",
                                 default_value=1.0, min_value=0.0,
                                 max_value=1.0, format="%.4f")
            dpg.add_slider_float(label="КПД сопла ηс",
                                 tag="sp_eff_nozzle",
                                 default_value=1.0, min_value=0.0,
                                 max_value=1.0, format="%.4f",
                                 callback=self._update_overall_efficiency)
            dpg.add_text("Суммарный ηобщ = 1.0000", tag="lbl_eff_overall",
                         color=C_ACCENT)

        # Кнопка расчёта и статус перенесены в нижнюю панель правой колонки
        # (_build_action_bar). Здесь больше ничего не добавляем.

    def _build_geometry_input(self):
        """Панель геометрии сопла (для оси X и профиля)."""
        dpg.add_text("Размер камеры сгорания:", color=C_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Длина камеры",
                             tag="rb_chamber_len", default_value=True,
                             callback=lambda: self._on_chamber_size_mode_changed("rb_chamber_len"))
            dpg.add_checkbox(label="Характеристическая L*",
                             tag="rb_chamber_lstar", default_value=False,
                             callback=lambda: self._on_chamber_size_mode_changed("rb_chamber_lstar"))
        dpg.add_input_float(label="Длина камеры (м)",
                            tag="sp_L_chamber", default_value=0.100,
                            min_value=0.0, min_clamped=False, format="%.4f")
        dpg.add_input_float(label="Характер. L* (м)",
                            tag="sp_L_star", default_value=1.000,
                            min_value=0.0, min_clamped=False, format="%.4f")

        dpg.add_separator()
        dpg.add_text("Профиль сопла:", color=C_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Коническое", tag="rb_calc_conical",
                             default_value=False,
                             callback=lambda: self._on_calc_geom_type_changed("rb_calc_conical"))
            dpg.add_checkbox(label="Профилированное", tag="rb_calc_profiled",
                             default_value=True,
                             callback=lambda: self._on_calc_geom_type_changed("rb_calc_profiled"))
            dpg.add_checkbox(label="RPA (bell)", tag="rb_calc_rpa",
                             default_value=False,
                             callback=lambda: self._on_calc_geom_type_changed("rb_calc_rpa"))
        dpg.add_input_float(label="Rкр — горловина (м)", tag="sp_calc_Rthroat",
                            default_value=0.050, min_value=0.0001,
                            min_clamped=True, format="%.4f")
        dpg.add_input_float(label="Rкамеры / Rкр", tag="sp_calc_Rcham",
                            default_value=2.500, min_value=1.05,
                            min_clamped=True, format="%.3f")
        dpg.add_input_float(label="θвх (дозвук), °", tag="sp_calc_theta_in",
                            default_value=30.0, min_value=10.0,
                            max_value=45.0, format="%.2f")
        dpg.add_checkbox(label="θm, θa, длина — авто (Рис. 2.14)",
                         tag="chk_calc_auto_angles", default_value=True)
        dpg.add_input_float(label="θm (начало св/зв), °", tag="sp_calc_theta_max",
                            default_value=30.0, min_value=5.0,
                            max_value=50.0, format="%.2f")
        dpg.add_input_float(label="θa (срез), °", tag="sp_calc_theta_exit",
                            default_value=15.0, min_value=3.0,
                            max_value=25.0, format="%.2f")
        dpg.add_input_float(label="x̄a = L/Rкр", tag="sp_calc_len_ratio",
                            default_value=9.5, min_value=0.5,
                            max_value=200.0, format="%.2f")

    # ─── Вкладки результатов ─────────────────────────────────────────────

    def _build_results_tabs(self):
        with dpg.tab_bar(tag="main_tabs"):
            # Группа 1: Газодинамика
            with dpg.tab(label="Газодинамика"):
                with dpg.tab_bar():
                    with dpg.tab(label="Параметры по сечениям", tag="tab_stations"):
                        # Таблица помещена в прокручиваемый контейнер: при
                        # большом числе сечений колонки сохраняют читаемую
                        # ширину и прокручиваются ГОРИЗОНТАЛЬНО внутри панели,
                        # а не «распирают» всю вкладку.
                        with dpg.child_window(tag="stations_scroll", border=False,
                                              autosize_x=True, autosize_y=True,
                                              horizontal_scrollbar=True):
                            with dpg.table(tag="tbl_stations", header_row=True,
                                           resizable=True,
                                           policy=dpg.mvTable_SizingFixedFit,
                                           scrollX=True, scrollY=False):
                                dpg.add_table_column(label="Параметр")
                                dpg.add_table_column(label="Значение")
                                dpg.add_table_column(label="Ед.изм.")
                    with dpg.tab(label="Графики по длине сопла"):
                        self._build_plots_tab()
                    with dpg.tab(label="Тяговые характеристики"):
                        dpg.add_text("", tag="txt_perf", wrap=1000)

            # Группа 2: Равновесный состав
            with dpg.tab(label="Равновесный состав"):
                with dpg.tab_bar():
                    with dpg.tab(label="Состав продуктов сгорания", tag="tab_species_container"):
                        with dpg.group(horizontal=True):
                            dpg.add_text("Показывать:")
                            dpg.add_checkbox(label="Мольные доли",
                                             tag="rb_mole", default_value=True,
                                             callback=lambda: self._on_fraction_mode_changed("rb_mole"))
                            dpg.add_checkbox(label="Массовые доли",
                                             tag="rb_mass", default_value=False,
                                             callback=lambda: self._on_fraction_mode_changed("rb_mass"))
                            dpg.add_text("Топ:")
                            dpg.add_slider_int(tag="sp_topN", default_value=15,
                                               min_value=3, max_value=50,
                                               width=120,
                                               callback=lambda: self._refresh_species_view())
                        with dpg.table(tag="tbl_species", header_row=True,
                                       resizable=True, policy=dpg.mvTable_SizingStretchProp):
                            dpg.add_table_column(label="Компонент")
                            dpg.add_table_column(label="Доля")

            # Группа 3: Геометрия
            with dpg.tab(label="Геометрия"):
                self._build_geometry_tab()

            # Группа 4: Аналитический расчёт
            with dpg.tab(label="Аналитический расчёт"):
                self._build_analytic_tab()

    def _build_plots_tab(self):
        """Вкладка графиков: DPG plot вместо Plotly/matplotlib.

        Слева — графики (растягиваются), справа — панель оформления, между
        ними вертикальный сплиттер (hsplit), ширину которого можно тянуть.
        """
        with dpg.group(horizontal=True, tag="plots_row"):
            # Контейнер графиков растягивается на всё свободное место
            # (отрицательная ширина = «всё, кроме зарезервированного справа»).
            with dpg.child_window(tag="plots_container",
                                  border=False,
                                  width=-(self._side_width + 14),
                                  autosize_y=True):
                # Заголовок-индикатор
                dpg.add_text("Выполните расчёт сопла, чтобы построить графики.",
                             tag="plot_hint", color=C_MUTED)
                # Контейнер для plot-виджетов (создаются динамически)
                with dpg.group(tag="plots_group"):
                    pass

                # ── Горизонтальный сплиттер высоты графиков ──
                # Перетаскивание этой полосы вверх/вниз меняет высоту каждого
                # графика газодинамических параметров (self._plot_row_h).
                dpg.add_button(tag="plot_h_split", label="═ высота графиков ═",
                               width=-1, height=10)
                with dpg.theme() as ph_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, C_BORDER)
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_ACCENT)
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C_ACCENT_DARK)
                        dpg.add_theme_color(dpg.mvThemeCol_Text, C_MUTED)
                dpg.bind_item_theme("plot_h_split", ph_theme)

            # Вертикальный сплиттер между графиками и панелью стиля
            dpg.add_button(tag="hsplit", label="|", width=8, height=-1)
            with dpg.theme() as hsplit_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, C_BORDER)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, C_ACCENT)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C_ACCENT_DARK)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, C_MUTED)
            dpg.bind_item_theme("hsplit", hsplit_theme)

            # Боковая панель настроек (фиксированная ширина, меняется drag-ом)
            with dpg.child_window(tag="style_panel", border=True,
                                  width=self._side_width, autosize_y=True):
                self._build_style_panel()

    def _build_style_panel(self):
        """Панель оформления графиков (сплиттер вместо слайдера)."""
        dpg.add_text("Оформление графиков", color=C_ACCENT)
        # Выбор отображаемых графиков (checkboxes)
        dpg.add_text("Показать графики:")
        for key, label, unit, color in PLOT_PARAM_DEFS:
            dpg.add_checkbox(
                label=label + (f", {unit}" if unit else ""),
                tag=f"chk_plot_{key}",
                default_value=(key in PLOT_DEFAULT_KEYS),
                callback=lambda s, a, k=key: self._on_plot_param_toggle(k, a),
            )
        dpg.add_separator()
        dpg.add_checkbox(label="Профиль сопла на графиках",
                         tag="chk_show_profile", default_value=False,
                         callback=self._on_toggle_profile_1d)
        dpg.add_text("Размер каждого графика — сплиттерами:\n"
                     "• высота: полоса «═» под графиками (вверх/вниз);\n"
                     "• ширина: вертикальная полоса «|» между колонками\n"
                     "  (в 2-колоночной раскладке — тяните влево/вправо).",
                     color=C_MUTED, wrap=0)
        dpg.add_combo(["Авто", "1 колонка", "2 колонки"],
                      tag="cb_plot_cols", default_value="Авто",
                      callback=lambda: self._redraw_plots())
        dpg.add_button(label="↺ Сбросить ширину графиков", width=-1,
                       callback=self._reset_plot_width)
        dpg.add_separator()
        dpg.add_text("Шрифт/стиль:")
        dpg.add_input_float(label="Толщ. линий", tag="sp_lw",
                            default_value=1.8, min_value=0.1,
                            max_value=10.0, format="%.1f",
                            callback=lambda: self._redraw_plots())
        dpg.add_checkbox(label="Маркеры", tag="chk_markers",
                         default_value=True,
                         callback=lambda: self._redraw_plots())
        dpg.add_checkbox(label="Сглаживание", tag="chk_smooth",
                         default_value=False,
                         callback=lambda: self._redraw_plots())
        dpg.add_checkbox(label="Основная сетка", tag="chk_grid_major",
                         default_value=True,
                         callback=lambda: self._redraw_plots())
        dpg.add_checkbox(label="Доп. сетка", tag="chk_grid_minor",
                         default_value=True,
                         callback=lambda: self._redraw_plots())
        dpg.add_checkbox(label="Тёмный фон", tag="chk_dark_plot",
                         default_value=True,
                         callback=lambda: self._redraw_plots())
        dpg.add_button(label="↻ Обновить", width=-1,
                       callback=lambda: self._redraw_plots())
        dpg.add_button(label="⬇ Сохранить рисунки (PNG)", width=-1,
                       callback=self._save_figures)

    def _build_geometry_tab(self):
        """Вкладка построения контура сопла."""
        with dpg.group(horizontal=True):
            with dpg.child_window(border=True, width=380, autosize_y=True):
                dpg.add_text("Тип сопла (можно выбрать несколько):",
                             color=C_ACCENT)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Коническое", tag="rb_geom_conical",
                                     default_value=False)
                    dpg.add_checkbox(label="Профилированное",
                                     tag="rb_geom_profiled", default_value=True)
                    dpg.add_checkbox(label="RPA (bell)", tag="rb_geom_rpa",
                                     default_value=False)
                dpg.add_text("Отмеченные профили строятся вместе и "
                             "отображаются на графике разными цветами.",
                             color=C_MUTED, wrap=360)
                dpg.add_input_float(label="Rкр (м)", tag="sp_geom_Rthroat",
                                    default_value=0.050, min_value=0.0001,
                                    min_clamped=True, format="%.4f")
                dpg.add_input_float(label="Fa/Fкр", tag="sp_geom_AR",
                                    default_value=16.0, min_value=1.001,
                                    min_clamped=True, format="%.3f")
                dpg.add_input_float(label="Rкамеры / Rкр",
                                    tag="sp_geom_Rcham_factor",
                                    default_value=2.500, min_value=1.05,
                                    min_clamped=True, format="%.3f")
                dpg.add_input_float(label="θвх (°)", tag="sp_geom_theta_in",
                                    default_value=30.0)
                dpg.add_input_float(label="θa срез (°)", tag="sp_geom_theta_exit",
                                    default_value=15.0)
                dpg.add_input_float(label="θm св/зв (°)",
                                    tag="sp_geom_theta_max", default_value=30.0)
                dpg.add_checkbox(label="Авто-углы (Рис. 2.14)",
                                 tag="chk_geom_auto_angles", default_value=True)
                dpg.add_input_float(label="x̄a = L/Rкр",
                                    tag="sp_geom_len_ratio", default_value=9.5)
                dpg.add_separator()
                dpg.add_button(label="▶ Построить контур", width=-1,
                               height=36, tag="btn_geom_build",
                               callback=self.on_build_geometry)
                dpg.bind_item_theme("btn_geom_build", self._primary_theme)
                dpg.add_button(label="⤵ Взять Fa/Fкр из расчёта",
                               width=-1, callback=self.on_geometry_from_perf)
                dpg.add_button(label="💾 Экспорт контура (CSV)", width=-1,
                               callback=self.on_export_geometry_csv)
            with dpg.child_window(border=False, autosize_x=True,
                                  autosize_y=True):
                with dpg.plot(tag="geom_plot", height=-300, width=-1,
                              equal_aspects=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="x, м",
                                      tag="geom_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="r, м",
                                      tag="geom_y")
                dpg.add_text("", tag="txt_geom_summary", wrap=600)

    def _build_analytic_tab(self):
        """Вкладка аналитического (инженерного) расчёта."""
        with dpg.group(horizontal=True):
            with dpg.child_window(border=True, width=400, autosize_y=True):
                dpg.add_text("Инженерная методика РПА / Добровольского.",
                             color=C_MUTED, wrap=380)
                dpg.add_separator()
                dpg.add_input_float(label="Pн (тяга в пустот.), Н",
                                    tag="sp_an_thrust",
                                    default_value=7770000.0, format="%.1f")
                dpg.add_input_float(label="pк (камера), МПа",
                                    tag="sp_an_pk", default_value=7.0,
                                    format="%.4f")
                dpg.add_input_float(label="pa (срез), МПа",
                                    tag="sp_an_pa", default_value=0.0486,
                                    format="%.5f")
                dpg.add_input_float(label="Km (O/F массовое)",
                                    tag="sp_an_Km", default_value=2.27,
                                    format="%.4f")
                dpg.add_input_float(label="Iуд (пустот.), м/с",
                                    tag="sp_an_isp",
                                    default_value=3349.4838, format="%.4f")
                dpg.add_input_float(label="k (адиабата)", tag="sp_an_k",
                                    default_value=1.1343, format="%.4f")
                dpg.add_input_float(label="Rг, Дж/(кг·К)",
                                    tag="sp_an_Rg", default_value=346.2,
                                    format="%.3f")
                dpg.add_input_float(label="Tк (камера), К",
                                    tag="sp_an_Tk", default_value=3692.99,
                                    format="%.2f")
                dpg.add_input_float(label="α (справочно)",
                                    tag="sp_an_alpha", default_value=0.81,
                                    format="%.3f")
                dpg.add_input_float(label="φк (камера)",
                                    tag="sp_an_phik", default_value=0.99,
                                    format="%.4f")
                dpg.add_input_float(label="φс (сопло)",
                                    tag="sp_an_phic", default_value=0.98,
                                    format="%.4f")
                dpg.add_input_float(label="Wср (впрыск), м/с",
                                    tag="sp_an_winj", default_value=30.0,
                                    format="%.1f")
                dpg.add_input_float(label="ρ (скругление)",
                                    tag="sp_an_rho", default_value=2.0,
                                    format="%.2f")
                dpg.add_separator()
                dpg.add_button(label="Рассчитать", width=-1,
                               callback=self._on_analytic_compute)
                dpg.add_button(label="Из основного расчёта", width=-1,
                               callback=self._on_analytic_pull_from_main)
            with dpg.child_window(border=False, autosize_x=True,
                                  autosize_y=True):
                dpg.add_text("Задайте исходные данные слева и нажмите «Рассчитать».",
                             tag="txt_analytic", wrap=700)

    # ─── Обработчики UI ──────────────────────────────────────────────────

    def _on_solver_changed(self):
        own = dpg.get_value("rb_own")
        cea = dpg.get_value("rb_cea")
        if not own and not cea:
            dpg.set_value("rb_own", True)
            own = True
        if own and cea:
            dpg.set_value("rb_cea", False)
        self._solver = "cea" if cea else "own"

    @staticmethod
    def _enforce_radio(group_tags, clicked_tag=None):
        """Поведение «выбрать один из» для набора чекбоксов.

        Гарантирует, что ровно один чекбокс из group_tags включён.
        clicked_tag — тег чекбокса, по которому только что кликнули
        (если задан, именно он остаётся включённым).
        """
        checked = [t for t in group_tags
                   if dpg.does_item_exist(t) and dpg.get_value(t)]
        if clicked_tag is not None and dpg.get_value(clicked_tag):
            # Кликнули по чекбоксу и он стал включён → выключаем остальные
            for t in group_tags:
                if dpg.does_item_exist(t):
                    dpg.set_value(t, t == clicked_tag)
            return clicked_tag
        if not checked:
            # Запретили снять последний — возвращаем кликнутый (или первый)
            keep = clicked_tag if clicked_tag in group_tags else group_tags[0]
            dpg.set_value(keep, True)
            for t in group_tags:
                if dpg.does_item_exist(t) and t != keep:
                    dpg.set_value(t, False)
            return keep
        if len(checked) > 1:
            keep = checked[0]
            for t in checked[1:]:
                dpg.set_value(t, False)
            return keep
        return checked[0]

    def _on_chamber_size_mode_changed(self, clicked_tag=None):
        # Размер камеры сгорания: «выбрать один из» (длина / характ. L*)
        self._enforce_radio(["rb_chamber_len", "rb_chamber_lstar"], clicked_tag)

    def _on_fraction_mode_changed(self, clicked_tag=None):
        # Мольные / массовые доли: «выбрать один из»
        self._enforce_radio(["rb_mole", "rb_mass"], clicked_tag)
        self._refresh_species_view()

    def _on_calc_geom_type_changed(self, clicked_tag=None):
        # Тип сопла для оси X основного расчёта — единственная ось координат,
        # поэтому здесь «выбрать один из».
        self._enforce_radio(
            ["rb_calc_conical", "rb_calc_profiled", "rb_calc_rpa"], clicked_tag)
        if dpg.get_value("rb_calc_conical"):
            self._calc_geom_type, self._calc_use_rpa = "conical", False
        elif dpg.get_value("rb_calc_rpa"):
            self._calc_geom_type, self._calc_use_rpa = "rpa", True
        else:
            self._calc_geom_type, self._calc_use_rpa = "profiled", False

    def _update_overall_efficiency(self):
        try:
            eta_r = float(dpg.get_value("sp_eff_reaction"))
            eta_n = float(dpg.get_value("sp_eff_nozzle"))
        except Exception:
            return
        dpg.set_value("lbl_eff_overall", f"Суммарный ηобщ = {eta_r*eta_n:.4f}")

    def _on_plot_param_toggle(self, key, checked):
        if checked and key not in self._plot_keys:
            self._plot_keys.append(key)
        elif not checked and key in self._plot_keys:
            self._plot_keys.remove(key)
        self._redraw_plots()

    def _on_toggle_profile_1d(self):
        self._show_profile_1d = bool(dpg.get_value("chk_show_profile"))
        self._redraw_plots()

    # ─── Логика смеси / O/F ──────────────────────────────────────────────

    def _mix_mode(self) -> str:
        idx = dpg.get_value("cb_mix_mode") or ""
        if "α" in idx or "alpha" in idx.lower():
            return "alpha"
        if "оптимум" in idx.lower() or "optimum" in idx.lower():
            return "optimum"
        return "km"

    @staticmethod
    def _get_float_field(tag) -> Optional[float]:
        try:
            txt = (dpg.get_value(tag) or "").strip().replace(",", ".")
        except Exception:
            return None
        if not txt:
            return None
        try:
            val = float(txt)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val <= 0:
            return None
        return val

    def _get_mix_value(self) -> Optional[float]:
        return self._get_float_field("ed_mix_value")

    def _compute_km0(self) -> float:
        if not self.species_db:
            return float("nan")
        mixture = self.mixture_widget.get_mixture() if self.mixture_widget else {}
        ox_names = [c["name"] for c in mixture.get("ox_components", []) if c.get("name")]
        fu_names = [c["name"] for c in mixture.get("fuel_components", []) if c.get("name")]
        if not ox_names or not fu_names:
            return float("nan")
        try:
            oxidizers = [self.species_db[n] for n in ox_names if n in self.species_db]
            fuels = [self.species_db[n] for n in fu_names if n in self.species_db]
            if not oxidizers or not fuels:
                return float("nan")
            return stoichiometric_OF(oxidizers, fuels)
        except Exception:
            return float("nan")

    def _resolve_of_ratio(self) -> Optional[float]:
        mode = self._mix_mode()
        if mode == "optimum":
            return None
        val = self._get_mix_value()
        if val is None:
            return None
        if mode == "km":
            return val
        km0 = self._compute_km0()
        if not (math.isfinite(km0) and km0 > 0):
            return None
        return val * km0

    def _update_of_from_mixture(self):
        mode = self._mix_mode()
        km0 = self._compute_km0()
        km0_str = f"{km0:.4f}" if math.isfinite(km0) else "—"
        if mode == "optimum":
            dpg.set_value("lbl_of", f"Km0 = {km0_str}  (Km → max Isp)")
            return
        val = self._get_mix_value()
        if val is None:
            dpg.set_value("lbl_of", f"Km0 = {km0_str}")
        elif mode == "km" and math.isfinite(km0) and km0 > 0:
            dpg.set_value("lbl_of",
                          f"Km = {val:.4f}, α = {val/km0:.4f}  (Km0 = {km0_str})")
        elif mode == "alpha" and math.isfinite(km0) and km0 > 0:
            dpg.set_value("lbl_of",
                          f"α = {val:.4f}, Km = {val*km0:.4f}  (Km0 = {km0_str})")
        else:
            dpg.set_value("lbl_of", f"Km0 = {km0_str}")

    def _on_mix_mode_changed(self):
        mode = self._mix_mode()
        if mode == "optimum":
            dpg.set_value("ed_mix_value", "")
            dpg.configure_item("ed_mix_value", hint="подбирается автоматически")
        else:
            hint = "Km (O/F)" if mode == "km" else "α (Km/Km0)"
            dpg.configure_item("ed_mix_value", hint=hint)
        self._update_of_from_mixture()

    # ─── Длина камеры / геометрия для оси X ──────────────────────────────

    def _chamber_length_m(self) -> float:
        use_lstar = dpg.get_value("rb_chamber_lstar") if dpg.does_item_exist("rb_chamber_lstar") else False
        if not use_lstar:
            return float(dpg.get_value("sp_L_chamber") or 0.1)
        L_star = float(dpg.get_value("sp_L_star") or 1.0)
        try:
            rcham_ratio = max(float(dpg.get_value("sp_calc_Rcham") or 1.0), 1.0)
        except Exception:
            rcham_ratio = 1.0
        return L_star / (rcham_ratio * rcham_ratio)

    def _auto_conv_div_lengths(self) -> tuple:
        try:
            R_throat = max(float(dpg.get_value("sp_calc_Rthroat") or 0.05), 1e-4)
        except Exception:
            R_throat = 0.05
        try:
            rcham_ratio = max(float(dpg.get_value("sp_calc_Rcham") or 3.0), 1.05)
        except Exception:
            rcham_ratio = 3.0
        R_cham = rcham_ratio * R_throat
        area_ratio = 6.0
        if self.perf is not None and getattr(self.perf, "stations", None):
            try:
                ar = float(self.perf.stations[-1].Ae_At)
                if math.isfinite(ar) and ar > 1.0:
                    area_ratio = ar
            except Exception:
                pass
        R_exit = R_throat * math.sqrt(area_ratio)
        theta_in = math.radians(max(5.0, float(dpg.get_value("sp_calc_theta_in") or 30.0)))
        theta_exit = math.radians(max(3.0, float(dpg.get_value("sp_calc_theta_exit") or 15.0)))
        L_conv = (R_cham - R_throat) / math.tan(theta_in) if theta_in > 0 else 0.0
        L_div = (R_exit - R_throat) / math.tan(theta_exit) if theta_exit > 0 else 0.0
        return max(L_conv, 1e-4), max(L_div, 1e-4)

    def _build_calc_geometry(self, perf):
        """Строит геометрию по выбранному типу."""
        try:
            ar = float(perf.stations[-1].Ae_At)
            if not (math.isfinite(ar) and ar > 1.0):
                return None
            R_throat = max(float(dpg.get_value("sp_calc_Rthroat") or 0.05), 1e-4)
            R_cham = float(dpg.get_value("sp_calc_Rcham") or 2.5) * R_throat
            self._on_calc_geom_type_changed()
            if self._calc_use_rpa:
                return build_rpa_parabolic_nozzle(
                    R_throat, ar, R_chamber_m=R_cham,
                    contraction_angle_deg=30.0,
                    R1_over_Rt=1.5, Rn_over_Rt=0.382,
                    R2_over_R2max=0.5)
            if self._calc_geom_type == "conical":
                return build_conical_nozzle(
                    R_throat, ar, R_chamber_m=R_cham,
                    theta_exit_deg=float(dpg.get_value("sp_calc_theta_exit") or 15.0),
                    theta_in_deg=float(dpg.get_value("sp_calc_theta_in") or 30.0))
            auto = dpg.get_value("chk_calc_auto_angles")
            return build_profiled_nozzle(
                R_throat, ar, R_chamber_m=R_cham,
                theta_exit_deg=(None if auto else float(dpg.get_value("sp_calc_theta_exit") or 15.0)),
                theta_max_deg=(None if auto else float(dpg.get_value("sp_calc_theta_max") or 30.0)),
                length_ratio=(None if auto else float(dpg.get_value("sp_calc_len_ratio") or 9.5)),
                theta_in_deg=float(dpg.get_value("sp_calc_theta_in") or 30.0))
        except Exception:
            return None

    def _series_geometry(self):
        """Геометрия выбранного типа для согласования профиля и параметров.

        Источник оси X и профиля r/R_кр в section_series, чтобы кривые
        газодинамики и наложенный профиль соответствовали тому же контуру,
        что и на вкладке «Геометрия». None → обобщённая разбивка.
        """
        if self.perf is None:
            return None
        try:
            return self._build_calc_geometry(self.perf)
        except Exception:
            return None

    # ─── Расчёт ──────────────────────────────────────────────────────────

    def on_calculate(self):
        if self.mixture_widget is None:
            return
        mixture = self.mixture_widget.get_mixture()
        if not mixture.get("ox_components") or not mixture.get("fuel_components"):
            dpg.set_value("status_text",
                          "Укажите хотя бы один компонент окислителя и горючего.")
            return

        mix_mode = self._mix_mode()
        optimize_of = (mix_mode == "optimum")
        of_ratio = self._resolve_of_ratio()
        P_chamber = self._get_float_field("ed_Pc")
        P_exit = self._get_float_field("ed_Pe")

        def pv_to_pa(val, unit):
            if unit == "Па": return val
            if unit == "кПа": return val * 1e3
            if unit == "МПа": return val * 1e6
            if unit == "бар": return val * 1e5
            if unit == "атм": return val * 101325.0
            return val

        missing = []
        if P_chamber is None:
            missing.append("давление в камере (Pк)")
        if P_exit is None:
            missing.append("давление на срезе (Pс)")
        if not optimize_of and of_ratio is None:
            missing.append("соотношение компонентов")
        if missing:
            dpg.set_value("status_text",
                          "Не заданы: " + ", ".join(missing))
            return

        params = {
            "ox_components": mixture["ox_components"],
            "fuel_components": mixture["fuel_components"],
            "of_ratio": of_ratio if of_ratio is not None else 1.0,
            "optimize_of": optimize_of,
            "of_stoich": self._compute_km0(),
            "P_chamber": pv_to_pa(P_chamber, dpg.get_value("cb_Pc_unit")),
            "P_exit": pv_to_pa(P_exit, dpg.get_value("cb_Pe_unit")),
            "n_inter": int(dpg.get_value("sp_n_inter") or 8),
            "include_condensed": bool(dpg.get_value("chk_condensed")),
            "injection_velocity": float(dpg.get_value("sp_inj_velocity") or 0.0),
            "chamber_pressure_drop_frac": float(dpg.get_value("sp_chamber_dp") or 0.0) / 100.0,
            "precision": PRECISION_MAP.get(dpg.get_value("cb_precision"), "balanced"),
        }

        self._solver = "cea" if dpg.get_value("rb_cea") else "own"
        if self._solver == "own" and self.species_db is None:
            dpg.set_value("status_text",
                          "База NASA-9 ещё не загружена. Подождите 1-2 секунды.")
            return

        dpg.configure_item("btn_calc", enabled=False)
        dpg.set_value("status_text", f"Расчёт ({self._solver})... подождите.")
        dpg.set_value("progress_text", "⏳ Выполняется расчёт...")
        dpg.set_value("iter_text", "")
        of_desc = f"of={of_ratio:.3f}" if of_ratio is not None else "optimize"
        ActionLogger.info(
            "Расчёт запущен",
            solver=self._solver,
            P_chamber_Mpa=f"{params['P_chamber']/1e6:.3f}",
            P_exit_Mpa=f"{params['P_exit']/1e6:.3f}",
            of=of_desc,
        )

        self.worker = NozzleSolverWorker(params, self._solver, self.species_db)
        self.worker.start()

    def _poll_worker(self):
        """Опрос результатов фонового расчёта (вызывается каждый кадр)."""
        if self.worker is None:
            return
        while True:
            msg = self.worker.poll()
            if msg is None:
                break
            if msg["type"] == "progress":
                m = msg['msg']
                if "·" in m:
                    stage, detail = m.split("·", 1)
                    dpg.set_value("progress_text", f"⏳ {stage.strip()}")
                    dpg.set_value("iter_text", detail.strip())
                else:
                    dpg.set_value("progress_text", f"⏳ {m}")
                    dpg.set_value("iter_text", "")
            elif msg["type"] == "ok":
                try:
                    self._on_calc_done(msg["perf"])
                except Exception as e:
                    ActionLogger.error("Краш в _on_calc_done", detail=str(e))
                    import traceback
                    ActionLogger.error("Traceback", detail=traceback.format_exc())
                    dpg.set_value("status_text", f"Ошибка отображения: {e}")
                self.worker = None
                break
            elif msg["type"] == "error":
                self._on_calc_failed(msg["msg"])
                self.worker = None
                break

    def _on_calc_done(self, perf: RocketPerformance):
        self.perf = perf
        dpg.configure_item("btn_calc", enabled=True)
        dpg.set_value("progress_text", "")
        dpg.set_value("iter_text", "")
        st0 = perf.stations[0]
        dpg.set_value("status_text",
                      f"Готово. Tкамеры = {st0.T_K:.1f} К, "
                      f"Isp = {perf.Isp_s:.2f} с, "
                      f"C* = {perf.Cstar_m_per_s:.1f} м/с")
        ActionLogger.info(
            "Расчёт завершён",
            T_chamber_K=f"{st0.T_K:.1f}",
            Isp_s=f"{perf.Isp_s:.2f}",
            Cstar_ms=f"{perf.Cstar_m_per_s:.1f}",
            O_F=f"{perf.O_F:.4f}",
            stations_count=len(perf.stations),
        )
        try:
            ActionLogger.info("Шаг 1: таблица станций")
            self._fill_stations_table(perf)
            ActionLogger.info("Шаг 2: текст характеристик")
            self._fill_perf_text(perf)
            ActionLogger.info("Шаг 3: состав")
            self._refresh_species_view()
            ActionLogger.info("Шаг 4: графики")
            self._redraw_plots()
            ActionLogger.info("Все шаги отображения завершены")
        except Exception as e:
            ActionLogger.error("Ошибка отображения результатов", detail=str(e))
            import traceback
            ActionLogger.error("Traceback", detail=traceback.format_exc())
            dpg.set_value("status_text", f"Ошибка отображения: {e}")

    def _on_calc_failed(self, msg: str):
        dpg.configure_item("btn_calc", enabled=True)
        dpg.set_value("progress_text", "")
        dpg.set_value("iter_text", "")
        dpg.set_value("status_text", "Ошибка расчёта.")
        dpg.set_value("txt_perf", f"Ошибка расчёта:\n{msg[:2000]}")
        ActionLogger.error("Расчёт завершился ошибкой", detail=msg[:500])

    # ─── Заполнение таблиц ───────────────────────────────────────────────

    def _fill_stations_table(self, perf: RocketPerformance):
        ActionLogger.info("Заполнение таблицы станций")
        stations = perf.stations
        # Очищаем таблицу (DPG 2.x: children — dict, удаляем безопасно)
        if dpg.does_item_exist("tbl_stations"):
            children = dpg.get_item_info("tbl_stations").get("children", {})
            if isinstance(children, dict):
                for child_list in children.values():
                    for child in child_list:
                        dpg.delete_item(child)
            dpg.delete_item("tbl_stations")
        # Перестраиваем колонки: параметр + по станциям + ед.изм.
        # (DPG не поддерживает динамическое добавление колонок в существующую
        #  таблицу — пересоздаём таблицу целиком.)
        # Фиксированная ширина колонок + горизонтальная прокрутка: при
        # большом числе сечений таблица прокручивается внутри своего
        # контейнера (stations_scroll), а не растягивает всю вкладку.
        with dpg.table(tag="tbl_stations", header_row=True,
                       resizable=True, policy=dpg.mvTable_SizingFixedFit,
                       scrollX=True, scrollY=False,
                       parent=self._stations_parent()):
            dpg.add_table_column(label="Параметр",
                                 init_width_or_weight=160)
            for s in stations:
                dpg.add_table_column(label=s.label,
                                     init_width_or_weight=110)
            dpg.add_table_column(label="Ед.изм.",
                                 init_width_or_weight=110)
            params = [
                ("Давление", lambda s: f"{s.P_Pa/1e6:.4f}", "МПа"),
                ("Температура", lambda s: f"{s.T_K:.4f}", "К"),
                ("Энтальпия", lambda s: f"{s.H_J_per_kg/1000:.4f}", "кДж/кг"),
                ("Энтропия", lambda s: f"{s.S_J_per_kgK/1000:.4f}", "кДж/(кг·К)"),
                ("γ (eq.)", lambda s: f"{s.gamma_eq:.4f}", ""),
                ("γs (изэнтр.)", lambda s: f"{s.gamma_s:.4f}", ""),
                ("Газовая пост.", lambda s: f"{s.R_specific_J_per_kgK/1000:.4f}", "кДж/(кг·К)"),
                ("Молярная масса", lambda s: f"{s.mw_g_per_mol:.4f}", "кг/кмоль"),
                ("Плотность", lambda s: f"{s.rho_kg_per_m3:.4f}", "кг/м³"),
                ("Скорость звука", lambda s: f"{s.a_m_per_s:.4f}", "м/с"),
                ("Скорость потока", lambda s: f"{s.V_m_per_s:.4f}", "м/с"),
                ("Число Маха", lambda s: f"{s.M:.4f}", ""),
                ("Ae/At", lambda s: ("∞" if (not math.isfinite(s.Ae_At) or s.Ae_At > 1e5) else f"{s.Ae_At:.4f}"), ""),
            ]
            for name, fn, unit in params:
                with dpg.table_row():
                    dpg.add_text(name)
                    for s in stations:
                        dpg.add_text(fn(s))
                    dpg.add_text(unit)

    def _stations_parent(self) -> str:
        """Родительский тег для таблицы станций (прокручиваемый контейнер)."""
        return "stations_scroll" if dpg.does_item_exist("stations_scroll") else "tab_stations"

    def _fill_perf_text(self, perf: RocketPerformance):
        ActionLogger.info("Заполнение текста тяговых характеристик")
        lines = []
        lines.append("═" * 70)
        lines.append("  ТЯГОВЫЕ ХАРАКТЕРИСТИКИ")
        lines.append("═" * 70)
        lines.append("")
        lines.append(f"  Массовое O/F:         {perf.O_F:.4f}")
        if not math.isnan(perf.O_F_stoich):
            lines.append(f"  Стехиометр. O/F:      {perf.O_F_stoich:.4f}")
        if not math.isnan(perf.alpha):
            lines.append(f"  α (избыток окисл.):   {perf.alpha:.4f}")
        lines.append("")
        lines.append(f"  Давление в камере:    {perf.stations[0].P_Pa/1e6:.4f} МПа")
        lines.append(f"  Давление на срезе:    {perf.stations[-1].P_Pa/1e6:.4f} МПа")
        lines.append(f"  Геометрич. степень:   Ae/At = {perf.stations[-1].Ae_At:.4f}")
        lines.append("")
        lines.append("─" * 70)
        lines.append(f"  Isp (срез):           {perf.Isp_s:8.4f} с")
        lines.append(f"  Isp (вакуум):         {perf.Isp_vac_s:8.4f} с")
        lines.append(f"  C* (характер.):       {perf.Cstar_m_per_s:8.4f} м/с")
        lines.append(f"  CF (коэф. тяги):      {perf.CF:8.4f}")
        lines.append(f"  Ve (скор. на срезе):  {perf.stations[-1].V_m_per_s:8.4f} м/с")
        lines.append("")
        st_c, st_e = perf.stations[0], perf.stations[-1]
        lines.append("═" * 70)
        lines.append("  ПАРАМЕТРЫ В КАМЕРЕ И НА СРЕЗЕ")
        lines.append("═" * 70)
        lines.append(f"  T_камеры:  {st_c.T_K:8.2f} К   |  T_выход:  {st_e.T_K:8.2f} К")
        lines.append(f"  ρ_камеры:  {st_c.rho_kg_per_m3:8.4f}    |  ρ_выход:  {st_e.rho_kg_per_m3:8.4f}")
        lines.append(f"  M_камеры:  {st_c.M:8.4f}     |  M_выход:  {st_e.M:8.4f}")
        lines.append("")
        dpg.set_value("txt_perf", "\n".join(lines))

    def _get_composition_station_indices(self, stations):
        target_labels = ["injector", "nozzle inlet", "nozzle throat", "nozzle exit"]
        idx_by_label = {str(st.label).strip().lower(): i for i, st in enumerate(stations)}
        indices = []
        for lbl in target_labels:
            idx = idx_by_label.get(lbl)
            if idx is not None and idx not in indices:
                indices.append(idx)
        if not indices and stations:
            indices = [0, len(stations) - 1]
        return sorted(indices)

    def _refresh_species_view(self):
        ActionLogger.info("Обновление таблицы состава")
        if self.perf is None:
            ActionLogger.warning("_refresh_species_view: perf is None")
            return
        stations = self.perf.stations
        comp_idx = self._get_composition_station_indices(stations)
        comp_stations = [stations[i] for i in comp_idx]
        if not comp_stations:
            return
        sp_names = comp_stations[0].species_names
        N = len(sp_names)
        use_mole = dpg.get_value("rb_mole") if dpg.does_item_exist("rb_mole") else True
        topN = int(dpg.get_value("sp_topN") or 15)
        max_frac = np.zeros(N)
        for st in comp_stations:
            frac = st.mole_fractions if use_mole else st.mass_fractions
            if frac is not None and len(frac) == N:
                max_frac = np.maximum(max_frac, frac)
        order = np.argsort(-max_frac)[:topN]
        order = [i for i in order if max_frac[i] > 1e-9]

        # Перестроить таблицу
        if dpg.does_item_exist("tbl_species"):
            dpg.delete_item("tbl_species")
        with dpg.table(tag="tbl_species", header_row=True,
                       resizable=True, policy=dpg.mvTable_SizingStretchProp,
                       parent="tab_species_container"):
            dpg.add_table_column(label="Компонент")
            for st in comp_stations:
                dpg.add_table_column(label=st.label)
            for idx in order:
                with dpg.table_row():
                    dpg.add_text(sp_names[idx])
                    for st in comp_stations:
                        frac = st.mole_fractions if use_mole else st.mass_fractions
                        v = frac[idx] if frac is not None and idx < len(frac) else 0.0
                        dpg.add_text(f"{v:.6e}")

    # ─── Графики (DPG plot) ──────────────────────────────────────────────

    def _collect_style(self) -> dict:
        return {
            "lw": float(dpg.get_value("sp_lw") or 1.8),
            "markers": bool(dpg.get_value("chk_markers")),
            "smooth": bool(dpg.get_value("chk_smooth")),
            "grid_major": bool(dpg.get_value("chk_grid_major")),
            "grid_minor": bool(dpg.get_value("chk_grid_minor")),
            "dark": bool(dpg.get_value("chk_dark_plot")),
        }

    def _plot_columns(self) -> int:
        """Число колонок графиков по выбору пользователя (combo cb_plot_cols)."""
        mode = dpg.get_value("cb_plot_cols") if dpg.does_item_exist("cb_plot_cols") else "Авто"
        if mode == "1 колонка":
            return 1
        if mode == "2 колонки":
            return 2
        # «Авто»: 2 колонки, если графиков больше трёх, иначе 1.
        n = len([k for k in self._plot_keys])
        return 2 if n > 3 else 1

    def _redraw_plots(self):
        ActionLogger.info("Перерисовка графиков")
        if self.perf is None:
            ActionLogger.warning("_redraw_plots: perf is None")
            return
        ser = section_series(self.perf, self._chamber_length_m(),
                             self._auto_conv_div_lengths(),
                             geometry=self._series_geometry())
        if not ser:
            return
        # Очищаем контейнер графиков
        grp = "plots_group"
        if dpg.does_item_exist(grp):
            children = dpg.get_item_info(grp).get("children", {})
            if isinstance(children, dict):
                for child_list in children.values():
                    for child in child_list:
                        dpg.delete_item(child)
        # Чистим темы графиков прошлой перерисовки (они живут в корне и не
        # удаляются вместе с графиками — иначе утечка и деградация рендера).
        for tid in self._plot_theme_ids:
            try:
                if dpg.does_item_exist(tid):
                    dpg.delete_item(tid)
            except Exception:
                pass
        self._plot_theme_ids = []

        x = ser["x_m"]
        keys = [k for k in self._plot_keys if plot_param_value(k, ser) is not None]
        if not keys:
            dpg.add_text("Выберите параметры для отображения",
                         parent=grp, color=C_MUTED)
            return
        dpg.set_value("plot_hint", "")

        style = self._collect_style()
        lw = max(0.5, style["lw"])
        row_h = int(self._plot_row_h)
        ncols = self._plot_columns()
        show_profile = bool(self._show_profile_1d)
        r_rel = ser.get("r_rel")
        # Профиль рисуем (нормированный r/r_кр) только если есть валидные данные.
        if r_rel is None or not np.any(np.isfinite(np.asarray(r_rel, dtype=float))):
            show_profile = False

        # Раскладка по колонкам: размещаем графики в строки по ncols штук.
        # Каждая строка — горизонтальная группа; ширина каждого графика
        # делится поровну (-1 «на всю оставшуюся ширину» внутри группы).
        # Ширина каждого графика: если задан сплиттер ширины
        # (self._plot_col_w > 0) — фиксированные px одинаково для всех;
        # иначе старое поведение (1 колонка → -1, иначе делить поровну).
        col_w = int(getattr(self, "_plot_col_w", 0) or 0)
        if col_w > 0:
            plot_w = col_w
        elif ncols <= 1:
            plot_w = -1
        else:
            # 2+ колонки без ручного сплиттера: явно делим доступную ширину
            # контейнера поровну. Передавать width=0 нельзя — Dear PyGui
            # отрисует график нулевой ширины, и добавленные графики «не видны».
            plot_w = self._even_split_plot_width(ncols)

        rows = [keys[i:i + ncols] for i in range(0, len(keys), ncols)]
        for r_i, row_keys in enumerate(rows):
            row_tag = f"plot_row_{r_i}"
            with dpg.group(parent=grp, tag=row_tag, horizontal=(ncols > 1)):
                for c_i, key in enumerate(row_keys):
                    self._draw_single_plot(key, ser, x, style, lw, row_h,
                                           plot_w, show_profile, r_rel)
                    # Вертикальный сплиттер ширины между колонками
                    # (только при 2+ колонках и не после последнего графика).
                    if ncols > 1 and c_i < len(row_keys) - 1:
                        wsplit_tag = f"plot_wsplit_{r_i}_{c_i}"
                        dpg.add_button(tag=wsplit_tag, label="|",
                                       width=8, height=row_h)
                        with dpg.theme() as ws_theme:
                            with dpg.theme_component(dpg.mvButton):
                                dpg.add_theme_color(dpg.mvThemeCol_Button,
                                                    C_BORDER)
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_ButtonHovered, C_ACCENT)
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_ButtonActive, C_ACCENT_DARK)
                                dpg.add_theme_color(dpg.mvThemeCol_Text,
                                                    C_MUTED)
                        dpg.bind_item_theme(wsplit_tag, ws_theme)
                        self._plot_theme_ids.append(ws_theme)

    def _draw_single_plot(self, key, ser, x, style, lw, row_h, plot_w,
                          show_profile, r_rel):
        """Отрисовка одного графика газодинамического параметра.

        ``plot_w`` — ширина графика в px (или -1/0 для авто-доли строки),
        вычисляется в _redraw_plots с учётом сплиттера ширины.
        """
        label, unit, color = next((l, u, c) for k, l, u, c in PLOT_PARAM_DEFS if k == key)
        y = plot_param_value(key, ser)
        if y is None:
            return
        plot_tag = f"plot_{key}"
        x_axis = f"plot_{key}_x"
        y_axis = f"plot_{key}_y"
        y2_axis = f"plot_{key}_y2"
        # Флаги сеток: основная/доп. реализуются через no_gridlines у осей и
        # толщину линий (MajorGridSize / MinorGridSize) в теме плота.
        no_grid = not (style["grid_major"] or style["grid_minor"])
        # Плот добавляется в текущий активный контейнер (горизонтальную
        # группу строки), открытый в _redraw_plots через `with dpg.group(...)`.
        with dpg.plot(tag=plot_tag,
                      label=label + (f", {unit}" if unit else ""),
                      height=row_h, width=plot_w):
            # Тема плота: управляем толщиной основной/доп. сетки.
            with dpg.theme() as plot_theme:
                with dpg.theme_component(dpg.mvPlot):
                    major = 1.2 if style["grid_major"] else 0.0
                    minor = 0.7 if style["grid_minor"] else 0.0
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MajorGridSize,
                                        major, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MinorGridSize,
                                        minor, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(plot_tag, plot_theme)
            self._plot_theme_ids.append(plot_theme)

            dpg.add_plot_axis(dpg.mvXAxis, label="x, м", tag=x_axis,
                              no_gridlines=no_grid)
            dpg.add_plot_axis(dpg.mvYAxis,
                              label=(unit if unit else label), tag=y_axis,
                              no_gridlines=no_grid)
            dpg.set_axis_limits(y_axis,
                                ymin=float(np.nanmin(y)),
                                ymax=float(np.nanmax(y)))
            # Theme for line series with color and weight
            line_series_tag = f"ls_{key}"
            scatter_series_tag = f"ss_{key}"
            dpg.add_line_series(list(x), list(y),
                                parent=y_axis, tag=line_series_tag)
            with dpg.theme() as line_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight,
                                        float(lw),
                                        category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(line_series_tag, line_theme)
            self._plot_theme_ids.append(line_theme)
            if style["markers"]:
                dpg.add_scatter_series(list(x), list(y),
                                       parent=y_axis, tag=scatter_series_tag)
                with dpg.theme() as scatter_theme:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color,
                                            category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color,
                                            category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(scatter_series_tag, scatter_theme)
                self._plot_theme_ids.append(scatter_theme)
            if key == "M":
                hl_tag = f"hl_M_{key}"
                try:
                    dpg.add_inf_line_series([1.0], parent=y_axis, tag=hl_tag, horizontal=True)
                except Exception as e_inf:
                    ActionLogger.warning("add_inf_line_series (M) failed", detail=str(e_inf))
                    hl_tag = None
                if hl_tag:
                    with dpg.theme() as hl_theme:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, C_MUTED,
                                                category=dpg.mvThemeCat_Plots)
                    dpg.bind_item_theme(hl_tag, hl_theme)
                    self._plot_theme_ids.append(hl_theme)
            x_thr = ser.get("x_throat_m")
            if x_thr is not None:
                vl_tag = f"vl_throat_{key}"
                try:
                    dpg.add_inf_line_series([x_thr], parent=y_axis, tag=vl_tag, horizontal=False)
                except Exception as e_inf:
                    ActionLogger.warning("add_inf_line_series (throat) failed", detail=str(e_inf))
                    vl_tag = None
                if vl_tag:
                    with dpg.theme() as vl_theme:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, C_MUTED,
                                                category=dpg.mvThemeCat_Plots)
                    dpg.bind_item_theme(vl_tag, vl_theme)
                    self._plot_theme_ids.append(vl_theme)

            # ── Наложение профиля сопла (r/r_кр) на отдельной правой оси ──
            if show_profile and r_rel is not None:
                try:
                    rr = np.asarray(r_rel, dtype=float)
                    dpg.add_plot_axis(dpg.mvYAxis2, label="r/rкр",
                                      tag=y2_axis, no_gridlines=True,
                                      opposite=True)
                    rmax = float(np.nanmax(rr[np.isfinite(rr)])) if np.any(np.isfinite(rr)) else 1.0
                    dpg.set_axis_limits(y2_axis, ymin=0.0, ymax=max(rmax * 1.15, 1e-6))
                    prof_top = f"prof_top_{key}"
                    prof_bot = f"prof_bot_{key}"
                    dpg.add_line_series(list(x), list(rr),
                                        parent=y2_axis, tag=prof_top)
                    dpg.add_line_series(list(x), list(-rr),
                                        parent=y2_axis, tag=prof_bot)
                    with dpg.theme() as prof_theme:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line,
                                                (130, 130, 128, 160),
                                                category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight,
                                                1.5,
                                                category=dpg.mvThemeCat_Plots)
                    dpg.bind_item_theme(prof_top, prof_theme)
                    dpg.bind_item_theme(prof_bot, prof_theme)
                    self._plot_theme_ids.append(prof_theme)
                except Exception as e_prof:
                    ActionLogger.warning("Наложение профиля сопла не удалось",
                                         detail=str(e_prof))

            dpg.fit_axis_data(x_axis)
            dpg.fit_axis_data(y_axis)
            if show_profile and dpg.does_item_exist(y2_axis):
                dpg.fit_axis_data(y2_axis)

    def _save_figures(self):
        """Сохранение графиков через matplotlib (экспорт в PNG)."""
        if self.perf is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            dpg.set_value("status_text", "matplotlib недоступен для экспорта.")
            return
        ser = section_series(self.perf, self._chamber_length_m(),
                             self._auto_conv_div_lengths(),
                             geometry=self._series_geometry())
        if not ser:
            return
        # Простой экспорт: все выбранные графики в один PNG
        keys = [k for k in self._plot_keys if plot_param_value(k, ser) is not None]
        if not keys:
            return
        x = ser["x_m"]
        n = len(keys)
        ncols = 2 if n > 1 else 1
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4*nrows),
                                 dpi=150)
        if n == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        for i, key in enumerate(keys):
            row, col = i // ncols, i % ncols
            ax = axes[row][col]
            label, unit, color = next((l, u, c) for k, l, u, c in PLOT_PARAM_DEFS if k == key)
            y = plot_param_value(key, ser)
            ax.plot(x, y, '-', color=color, lw=1.8)
            ax.set_title(label + (f", {unit}" if unit else ""))
            ax.set_xlabel("x, м")
            ax.grid(True, alpha=0.3)
        # Скрываем лишние подграфики
        for i in range(n, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row][col].set_visible(False)
        fig.tight_layout()
        path = os.path.join(os.path.expanduser("~"), "nozzle_plots.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        dpg.set_value("status_text", f"Рисунок сохранён: {path}")

    # ─── Геометрия ───────────────────────────────────────────────────────

    # Цвета профилей сопла на графике геометрии (по типу)
    GEOM_TYPE_COLORS = {
        "conical": (106, 176, 255),   # синий
        "profiled": (204, 120, 92),   # акцент (оранжевый)
        "rpa": (130, 210, 122),       # зелёный
    }
    GEOM_TYPE_LABELS = {
        "conical": "Коническое",
        "profiled": "Профилированное",
        "rpa": "RPA (bell)",
    }

    def _build_single_geometry(self, geom_type, R_throat, ar, R_cham):
        """Строит одну геометрию заданного типа."""
        if geom_type == "conical":
            return build_conical_nozzle(
                R_throat, ar, R_chamber_m=R_cham,
                theta_exit_deg=float(dpg.get_value("sp_geom_theta_exit") or 15.0),
                theta_in_deg=float(dpg.get_value("sp_geom_theta_in") or 30.0))
        if geom_type == "rpa":
            return build_rpa_parabolic_nozzle(
                R_throat, ar, R_chamber_m=R_cham,
                contraction_angle_deg=30.0)
        return build_profiled_nozzle(
            R_throat, ar, R_chamber_m=R_cham,
            theta_exit_deg=float(dpg.get_value("sp_geom_theta_exit") or 15.0),
            theta_in_deg=float(dpg.get_value("sp_geom_theta_in") or 30.0))

    def on_build_geometry(self):
        # Какие типы отмечены (можно несколько)
        selected = []
        if dpg.get_value("rb_geom_conical"):
            selected.append("conical")
        if dpg.get_value("rb_geom_profiled"):
            selected.append("profiled")
        if dpg.get_value("rb_geom_rpa"):
            selected.append("rpa")
        if not selected:
            dpg.set_value("txt_geom_summary",
                          "Выберите хотя бы один тип сопла (галочкой).")
            self._last_geometries = {}
            self._last_geometry = None
            self._render_geometry({})
            return
        try:
            R_throat = float(dpg.get_value("sp_geom_Rthroat") or 0.05)
            ar = float(dpg.get_value("sp_geom_AR") or 16.0)
            R_cham = float(dpg.get_value("sp_geom_Rcham_factor") or 2.5) * R_throat
            geometries = {}
            errors = {}
            for gtype in selected:
                try:
                    geometries[gtype] = self._build_single_geometry(
                        gtype, R_throat, ar, R_cham)
                except Exception as e:
                    errors[gtype] = str(e)
            self._last_geometries = geometries
            # Совместимость: «текущая» геометрия — первая успешная
            self._last_geometry = next(iter(geometries.values()), None)
            self._render_geometry(geometries)
            self._update_geometry_summary(geometries, errors)
        except Exception as e:
            dpg.set_value("txt_geom_summary", f"Ошибка: {e}")

    def _render_geometry(self, geometries):
        """Отрисовка одного или нескольких профилей сопла.

        geometries — dict {geom_type: NozzleGeometry}. Каждый профиль рисуется
        своим цветом (верхняя и нижняя ветви контура).
        """
        # Очищаем plot (все серии оси Y)
        if dpg.does_item_exist("geom_y"):
            children = dpg.get_item_info("geom_y").get("children", {})
            series = children.get(1, []) if isinstance(children, dict) else []
            for child in list(series):
                dpg.delete_item(child)
        if not geometries:
            return
        for gtype, geom in geometries.items():
            if geom is None:
                continue
            x_arr, r_arr = geom.as_xy_arrays()
            color = self.GEOM_TYPE_COLORS.get(gtype, C_ACCENT)
            label = self.GEOM_TYPE_LABELS.get(gtype, gtype)
            ls_pos = dpg.add_line_series(list(x_arr), list(r_arr),
                                         parent="geom_y", label=label)
            ls_neg = dpg.add_line_series(list(x_arr), list(-np.asarray(r_arr)),
                                         parent="geom_y")
            # ВАЖНО: цвет линии серии в ImPlot применяется только с
            # категорией mvThemeCat_Plots. Без неё серия (и маркер в
            # легенде) берёт цвет из дефолтной палитры ImPlot — отсюда
            # расхождение легенды и линий. Цвет в легенде = цвет линии той
            # серии, у которой задан label (положительная ветвь).
            with dpg.theme() as geom_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0,
                                        category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(ls_pos, geom_theme)
            dpg.bind_item_theme(ls_neg, geom_theme)
        dpg.fit_axis_data("geom_x")
        dpg.fit_axis_data("geom_y")

    def _update_geometry_summary(self, geometries, errors=None):
        errors = errors or {}
        blocks = []
        for gtype, geom in geometries.items():
            if geom is None:
                continue
            label = self.GEOM_TYPE_LABELS.get(gtype, gtype)
            s = [f"━━ {label} ━━"]
            s.append(f"Тип: {geom.method}")
            s.append(f"Rкр = {geom.R_throat_m*1e3:.3f} мм")
            s.append(f"Ra = {geom.R_exit_m*1e3:.3f} мм")
            s.append(f"Fa/Fкр = {geom.area_ratio:.4f}")
            s.append(f"θa = {geom.theta_exit_deg:.2f}°")
            s.append(f"φрас = {geom.phi_dispersion:.4f}")
            s.append(f"Длина полная = {geom.length_total_m*1e3:.2f} мм")
            blocks.append("\n".join(s))
        for gtype, err in errors.items():
            label = self.GEOM_TYPE_LABELS.get(gtype, gtype)
            blocks.append(f"━━ {label} ━━\nОшибка: {err}")
        dpg.set_value("txt_geom_summary",
                      "\n\n".join(blocks) if blocks else "Нет данных.")

    def on_geometry_from_perf(self):
        if self.perf is None:
            ActionLogger.warning("Перенос геометрии из расчёта прерван — нет расчёта")
            return
        ActionLogger.info("Геометрия из расчёта")
        try:
            ar = float(self.perf.stations[-1].Ae_At)
            if math.isfinite(ar) and ar > 1.0:
                dpg.set_value("sp_geom_AR", ar)
                dpg.set_value("status_text",
                              f"Степень расширения Fa/Fкр = {ar:.3f} взята из расчёта")
        except Exception:
            pass

    def on_export_geometry_csv(self):
        geometries = self._last_geometries or (
            {"profiled": self._last_geometry} if self._last_geometry else {})
        if not geometries:
            ActionLogger.warning("Экспорт геометрии прерван — контур не построен")
            dpg.set_value("status_text", "Сначала постройте контур сопла.")
            return
        ActionLogger.info("Экспорт контура сопла в CSV",
                          types=",".join(geometries.keys()))
        try:
            saved = []
            for gtype, geom in geometries.items():
                if geom is None:
                    continue
                fname = f"nozzle_contour_{gtype}.csv"
                path = os.path.join(os.path.expanduser("~"), fname)
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    wr = csv.writer(f, delimiter=";")
                    wr.writerow(["x_m", "r_m"])
                    for p in geom.points:
                        wr.writerow([f"{p.x_m:.6f}", f"{p.r_m:.6f}"])
                saved.append(fname)
            dpg.set_value("status_text",
                          "Контур(ы) сохранены: " + ", ".join(saved))
        except Exception as e:
            dpg.set_value("status_text", f"Ошибка экспорта: {e}")

    # ─── Аналитический расчёт ────────────────────────────────────────────

    def _on_analytic_compute(self):
        ActionLogger.info("Аналитический расчёт запущен")
        try:
            inp = AnalyticSizingInput(
                thrust_vac_N=float(dpg.get_value("sp_an_thrust") or 7.77e6),
                p_chamber_Pa=float(dpg.get_value("sp_an_pk") or 7.0) * 1e6,
                p_exit_Pa=float(dpg.get_value("sp_an_pa") or 0.0486) * 1e6,
                Km=float(dpg.get_value("sp_an_Km") or 2.27),
                Isp_vac_m_s=float(dpg.get_value("sp_an_isp") or 3349.48),
                k_adiabatic=float(dpg.get_value("sp_an_k") or 1.1343),
                R_gas_J_kgK=float(dpg.get_value("sp_an_Rg") or 346.2),
                T_chamber_K=float(dpg.get_value("sp_an_Tk") or 3692.99),
                phi_k=float(dpg.get_value("sp_an_phik") or 0.99),
                phi_c=float(dpg.get_value("sp_an_phic") or 0.98),
                alpha=float(dpg.get_value("sp_an_alpha") or 0.81),
                W_inj_mean_m_s=float(dpg.get_value("sp_an_winj") or 30.0),
                rho_curvature=float(dpg.get_value("sp_an_rho") or 2.0),
            )
            res = compute_analytic_sizing(inp)
            dpg.set_value("txt_analytic",
                          self._format_analytic_result(inp, res))
            self._last_analytic_result = res
        except Exception as exc:
            dpg.set_value("txt_analytic",
                          f"Ошибка расчёта:\n{exc}\n\n{traceback.format_exc()}")

    @staticmethod
    def _format_analytic_result(inp, r):
        def fnum(x, d=4):
            return f"{x:.{d}f}"
        L = []
        L.append("═" * 64)
        L.append("  АНАЛИТИЧЕСКИЙ РАСЧЁТ ПРОФИЛЯ СОПЛА (РПА / Добровольский)")
        L.append("═" * 64)
        L.append("")
        L.append("ИСХОДНЫЕ ДАННЫЕ")
        L.append("─" * 64)
        L.append(f"  Тяга в пустот. Pн ........... {inp.thrust_vac_N:,.1f} Н")
        L.append(f"  Давление в камере pк ........ {inp.p_chamber_Pa/1e6:.4f} МПа")
        L.append(f"  Соотношение Km .............. {inp.Km:.4f}")
        L.append(f"  Удельный импульс Iуд ........ {inp.Isp_vac_m_s:.4f} м/с")
        L.append(f"  Показатель адиабаты k ....... {inp.k_adiabatic:.4f}")
        L.append(f"  Газовая постоянная Rг ....... {inp.R_gas_J_kgK:.3f} Дж/(кг·К)")
        L.append(f"  Температура в камере Tк ..... {inp.T_chamber_K:.2f} К")
        L.append("")
        L.append("1. ЭНЕРГЕТИЧЕСКИЕ ПОКАЗАТЕЛИ")
        L.append("─" * 64)
        L.append(f"  Характеристическая ск. C* ... {fnum(r.Cstar_m_s, 2)} м/с")
        L.append(f"  Ожидаемая C*ож .............. {fnum(r.Cstar_exp_m_s, 2)} м/с")
        L.append(f"  Ожидаемый Iуд.ож ............ {fnum(r.Isp_exp_m_s, 2)} м/с")
        L.append("")
        L.append("2. РАСХОДЫ ТОПЛИВА")
        L.append("─" * 64)
        L.append(f"  Суммарный расход ṁ .......... {fnum(r.mdot_total_kg_s, 2)} кг/с")
        L.append(f"  Расход горючего ṁг .......... {fnum(r.mdot_fuel_kg_s, 2)} кг/с")
        L.append(f"  Расход окислителя ṁо ........ {fnum(r.mdot_ox_kg_s, 2)} кг/с")
        L.append("")
        L.append("3. ПЛОЩАДИ И ДИАМЕТРЫ")
        L.append("─" * 64)
        L.append(f"  Fкр (критика) ............... {fnum(r.F_throat_m2, 4)} м²")
        L.append(f"  Dкр (критика) ............... {fnum(r.D_throat_m, 4)} м")
        L.append(f"  Площадь среза Fa ............ {fnum(r.F_exit_m2, 4)} м²")
        L.append(f"  Диаметр среза Da ............ {fnum(r.D_exit_m, 4)} м")
        L.append("")
        L.append("4. ГЕОМЕТРИЯ КАМЕРЫ")
        L.append("─" * 64)
        L.append(f"  Диаметр камеры Dк ........... {fnum(r.D_chamber_m, 4)} м")
        L.append(f"  Длина цил. участка Lц ....... {fnum(r.L_cyl_m, 4)} м")
        return "\n".join(L)

    def _on_analytic_pull_from_main(self):
        if self.perf is None:
            dpg.set_value("txt_analytic",
                          "Сначала выполните основной (термодинамический) расчёт.")
            return
        try:
            g0 = 9.80665
            isp_vac_s = getattr(self.perf, "Isp_vac_s", None)
            if isp_vac_s:
                dpg.set_value("sp_an_isp", float(isp_vac_s) * g0)
            of = getattr(self.perf, "O_F", None)
            if of:
                dpg.set_value("sp_an_Km", float(of))
            stations = getattr(self.perf, "stations", None) or []
            chamber = None
            for st in stations:
                lbl = (getattr(st, "label", "") or "").lower()
                if "inject" in lbl or "chamber" in lbl:
                    chamber = st
                    break
            if chamber is None and stations:
                chamber = stations[0]
            if chamber is not None:
                k = getattr(chamber, "gamma_eq", None) or getattr(chamber, "gamma_s", None)
                if k:
                    dpg.set_value("sp_an_k", float(k))
                Rg = getattr(chamber, "R_specific_J_per_kgK", None)
                if Rg:
                    dpg.set_value("sp_an_Rg", float(Rg))
                Tk = getattr(chamber, "T_K", None)
                if Tk:
                    dpg.set_value("sp_an_Tk", float(Tk))
                Pc = getattr(chamber, "P_Pa", None)
                if Pc:
                    dpg.set_value("sp_an_pk", float(Pc) / 1e6)
            if stations:
                Pe = getattr(stations[-1], "P_Pa", None)
                if Pe:
                    dpg.set_value("sp_an_pa", float(Pe) / 1e6)
            dpg.set_value("txt_analytic",
                          "Параметры подставлены из последнего расчёта. "
                          "Проверьте значения перед расчётом.")
        except Exception as exc:
            dpg.set_value("txt_analytic", f"Ошибка: {exc}")

    # ─── Экспорт ─────────────────────────────────────────────────────────

    def on_export_csv(self):
        if self.perf is None:
            ActionLogger.warning("Экспорт CSV прерван — нет данных расчёта")
            return
        path = os.path.join(os.path.expanduser("~"), "nozzle_export.csv")
        ActionLogger.info("Экспорт CSV", path=path)
        stations = self.perf.stations
        x = build_axial_coordinates(
            stations, L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1])
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["Сечение", "x_м", "P_МПа", "T_К",
                            "rho_кг/м³", "V_м/с", "M", "gamma_s", "Ae/At"])
                for i, s in enumerate(stations):
                    w.writerow([
                        s.label, f"{x[i]:.6f}", f"{s.P_Pa/1e6:.6f}",
                        f"{s.T_K:.4f}", f"{s.rho_kg_per_m3:.5f}",
                        f"{s.V_m_per_s:.4f}", f"{s.M:.5f}",
                        f"{s.gamma_s:.5f}",
                        ("inf" if not math.isfinite(s.Ae_At)
                         else f"{s.Ae_At:.5f}"),
                    ])
            dpg.set_value("status_text", f"CSV сохранён: {path}")
        except Exception as e:
            dpg.set_value("status_text", f"Ошибка: {e}")

    def on_export_amesim(self):
        if self.perf is None:
            ActionLogger.warning("Экспорт Amesim прерван — нет данных расчёта")
            return
        path = os.path.join(os.path.expanduser("~"), "nozzle_amesim.data")
        ActionLogger.info("Экспорт Amesim", path=path)
        stations = self.perf.stations
        x = build_axial_coordinates(
            stations, L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1])
        signals = [
            ("Pressure", "MPa", [s.P_Pa / 1e6 for s in stations]),
            ("Temperature", "K", [s.T_K for s in stations]),
            ("Density", "kg/m^3", [s.rho_kg_per_m3 for s in stations]),
            ("Velocity", "m/s", [s.V_m_per_s for s in stations]),
            ("Mach number", "", [s.M for s in stations]),
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Amesim XY export from {APP_NAME}\n")
                f.write("# Table format: XY\n")
                f.write("# axis1_unit = m\n")
                for k, (title, unit, _) in enumerate(signals, start=2):
                    f.write(f"# axis{k}_unit = {unit}\n")
                    f.write(f"# axis{k}_title = {title}\n")
                for i in range(len(stations)):
                    row = [f"{x[i]:.6f}"]
                    for _, _, vals in signals:
                        row.append(f"{vals[i]:.6e}")
                    f.write("\t".join(row) + "\n")
            dpg.set_value("status_text", f"Amesim .data сохранён: {path}")
        except Exception as e:
            dpg.set_value("status_text", f"Ошибка: {e}")

    # ─── Конфигурация ────────────────────────────────────────────────────

    def on_save_config(self):
        """Открывает проводник для выбора места сохранения конфигурации."""
        if self.mixture_widget is None:
            ActionLogger.warning("Сохранение конфигурации прервано — нет mixture_widget")
            dpg.set_value("status_text", "Нет данных для сохранения.")
            return
        ActionLogger.info("Открытие диалога сохранения конфигурации")
        if dpg.does_item_exist("dlg_save_config"):
            dpg.show_item("dlg_save_config")

    def on_load_config(self):
        """Открывает проводник для выбора файла конфигурации."""
        ActionLogger.info("Открытие диалога загрузки конфигурации")
        if dpg.does_item_exist("dlg_load_config"):
            dpg.show_item("dlg_load_config")

    @staticmethod
    def _path_from_dialog(app_data) -> Optional[str]:
        """Извлекает путь к файлу из app_data диалога файлов DPG."""
        if not isinstance(app_data, dict):
            return None
        # Предпочитаем полный путь, выбранный пользователем.
        path = app_data.get("file_path_name")
        if path:
            return path
        # Резерв: каталог + первый выбранный файл из selections.
        sels = app_data.get("selections") or {}
        if sels:
            return next(iter(sels.values()))
        cur = app_data.get("current_path")
        name = app_data.get("file_name")
        if cur and name:
            return os.path.join(cur, name)
        return None

    def _on_save_config_selected(self, sender, app_data):
        path = self._path_from_dialog(app_data)
        if not path:
            dpg.set_value("status_text", "Сохранение отменено.")
            return
        # Гарантируем расширение .json.
        if not os.path.splitext(path)[1]:
            path += ".json"
        self._do_save_config(path)

    def _on_load_config_selected(self, sender, app_data):
        path = self._path_from_dialog(app_data)
        if not path or not os.path.exists(path):
            dpg.set_value("status_text", "Файл конфигурации не выбран.")
            return
        self._do_load_config(path)

    def _do_save_config(self, path: str):
        if self.mixture_widget is None:
            return
        ActionLogger.info("Сохранение конфигурации", path=path)
        cfg = {
            "mixture": self.mixture_widget.get_mixture(),
            "mix_mode": self._mix_mode(),
            "mix_value": dpg.get_value("ed_mix_value") or "",
            "Pc_field": dpg.get_value("ed_Pc") or "",
            "Pe_field": dpg.get_value("ed_Pe") or "",
            "Pc_unit": dpg.get_value("cb_Pc_unit") or "МПа",
            "Pe_unit": dpg.get_value("cb_Pe_unit") or "МПа",
            "n_inter": int(dpg.get_value("sp_n_inter") or 8),
            "include_condensed": bool(dpg.get_value("chk_condensed")),
            "injection_velocity": float(dpg.get_value("sp_inj_velocity") or 0.0),
            "chamber_pressure_drop": float(dpg.get_value("sp_chamber_dp") or 0.0),
            "solver": self._solver,
            "precision": dpg.get_value("cb_precision") or "Сбалансировано",
            "L_chamber": float(dpg.get_value("sp_L_chamber") or 0.1),
            "L_star": float(dpg.get_value("sp_L_star") or 1.0),
            "losses": {
                "reaction_eff": float(dpg.get_value("sp_eff_reaction") or 1.0),
                "nozzle_eff": float(dpg.get_value("sp_eff_nozzle") or 1.0),
            },
            "style": {
                "lw": float(dpg.get_value("sp_lw") or 1.8),
                "markers": bool(dpg.get_value("chk_markers")),
                "smooth": bool(dpg.get_value("chk_smooth")),
                "grid_major": bool(dpg.get_value("chk_grid_major")),
                "grid_minor": bool(dpg.get_value("chk_grid_minor")),
                "dark": bool(dpg.get_value("chk_dark_plot")),
                "show_profile": bool(dpg.get_value("chk_show_profile")),
                "plot_cols": dpg.get_value("cb_plot_cols") or "Авто",
                "plot_row_h": int(self._plot_row_h),
                "plot_col_w": int(self._plot_col_w),
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            dpg.set_value("status_text", f"Конфигурация сохранена: {path}")
        except Exception as e:
            ActionLogger.error("Ошибка сохранения конфигурации", detail=str(e))
            dpg.set_value("status_text", f"Ошибка: {e}")

    def _do_load_config(self, path: str):
        ActionLogger.info("Загрузка конфигурации", path=path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mixture = cfg.get("mixture")
            if mixture is not None and self.mixture_widget:
                self.mixture_widget.set_mixture(mixture)
            dpg.set_value("ed_mix_value", str(cfg.get("mix_value", "")))
            dpg.set_value("ed_Pc", str(cfg.get("Pc_field", "")))
            dpg.set_value("ed_Pe", str(cfg.get("Pe_field", "")))
            dpg.set_value("cb_Pc_unit", cfg.get("Pc_unit", "МПа"))
            dpg.set_value("cb_Pe_unit", cfg.get("Pe_unit", "МПа"))
            dpg.set_value("sp_n_inter", int(cfg.get("n_inter", 8)))
            dpg.set_value("cb_precision", cfg.get("precision", "Сбалансировано"))
            dpg.set_value("chk_condensed", bool(cfg.get("include_condensed", True)))
            dpg.set_value("sp_inj_velocity", float(cfg.get("injection_velocity", 0.0)))
            dpg.set_value("sp_chamber_dp", float(cfg.get("chamber_pressure_drop", 0.0)))
            dpg.set_value("sp_L_chamber", float(cfg.get("L_chamber", 0.1)))
            dpg.set_value("sp_L_star", float(cfg.get("L_star", 1.0)))
            losses = cfg.get("losses", {})
            dpg.set_value("sp_eff_reaction", float(losses.get("reaction_eff", 1.0)))
            dpg.set_value("sp_eff_nozzle", float(losses.get("nozzle_eff", 1.0)))
            st = cfg.get("style", {})
            dpg.set_value("sp_lw", float(st.get("lw", 1.8)))
            dpg.set_value("chk_markers", bool(st.get("markers", True)))
            if dpg.does_item_exist("chk_smooth"):
                dpg.set_value("chk_smooth", bool(st.get("smooth", False)))
            dpg.set_value("chk_grid_major", bool(st.get("grid_major", True)))
            dpg.set_value("chk_grid_minor", bool(st.get("grid_minor", True)))
            dpg.set_value("chk_dark_plot", bool(st.get("dark", True)))
            if dpg.does_item_exist("chk_show_profile"):
                self._show_profile_1d = bool(st.get("show_profile", False))
                dpg.set_value("chk_show_profile", self._show_profile_1d)
            if dpg.does_item_exist("cb_plot_cols"):
                dpg.set_value("cb_plot_cols", st.get("plot_cols", "Авто"))
            self._plot_row_h = int(st.get("plot_row_h", self._plot_row_h))
            self._plot_col_w = int(st.get("plot_col_w", self._plot_col_w))
            self._update_of_from_mixture()
            self._update_overall_efficiency()
            if self.perf is not None:
                self._redraw_plots()
            dpg.set_value("status_text", f"Конфигурация загружена: {path}")
        except Exception as e:
            ActionLogger.error("Ошибка загрузки конфигурации", detail=str(e))
            dpg.set_value("status_text", f"Ошибка: {e}")

    def _about(self):
        ActionLogger.info("Вызван диалог \"О программе\"")
        dpg.set_value("status_text",
                      f"{APP_NAME} v{APP_VERSION}. "
                      f"Расчёт газодинамики ракетного сопла в равновесном приближении. "
                      f"Решатели: собственный (NASA-9) и CEA (Cantera). "
                      f"GUI: Dear PyGui.")

    # ─── Предзагрузка базы NASA-9 ────────────────────────────────────────

    def _preload_species_db(self):
        """Фоновая загрузка базы NASA-9 (в отдельном потоке)."""
        def _load():
            try:
                db_path = find_thermo_db()
                db = parse_thermo_file(db_path)
                clear_equilibrium_cache()
                self.species_db = db
                if self.mixture_widget:
                    self.mixture_widget.species_db = db
                    if self.mixture_widget.oxidizer_list:
                        self.mixture_widget.oxidizer_list.species_db = db
                    if self.mixture_widget.fuel_list:
                        self.mixture_widget.fuel_list.species_db = db
                    # Стандартная смесь
                    self.mixture_widget.set_mixture({
                        "ox_components": [{"name": "O2(L)", "mass": 1.0, "T": 0}],
                        "fuel_components": [{"name": "H2(L)", "mass": 1.0, "T": 0}],
                    })
                self._db_loaded = True
            except Exception as e:
                self._db_error = str(e)
        self._db_loaded = False
        self._db_error = None
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _poll_db_load(self):
        """Проверка загрузки базы (вызывается каждый кадр)."""
        if getattr(self, "_db_loaded", False):
            self._db_loaded = False
            db_count = len(self.species_db) if self.species_db else 0
            ActionLogger.info("База NASA-9 загружена", species_count=db_count)
            dpg.set_value("status_text",
                          f"База NASA-9 загружена: "
                          f"{db_count} веществ. Готово.")
            self._update_of_from_mixture()
        if getattr(self, "_db_error", None):
            err = self._db_error
            self._db_error = None
            ActionLogger.error("Ошибка загрузки базы NASA-9", detail=err)
            dpg.set_value("status_text", f"Ошибка загрузки базы: {err}")


# ═══════════════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════════════

def _load_cyrillic_font():
    import os as _os
    candidates = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    try:
        import matplotlib
        mpl_dir = _os.path.dirname(matplotlib.get_data_path())
        candidates.append(_os.path.join(mpl_dir, "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    path = next((p for p in candidates if _os.path.exists(p)), None)
    if path is None:
        return
    with dpg.font_registry():
        font = dpg.add_font(path, 15)
        # Диапазоны символов в новых версиях Dear PyGui добавляются автоматически.
    dpg.bind_font(font)


def _frame_callback(main_win: MainWindow):
    """Callback, вызываемый каждый кадр рендера: опрос worker и БД."""
    def _cb():
        main_win._poll_worker()
        main_win._poll_db_load()
    return _cb


def _global_excepthook(exc_type, exc_value, exc_traceback):
    import traceback
    ActionLogger.error("Необработанное исключение", detail=str(exc_value))
    ActionLogger.error("Traceback", detail="".join(traceback.format_tb(exc_traceback)))


def main():
    """Точка входа GUI на Dear PyGui."""
    import sys
    import threading
    sys.excepthook = _global_excepthook
    threading.excepthook = _global_excepthook
    dpg.create_context()

    # Шрифт с кириллицей — ДО создания виджетов!
    _load_cyrillic_font()

    apply_dark_theme()

    main_win = MainWindow()
    dpg.create_viewport(title=APP_NAME)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Предзагрузка базы NASA-9 в фоне
    main_win._preload_species_db()

    # Главный цикл: рендер + опрос фоновых задач
    frame_cb = _frame_callback(main_win)
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        frame_cb()

    dpg.destroy_context()


if __name__ == "__main__":
    main()