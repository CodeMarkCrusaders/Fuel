#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPA-Style Rocket Nozzle Calculator — GUI приложение для расчёта газодинамических
параметров по длине сопла ракетного двигателя.

Возможности:
  • Расчёт параметров по длине сопла (P, T, V, M, ρ, гамма, состав)
  • Два решателя: собственный (Gibbs minimisation) и CEA (Cantera)
  • Тёмная тема в стиле Claude.ai (из MILF PLOTTER)
  • Экспорт точек в CSV (формат, совместимый с Amesim)
  • Графики с настраиваемым ГОСТ-оформлением
  • Сохранение/загрузка конфигурации
"""

import sys
import os
import io
import json
import csv
import math
import traceback
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple

import numpy as np

# Проверка зависимостей
_missing = []
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, pyqtSignal, QThread
except ImportError:
    _missing.append('PyQt5')
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
except ImportError:
    _missing.append('matplotlib')

if _missing:
    print(f"Missing packages: {_missing}")
    print(f"Install: pip install {' '.join(_missing)}")
    sys.exit(1)

# Импорт решателей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nozzle_flow import (
    Propellant, StationResult, RocketPerformance,
    solve_rocket_nozzle, print_nozzle_table,
)
from nasa9_parser import parse_thermo_file
from equilibrium import find_thermo_db
from iteration_logger import IterationLogger, NullLogger
from component_selector import ComponentSelectorDialog, ComponentListWidget, MixturePropellantWidget

try:
    from cea_solver import (
        solve_rocket_nozzle_cea, build_nozzle_geometry,
        nozzle_radius, CANTERA_AVAILABLE,
    )
except ImportError:
    CANTERA_AVAILABLE = False
    def build_nozzle_geometry(stations, **kw):
        return np.linspace(0, 1, len(stations))
    def nozzle_radius(stations):
        return np.ones(len(stations))


APP_NAME = "RPA-Style Rocket Nozzle Calculator"
APP_VERSION = "1.0"


# ═══════════════════════════════════════════════════════════════════════════
# Тема: тёмный QSS из MILF PLOTTER
# ═══════════════════════════════════════════════════════════════════════════

def _make_icons() -> dict:
    """Создаёт иконки (галочку и точку для радиокнопки) и возвращает их пути."""
    paths = {'check': '', 'radio': ''}
    try:
        from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
        from PyQt5.QtCore import Qt as _Qt
        import tempfile

        # Галочка для QCheckBox
        size = 16
        pm = QPixmap(size, size)
        pm.fill(_Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor('#1e1e1c'))
        pen.setWidth(3)
        pen.setCapStyle(_Qt.RoundCap)
        pen.setJoinStyle(_Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(3, 8, 7, 12)
        p.drawLine(7, 12, 13, 4)
        p.end()
        path1 = os.path.join(tempfile.gettempdir(), 'rpa_check_icon.png')
        pm.save(path1, 'PNG')
        paths['check'] = path1.replace('\\', '/')

        # Точка для QRadioButton
        pm2 = QPixmap(size, size)
        pm2.fill(_Qt.transparent)
        p = QPainter(pm2)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(_Qt.NoPen)
        p.setBrush(QBrush(QColor('#1e1e1c')))
        p.drawEllipse(4, 4, 8, 8)
        p.end()
        path2 = os.path.join(tempfile.gettempdir(), 'rpa_radio_icon.png')
        pm2.save(path2, 'PNG')
        paths['radio'] = path2.replace('\\', '/')
    except Exception:
        pass
    return paths


DARK_QSS = """
QMainWindow, QWidget {
    background-color: #262624;
    color: #fafaf9;
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 11px;
}
QMenuBar {
    background-color: #1e1e1c;
    color: #fafaf9;
    border-bottom: 1px solid #3a3a37;
}
QMenuBar::item:selected { background-color: #3a3a37; }
QMenu {
    background-color: #30302e;
    color: #fafaf9;
    border: 1px solid #3a3a37;
}
QMenu::item:selected { background-color: #cc785c; color: #1e1e1c; }

QLabel { color: #fafaf9; background: transparent; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background-color: #1e1e1c;
    color: #fafaf9;
    border: 1px solid #3a3a37;
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: #cc785c;
    selection-color: #1e1e1c;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #cc785c;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: #6a6a66;
    background-color: #262624;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e1c;
    color: #fafaf9;
    border: 1px solid #3a3a37;
    selection-background-color: #cc785c;
    selection-color: #1e1e1c;
}

QPushButton {
    background-color: #3a3a37;
    color: #fafaf9;
    border: 1px solid #4a4a47;
    border-radius: 3px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #4a4a47;
    border-color: #cc785c;
}
QPushButton:pressed { background-color: #2a2a27; }
QPushButton:disabled {
    color: #6a6a66;
    background-color: #2a2a27;
    border-color: #3a3a37;
}
QPushButton#primary {
    background-color: #cc785c;
    color: #1e1e1c;
    font-weight: bold;
}
QPushButton#primary:hover { background-color: #d88c70; }
QPushButton#primary:pressed { background-color: #b86b50; }

QGroupBox {
    border: 1px solid #3a3a37;
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 6px;
    color: #cc785c;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}

QCheckBox, QRadioButton {
    color: #fafaf9;
    background: transparent;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #6a6a66;
    background-color: #1e1e1c;
    border-radius: 3px;
}
QCheckBox::indicator:hover { border-color: #cc785c; }
QCheckBox::indicator:checked {
    background-color: #cc785c;
    border-color: #cc785c;
    image: url(__CHECKICON__);
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #6a6a66;
    background-color: #1e1e1c;
    border-radius: 8px;
}
QRadioButton::indicator:hover {
    border-color: #cc785c;
}
QRadioButton::indicator:checked {
    background-color: #cc785c;
    border: 2px solid #cc785c;
    border-radius: 8px;
    image: url(__RADIOICON__);
}

QListWidget, QTreeWidget, QTableWidget {
    background-color: #1e1e1c;
    color: #fafaf9;
    border: 1px solid #3a3a37;
    alternate-background-color: #262624;
    gridline-color: #3a3a37;
}
QTableWidget::item { padding: 4px; }
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #cc785c;
    color: #1e1e1c;
}
QHeaderView::section {
    background-color: #30302e;
    color: #fafaf9;
    border: 1px solid #3a3a37;
    padding: 4px 6px;
    font-weight: bold;
}

QScrollBar:vertical {
    background: #1e1e1c;
    width: 12px;
    border: 1px solid #3a3a37;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #6a6a66;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #cc785c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #1e1e1c;
    height: 12px;
    border: 1px solid #3a3a37;
    border-radius: 3px;
}
QScrollBar::handle:horizontal {
    background: #6a6a66;
    border-radius: 3px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #cc785c; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QScrollArea {
    border: 1px solid #3a3a37;
    background: #262624;
}

QSplitter::handle { background-color: #3a3a37; }
QSplitter::handle:hover { background-color: #cc785c; }

QStatusBar {
    background-color: #1e1e1c;
    color: #a8a29e;
    border-top: 1px solid #3a3a37;
}

QToolTip {
    background-color: #1e1e1c;
    color: #fafaf9;
    border: 1px solid #cc785c;
    padding: 4px;
}

QTabWidget::pane {
    border: 1px solid #3a3a37;
    background: #262624;
    top: -1px;
}
QTabBar::tab {
    background: #30302e;
    color: #a8a29e;
    border: 1px solid #3a3a37;
    padding: 6px 18px;
    margin-right: 2px;
    min-width: 160px;
}
QTabBar::tab:selected {
    background: #262624;
    color: #cc785c;
    border-bottom-color: #262624;
    font-weight: bold;
}
QTabBar::tab:hover:!selected { background: #3a3a37; color: #fafaf9; }

QProgressBar {
    background-color: #1e1e1c;
    border: 1px solid #3a3a37;
    border-radius: 3px;
    text-align: center;
    color: #fafaf9;
}
QProgressBar::chunk {
    background-color: #cc785c;
    border-radius: 2px;
}
"""


def build_dark_qss() -> str:
    icons = _make_icons()
    return (DARK_QSS
            .replace('__CHECKICON__', icons.get('check', ''))
            .replace('__RADIOICON__', icons.get('radio', '')))


# ═══════════════════════════════════════════════════════════════════════════
# Стиль графиков (из MILF PLOTTER, упрощён под наши задачи)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlotStyle:
    """Параметры оформления графика."""
    font_family: str = 'DejaVu Serif'
    font_size_axis: int = 12
    font_size_tick: int = 11
    font_size_legend: int = 10
    font_size_title: int = 13

    title: str = ''
    xlabel: str = ''
    ylabel: str = ''

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_log: bool = False
    y_log: bool = False

    grid_major: bool = True
    grid_minor: bool = True
    grid_alpha_major: float = 0.45
    grid_alpha_minor: float = 0.22

    spine_linewidth: float = 1.2
    tick_direction: str = 'in'

    legend_show: bool = True
    legend_loc: str = 'best'

    dark_plot: bool = True   # тёмная тема и для графика

    line_width: float = 1.8
    marker_size: float = 6.0
    show_markers: bool = True


def apply_plot_style(fig, ax, style: PlotStyle):
    """Применяет стиль PlotStyle к matplotlib-фигуре и осям."""
    # Цвета зависят от темы
    if style.dark_plot:
        bg = '#262624'
        fg = '#fafaf9'
        grid_color = '#5a5a57'
    else:
        bg = '#ffffff'
        fg = '#000000'
        grid_color = '#888888'

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Шрифт
    plt.rcParams['font.family'] = style.font_family
    for txt in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        txt.set_color(fg)
        txt.set_fontsize(style.font_size_axis)
        txt.set_fontfamily(style.font_family)

    if style.title:
        ax.set_title(style.title, fontsize=style.font_size_title,
                     color=fg, fontfamily=style.font_family)
    if style.xlabel:
        ax.set_xlabel(style.xlabel, fontsize=style.font_size_axis,
                      color=fg, fontfamily=style.font_family)
    if style.ylabel:
        ax.set_ylabel(style.ylabel, fontsize=style.font_size_axis,
                      color=fg, fontfamily=style.font_family)

    # Тики
    ax.tick_params(axis='both', which='major',
                   labelsize=style.font_size_tick,
                   direction=style.tick_direction,
                   length=5, width=1.0, color=fg, labelcolor=fg)
    ax.tick_params(axis='both', which='minor',
                   direction=style.tick_direction,
                   length=3, width=0.7, color=fg)

    # Рамка
    for spine in ax.spines.values():
        spine.set_linewidth(style.spine_linewidth)
        spine.set_color(fg)

    # Лимиты
    if style.x_min is not None or style.x_max is not None:
        cur = ax.get_xlim()
        ax.set_xlim(style.x_min if style.x_min is not None else cur[0],
                    style.x_max if style.x_max is not None else cur[1])
    if style.y_min is not None or style.y_max is not None:
        cur = ax.get_ylim()
        ax.set_ylim(style.y_min if style.y_min is not None else cur[0],
                    style.y_max if style.y_max is not None else cur[1])

    # Лог. шкалы
    if style.x_log:
        ax.set_xscale('log')
    if style.y_log:
        ax.set_yscale('log')

    # Сетка
    if style.grid_major:
        ax.grid(True, which='major', color=grid_color,
                alpha=style.grid_alpha_major, linewidth=0.7)
    if style.grid_minor:
        if not style.x_log:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if not style.y_log:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', color=grid_color,
                alpha=style.grid_alpha_minor, linewidth=0.4)

    # Легенда
    leg = ax.get_legend()
    if style.legend_show and leg is not None:
        leg.set_visible(True)
        frame = leg.get_frame()
        frame.set_facecolor(bg)
        frame.set_edgecolor(fg)
        frame.set_alpha(0.85)
        for txt in leg.get_texts():
            txt.set_color(fg)
            txt.set_fontfamily(style.font_family)
            txt.set_fontsize(style.font_size_legend)


# ═══════════════════════════════════════════════════════════════════════════
# Matplotlib-canvas
# ═══════════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6.0, height=4.0, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)


# ═══════════════════════════════════════════════════════════════════════════
# Worker для асинхронного расчёта
# ═══════════════════════════════════════════════════════════════════════════

class NozzleSolverWorker(QThread):
    finished_ok = pyqtSignal(object)         # RocketPerformance
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params: dict, solver: str, species_db=None):
        super().__init__()
        self.params = params
        self.solver = solver
        self.species_db = species_db

    @staticmethod
    def _components_to_propellants(components: List[Dict]) -> List[Propellant]:
        """Преобразовать список компонент GUI в список Propellant."""
        propellants: List[Propellant] = []
        for comp in components:
            mass = float(comp.get('mass', 0.0) or 0.0)
            if mass <= 0.0:
                continue
            T_raw = float(comp.get('T', 0.0) or 0.0)
            propellants.append(
                Propellant(
                    name=comp['name'],
                    mass_kg=mass,
                    T_K=T_raw if T_raw > 0.0 else None,
                )
            )
        return propellants

    @staticmethod
    def _scaled_components(components: List[Dict], target_total_mass: float) -> List[Dict]:
        """Масштабировать список компонентов до заданной суммарной массы."""
        if not components:
            return []

        positive = [max(float(c.get('mass', 0.0) or 0.0), 0.0) for c in components]
        src_total = sum(positive)
        if src_total <= 1e-12:
            # fallback: равные доли
            frac = 1.0 / len(components)
            return [
                {'name': c['name'], 'mass': target_total_mass * frac, 'T': float(c.get('T', 0.0) or 0.0)}
                for c in components
            ]

        out: List[Dict] = []
        for c, m in zip(components, positive):
            out.append({
                'name': c['name'],
                'mass': target_total_mass * (m / src_total),
                'T': float(c.get('T', 0.0) or 0.0),
            })
        return out

    @staticmethod
    def _single_propellant_from_mix(components: List[Dict]) -> Propellant:
        """Сжать смесь до одного компонента (для CEA fallback)."""
        if not components:
            raise ValueError("Список компонентов пуст.")

        total_mass = sum(max(float(c.get('mass', 0.0) or 0.0), 0.0) for c in components)
        if total_mass <= 0:
            raise ValueError("Суммарная масса компонентов должна быть > 0.")

        main = max(components, key=lambda c: float(c.get('mass', 0.0) or 0.0))
        weighted_T_num = 0.0
        weighted_T_den = 0.0
        for c in components:
            m = max(float(c.get('mass', 0.0) or 0.0), 0.0)
            t = float(c.get('T', 0.0) or 0.0)
            if m > 0 and t > 0:
                weighted_T_num += m * t
                weighted_T_den += m

        T_mix = (weighted_T_num / weighted_T_den) if weighted_T_den > 0 else 0.0
        return Propellant(
            name=main['name'],
            mass_kg=total_mass,
            T_K=T_mix if T_mix > 0 else None,
        )

    def _solve_once(self, ox_components: List[Dict], fu_components: List[Dict], p: Dict) -> RocketPerformance:
        if self.solver == 'cea':
            if len(ox_components) > 1 or len(fu_components) > 1:
                self.progress.emit(
                    "CEA-режим не поддерживает точные смеси: используется доминирующий компонент каждой группы."
                )

            ox = self._single_propellant_from_mix(ox_components)
            fu = self._single_propellant_from_mix(fu_components)

            self.progress.emit("Запуск CEA-решателя (Cantera)...")
            return solve_rocket_nozzle_cea(
                oxidizer=ox, fuel=fu,
                P_chamber=p['P_chamber'],
                P_exit=p['P_exit'],
                n_intermediate_stations=p.get('n_inter', 5),
                include_condensed=p.get('include_condensed', False),
                verbose=False,
                progress_cb=lambda msg: self.progress.emit(msg),
            )

        self.progress.emit("Запуск собственного решателя (Gibbs)...")
        ox_propellants = self._components_to_propellants(ox_components)
        fu_propellants = self._components_to_propellants(fu_components)
        if not ox_propellants or not fu_propellants:
            raise ValueError("Не заданы компоненты окислителя/горючего с положительной массой.")

        return solve_rocket_nozzle(
            oxidizer=ox_propellants,
            fuel=fu_propellants,
            P_chamber=p['P_chamber'],
            P_exit=p['P_exit'],
            species_db=self.species_db,
            n_intermediate_stations=p.get('n_inter', 5),
            include_condensed=p.get('include_condensed', True),
            verbose=False,
            logger=NullLogger(),
        )

    def _search_optimal_of(self, p: Dict) -> Tuple[RocketPerformance, float, List[Dict], List[Dict]]:
        """Поиск оптимального O/F по максимуму Isp."""
        of_min = max(float(p.get('of_min', 0.1) or 0.1), 1e-3)
        of_max = max(float(p.get('of_max', 20.0) or 20.0), of_min + 1e-3)
        n_steps = max(int(p.get('of_steps', 12) or 12), 2)

        base_ox = p['ox_components']
        base_fu = p['fuel_components']

        best_metric = -float('inf')
        best_perf = None
        best_of = None
        best_ox = None
        best_fu = None

        for i in range(n_steps):
            frac = i / (n_steps - 1)
            of_try = of_min + (of_max - of_min) * frac

            fu_try = self._scaled_components(base_fu, target_total_mass=1.0)
            ox_try = self._scaled_components(base_ox, target_total_mass=of_try)

            self.progress.emit(
                f"Оптимизация O/F: шаг {i + 1}/{n_steps}, O/F={of_try:.4f}"
            )

            try:
                perf_try = self._solve_once(ox_try, fu_try, p)
            except Exception as exc:
                self.progress.emit(f"  Пропуск O/F={of_try:.4f}: {exc}")
                continue

            metric = perf_try.Isp_s
            if metric > best_metric:
                best_metric = metric
                best_perf = perf_try
                best_of = of_try
                best_ox = ox_try
                best_fu = fu_try

        if best_perf is None:
            raise RuntimeError("Не удалось подобрать оптимальное O/F: все расчёты завершились ошибкой.")

        self.progress.emit(
            f"Оптимум найден: O/F={best_of:.4f}, Isp={best_perf.Isp_s:.4f} с"
        )
        return best_perf, best_of, best_ox, best_fu

    def run(self):
        try:
            p = dict(self.params)
            if p.get('optimize_of', False):
                perf, best_of, best_ox, best_fu = self._search_optimal_of(p)
                setattr(perf, 'optimized_of', best_of)
                setattr(perf, 'optimized_ox_components', best_ox)
                setattr(perf, 'optimized_fuel_components', best_fu)
            else:
                perf = self._solve_once(p['ox_components'], p['fuel_components'], p)

            self.finished_ok.emit(perf)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


class EngineParamsDialog(QtWidgets.QDialog):
    """Отдельное окно ручного задания параметров двигателя."""

    def __init__(self, params: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры двигателя")
        self.setModal(True)
        self.resize(440, 420)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.sp_Pc = QtWidgets.QDoubleSpinBox()
        self.sp_Pc.setRange(0.01, 1000.0)
        self.sp_Pc.setDecimals(4)
        self.sp_Pc.setValue(float(params.get('Pc_MPa', 10.0)))
        self.sp_Pc.setSuffix(" МПа")

        self.sp_Pe = QtWidgets.QDoubleSpinBox()
        self.sp_Pe.setRange(0.0001, 100.0)
        self.sp_Pe.setDecimals(5)
        self.sp_Pe.setValue(float(params.get('Pe_MPa', 0.1013)))
        self.sp_Pe.setSuffix(" МПа")

        self.sp_n_inter = QtWidgets.QSpinBox()
        self.sp_n_inter.setRange(0, 50)
        self.sp_n_inter.setValue(int(params.get('n_inter', 8)))

        self.chk_condensed = QtWidgets.QCheckBox("Учитывать конденсат")
        self.chk_condensed.setChecked(bool(params.get('include_condensed', True)))

        self.sp_L_chamber = QtWidgets.QDoubleSpinBox()
        self.sp_L_chamber.setRange(0.001, 10.0)
        self.sp_L_chamber.setDecimals(3)
        self.sp_L_chamber.setValue(float(params.get('L_chamber', 0.1)))
        self.sp_L_chamber.setSuffix(" м")

        self.sp_L_conv = QtWidgets.QDoubleSpinBox()
        self.sp_L_conv.setRange(0.001, 10.0)
        self.sp_L_conv.setDecimals(3)
        self.sp_L_conv.setValue(float(params.get('L_conv', 0.05)))
        self.sp_L_conv.setSuffix(" м")

        self.sp_L_div = QtWidgets.QDoubleSpinBox()
        self.sp_L_div.setRange(0.001, 10.0)
        self.sp_L_div.setDecimals(3)
        self.sp_L_div.setValue(float(params.get('L_div', 0.2)))
        self.sp_L_div.setSuffix(" м")

        self.chk_optimize_of = QtWidgets.QCheckBox("Искать оптимальное O/F автоматически")
        self.chk_optimize_of.setChecked(bool(params.get('optimize_of', False)))

        self.sp_of_min = QtWidgets.QDoubleSpinBox()
        self.sp_of_min.setRange(0.01, 100.0)
        self.sp_of_min.setDecimals(4)
        self.sp_of_min.setValue(float(params.get('of_min', 2.0)))

        self.sp_of_max = QtWidgets.QDoubleSpinBox()
        self.sp_of_max.setRange(0.01, 100.0)
        self.sp_of_max.setDecimals(4)
        self.sp_of_max.setValue(float(params.get('of_max', 12.0)))

        self.sp_of_steps = QtWidgets.QSpinBox()
        self.sp_of_steps.setRange(2, 200)
        self.sp_of_steps.setValue(int(params.get('of_steps', 15)))

        form.addRow("Давление в камере:", self.sp_Pc)
        form.addRow("Давление на срезе:", self.sp_Pe)
        form.addRow("Промежуточных сечений:", self.sp_n_inter)
        form.addRow("", self.chk_condensed)
        form.addRow("Длина камеры:", self.sp_L_chamber)
        form.addRow("Длина конфузора:", self.sp_L_conv)
        form.addRow("Длина дивергента:", self.sp_L_div)
        form.addRow("", self.chk_optimize_of)
        form.addRow("O/F мин:", self.sp_of_min)
        form.addRow("O/F макс:", self.sp_of_max)
        form.addRow("Шагов поиска:", self.sp_of_steps)

        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict:
        return {
            'Pc_MPa': self.sp_Pc.value(),
            'Pe_MPa': self.sp_Pe.value(),
            'n_inter': self.sp_n_inter.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'L_chamber': self.sp_L_chamber.value(),
            'L_conv': self.sp_L_conv.value(),
            'L_div': self.sp_L_div.value(),
            'optimize_of': self.chk_optimize_of.isChecked(),
            'of_min': self.sp_of_min.value(),
            'of_max': self.sp_of_max.value(),
            'of_steps': self.sp_of_steps.value(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Главное окно
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1500, 920)

        self.perf: Optional[RocketPerformance] = None
        self.species_db = None
        self.plot_style = PlotStyle()
        self.worker: Optional[NozzleSolverWorker] = None
        self.optimize_of = False
        self.of_min = 2.0
        self.of_max = 12.0
        self.of_steps = 15

        self._build_ui()
        self._apply_engine_params(self._current_engine_params())
        self._build_menu()
        self.statusBar().showMessage("Готово. Введите параметры и нажмите «Рассчитать».")

        # Предзагрузка базы NASA-9 (в фоне, чтобы не блокировать UI)
        QtCore.QTimer.singleShot(100, self._preload_species_db)

    # ───────────────────────── UI ─────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # Splitter: слева панель ввода, справа результаты
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ─── Левая колонка: входные данные ───
        left_widget = self._build_input_panel()
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(400)
        left_scroll.setMaximumWidth(460)
        splitter.addWidget(left_scroll)

        # ─── Правая часть: вкладки результатов ───
        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)

        # Вкладка: Параметры по сечениям (таблица RPA-style)
        self.tab_table = self._build_table_tab()
        self.tabs.addTab(self.tab_table, "Параметры по сечениям")

        # Вкладка: Графики по длине сопла
        self.tab_plots = self._build_plot_tab()
        self.tabs.addTab(self.tab_plots, "Графики по длине сопла")

        # Вкладка: Состав продуктов
        self.tab_species = self._build_species_tab()
        self.tabs.addTab(self.tab_species, "Состав продуктов сгорания")

        # Вкладка: Тяговые характеристики
        self.tab_perf = self._build_perf_tab()
        self.tabs.addTab(self.tab_perf, "Тяговые характеристики")

        splitter.setSizes([400, 1100])

    def _build_input_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # ─── Топливо (RPA-style с поддержкой смеси) ───
        gb_fuel = QtWidgets.QGroupBox("Топливо (RPA-style)")
        gb_fuel_layout = QtWidgets.QVBoxLayout(gb_fuel)
        gb_fuel_layout.setSpacing(10)
        
        # Виджет для управления смесью компонентов
        self.mixture_widget = MixturePropellantWidget(species_db=None)
        self.mixture_widget.mixture_changed.connect(self._update_of_from_mixture)
        gb_fuel_layout.addWidget(self.mixture_widget)
        
        # Отношение O/F (информационное)
        self.lbl_of = QtWidgets.QLabel("O/F = —")
        self.lbl_of.setStyleSheet("color: #cc785c; font-weight: bold;")
        of_layout = QtWidgets.QHBoxLayout()
        of_layout.addWidget(QtWidgets.QLabel("Отношение:"))
        of_layout.addWidget(self.lbl_of)
        of_layout.addStretch()
        gb_fuel_layout.addLayout(of_layout)

        layout.addWidget(gb_fuel)

        # ─── Условия в сопле ───
        gb_cond = QtWidgets.QGroupBox("Условия")
        form2 = QtWidgets.QFormLayout(gb_cond)
        form2.setSpacing(6)

        self.sp_Pc = QtWidgets.QDoubleSpinBox()
        self.sp_Pc.setRange(0.01, 1000.0)
        self.sp_Pc.setDecimals(4)
        self.sp_Pc.setValue(10.0)
        self.sp_Pc.setSuffix(" МПа")
        self.sp_Pc.setSingleStep(0.5)

        self.sp_Pe = QtWidgets.QDoubleSpinBox()
        self.sp_Pe.setRange(0.0001, 100.0)
        self.sp_Pe.setDecimals(5)
        self.sp_Pe.setValue(0.1013)
        self.sp_Pe.setSuffix(" МПа")
        self.sp_Pe.setSingleStep(0.01)

        self.sp_n_inter = QtWidgets.QSpinBox()
        self.sp_n_inter.setRange(0, 50)
        self.sp_n_inter.setValue(8)
        self.sp_n_inter.setToolTip(
            "Число промежуточных сечений между горловиной и срезом.\n"
            "Чем больше — тем гладче графики, но дольше расчёт."
        )

        self.chk_condensed = QtWidgets.QCheckBox("Учитывать конденсат")
        self.chk_condensed.setChecked(True)

        form2.addRow("Давление в камере:", self.sp_Pc)
        form2.addRow("Давление на срезе:", self.sp_Pe)
        form2.addRow("Промежут. сечений:", self.sp_n_inter)
        form2.addRow("", self.chk_condensed)
        layout.addWidget(gb_cond)

        # ─── Решатель ───
        gb_solver = QtWidgets.QGroupBox("Решатель")
        form3 = QtWidgets.QVBoxLayout(gb_solver)
        form3.setSpacing(4)
        self.rb_own = QtWidgets.QRadioButton("Собственный (минимизация G, NASA-9)")
        self.rb_own.setChecked(True)
        self.rb_cea = QtWidgets.QRadioButton(
            "CEA (Cantera, эквивалент NASA CEA)" +
            ("" if CANTERA_AVAILABLE else "  ⛔ не установлен")
        )
        self.rb_cea.setEnabled(CANTERA_AVAILABLE)
        form3.addWidget(self.rb_own)
        form3.addWidget(self.rb_cea)

        info = QtWidgets.QLabel(
            "Собственный решатель использует NASA-9 полиномы и SLSQP.\n"
            "CEA-решатель (Cantera) даёт идентичные результаты NASA CEA."
        )
        info.setStyleSheet("color: #a8a29e; font-size: 10px;")
        info.setWordWrap(True)
        form3.addWidget(info)
        layout.addWidget(gb_solver)

        # ─── Геометрия сопла (для оси X на графиках) ───
        gb_geom = QtWidgets.QGroupBox("Геометрия сопла (для оси X)")
        form4 = QtWidgets.QFormLayout(gb_geom)
        form4.setSpacing(6)
        self.sp_L_chamber = QtWidgets.QDoubleSpinBox()
        self.sp_L_chamber.setRange(0.001, 10.0)
        self.sp_L_chamber.setDecimals(3)
        self.sp_L_chamber.setValue(0.100)
        self.sp_L_chamber.setSuffix(" м")
        self.sp_L_conv = QtWidgets.QDoubleSpinBox()
        self.sp_L_conv.setRange(0.001, 10.0)
        self.sp_L_conv.setDecimals(3)
        self.sp_L_conv.setValue(0.050)
        self.sp_L_conv.setSuffix(" м")
        self.sp_L_div = QtWidgets.QDoubleSpinBox()
        self.sp_L_div.setRange(0.001, 10.0)
        self.sp_L_div.setDecimals(3)
        self.sp_L_div.setValue(0.200)
        self.sp_L_div.setSuffix(" м")
        form4.addRow("Длина камеры:", self.sp_L_chamber)
        form4.addRow("Конфузор:", self.sp_L_conv)
        form4.addRow("Дивергент:", self.sp_L_div)
        layout.addWidget(gb_geom)

        self.btn_engine_params = QtWidgets.QPushButton("⚙  Параметры двигателя…")
        self.btn_engine_params.clicked.connect(self._open_engine_params_dialog)
        layout.addWidget(self.btn_engine_params)

        self.lbl_optimize_hint = QtWidgets.QLabel("Оптимизация O/F: выключена")
        self.lbl_optimize_hint.setStyleSheet("color: #a8a29e;")
        layout.addWidget(self.lbl_optimize_hint)

        # ─── Кнопки ───
        self.btn_calc = QtWidgets.QPushButton("▶  Рассчитать сопло")
        self.btn_calc.setObjectName("primary")
        self.btn_calc.setMinimumHeight(40)
        self.btn_calc.clicked.connect(self.on_calculate)
        layout.addWidget(self.btn_calc)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.btn_export = QtWidgets.QPushButton("⬇  Экспорт в CSV…")
        self.btn_export.clicked.connect(self.on_export_csv)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        self.btn_export_amesim = QtWidgets.QPushButton("⬇  Экспорт в формате Amesim (.data)")
        self.btn_export_amesim.clicked.connect(self.on_export_amesim)
        self.btn_export_amesim.setEnabled(False)
        layout.addWidget(self.btn_export_amesim)

        layout.addStretch(1)
        return w

    def _build_table_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.tbl_stations = QtWidgets.QTableWidget(0, 0)
        self.tbl_stations.setAlternatingRowColors(True)
        self.tbl_stations.horizontalHeader().setStretchLastSection(False)
        self.tbl_stations.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.tbl_stations)
        return w

    def _build_plot_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(4, 4, 4, 4)

        # Слева — графики (сетка 2x2)
        plot_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plot_widget)
        grid.setSpacing(4)

        self.canvas_PT = MplCanvas(width=5, height=3.5)
        self.canvas_VM = MplCanvas(width=5, height=3.5)
        self.canvas_RHO = MplCanvas(width=5, height=3.5)
        self.canvas_PROFILE = MplCanvas(width=5, height=3.5)

        grid.addWidget(self.canvas_PT, 0, 0)
        grid.addWidget(self.canvas_VM, 0, 1)
        grid.addWidget(self.canvas_RHO, 1, 0)
        grid.addWidget(self.canvas_PROFILE, 1, 1)
        h.addWidget(plot_widget, 1)

        # Справа — панель настройки стиля
        side = QtWidgets.QWidget()
        side.setMaximumWidth(260)
        side_v = QtWidgets.QVBoxLayout(side)
        side_v.setContentsMargins(4, 4, 4, 4)

        gb_style = QtWidgets.QGroupBox("Оформление графиков")
        sf = QtWidgets.QFormLayout(gb_style)
        sf.setSpacing(4)

        self.cb_font = QtWidgets.QComboBox()
        self.cb_font.addItems([
            'DejaVu Serif', 'DejaVu Sans', 'Times New Roman',
            'Liberation Serif', 'Liberation Sans', 'Arial', 'CMU Serif',
        ])
        self.cb_font.setEditable(True)
        sf.addRow("Шрифт:", self.cb_font)

        self.sp_font_axis = QtWidgets.QSpinBox()
        self.sp_font_axis.setRange(6, 30)
        self.sp_font_axis.setValue(12)
        sf.addRow("Подписи осей:", self.sp_font_axis)

        self.sp_font_tick = QtWidgets.QSpinBox()
        self.sp_font_tick.setRange(6, 30)
        self.sp_font_tick.setValue(10)
        sf.addRow("Тики:", self.sp_font_tick)

        self.sp_font_leg = QtWidgets.QSpinBox()
        self.sp_font_leg.setRange(6, 30)
        self.sp_font_leg.setValue(9)
        sf.addRow("Легенда:", self.sp_font_leg)

        self.sp_lw = QtWidgets.QDoubleSpinBox()
        self.sp_lw.setRange(0.1, 10.0)
        self.sp_lw.setSingleStep(0.1)
        self.sp_lw.setValue(1.8)
        sf.addRow("Толщ. линий:", self.sp_lw)

        self.chk_markers = QtWidgets.QCheckBox("Маркеры на точках")
        self.chk_markers.setChecked(True)
        sf.addRow("", self.chk_markers)

        self.chk_grid_major = QtWidgets.QCheckBox("Основная сетка")
        self.chk_grid_major.setChecked(True)
        sf.addRow("", self.chk_grid_major)

        self.chk_grid_minor = QtWidgets.QCheckBox("Доп. сетка")
        self.chk_grid_minor.setChecked(True)
        sf.addRow("", self.chk_grid_minor)

        self.chk_dark_plot = QtWidgets.QCheckBox("Тёмный фон графиков")
        self.chk_dark_plot.setChecked(True)
        sf.addRow("", self.chk_dark_plot)

        self.cb_tick_dir = QtWidgets.QComboBox()
        self.cb_tick_dir.addItems(['in (ГОСТ)', 'out', 'inout'])
        sf.addRow("Тики направл.:", self.cb_tick_dir)

        btn_apply = QtWidgets.QPushButton("Применить стиль")
        btn_apply.clicked.connect(self._redraw_plots)
        sf.addRow("", btn_apply)

        btn_save = QtWidgets.QPushButton("⬇  Сохранить рисунки (PNG)")
        btn_save.clicked.connect(self._save_figures)
        sf.addRow("", btn_save)

        side_v.addWidget(gb_style)
        side_v.addStretch(1)
        h.addWidget(side)
        return w

    def _build_species_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Показывать:"))
        self.rb_mole = QtWidgets.QRadioButton("Мольные доли")
        self.rb_mole.setChecked(True)
        self.rb_mass = QtWidgets.QRadioButton("Массовые доли")
        self.rb_mole.toggled.connect(self._refresh_species_view)
        ctrl.addWidget(self.rb_mole)
        ctrl.addWidget(self.rb_mass)
        ctrl.addStretch(1)
        ctrl.addWidget(QtWidgets.QLabel("Топ компонентов:"))
        self.sp_topN = QtWidgets.QSpinBox()
        self.sp_topN.setRange(3, 50)
        self.sp_topN.setValue(15)
        self.sp_topN.valueChanged.connect(self._refresh_species_view)
        ctrl.addWidget(self.sp_topN)
        v.addLayout(ctrl)

        self.tbl_species = QtWidgets.QTableWidget(0, 0)
        self.tbl_species.setAlternatingRowColors(True)
        v.addWidget(self.tbl_species, 2)

        # График состава по длине
        self.canvas_species = MplCanvas(width=6, height=3.5)
        v.addWidget(self.canvas_species, 3)
        return w

    def _build_perf_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)

        self.txt_perf = QtWidgets.QPlainTextEdit()
        self.txt_perf.setReadOnly(True)
        self.txt_perf.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas','DejaVu Sans Mono',monospace; "
            "font-size: 11pt; }"
        )
        v.addWidget(self.txt_perf)
        return w

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&Файл")
        a_save_csv = m_file.addAction("Экспорт CSV…")
        a_save_csv.triggered.connect(self.on_export_csv)
        a_save_amesim = m_file.addAction("Экспорт Amesim (.data)…")
        a_save_amesim.triggered.connect(self.on_export_amesim)
        m_file.addSeparator()
        a_save_cfg = m_file.addAction("Сохранить конфигурацию…")
        a_save_cfg.triggered.connect(self.on_save_config)
        a_load_cfg = m_file.addAction("Загрузить конфигурацию…")
        a_load_cfg.triggered.connect(self.on_load_config)
        m_file.addSeparator()
        a_exit = m_file.addAction("Выход")
        a_exit.triggered.connect(self.close)

        m_help = mb.addMenu("&Справка")
        a_about = m_help.addAction("О программе…")
        a_about.triggered.connect(self._about)

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, "О программе",
            f"<h3>{APP_NAME}</h3>"
            f"<p>Версия {APP_VERSION}</p>"
            "<p>Расчёт газодинамики ракетного сопла в равновесном приближении.</p>"
            "<p>Решатели: собственный (NASA-9, минимизация G) и CEA (Cantera).</p>"
            "<p>Интерфейс выполнен по мотивам RPA (Rocket Propulsion Analysis).</p>"
        )

    # ───────────────────────── Логика ─────────────────────────

    def _preload_species_db(self):
        try:
            self.statusBar().showMessage("Загрузка базы NASA-9 (thermo.inp)...")
            QtWidgets.QApplication.processEvents()
            db_path = find_thermo_db()
            self.species_db = parse_thermo_file(db_path)
            
            # Обновить mixture_widget с загруженной базой
            self.mixture_widget.species_db = self.species_db
            # Обновить species_db в обоих списках компонентов
            self.mixture_widget.oxidizer_list.species_db = self.species_db
            self.mixture_widget.fuel_list.species_db = self.species_db
            
            # Инициализировать с стандартными компонентами
            self.mixture_widget.set_mixture({
                'ox_components': [{'name': 'O2(L)', 'mass': 7.937, 'T': 0}],
                'fuel_components': [{'name': 'H2(L)', 'mass': 1.000, 'T': 0}],
            })
            self._update_of_from_mixture()
            
            self.statusBar().showMessage(
                f"База NASA-9 загружена: {len(self.species_db)} веществ. Готово."
            )
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки базы: {e}")

    def _update_of_label(self):
        """(Устарело) Обновить O/F (заменено на _update_of_from_mixture)."""
        pass

    def _update_of_from_mixture(self):
        """Обновить O/F из текущей смеси компонентов."""
        mixture = self.mixture_widget.get_mixture()
        ox_mass = sum(c['mass'] for c in mixture['ox_components'])
        fu_mass = sum(c['mass'] for c in mixture['fuel_components'])
        if fu_mass > 1e-9:
            of = ox_mass / fu_mass
            self.lbl_of.setText(f"O/F = {of:.4f}")
        else:
            self.lbl_of.setText("O/F = ∞")

    def _current_engine_params(self) -> Dict:
        return {
            'Pc_MPa': self.sp_Pc.value(),
            'Pe_MPa': self.sp_Pe.value(),
            'n_inter': self.sp_n_inter.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'L_chamber': self.sp_L_chamber.value(),
            'L_conv': self.sp_L_conv.value(),
            'L_div': self.sp_L_div.value(),
            'optimize_of': self.optimize_of,
            'of_min': self.of_min,
            'of_max': self.of_max,
            'of_steps': self.of_steps,
        }

    def _apply_engine_params(self, values: Dict):
        self.sp_Pc.setValue(values['Pc_MPa'])
        self.sp_Pe.setValue(values['Pe_MPa'])
        self.sp_n_inter.setValue(values['n_inter'])
        self.chk_condensed.setChecked(values['include_condensed'])
        self.sp_L_chamber.setValue(values['L_chamber'])
        self.sp_L_conv.setValue(values['L_conv'])
        self.sp_L_div.setValue(values['L_div'])

        self.optimize_of = bool(values.get('optimize_of', False))
        self.of_min = float(values.get('of_min', 2.0))
        self.of_max = float(values.get('of_max', 12.0))
        self.of_steps = int(values.get('of_steps', 15))

        if self.optimize_of:
            self.lbl_optimize_hint.setText(
                f"Оптимизация O/F: ON ({self.of_min:.3f} … {self.of_max:.3f}, шагов: {self.of_steps})"
            )
        else:
            self.lbl_optimize_hint.setText("Оптимизация O/F: выключена")

    def _open_engine_params_dialog(self):
        dialog = EngineParamsDialog(self._current_engine_params(), parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            values = dialog.values()
            if values['of_max'] <= values['of_min']:
                QtWidgets.QMessageBox.warning(
                    self, "Некорректный диапазон O/F",
                    "Параметр O/F макс должен быть больше O/F мин."
                )
                return
            self._apply_engine_params(values)

    def _format_components_for_display(self, components: List[Dict]) -> str:
        if not components:
            return "—"
        return ", ".join(
            f"{c['name']} ({float(c.get('mass', 0.0)):.4g} кг)" for c in components
        )

    def _open_component_selector(self):
        """(Устарено) Открыть диалог выбора компонентов."""
        pass

    def on_calculate(self):
        # Получить смесь из виджета
        mixture = self.mixture_widget.get_mixture()
        
        # Проверить, что указаны компоненты
        if not mixture['ox_components'] or not mixture['fuel_components']:
            QtWidgets.QMessageBox.warning(
                self, "Ошибка",
                "Укажите хотя бы один компонент окислителя и горючего."
            )
            return
        
        params = {
            'ox_components': mixture['ox_components'],
            'fuel_components': mixture['fuel_components'],
            'P_chamber': self.sp_Pc.value() * 1e6,
            'P_exit': self.sp_Pe.value() * 1e6,
            'n_inter': self.sp_n_inter.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'optimize_of': self.optimize_of,
            'of_min': self.of_min,
            'of_max': self.of_max,
            'of_steps': self.of_steps,
        }
        solver = 'cea' if self.rb_cea.isChecked() else 'own'

        if solver == 'own' and self.species_db is None:
            QtWidgets.QMessageBox.warning(
                self, "База NASA-9 не загружена",
                "База NASA-9 ещё не загружена или не найдена.\n"
                "Подождите 1–2 секунды и повторите."
            )
            return

        self.btn_calc.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_export_amesim.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage(f"Расчёт ({solver})... подождите.")

        self.worker = NozzleSolverWorker(params, solver, self.species_db)
        self.worker.progress.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.finished_ok.connect(self._on_calc_done)
        self.worker.failed.connect(self._on_calc_failed)
        self.worker.start()

    def _on_calc_done(self, perf: RocketPerformance):
        self.perf = perf
        self.btn_calc.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_amesim.setEnabled(True)
        self.progress.setVisible(False)

        if hasattr(perf, 'optimized_ox_components') and hasattr(perf, 'optimized_fuel_components'):
            self.mixture_widget.set_mixture({
                'ox_components': list(getattr(perf, 'optimized_ox_components')),
                'fuel_components': list(getattr(perf, 'optimized_fuel_components')),
            })
            self._update_of_from_mixture()

        suffix = ""
        if hasattr(perf, 'optimized_of'):
            suffix = f", O/F_opt = {getattr(perf, 'optimized_of'):.4f}"

        self.statusBar().showMessage(
            f"Готово. T_камеры = {perf.stations[0].T_K:.1f} К, "
            f"Isp = {perf.Isp_s:.2f} с, Cstar = {perf.Cstar_m_per_s:.1f} м/с{suffix}"
        )
        self._fill_stations_table(perf)
        self._fill_perf_text(perf)
        self._fill_species_table(perf)
        self._redraw_plots()
        self._draw_species_plot()

    def _on_calc_failed(self, msg: str):
        self.btn_calc.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Ошибка расчёта.")
        QtWidgets.QMessageBox.critical(self, "Ошибка расчёта", msg[:2000])

    # ─── Заполнение таблиц и текста ───

    def _fill_stations_table(self, perf: RocketPerformance):
        stations = perf.stations
        params = [
            ("Параметр",         lambda s: "",                                  ""),
            ("Давление",         lambda s: f"{s.P_Pa/1e6:.4f}",                 "МПа"),
            ("Температура",      lambda s: f"{s.T_K:.4f}",                      "К"),
            ("Энтальпия",        lambda s: f"{s.H_J_per_kg/1000:.4f}",          "кДж/кг"),
            ("Энтропия",         lambda s: f"{s.S_J_per_kgK/1000:.4f}",         "кДж/(кг·К)"),
            ("Внутр. энергия",   lambda s: f"{s.U_J_per_kg/1000:.4f}",          "кДж/кг"),
            ("Cp (eq.)",         lambda s: f"{s.cp_eq_J_per_kgK/1000:.4f}",     "кДж/(кг·К)"),
            ("Cv (eq.)",         lambda s: f"{s.cv_eq_J_per_kgK/1000:.4f}",     "кДж/(кг·К)"),
            ("γ (eq.)",          lambda s: f"{s.gamma_eq:.4f}",                 ""),
            ("γₛ (изентр.)",     lambda s: f"{s.gamma_s:.4f}",                  ""),
            ("Газовая постоянная", lambda s: f"{s.R_specific_J_per_kgK/1000:.4f}", "кДж/(кг·К)"),
            ("Молярная масса",   lambda s: f"{s.mw_g_per_mol:.4f}",             "кг/кмоль"),
            ("Плотность",        lambda s: f"{s.rho_kg_per_m3:.4f}",            "кг/м³"),
            ("Скорость звука",   lambda s: f"{s.a_m_per_s:.4f}",                "м/с"),
            ("Скорость потока",  lambda s: f"{s.V_m_per_s:.4f}",                "м/с"),
            ("Число Маха",       lambda s: f"{s.M:.4f}",                        ""),
            ("Ae/At",            lambda s: ("∞" if (not math.isfinite(s.Ae_At) or s.Ae_At > 1e5)
                                              else f"{s.Ae_At:.4f}"),           ""),
            ("Массовый поток ρV", lambda s: f"{s.mass_flux_kg_per_m2_s:.4f}",   "кг/(м²·с)"),
        ]

        t = self.tbl_stations
        t.clear()
        t.setRowCount(len(params) - 1)
        t.setColumnCount(len(stations) + 2)
        headers = ["Параметр"] + [s.label for s in stations] + ["Ед.изм."]
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)

        for r, (name, fn, unit) in enumerate(params[1:]):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            for c, s in enumerate(stations, start=1):
                item = QtWidgets.QTableWidgetItem(fn(s))
                item.setTextAlignment(Qt.AlignCenter)
                t.setItem(r, c, item)
            t.setItem(r, len(stations) + 1, QtWidgets.QTableWidgetItem(unit))
        t.resizeColumnsToContents()
        # Установим минимальную ширину колонки 80
        for c in range(t.columnCount()):
            if t.columnWidth(c) < 80:
                t.setColumnWidth(c, 80)

    def _fill_perf_text(self, perf: RocketPerformance):
        mixture = self.mixture_widget.get_mixture()

        s = []
        s.append("═" * 70)
        s.append("  ТЯГОВЫЕ ХАРАКТЕРИСТИКИ")
        s.append("═" * 70)
        s.append("")
        s.append(f"  Окислитель:           {self._format_components_for_display(mixture['ox_components'])}")
        s.append(f"  Горючее:              {self._format_components_for_display(mixture['fuel_components'])}")
        s.append(f"  Массовое O/F:         {perf.O_F:.4f}")
        if hasattr(perf, 'optimized_of'):
            s.append(f"  O/F (оптимум):        {getattr(perf, 'optimized_of'):.4f}")
        if not math.isnan(perf.O_F_stoich):
            s.append(f"  Стехиометр. O/F:      {perf.O_F_stoich:.4f}")
        if not math.isnan(perf.alpha):
            s.append(f"  α (избыток окисл.):   {perf.alpha:.4f}")
        if not math.isnan(perf.phi):
            s.append(f"  φ (эквив. отнош.):    {perf.phi:.4f}")
        s.append("")
        s.append(f"  Давление в камере:    {perf.stations[0].P_Pa/1e6:.4f} МПа")
        s.append(f"  Давление на срезе:    {perf.stations[-1].P_Pa/1e6:.4f} МПа")
        s.append(f"  Геометрич. степень:   Ae/At = {perf.stations[-1].Ae_At:.4f}")
        s.append("")
        s.append("─" * 70)
        s.append("  Удельный импульс / характеристическая скорость:")
        s.append("─" * 70)
        s.append(f"  Isp (срез,   P_amb=0):    {perf.Isp_s:8.4f} с")
        s.append(f"  Isp (вакуум):             {perf.Isp_vac_s:8.4f} с")
        s.append(f"  C* (характеристическая):  {perf.Cstar_m_per_s:8.4f} м/с")
        s.append(f"  CF (коэф. тяги):          {perf.CF:8.4f}")
        s.append(f"  Ve (скорость на срезе):   {perf.stations[-1].V_m_per_s:8.4f} м/с")
        s.append("")
        s.append("═" * 70)
        s.append("  ПАРАМЕТРЫ В КАМЕРЕ И НА СРЕЗЕ")
        s.append("═" * 70)
        st_c, st_e = perf.stations[0], perf.stations[-1]
        s.append(f"  T_camera:  {st_c.T_K:8.2f} К      |  T_exit:  {st_e.T_K:8.2f} К")
        s.append(f"  ρ_camera:  {st_c.rho_kg_per_m3:8.4f} кг/м³ |  ρ_exit:  {st_e.rho_kg_per_m3:8.4f} кг/м³")
        s.append(f"  γs_camera: {st_c.gamma_s:8.4f}      |  γs_exit: {st_e.gamma_s:8.4f}")
        s.append(f"  M_camera:  {st_c.M:8.4f}      |  M_exit:  {st_e.M:8.4f}")
        s.append(f"  MW_camera: {st_c.mw_g_per_mol:8.4f} г/моль|  MW_exit: {st_e.mw_g_per_mol:8.4f} г/моль")
        s.append("")
        self.txt_perf.setPlainText("\n".join(s))

    def _fill_species_table(self, perf: RocketPerformance):
        self._refresh_species_view()

    def _refresh_species_view(self):
        if self.perf is None:
            return
        stations = self.perf.stations
        sp_names = stations[0].species_names
        N = len(sp_names)

        use_mole = self.rb_mole.isChecked()
        topN = self.sp_topN.value()

        # Найдём top по максимальной доле во всех станциях
        max_frac = np.zeros(N)
        for st in stations:
            frac = st.mole_fractions if use_mole else st.mass_fractions
            if frac is not None and len(frac) == N:
                max_frac = np.maximum(max_frac, frac)

        order = np.argsort(-max_frac)[:topN]
        order = [i for i in order if max_frac[i] > 1e-9]

        t = self.tbl_species
        t.clear()
        t.setRowCount(len(order))
        t.setColumnCount(len(stations) + 1)
        headers = ["Компонент"] + [s.label for s in stations]
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)

        for r, idx in enumerate(order):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(sp_names[idx]))
            for c, st in enumerate(stations, start=1):
                frac = st.mole_fractions if use_mole else st.mass_fractions
                v = frac[idx] if frac is not None and idx < len(frac) else 0.0
                item = QtWidgets.QTableWidgetItem(f"{v:.6e}")
                item.setTextAlignment(Qt.AlignCenter)
                t.setItem(r, c, item)
        t.resizeColumnsToContents()

        self._draw_species_plot()

    # ─── Графики ───

    def _collect_style(self) -> PlotStyle:
        s = PlotStyle()
        s.font_family = self.cb_font.currentText().strip() or 'DejaVu Serif'
        s.font_size_axis = self.sp_font_axis.value()
        s.font_size_tick = self.sp_font_tick.value()
        s.font_size_legend = self.sp_font_leg.value()
        s.line_width = self.sp_lw.value()
        s.show_markers = self.chk_markers.isChecked()
        s.grid_major = self.chk_grid_major.isChecked()
        s.grid_minor = self.chk_grid_minor.isChecked()
        s.dark_plot = self.chk_dark_plot.isChecked()
        s.tick_direction = self.cb_tick_dir.currentText().split()[0]
        return s

    def _redraw_plots(self):
        if self.perf is None:
            return
        style = self._collect_style()
        self.plot_style = style

        stations = self.perf.stations
        # X — координата по длине сопла
        x = build_nozzle_geometry(
            stations,
            L_chamber=self.sp_L_chamber.value(),
            L_conv=self.sp_L_conv.value(),
            L_div=self.sp_L_div.value(),
        )

        # ─── Плот 1: P, T ───
        c = self.canvas_PT
        c.fig.clear()
        ax1 = c.fig.add_subplot(111)
        P = np.array([s.P_Pa / 1e6 for s in stations])
        T = np.array([s.T_K for s in stations])

        l1, = ax1.plot(x, P, 'o-' if style.show_markers else '-',
                       color='#cc785c', lw=style.line_width, ms=style.marker_size,
                       label='P, МПа')
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("Давление P, МПа")

        ax2 = ax1.twinx()
        l2, = ax2.plot(x, T, 's--' if style.show_markers else '--',
                       color='#6ab0ff', lw=style.line_width, ms=style.marker_size,
                       label='T, К')
        ax2.set_ylabel("Температура T, К")
        ax2.legend(handles=[l1, l2], loc='best')

        style.title = "Давление и температура по длине сопла"
        style.xlabel = "Координата x, м"
        style.ylabel = "Давление P, МПа"
        apply_plot_style(c.fig, ax1, style)
        # Применим dark fg к ax2 тоже
        self._style_twinx(ax2, style)
        c.fig.tight_layout()
        c.draw()

        # ─── Плот 2: V, M ───
        c = self.canvas_VM
        c.fig.clear()
        ax1 = c.fig.add_subplot(111)
        V = np.array([s.V_m_per_s for s in stations])
        M = np.array([s.M for s in stations])
        l1, = ax1.plot(x, V, 'o-' if style.show_markers else '-',
                       color='#82d27a', lw=style.line_width, ms=style.marker_size,
                       label='V, м/с')
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("�