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

# Импорт решателей (всё через пакет fuel_equilibrium)
from ..rocket.nozzle_flow import (
    Propellant, StationResult, RocketPerformance,
    solve_rocket_nozzle,
)
from ..io.reporting import print_nozzle_table
from ..core.nasa9_parser import parse_thermo_file
from ..core.equilibrium import find_thermo_db
from ..io.iteration_logger import IterationLogger, NullLogger
from .component_selector import ComponentSelectorDialog, ComponentListWidget, MixturePropellantWidget

try:
    from ..rocket.cea_solver import (
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

    def run(self):
        try:
            p = self.params
            
            # Внутри окислителя/горючего «масса» задаёт долю компонента (0.001..1).
            # Суммарное O/F задаётся отдельно и не зависит от этих долей.
            ox_components = p['ox_components']  # List[Dict{'name', 'mass', 'T'}]
            fu_components = p['fuel_components']
            of_ratio = max(float(p.get('of_ratio', 1.0)), 1e-9)

            # Нормировка на 1 кг суммарной смеси по заданному O/F.
            fuel_mass_kg = 1.0 / (1.0 + of_ratio)
            oxidizer_mass_kg = of_ratio / (1.0 + of_ratio)

            # На текущем этапе решатель принимает по одному «эквивалентному» компоненту
            # окислителя и горючего; берем первый в каждом списке.
            ox_comp = ox_components[0]
            ox_T = ox_comp['T'] if ox_comp['T'] > 0 else None
            ox = Propellant(name=ox_comp['name'], mass_kg=oxidizer_mass_kg, T_K=ox_T)

            fu_comp = fu_components[0]
            fu_T = fu_comp['T'] if fu_comp['T'] > 0 else None
            fu = Propellant(name=fu_comp['name'], mass_kg=fuel_mass_kg, T_K=fu_T)

            if self.solver == 'cea':
                self.progress.emit("Запуск CEA-решателя (Cantera)...")
                perf = solve_rocket_nozzle_cea(
                    oxidizer=ox, fuel=fu,
                    P_chamber=p['P_chamber'],
                    P_exit=p['P_exit'],
                    n_intermediate_stations=p.get('n_inter', 5),
                    include_condensed=p.get('include_condensed', False),
                    verbose=False,
                    progress_cb=lambda s: self.progress.emit(s),
                )
            else:
                self.progress.emit("Запуск собственного решателя (Gibbs)...")
                perf = solve_rocket_nozzle(
                    oxidizer=ox, fuel=fu,
                    P_chamber=p['P_chamber'],
                    P_exit=p['P_exit'],
                    species_db=self.species_db,
                    n_intermediate_stations=p.get('n_inter', 5),
                    include_condensed=p.get('include_condensed', True),
                    verbose=False,
                    logger=NullLogger(),
                )
            self.finished_ok.emit(perf)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")


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

        self._build_ui()
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
        
        # Отношение O/F задаётся отдельно от внутритопливных долей компонентов.
        self.sp_of_ratio = QtWidgets.QDoubleSpinBox()
        self.sp_of_ratio.setRange(0.01, 1000.0)
        self.sp_of_ratio.setDecimals(4)
        self.sp_of_ratio.setValue(7.9370)
        self.sp_of_ratio.setSingleStep(0.1)
        self.sp_of_ratio.setToolTip(
            "Массовое отношение окислителя к горючему (O/F).\n"
            "Это значение задаётся отдельно от долей компонентов внутри\n"
            "окислителя и горючего."
        )
        self.sp_of_ratio.valueChanged.connect(self._update_of_from_mixture)

        self.lbl_of = QtWidgets.QLabel("O/F = 7.9370")
        self.lbl_of.setStyleSheet("color: #cc785c; font-weight: bold;")
        of_layout = QtWidgets.QHBoxLayout()
        of_layout.addWidget(QtWidgets.QLabel("Заданное O/F:"))
        of_layout.addWidget(self.sp_of_ratio)
        of_layout.addWidget(self.lbl_of)
        of_layout.addStretch()
        gb_fuel_layout.addLayout(of_layout)

        layout.addWidget(gb_fuel)

        # ─── Условия в сопле ───
        gb_cond = QtWidgets.QGroupBox("Условия")
        form2 = QtWidgets.QFormLayout(gb_cond)
        form2.setSpacing(6)

        self.sp_Pc = QtWidgets.QDoubleSpinBox()
        self.sp_Pc.setRange(0.000001, 1e6)
        self.sp_Pc.setDecimals(6)
        self.sp_Pc.setValue(10.0)
        self.sp_Pc.setSingleStep(0.5)
        # контейнер со списком единиц
        w_Pc = QtWidgets.QWidget()
        h_Pc = QtWidgets.QHBoxLayout(w_Pc)
        h_Pc.setContentsMargins(0, 0, 0, 0)
        h_Pc.addWidget(self.sp_Pc)
        self.cb_Pc_unit = QtWidgets.QComboBox()
        self.cb_Pc_unit.addItems(["Па", "кПа", "МПа", "бар", "атм"])
        self.cb_Pc_unit.setCurrentText("МПа")
        h_Pc.addWidget(self.cb_Pc_unit)

        self.sp_Pe = QtWidgets.QDoubleSpinBox()
        self.sp_Pe.setRange(0.0000001, 1e6)
        self.sp_Pe.setDecimals(6)
        self.sp_Pe.setValue(0.1013)
        self.sp_Pe.setSingleStep(0.01)
        w_Pe = QtWidgets.QWidget()
        h_Pe = QtWidgets.QHBoxLayout(w_Pe)
        h_Pe.setContentsMargins(0, 0, 0, 0)
        h_Pe.addWidget(self.sp_Pe)
        self.cb_Pe_unit = QtWidgets.QComboBox()
        self.cb_Pe_unit.addItems(["Па", "кПа", "МПа", "бар", "атм"])
        self.cb_Pe_unit.setCurrentText("МПа")
        h_Pe.addWidget(self.cb_Pe_unit)

        self.sp_n_inter = QtWidgets.QSpinBox()
        self.sp_n_inter.setRange(0, 50)
        self.sp_n_inter.setValue(8)
        self.sp_n_inter.setToolTip(
            "Число промежуточных сечений только для газодинамических параметров\n"
            "между горловиной и срезом.\n"
            "Состав продуктов всегда показывается на 4 сечениях:\n"
            "Injector, Nozzle inlet, Nozzle throat, Nozzle exit."
        )

        self.chk_condensed = QtWidgets.QCheckBox("Учитывать конденсат")
        self.chk_condensed.setChecked(True)

        form2.addRow("Давление в камере:", w_Pc)
        form2.addRow("Давление на срезе:", w_Pe)
        form2.addRow("Промежут. сечений (газодин.):", self.sp_n_inter)
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
        self.sp_L_chamber.setRange(0.000, 1000.0)
        self.sp_L_chamber.setDecimals(4)
        self.sp_L_chamber.setValue(0.100)
        self.sp_L_chamber.setSingleStep(0.01)
        w_L_ch = QtWidgets.QWidget()
        h_L_ch = QtWidgets.QHBoxLayout(w_L_ch)
        h_L_ch.setContentsMargins(0,0,0,0)
        h_L_ch.addWidget(self.sp_L_chamber)
        self.cb_L_chamber_unit = QtWidgets.QComboBox()
        self.cb_L_chamber_unit.addItems(["м", "см", "мм"])
        self.cb_L_chamber_unit.setCurrentText("м")
        h_L_ch.addWidget(self.cb_L_chamber_unit)

        self.sp_L_conv = QtWidgets.QDoubleSpinBox()
        self.sp_L_conv.setRange(0.000, 1000.0)
        self.sp_L_conv.setDecimals(4)
        self.sp_L_conv.setValue(0.050)
        self.sp_L_conv.setSingleStep(0.01)
        w_L_co = QtWidgets.QWidget()
        h_L_co = QtWidgets.QHBoxLayout(w_L_co)
        h_L_co.setContentsMargins(0,0,0,0)
        h_L_co.addWidget(self.sp_L_conv)
        self.cb_L_conv_unit = QtWidgets.QComboBox()
        self.cb_L_conv_unit.addItems(["м", "см", "мм"])
        self.cb_L_conv_unit.setCurrentText("м")
        h_L_co.addWidget(self.cb_L_conv_unit)

        self.sp_L_div = QtWidgets.QDoubleSpinBox()
        self.sp_L_div.setRange(0.000, 1000.0)
        self.sp_L_div.setDecimals(4)
        self.sp_L_div.setValue(0.200)
        self.sp_L_div.setSingleStep(0.01)
        w_L_di = QtWidgets.QWidget()
        h_L_di = QtWidgets.QHBoxLayout(w_L_di)
        h_L_di.setContentsMargins(0,0,0,0)
        h_L_di.addWidget(self.sp_L_div)
        self.cb_L_div_unit = QtWidgets.QComboBox()
        self.cb_L_div_unit.addItems(["м", "см", "мм"])
        self.cb_L_div_unit.setCurrentText("м")
        h_L_di.addWidget(self.cb_L_div_unit)

        form4.addRow("Длина камеры:", w_L_ch)
        form4.addRow("Конфузор:", w_L_co)
        form4.addRow("Дивергент:", w_L_di)
        layout.addWidget(gb_geom)

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
            
            # Инициализировать стандартной смесью (внутренние доли 1.0/1.0,
            # массовое O/F задаётся отдельно через self.sp_of_ratio).
            self.mixture_widget.set_mixture({
                'ox_components': [{'name': 'O2(L)', 'mass': 1.000, 'T': 0}],
                'fuel_components': [{'name': 'H2(L)', 'mass': 1.000, 'T': 0}],
            })
            self.sp_of_ratio.setValue(7.9370)
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
        """Обновить подпись O/F (задаётся отдельным полем)."""
        of = self.sp_of_ratio.value()
        mixture = self.mixture_widget.get_mixture()
        ox_parts = sum(c['mass'] for c in mixture['ox_components'])
        fu_parts = sum(c['mass'] for c in mixture['fuel_components'])
        self.lbl_of.setText(f"O/F = {of:.4f}  (доли: OX={ox_parts:.3f}, FUEL={fu_parts:.3f})")

    def _open_component_selector(self):
        """(Устарено) Открыть диалог выбора компонентов."""
        pass

    def _get_mixture_summary(self) -> Tuple[str, str]:
        """Строки состава окислителя и горючего в формате Name(frac)."""
        mixture = self.mixture_widget.get_mixture()

        def _fmt(parts: List[Dict]) -> str:
            if not parts:
                return "—"
            total = sum(max(0.0, float(p.get('mass', 0.0))) for p in parts)
            if total <= 1e-12:
                return " + ".join(f"{p.get('name', '?')}(0.000)" for p in parts)
            chunks = []
            for p in parts:
                frac = max(0.0, float(p.get('mass', 0.0))) / total
                chunks.append(f"{p.get('name', '?')}({frac:.3f})")
            return " + ".join(chunks)

        return _fmt(mixture.get('ox_components', [])), _fmt(mixture.get('fuel_components', []))

    @staticmethod
    def _get_composition_station_indices(stations: List[StationResult]) -> List[int]:
        """Индексы 4 сечений для состава: камера, вход в сопло, горловина, срез."""
        target_labels = ["injector", "nozzle inlet", "nozzle throat", "nozzle exit"]
        idx_by_label = {
            str(st.label).strip().lower(): i
            for i, st in enumerate(stations)
        }

        indices: List[int] = []
        for lbl in target_labels:
            idx = idx_by_label.get(lbl)
            if idx is not None and idx not in indices:
                indices.append(idx)

        if not indices and stations:
            indices = [0, len(stations) - 1]

        return sorted(indices)

    def _get_composition_stations(self, stations: List[StationResult]) -> List[StationResult]:
        indices = self._get_composition_station_indices(stations)
        return [stations[i] for i in indices]

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
        
        def pv_to_pa(val, unit):
            if unit == 'Па':
                return val
            if unit == 'кПа':
                return val * 1e3
            if unit == 'МПа':
                return val * 1e6
            if unit == 'бар':
                return val * 1e5
            if unit == 'атм':
                return val * 101325.0
            return val

        params = {
            'ox_components': mixture['ox_components'],
            'fuel_components': mixture['fuel_components'],
            'of_ratio': self.sp_of_ratio.value(),
            'P_chamber': pv_to_pa(self.sp_Pc.value(), self.cb_Pc_unit.currentText()),
            'P_exit': pv_to_pa(self.sp_Pe.value(), self.cb_Pe_unit.currentText()),
            'n_inter': self.sp_n_inter.value(),
            'include_condensed': self.chk_condensed.isChecked(),
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
        self.statusBar().showMessage(
            f"Готово. T_камеры = {perf.stations[0].T_K:.1f} К, "
            f"Isp = {perf.Isp_s:.2f} с, Cstar = {perf.Cstar_m_per_s:.1f} м/с"
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
        s = []
        s.append("═" * 70)
        s.append("  ТЯГОВЫЕ ХАРАКТЕРИСТИКИ")
        s.append("═" * 70)
        s.append("")
        ox_desc, fu_desc = self._get_mixture_summary()
        s.append(f"  Окислитель:           {ox_desc}")
        s.append(f"  Горючее:              {fu_desc}")
        s.append(f"  Массовое O/F:         {perf.O_F:.4f}")
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
        stations = self._get_composition_stations(self.perf.stations)
        if not stations:
            return
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
        def length_to_m(v, unit):
            if unit == 'м':
                return v
            if unit == 'см':
                return v * 0.01
            if unit == 'мм':
                return v * 0.001
            return v

        x = build_nozzle_geometry(
            stations,
            L_chamber=length_to_m(self.sp_L_chamber.value(), self.cb_L_chamber_unit.currentText()),
            L_conv=length_to_m(self.sp_L_conv.value(), self.cb_L_conv_unit.currentText()),
            L_div=length_to_m(self.sp_L_div.value(), self.cb_L_div_unit.currentText()),
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
        ax1.set_ylabel("Скорость потока V, м/с")
        ax2 = ax1.twinx()
        l2, = ax2.plot(x, M, 'D--' if style.show_markers else '--',
                       color='#e6b800', lw=style.line_width, ms=style.marker_size,
                       label='M')
        ax2.set_ylabel("Число Маха M")
        # горизонталь M=1
        ax2.axhline(1.0, color='#a8a29e', lw=0.8, ls=':')
        ax2.legend(handles=[l1, l2], loc='best')
        style2 = self._collect_style()
        style2.title = "Скорость потока и число Маха"
        style2.xlabel = "Координата x, м"
        style2.ylabel = "V, м/с"
        apply_plot_style(c.fig, ax1, style2)
        self._style_twinx(ax2, style2)
        c.fig.tight_layout()
        c.draw()

        # ─── Плот 3: ρ, γs ───
        c = self.canvas_RHO
        c.fig.clear()
        ax1 = c.fig.add_subplot(111)
        rho = np.array([s.rho_kg_per_m3 for s in stations])
        gs = np.array([s.gamma_s for s in stations])
        l1, = ax1.plot(x, rho, 'o-' if style.show_markers else '-',
                       color='#cc785c', lw=style.line_width, ms=style.marker_size,
                       label='ρ, кг/м³')
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("Плотность ρ, кг/м³")
        ax2 = ax1.twinx()
        l2, = ax2.plot(x, gs, '^--' if style.show_markers else '--',
                       color='#c084fc', lw=style.line_width, ms=style.marker_size,
                       label='γₛ')
        ax2.set_ylabel("Изэнтр. показатель γₛ")
        ax2.legend(handles=[l1, l2], loc='best')
        style3 = self._collect_style()
        style3.title = "Плотность и изэнтропич. показатель"
        style3.xlabel = "Координата x, м"
        style3.ylabel = "ρ, кг/м³"
        apply_plot_style(c.fig, ax1, style3)
        self._style_twinx(ax2, style3)
        c.fig.tight_layout()
        c.draw()

        # ─── Плот 4: профиль сопла (геометрия) ───
        c = self.canvas_PROFILE
        c.fig.clear()
        ax = c.fig.add_subplot(111)
        r = nozzle_radius(stations)
        # ограничим радиус камеры разумным значением
        r_max = float(np.max(r))

        # Гладкий контур: интерполируем между ключевыми точками
        x_smooth, r_smooth = self._smooth_profile(stations, x, r)

        ax.plot(x_smooth, r_smooth, '-', color='#cc785c',
                lw=style.line_width * 1.2)
        ax.plot(x_smooth, -r_smooth, '-', color='#cc785c',
                lw=style.line_width * 1.2)
        ax.fill_between(x_smooth, -r_smooth, r_smooth, alpha=0.15, color='#cc785c')

        # Точки сечений
        if style.show_markers:
            ax.plot(x, r, 'o', color='#cc785c', ms=style.marker_size,
                    markeredgecolor='#fafaf9' if style.dark_plot else '#000000',
                    markeredgewidth=0.6)
            ax.plot(x, -r, 'o', color='#cc785c', ms=style.marker_size,
                    markeredgecolor='#fafaf9' if style.dark_plot else '#000000',
                    markeredgewidth=0.6)

        # Подписи сечений (компактные)
        for i, st in enumerate(stations):
            short = st.label.replace('Nozzle ', '').replace('Section ', 'S')[:10]
            ax.annotate(short, (x[i], r[i] + r_max * 0.07),
                        ha='center', fontsize=8, rotation=45,
                        color='#fafaf9' if style.dark_plot else '#000000')
        # Вертикальные пунктирные линии в сечениях
        for i in range(len(stations)):
            ax.axvline(x[i], color='#a8a29e', lw=0.4, ls=':', alpha=0.5)

        style4 = self._collect_style()
        style4.title = "Профиль сопла (R/Rₜ относит.)"
        style4.xlabel = "Координата x, м"
        style4.ylabel = "R / Rₜ"
        ax.set_ylim(-r_max * 1.3, r_max * 1.4)
        apply_plot_style(c.fig, ax, style4)
        c.fig.tight_layout()
        c.draw()

    def _smooth_profile(self, stations, x, r):
        """Создаёт сглаженный профиль сопла из дискретных точек.

        Камера → линейно;
        Конфузор: камера → горловина — параболический спад;
        Дивергент: горловина → срез — конический рост.
        """
        # Найдём индексы ключевых сечений
        labels = [s.label.lower() for s in stations]
        try:
            i_inlet = labels.index('nozzle inlet')
        except ValueError:
            i_inlet = 1
        try:
            i_throat = labels.index('nozzle throat')
        except ValueError:
            i_throat = 2

        n_smooth = 200
        x_out = []
        r_out = []
        # Камера: linearly to inlet
        x_out.extend(np.linspace(x[0], x[i_inlet], 20))
        r_out.extend([r[0]] * 20)
        # Конфузор: cubic from inlet to throat
        xs = np.linspace(x[i_inlet], x[i_throat], 50)
        t = (xs - x[i_inlet]) / max(x[i_throat] - x[i_inlet], 1e-9)
        # smooth-step: 3t^2 - 2t^3 от 1 к 0 (масштаб r[inlet] -> r[throat])
        smooth = 1.0 - (3 * t**2 - 2 * t**3)
        rs = r[i_throat] + (r[i_inlet] - r[i_throat]) * smooth
        x_out.extend(xs)
        r_out.extend(rs)
        # Дивергент: от throat до exit — линейно по x от точек (можно cubic spline)
        if i_throat + 1 < len(x):
            from scipy.interpolate import PchipInterpolator
            try:
                pchip = PchipInterpolator(x[i_throat:], r[i_throat:])
                xs = np.linspace(x[i_throat], x[-1], 100)
                rs = pchip(xs)
                x_out.extend(xs)
                r_out.extend(rs)
            except Exception:
                x_out.extend(x[i_throat:])
                r_out.extend(r[i_throat:])
        return np.array(x_out), np.array(r_out)

    def _style_twinx(self, ax2, style):
        """Применяет dark/light стиль ко второй (правой) оси twinx."""
        fg = '#fafaf9' if style.dark_plot else '#000000'
        ax2.tick_params(axis='y', which='major',
                        labelsize=style.font_size_tick,
                        direction=style.tick_direction,
                        length=5, width=1.0, color=fg, labelcolor=fg)
        ax2.tick_params(axis='y', which='minor',
                        direction=style.tick_direction,
                        length=3, width=0.7, color=fg)
        ax2.yaxis.label.set_color(fg)
        ax2.yaxis.label.set_fontsize(style.font_size_axis)
        ax2.yaxis.label.set_fontfamily(style.font_family)
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        for spine in ax2.spines.values():
            spine.set_color(fg)
            spine.set_linewidth(style.spine_linewidth)
        leg = ax2.get_legend()
        if leg is not None:
            bg = '#262624' if style.dark_plot else '#ffffff'
            leg.get_frame().set_facecolor(bg)
            leg.get_frame().set_edgecolor(fg)
            leg.get_frame().set_alpha(0.85)
            for txt in leg.get_texts():
                txt.set_color(fg)
                txt.set_fontfamily(style.font_family)
                txt.set_fontsize(style.font_size_legend)

    def _draw_species_plot(self):
        if self.perf is None:
            return
        style = self._collect_style()
        all_stations = self.perf.stations

        def length_to_m(v, unit):
            if unit == 'м':
                return v
            if unit == 'см':
                return v * 0.01
            if unit == 'мм':
                return v * 0.001
            return v

        x_all = build_nozzle_geometry(
            all_stations,
            L_chamber=length_to_m(self.sp_L_chamber.value(), self.cb_L_chamber_unit.currentText()),
            L_conv=length_to_m(self.sp_L_conv.value(), self.cb_L_conv_unit.currentText()),
            L_div=length_to_m(self.sp_L_div.value(), self.cb_L_div_unit.currentText()),
        )
        comp_idx = self._get_composition_station_indices(all_stations)
        stations = [all_stations[i] for i in comp_idx]
        x = np.array([x_all[i] for i in comp_idx])
        if not stations:
            return

        sp_names = stations[0].species_names
        N = len(sp_names)
        use_mole = self.rb_mole.isChecked()
        topN = self.sp_topN.value()

        max_frac = np.zeros(N)
        for st in stations:
            frac = st.mole_fractions if use_mole else st.mass_fractions
            if frac is not None and len(frac) == N:
                max_frac = np.maximum(max_frac, frac)
        order = np.argsort(-max_frac)[:topN]
        order = [i for i in order if max_frac[i] > 1e-9]

        c = self.canvas_species
        c.fig.clear()
        ax = c.fig.add_subplot(111)

        palette = plt.get_cmap('tab20').colors
        for k, idx in enumerate(order):
            ys = np.array([
                (st.mole_fractions[idx] if use_mole else st.mass_fractions[idx])
                if (st.mole_fractions is not None and idx < len(st.mole_fractions)) else 0.0
                for st in stations
            ])
            ax.plot(x, ys, '-o' if style.show_markers else '-',
                    color=palette[k % len(palette)],
                    lw=style.line_width, ms=style.marker_size * 0.8,
                    label=sp_names[idx])

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=8)
        s2 = self._collect_style()
        s2.title = ("Мольные" if use_mole else "Массовые") + " доли по длине сопла"
        s2.xlabel = "Координата x, м"
        s2.ylabel = "Мольная доля" if use_mole else "Массовая доля"
        s2.y_log = True
        s2.y_min = max(1e-7, min(1e-3, max_frac.max() / 1e6))
        s2.y_max = 1.0
        apply_plot_style(c.fig, ax, s2)
        try:
            c.fig.tight_layout()
        except Exception:
            pass
        c.draw()

    def _save_figures(self):
        if self.perf is None:
            return
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Папка для сохранения рисунков"
        )
        if not dir_path:
            return
        for name, canvas in [
            ('PT', self.canvas_PT), ('VM', self.canvas_VM),
            ('rho_gamma', self.canvas_RHO), ('profile', self.canvas_PROFILE),
            ('species', self.canvas_species),
        ]:
            path = os.path.join(dir_path, f"nozzle_{name}.png")
            try:
                canvas.fig.savefig(path, dpi=200,
                                   facecolor=canvas.fig.get_facecolor(),
                                   bbox_inches='tight')
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка сохранения", f"{path}\n{e}"
                )
                return
        self.statusBar().showMessage(f"Рисунки сохранены в {dir_path}")

    # ─── Экспорт CSV ───

    def on_export_csv(self):
        if self.perf is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить CSV",
            "nozzle_export.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        stations = self.perf.stations
        def length_to_m(v, unit):
            if unit == 'м':
                return v
            if unit == 'см':
                return v * 0.01
            if unit == 'мм':
                return v * 0.001
            return v

        x = build_nozzle_geometry(
            stations,
            L_chamber=length_to_m(self.sp_L_chamber.value(), self.cb_L_chamber_unit.currentText()),
            L_conv=length_to_m(self.sp_L_conv.value(), self.cb_L_conv_unit.currentText()),
            L_div=length_to_m(self.sp_L_div.value(), self.cb_L_div_unit.currentText()),
        )

        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f, delimiter=';')
                # Заголовок
                w.writerow(['# RPA-Style Rocket Nozzle Calculator — Export'])
                ox_desc, fu_desc = self._get_mixture_summary()
                w.writerow([f'# Окислитель: {ox_desc}'])
                w.writerow([f'# Горючее: {fu_desc}'])
                w.writerow([f'# Pc = {self.sp_Pc.value()} МПа, Pe = {self.sp_Pe.value()} МПа, O/F(set) = {self.sp_of_ratio.value():.4f}'])
                w.writerow([f'# O/F = {self.perf.O_F:.4f}, '
                            f'alpha = {self.perf.alpha:.4f}, '
                            f'Isp = {self.perf.Isp_s:.4f} c, '
                            f'C* = {self.perf.Cstar_m_per_s:.4f} м/с'])
                w.writerow([f'# Решатель: {"CEA (Cantera)" if self.rb_cea.isChecked() else "Собственный (NASA-9)"}'])
                w.writerow([])
                # Заголовок столбцов
                w.writerow([
                    'Сечение', 'x_м', 'P_МПа', 'T_К',
                    'H_кДж/кг', 'S_кДж/кгК', 'U_кДж/кг',
                    'Cp_eq_кДж/кгК', 'Cv_eq_кДж/кгК', 'gamma_eq',
                    'gamma_s_изентр', 'R_кДж/кгК', 'MW_кг/кмоль',
                    'rho_кг/м³', 'a_м/с', 'V_м/с', 'M', 'Ae/At', 'rho*V_кг/(м²с)',
                ])
                for i, s in enumerate(stations):
                    w.writerow([
                        s.label, f"{x[i]:.6f}",
                        f"{s.P_Pa/1e6:.6f}", f"{s.T_K:.4f}",
                        f"{s.H_J_per_kg/1000:.4f}",
                        f"{s.S_J_per_kgK/1000:.4f}",
                        f"{s.U_J_per_kg/1000:.4f}",
                        f"{s.cp_eq_J_per_kgK/1000:.4f}",
                        f"{s.cv_eq_J_per_kgK/1000:.4f}",
                        f"{s.gamma_eq:.5f}", f"{s.gamma_s:.5f}",
                        f"{s.R_specific_J_per_kgK/1000:.5f}",
                        f"{s.mw_g_per_mol:.4f}",
                        f"{s.rho_kg_per_m3:.5f}", f"{s.a_m_per_s:.4f}",
                        f"{s.V_m_per_s:.4f}", f"{s.M:.5f}",
                        ("inf" if (not math.isfinite(s.Ae_At) or s.Ae_At > 1e6)
                         else f"{s.Ae_At:.5f}"),
                        f"{s.mass_flux_kg_per_m2_s:.4f}",
                    ])

                # Состав: только 4 ключевых сечения
                w.writerow([])
                w.writerow(['# Мольные доли продуктов сгорания (Injector / Nozzle inlet / Nozzle throat / Nozzle exit)'])
                comp_stations = self._get_composition_stations(stations)
                sp_names = comp_stations[0].species_names
                w.writerow(['Компонент'] + [s.label for s in comp_stations])
                # Сортируем по максимуму
                max_frac = np.zeros(len(sp_names))
                for st in comp_stations:
                    if st.mole_fractions is not None:
                        max_frac = np.maximum(max_frac, st.mole_fractions)
                for idx in np.argsort(-max_frac):
                    if max_frac[idx] < 1e-8:
                        continue
                    row = [sp_names[idx]]
                    for st in comp_stations:
                        v = st.mole_fractions[idx] if st.mole_fractions is not None else 0.0
                        row.append(f"{v:.6e}")
                    w.writerow(row)

            self.statusBar().showMessage(f"CSV сохранён: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить CSV:\n{e}")

    def on_export_amesim(self):
        """Экспорт в формате Amesim .data (X-Y таблица с метаданными).

        Формат, который понимает MILF PLOTTER:
            # Table format: XY
            # axis1_unit = m
            # axis2_unit = MPa
            # axis1_title = Координата x, м
            # axis2_title = Давление P
            # ...
            x1  y1  y2  y3 ...
        """
        if self.perf is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить в формате Amesim",
            "nozzle_amesim.data",
            "Amesim data (*.data *.dat);;All files (*)"
        )
        if not path:
            return

        stations = self.perf.stations
        x = build_nozzle_geometry(
            stations,
            L_chamber=self.sp_L_chamber.value(),
            L_conv=self.sp_L_conv.value(),
            L_div=self.sp_L_div.value(),
        )

        # Сигналы для экспорта (порядок = порядок столбцов в .data)
        signals = [
            ("Pressure",         "MPa",    [s.P_Pa/1e6 for s in stations]),
            ("Temperature",      "K",      [s.T_K for s in stations]),
            ("Density",          "kg/m^3", [s.rho_kg_per_m3 for s in stations]),
            ("Velocity",         "m/s",    [s.V_m_per_s for s in stations]),
            ("Mach number",      "",       [s.M for s in stations]),
            ("Sonic velocity",   "m/s",    [s.a_m_per_s for s in stations]),
            ("Specific heat Cp", "kJ/kgK", [s.cp_eq_J_per_kgK/1000 for s in stations]),
            ("Gamma (isentr.)",  "",       [s.gamma_s for s in stations]),
            ("Molar mass",       "kg/kmol",[s.mw_g_per_mol for s in stations]),
            ("Area ratio",       "",       [
                s.Ae_At if math.isfinite(s.Ae_At) and s.Ae_At < 1e6 else 1e6
                for s in stations
            ]),
        ]

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"# Amesim XY export from {APP_NAME}\n")
                ox_desc, fu_desc = self._get_mixture_summary()
                f.write(f"# {ox_desc} / {fu_desc},  "
                        f"Pc={self.sp_Pc.value()} MPa, Pe={self.sp_Pe.value()} MPa\n")
                f.write(f"# O/F = {self.perf.O_F:.4f},  Isp = {self.perf.Isp_s:.4f} s,  "
                        f"C* = {self.perf.Cstar_m_per_s:.4f} m/s\n")
                f.write(f"# Table format: XY\n")
                # axis1 = x
                f.write(f"# axis1_unit = m\n")
                f.write(f"# axis1_title = Координата по длине сопла, м\n")
                # axis2... — данные
                for k, (title, unit, _) in enumerate(signals, start=2):
                    f.write(f"# axis{k}_unit = {unit}\n")
                    f.write(f"# axis{k}_title = {title}\n")
                # Данные
                for i in range(len(stations)):
                    row = [f"{x[i]:.6f}"]
                    for (_, _, vals) in signals:
                        row.append(f"{vals[i]:.6e}")
                    f.write("\t".join(row) + "\n")

            self.statusBar().showMessage(f"Amesim .data сохранён: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка", f"Не удалось сохранить Amesim .data:\n{e}"
            )

    # ─── Конфигурация ───

    def on_save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить конфигурацию",
            "rpa_config.json", "JSON (*.json)"
        )
        if not path:
            return
        cfg = {
            'mixture': self.mixture_widget.get_mixture(),
            'of_ratio': self.sp_of_ratio.value(),
            'Pc_MPa': self.sp_Pc.value(),
            'Pe_MPa': self.sp_Pe.value(),
            'n_inter': self.sp_n_inter.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'solver': 'cea' if self.rb_cea.isChecked() else 'own',
            'L_chamber': self.sp_L_chamber.value(),
            'L_conv': self.sp_L_conv.value(),
            'L_div': self.sp_L_div.value(),
            'style': {
                'font': self.cb_font.currentText(),
                'font_axis': self.sp_font_axis.value(),
                'font_tick': self.sp_font_tick.value(),
                'font_leg': self.sp_font_leg.value(),
                'lw': self.sp_lw.value(),
                'markers': self.chk_markers.isChecked(),
                'grid_major': self.chk_grid_major.isChecked(),
                'grid_minor': self.chk_grid_minor.isChecked(),
                'dark': self.chk_dark_plot.isChecked(),
            }
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"Конфигурация сохранена: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    def on_load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Загрузить конфигурацию", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            mixture = cfg.get('mixture')
            if mixture is not None:
                self.mixture_widget.set_mixture(mixture)
            else:
                # Совместимость со старым форматом конфигурации.
                self.mixture_widget.set_mixture({
                    'ox_components': [{
                        'name': cfg.get('oxidizer', 'O2(L)'),
                        'mass': 1.0,
                        'T': cfg.get('ox_T', 0.0),
                    }],
                    'fuel_components': [{
                        'name': cfg.get('fuel', 'H2(L)'),
                        'mass': 1.0,
                        'T': cfg.get('fu_T', 0.0),
                    }],
                })
            self.sp_of_ratio.setValue(cfg.get('of_ratio', 7.9370))
            self.sp_Pc.setValue(cfg.get('Pc_MPa', 10.0))
            self.sp_Pe.setValue(cfg.get('Pe_MPa', 0.1013))
            self.sp_n_inter.setValue(cfg.get('n_inter', 8))
            self.chk_condensed.setChecked(cfg.get('include_condensed', True))
            if cfg.get('solver') == 'cea' and CANTERA_AVAILABLE:
                self.rb_cea.setChecked(True)
            else:
                self.rb_own.setChecked(True)
            self.sp_L_chamber.setValue(cfg.get('L_chamber', 0.1))
            self.sp_L_conv.setValue(cfg.get('L_conv', 0.05))
            self.sp_L_div.setValue(cfg.get('L_div', 0.2))
            st = cfg.get('style', {})
            self.cb_font.setCurrentText(st.get('font', 'DejaVu Serif'))
            self.sp_font_axis.setValue(st.get('font_axis', 12))
            self.sp_font_tick.setValue(st.get('font_tick', 10))
            self.sp_font_leg.setValue(st.get('font_leg', 9))
            self.sp_lw.setValue(st.get('lw', 1.8))
            self.chk_markers.setChecked(st.get('markers', True))
            self.chk_grid_major.setChecked(st.get('grid_major', True))
            self.chk_grid_minor.setChecked(st.get('grid_minor', True))
            self.chk_dark_plot.setChecked(st.get('dark', True))
            self._update_of_from_mixture()
            self.statusBar().showMessage(f"Конфигурация загружена: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(build_dark_qss())

    # Палитра Fusion в тёмных тонах, чтобы стандартные виджеты тоже соответствовали
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,        QtGui.QColor('#262624'))
    pal.setColor(QtGui.QPalette.WindowText,    QtGui.QColor('#fafaf9'))
    pal.setColor(QtGui.QPalette.Base,          QtGui.QColor('#1e1e1c'))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor('#262624'))
    pal.setColor(QtGui.QPalette.Text,          QtGui.QColor('#fafaf9'))
    pal.setColor(QtGui.QPalette.Button,        QtGui.QColor('#3a3a37'))
    pal.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor('#fafaf9'))
    pal.setColor(QtGui.QPalette.Highlight,     QtGui.QColor('#cc785c'))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor('#1e1e1c'))
    pal.setColor(QtGui.QPalette.ToolTipBase,   QtGui.QColor('#1e1e1c'))
    pal.setColor(QtGui.QPalette.ToolTipText,   QtGui.QColor('#fafaf9'))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
