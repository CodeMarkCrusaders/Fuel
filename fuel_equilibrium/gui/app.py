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
from ..rocket.nozzle_geometry import (
    build_conical_nozzle, build_profiled_nozzle,
    build_geometry_from_performance, optimal_angles_from_area_ratio,
    dispersion_loss_coeff, NozzleGeometry,
    build_rpa_parabolic_nozzle, rao_reference_length_15deg, estimate_bell_angles,
)
from ..rocket.nozzle_flow_2d import solve_nozzle_2d, Nozzle2DResult
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


class CollapsibleSection(QtWidgets.QWidget):
    """Раскрывающаяся секция (как выпадающий список разделов настроек).

    Заголовок-кнопка со стрелкой ▶/▼; по клику тело секции скрывается/
    показывается. В тело через ``setContentLayout``/``setContentWidget``
    помещается любой layout или виджет.
    """

    def __init__(self, title: str = "", parent=None, expanded: bool = True):
        super().__init__(parent)
        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setStyleSheet(
            "QToolButton {"
            "  border: 1px solid #44403c;"
            "  border-radius: 4px;"
            "  background: #2a2724;"
            "  font-weight: bold;"
            "  text-align: left;"
            "  padding: 6px 8px;"
            "}"
            "QToolButton:hover { background: #353230; }"
        )
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setText(title)
        self._toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QtWidgets.QWidget(self)
        self._content.setVisible(expanded)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)

    def setContentLayout(self, content_layout):
        old = self._content.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)
        self._content.setLayout(content_layout)

    def setContentWidget(self, widget):
        """Помещает один виджет внутрь раскрывающейся секции."""
        lay = QtWidgets.QVBoxLayout()
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(widget)
        self.setContentLayout(lay)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)

    def setExpanded(self, expanded: bool):
        self._toggle.setChecked(expanded)
        self._on_toggled(expanded)

    def isExpanded(self) -> bool:
        return self._toggle.isChecked()


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
                    section_density_subsonic=p.get('density_sub', 1.0),
                    section_density_critical=p.get('density_crit', 1.0),
                    section_density_supersonic=p.get('density_sup', 1.0),
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
                    section_density_subsonic=p.get('density_sub', 1.0),
                    section_density_critical=p.get('density_crit', 1.0),
                    section_density_supersonic=p.get('density_sup', 1.0),
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
        # Минимальная ширина обеспечивает полную видимость панели настроек
        # по умолчанию; верхний предел снят, чтобы пользователь мог свободно
        # расширять панель слайдером.
        left_scroll.setMinimumWidth(440)
        # Не ограничиваем максимум — иначе панель «обрезается» справа.
        left_scroll.setMaximumWidth(16777215)
        splitter.addWidget(left_scroll)

        # ─── Правая часть: вкладки результатов, сгруппированы как в RPA ───
        # Верхний уровень: Газодинамика | Равновесный состав | Геометрия
        self.tabs = QtWidgets.QTabWidget()
        splitter.addWidget(self.tabs)

        # ═══ Группа 1: Газодинамические параметры (1D/2D) ═══
        self.tabs_gasdynamics = QtWidgets.QTabWidget()
        self.tabs_gasdynamics.setObjectName("subtabs")

        self.tab_table = self._build_table_tab()
        self.tabs_gasdynamics.addTab(self.tab_table, "Параметры по сечениям")

        # Вкладка «Графики по длине сопла» содержит внутри подвкладку
        # «Поле течения (2D)» (см. _build_plot_tab) — отдельной вкладки 2D нет.
        self.tab_plots = self._build_plot_tab()
        self.tabs_gasdynamics.addTab(self.tab_plots, "Графики по длине сопла")

        self.tab_perf = self._build_perf_tab()
        self.tabs_gasdynamics.addTab(self.tab_perf, "Тяговые характеристики")

        self.tabs.addTab(self.tabs_gasdynamics, "Газодинамика")

        # ═══ Группа 2: Равновесный состав продуктов сгорания ═══
        self.tabs_equilibrium = QtWidgets.QTabWidget()
        self.tabs_equilibrium.setObjectName("subtabs")
        self.tab_species = self._build_species_tab()
        self.tabs_equilibrium.addTab(self.tab_species, "Состав продуктов сгорания")
        self.tabs.addTab(self.tabs_equilibrium, "Равновесный состав")

        # ═══ Группа 3: Геометрия сопла ═══
        self.tabs_geometry = QtWidgets.QTabWidget()
        self.tabs_geometry.setObjectName("subtabs")
        self.tab_geometry = self._build_geometry_tab()
        self.tabs_geometry.addTab(self.tab_geometry, "Контур сопла (Size & Geometry)")
        self.tabs.addTab(self.tabs_geometry, "Геометрия")

        # По умолчанию слайдер выставлен так, чтобы панель настроек слева была
        # полностью видна (не обрезалась справа), а результаты занимали остаток.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([480, 1040])
        # Гарантируем, что главное окно достаточно широкое для полной панели.
        self.setMinimumWidth(1100)

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

        # Раскрывающаяся секция «Топливо» (заголовок убираем у группы, чтобы
        # не было двойного заголовка внутри раскрывающейся секции).
        gb_fuel.setTitle("")
        gb_fuel.setFlat(True)
        self.sec_fuel = CollapsibleSection("Топливо (RPA-style)", expanded=True)
        self.sec_fuel.setContentWidget(gb_fuel)
        layout.addWidget(self.sec_fuel)

        # ─── Условия + газодинамические настройки по вкладкам ───
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
        self.sp_n_inter.setRange(0, 1048)
        self.sp_n_inter.setValue(8)
        self.sp_n_inter.setToolTip(
            "Общее число промежуточных газодинамических сечений\n"
            "по всему соплу: дозвук + критика + сверхзвук.\n"
            "Состав продуктов всегда показывается на 4 сечениях:\n"
            "Injector, Nozzle inlet, Nozzle throat, Nozzle exit."
        )

        self.sp_density_sub = QtWidgets.QDoubleSpinBox()
        self.sp_density_sub.setRange(0.0, 20.0)
        self.sp_density_sub.setDecimals(2)
        self.sp_density_sub.setValue(1.0)
        self.sp_density_sub.setSingleStep(0.1)
        self.sp_density_sub.setToolTip("Относительная плотность сечений в дозвуковой зоне (камера → горловина).")

        self.sp_density_crit = QtWidgets.QDoubleSpinBox()
        self.sp_density_crit.setRange(0.0, 20.0)
        self.sp_density_crit.setDecimals(2)
        self.sp_density_crit.setValue(1.0)
        self.sp_density_crit.setSingleStep(0.1)
        self.sp_density_crit.setToolTip("Относительная плотность сечений в критической зоне (вблизи горловины).")

        self.sp_density_sup = QtWidgets.QDoubleSpinBox()
        self.sp_density_sup.setRange(0.0, 20.0)
        self.sp_density_sup.setDecimals(2)
        self.sp_density_sup.setValue(1.0)
        self.sp_density_sup.setSingleStep(0.1)
        self.sp_density_sup.setToolTip("Относительная плотность сечений в сверхзвуковой зоне (горловина → срез).")

        density_widget = QtWidgets.QWidget()
        density_layout = QtWidgets.QHBoxLayout(density_widget)
        density_layout.setContentsMargins(0, 0, 0, 0)
        density_layout.addWidget(QtWidgets.QLabel("дозвук"))
        density_layout.addWidget(self.sp_density_sub)
        density_layout.addWidget(QtWidgets.QLabel("критика"))
        density_layout.addWidget(self.sp_density_crit)
        density_layout.addWidget(QtWidgets.QLabel("сверхзвук"))
        density_layout.addWidget(self.sp_density_sup)

        self.chk_condensed = QtWidgets.QCheckBox("Учитывать конденсат")
        self.chk_condensed.setChecked(True)

        # Внутренний контейнер вместо QTabWidget: три ВЛОЖЕННЫХ раскрывающихся
        # подраздела (Исходные данные / Газодинамика / Геометрия), которые
        # ведут себя как выпадающие списки, аналогично основному разделу.
        self.input_tabs = QtWidgets.QWidget()
        input_tabs_layout = QtWidgets.QVBoxLayout(self.input_tabs)
        input_tabs_layout.setContentsMargins(0, 0, 0, 0)
        input_tabs_layout.setSpacing(6)

        tab_basic = QtWidgets.QWidget()
        form_basic = QtWidgets.QFormLayout(tab_basic)
        form_basic.setSpacing(6)
        form_basic.addRow("Давление в камере:", w_Pc)
        form_basic.addRow("Давление на срезе:", w_Pe)
        form_basic.addRow("", self.chk_condensed)

        tab_gasd = QtWidgets.QWidget()
        form_gasd = QtWidgets.QFormLayout(tab_gasd)
        form_gasd.setSpacing(6)
        form_gasd.addRow("Промежут. сечений:", self.sp_n_inter)
        form_gasd.addRow("Плотность сечений:", density_widget)

        gasd_hint = QtWidgets.QLabel(
            "Распределение промежуточных сечений выполняется\n"
            "на участках: дозвук, критика, сверхзвук."
        )
        gasd_hint.setStyleSheet("color: #a8a29e; font-size: 10px;")
        gasd_hint.setWordWrap(True)
        form_gasd.addRow("", gasd_hint)

        # ─── Выбор размерности расчёта газодинамики: 1D / 2D ───
        w_dim = QtWidgets.QWidget()
        h_dim = QtWidgets.QHBoxLayout(w_dim)
        h_dim.setContentsMargins(0, 0, 0, 0)
        self.rb_dim_1d = QtWidgets.QRadioButton("Одномерный (1D)")
        self.rb_dim_2d = QtWidgets.QRadioButton("Двумерный (2D)")
        self.rb_dim_1d.setChecked(True)
        self.rb_dim_1d.setToolTip(
            "Одномерный квазигазодинамический расчёт по оси сопла\n"
            "(стандартная модель равновесного течения)."
        )
        self.rb_dim_2d.setToolTip(
            "Двумерный (осесимметричный) расчёт поля течения.\n"
            "ЗАГОТОВКА: квази-2D обёртка 1D-профиля на сетку (n_r×n_x)\n"
            "с поправкой на угол стенки. Полный метод характеристик (MOC) — TODO."
        )
        self.rb_dim_1d.toggled.connect(self._on_dim_mode_changed)
        self.rb_dim_2d.toggled.connect(self._on_dim_mode_changed)
        h_dim.addWidget(self.rb_dim_1d)
        h_dim.addWidget(self.rb_dim_2d)
        h_dim.addStretch(1)
        form_gasd.addRow("Размерность расчёта:", w_dim)

        self.sp_n_radial_2d = QtWidgets.QSpinBox()
        self.sp_n_radial_2d.setRange(5, 121)
        self.sp_n_radial_2d.setValue(21)
        self.sp_n_radial_2d.setToolTip("Число радиальных узлов 2D-сетки (только для 2D).")
        form_gasd.addRow("Радиальных узлов (2D):", self.sp_n_radial_2d)

        # ── Пограничный слой (2D) ──
        self.chk_bl_2d = QtWidgets.QCheckBox("Учитывать пограничный слой")
        self.chk_bl_2d.setChecked(True)
        self.chk_bl_2d.setToolTip(
            "Вязкий пристеночный слой: условие прилипания (скорость → 0 у стенки),\n"
            "температура восстановления (вязкий нагрев), профиль 1/7."
        )
        self.chk_bl_2d.toggled.connect(lambda *_: self._update_field_2d_from_perf())
        form_gasd.addRow("Пограничный слой (2D):", self.chk_bl_2d)

        self.sp_bl_delta_2d = QtWidgets.QDoubleSpinBox()
        self.sp_bl_delta_2d.setRange(0.01, 0.50)
        self.sp_bl_delta_2d.setSingleStep(0.01)
        self.sp_bl_delta_2d.setDecimals(2)
        self.sp_bl_delta_2d.setValue(0.12)
        self.sp_bl_delta_2d.setToolTip(
            "Относительная толщина пограничного слоя δ/R на срезе сопла."
        )
        self.sp_bl_delta_2d.valueChanged.connect(lambda *_: self._update_field_2d_from_perf())
        form_gasd.addRow("Толщина δ/R на срезе (2D):", self.sp_bl_delta_2d)

        # Вложенные раскрывающиеся подразделы «Исходные данные» и «Газодинамика»
        self.sec_basic = CollapsibleSection("Исходные данные", expanded=True)
        self.sec_basic.setContentWidget(tab_basic)
        input_tabs_layout.addWidget(self.sec_basic)

        self.sec_gasd = CollapsibleSection("Газодинамика (1D/2D)", expanded=False)
        self.sec_gasd.setContentWidget(tab_gasd)
        input_tabs_layout.addWidget(self.sec_gasd)

        # ─── Третий подраздел исходных данных: Геометрия (Size & Geometry) ───
        self.tab_input_geom = QtWidgets.QWidget()
        self.form_input_geom = QtWidgets.QVBoxLayout(self.tab_input_geom)
        self.form_input_geom.setContentsMargins(2, 2, 2, 2)
        self.form_input_geom.setSpacing(8)
        geom_hdr = QtWidgets.QLabel(
            "Геометрия сопла (как в RPA: Size & Geometry).\n"
            "Здесь задаётся профиль сопла и габариты для оси X графиков."
        )
        geom_hdr.setStyleSheet("color: #a8a29e; font-size: 10px;")
        geom_hdr.setWordWrap(True)
        self.form_input_geom.addWidget(geom_hdr)

        self.sec_geom_input = CollapsibleSection(
            "Геометрия (Size & Geometry)", expanded=False
        )
        self.sec_geom_input.setContentWidget(self.tab_input_geom)
        input_tabs_layout.addWidget(self.sec_geom_input)

        # Раскрывающаяся секция «Параметры расчёта» содержит три вложенных
        # раскрывающихся подраздела.
        self.sec_params = CollapsibleSection("Параметры расчёта", expanded=True)
        self.sec_params.setContentWidget(self.input_tabs)
        layout.addWidget(self.sec_params)

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

        # Раскрывающаяся секция «Решатель» (по умолчанию свёрнута, как
        # второстепенная настройка).
        gb_solver.setTitle("")
        gb_solver.setFlat(True)
        self.sec_solver = CollapsibleSection("Решатель", expanded=False)
        self.sec_solver.setContentWidget(gb_solver)
        layout.addWidget(self.sec_solver)

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
        self.form_input_geom.addWidget(gb_geom)

        # ─── Профиль по Добровольскому (выбор типа + ручной ввод геометрии) ───
        gb_prof = QtWidgets.QGroupBox("Профиль сопла (Добровольский, гл. 2)")
        formP = QtWidgets.QFormLayout(gb_prof)
        formP.setSpacing(6)

        self.chk_use_dobro = QtWidgets.QCheckBox(
            "Строить профиль по выбранному типу и параметрам"
        )
        self.chk_use_dobro.setChecked(True)
        self.chk_use_dobro.setToolTip(
            "Если включено, профиль сопла на графике строится по методике\n"
            "Добровольского (коническое §2.3 / профилированное §2.6) с\n"
            "указанными ниже параметрами. Степень расширения F_a/F_кр\n"
            "берётся из результата газодинамического расчёта."
        )
        formP.addRow(self.chk_use_dobro)

        # Тип сопла
        w_type = QtWidgets.QWidget()
        h_type = QtWidgets.QHBoxLayout(w_type)
        h_type.setContentsMargins(0, 0, 0, 0)
        self.rb_calc_conical = QtWidgets.QRadioButton("Коническое")
        self.rb_calc_profiled = QtWidgets.QRadioButton("Профилированное")
        self.rb_calc_rpa = QtWidgets.QRadioButton("RPA (bell)")
        self.rb_calc_profiled.setChecked(True)
        self.rb_calc_conical.toggled.connect(self._on_calc_geom_type_changed)
        self.rb_calc_profiled.toggled.connect(self._on_calc_geom_type_changed)
        self.rb_calc_rpa.toggled.connect(self._on_calc_geom_type_changed)
        h_type.addWidget(self.rb_calc_conical)
        h_type.addWidget(self.rb_calc_profiled)
        h_type.addWidget(self.rb_calc_rpa)
        formP.addRow("Тип сопла:", w_type)

        # R_кр (горловина)
        w_Rkr = QtWidgets.QWidget()
        h_Rkr = QtWidgets.QHBoxLayout(w_Rkr)
        h_Rkr.setContentsMargins(0, 0, 0, 0)
        self.sp_calc_Rthroat = QtWidgets.QDoubleSpinBox()
        self.sp_calc_Rthroat.setRange(0.0001, 100.0)
        self.sp_calc_Rthroat.setDecimals(4)
        self.sp_calc_Rthroat.setValue(0.0500)
        self.sp_calc_Rthroat.setSingleStep(0.005)
        self.sp_calc_Rthroat.setToolTip("R_кр — радиус критического сечения (горловины)")
        h_Rkr.addWidget(self.sp_calc_Rthroat)
        self.cb_calc_Rthroat_unit = QtWidgets.QComboBox()
        self.cb_calc_Rthroat_unit.addItems(["м", "см", "мм"])
        self.cb_calc_Rthroat_unit.setCurrentText("м")
        h_Rkr.addWidget(self.cb_calc_Rthroat_unit)
        formP.addRow("R_кр (горловина):", w_Rkr)

        # R_камеры / R_кр
        self.sp_calc_Rcham = QtWidgets.QDoubleSpinBox()
        self.sp_calc_Rcham.setRange(1.05, 20.0)
        self.sp_calc_Rcham.setDecimals(3)
        self.sp_calc_Rcham.setValue(2.500)
        self.sp_calc_Rcham.setSingleStep(0.1)
        self.sp_calc_Rcham.setToolTip("R_камеры / R_кр")
        formP.addRow("R_камеры / R_кр:", self.sp_calc_Rcham)

        # Углы
        self.sp_calc_theta_in = QtWidgets.QDoubleSpinBox()
        self.sp_calc_theta_in.setRange(10.0, 45.0)
        self.sp_calc_theta_in.setDecimals(2)
        self.sp_calc_theta_in.setValue(30.0)
        self.sp_calc_theta_in.setSingleStep(1.0)
        self.sp_calc_theta_in.setToolTip("θ_вх — полуугол дозвукового конуса (2θ_вх=45…80°)")
        formP.addRow("θ_вх (дозвук), °:", self.sp_calc_theta_in)

        self.chk_calc_auto_angles = QtWidgets.QCheckBox(
            "θ_m, θ_a, длина — авто (Рис. 2.14)"
        )
        self.chk_calc_auto_angles.setChecked(True)
        self.chk_calc_auto_angles.setToolTip(
            "Для профилированного сопла: углы и длина берутся из семейства\n"
            "оптимальных контуров (Рис. 2.14, γ=1.23) по F_a/F_кр.\n"
            "Снимите галочку, чтобы задать θ_m, θ_a и длину вручную."
        )
        self.chk_calc_auto_angles.toggled.connect(self._on_calc_geom_auto_toggled)
        formP.addRow(self.chk_calc_auto_angles)

        self.sp_calc_theta_max = QtWidgets.QDoubleSpinBox()
        self.sp_calc_theta_max.setRange(5.0, 50.0)
        self.sp_calc_theta_max.setDecimals(2)
        self.sp_calc_theta_max.setValue(30.0)
        self.sp_calc_theta_max.setSingleStep(0.5)
        self.sp_calc_theta_max.setToolTip("θ_m — угол контура в начале св/зв части (профиль)")
        formP.addRow("θ_m (начало св/зв), °:", self.sp_calc_theta_max)

        self.sp_calc_theta_exit = QtWidgets.QDoubleSpinBox()
        self.sp_calc_theta_exit.setRange(3.0, 25.0)
        self.sp_calc_theta_exit.setDecimals(2)
        self.sp_calc_theta_exit.setValue(15.0)
        self.sp_calc_theta_exit.setSingleStep(0.5)
        self.sp_calc_theta_exit.setToolTip(
            "θ_a — угол на срезе.\n"
            "Конус: полуугол раствора (2θ_a=25…30° ⇒ 12.5…15°).\n"
            "Профиль: угол на срезе."
        )
        formP.addRow("θ_a (срез), °:", self.sp_calc_theta_exit)

        self.sp_calc_len_ratio = QtWidgets.QDoubleSpinBox()
        self.sp_calc_len_ratio.setRange(0.5, 200.0)
        self.sp_calc_len_ratio.setDecimals(2)
        self.sp_calc_len_ratio.setValue(9.5)
        self.sp_calc_len_ratio.setSingleStep(0.5)
        self.sp_calc_len_ratio.setToolTip("x̄_a = L_сверхзв / R_кр (профиль)")
        formP.addRow("x̄_a = L/R_кр:", self.sp_calc_len_ratio)

        # Скругления
        self.sp_calc_Rsub = QtWidgets.QDoubleSpinBox()
        self.sp_calc_Rsub.setRange(0.1, 5.0)
        self.sp_calc_Rsub.setDecimals(3)
        self.sp_calc_Rsub.setValue(1.500)
        self.sp_calc_Rsub.setSingleStep(0.05)
        self.sp_calc_Rsub.setToolTip(
            "R_скр перед горловиной. Конус: ×D_кр (0.65…1.5). Профиль: ×R_кр (≈1.5)."
        )
        formP.addRow("R_скр × (D_кр/R_кр):", self.sp_calc_Rsub)

        self.sp_calc_rsup = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rsup.setRange(0.05, 3.0)
        self.sp_calc_rsup.setDecimals(3)
        self.sp_calc_rsup.setValue(0.450)
        self.sp_calc_rsup.setSingleStep(0.05)
        self.sp_calc_rsup.setToolTip("r_скр за горловиной ×R_кр (≈0.45·R_кр)")
        formP.addRow("r_скр × R_кр:", self.sp_calc_rsup)

        self.sp_calc_R1 = QtWidgets.QDoubleSpinBox()
        self.sp_calc_R1.setRange(0.5, 10.0)
        self.sp_calc_R1.setDecimals(3)
        self.sp_calc_R1.setValue(3.000)
        self.sp_calc_R1.setSingleStep(0.25)
        self.sp_calc_R1.setToolTip("R_1 — скругление на входе из камеры ×D_кр (2…4)")
        formP.addRow("R_1 × D_кр:", self.sp_calc_R1)

        # ─── Поля RPA (параболический bell, нотация RPA Size & Geometry) ───
        self.sp_calc_rpa_b = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_b.setRange(10.0, 60.0)
        self.sp_calc_rpa_b.setDecimals(2)
        self.sp_calc_rpa_b.setValue(30.0)
        self.sp_calc_rpa_b.setSingleStep(1.0)
        self.sp_calc_rpa_b.setToolTip("Contraction angle b — угол сжатия конфузора, °")
        formP.addRow("b (contraction), °:", self.sp_calc_rpa_b)

        self.sp_calc_rpa_R1Rt = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_R1Rt.setRange(0.2, 10.0)
        self.sp_calc_rpa_R1Rt.setDecimals(3)
        self.sp_calc_rpa_R1Rt.setValue(1.500)
        self.sp_calc_rpa_R1Rt.setSingleStep(0.1)
        self.sp_calc_rpa_R1Rt.setToolTip("R1/Rt — скругление сходящейся стороны горловины")
        formP.addRow("R1/Rt:", self.sp_calc_rpa_R1Rt)

        self.sp_calc_rpa_R2 = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_R2.setRange(0.0, 1.0)
        self.sp_calc_rpa_R2.setDecimals(3)
        self.sp_calc_rpa_R2.setValue(0.500)
        self.sp_calc_rpa_R2.setSingleStep(0.05)
        self.sp_calc_rpa_R2.setToolTip("R2/R2max — относительный радиус входа в конфузор (0…1)")
        formP.addRow("R2/R2max (0…1):", self.sp_calc_rpa_R2)

        self.sp_calc_rpa_RnRt = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_RnRt.setRange(0.1, 2.0)
        self.sp_calc_rpa_RnRt.setDecimals(3)
        self.sp_calc_rpa_RnRt.setValue(0.382)
        self.sp_calc_rpa_RnRt.setSingleStep(0.01)
        self.sp_calc_rpa_RnRt.setToolTip("Rn/Rt — скругление расходящейся стороны горловины (RPA=0.382)")
        formP.addRow("Rn/Rt:", self.sp_calc_rpa_RnRt)

        self.chk_calc_rpa_auto = QtWidgets.QCheckBox("Tn, Te — авто (по ε, Le/Le15)")
        self.chk_calc_rpa_auto.setChecked(True)
        self.chk_calc_rpa_auto.toggled.connect(self._on_calc_geom_auto_toggled)
        formP.addRow(self.chk_calc_rpa_auto)

        self.sp_calc_rpa_Tn = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_Tn.setRange(5.0, 55.0)
        self.sp_calc_rpa_Tn.setDecimals(2)
        self.sp_calc_rpa_Tn.setValue(27.0)
        self.sp_calc_rpa_Tn.setSingleStep(0.5)
        self.sp_calc_rpa_Tn.setToolTip("Tn — начальный угол параболы, °")
        formP.addRow("Tn (нач. парабола), °:", self.sp_calc_rpa_Tn)

        self.sp_calc_rpa_Te = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_Te.setRange(1.0, 24.0)
        self.sp_calc_rpa_Te.setDecimals(2)
        self.sp_calc_rpa_Te.setValue(10.0)
        self.sp_calc_rpa_Te.setSingleStep(0.5)
        self.sp_calc_rpa_Te.setToolTip("Te — конечный угол параболы, °")
        formP.addRow("Te (кон. парабола), °:", self.sp_calc_rpa_Te)

        self.sp_calc_rpa_LeLe15 = QtWidgets.QDoubleSpinBox()
        self.sp_calc_rpa_LeLe15.setRange(50.0, 120.0)
        self.sp_calc_rpa_LeLe15.setDecimals(1)
        self.sp_calc_rpa_LeLe15.setValue(80.0)
        self.sp_calc_rpa_LeLe15.setSingleStep(1.0)
        self.sp_calc_rpa_LeLe15.setToolTip("Relative length Le/Le15, %")
        formP.addRow("Le/Le15, %:", self.sp_calc_rpa_LeLe15)

        # Списки полей для переключения видимости по типу сопла
        self._calc_rpa_widgets = [
            self.sp_calc_rpa_b, self.sp_calc_rpa_R1Rt, self.sp_calc_rpa_R2,
            self.sp_calc_rpa_RnRt, self.chk_calc_rpa_auto,
            self.sp_calc_rpa_Tn, self.sp_calc_rpa_Te, self.sp_calc_rpa_LeLe15,
        ]
        self._calc_dobro_widgets = [
            self.sp_calc_theta_in, self.chk_calc_auto_angles,
            self.sp_calc_theta_max, self.sp_calc_theta_exit, self.sp_calc_len_ratio,
            self.sp_calc_Rsub, self.sp_calc_rsup, self.sp_calc_R1,
        ]
        self._calc_formP = formP

        self.form_input_geom.addWidget(gb_prof)
        self.form_input_geom.addStretch(1)

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

        layout.addStretch(1)
        # начальная синхронизация активности полей геометрии профиля
        self._on_calc_geom_type_changed()
        self._on_dim_mode_changed()
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

        # Слева — внутренние подвкладки: графики 1D и поле течения 2D.
        self.plot_subtabs = QtWidgets.QTabWidget()
        self.plot_subtabs.setObjectName("subtabs")

        # Подвкладка 1: графики 1D (сетка 2x2)
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
        self.plot_subtabs.addTab(plot_widget, "Графики (1D)")

        # Подвкладка 2: поле течения (2D) — теперь живёт здесь, внутри
        # раздела «Графики по длине сопла».
        self.tab_field_2d = self._build_field_2d_tab()
        self.plot_subtabs.addTab(self.tab_field_2d, "Поле течения (2D)")

        h.addWidget(self.plot_subtabs, 1)

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

        # ─── Наложение графиков (сравнение нескольких 1D-расчётов) ───
        gb_overlay = QtWidgets.QGroupBox("Наложение графиков (1D)")
        of = QtWidgets.QVBoxLayout(gb_overlay)
        of.setSpacing(4)

        self.chk_overlay_show = QtWidgets.QCheckBox("Показывать наложения")
        self.chk_overlay_show.setChecked(True)
        self.chk_overlay_show.setToolTip(
            "Отображать ранее зафиксированные кривые поверх текущего расчёта "
            "для сравнения вариантов."
        )
        self.chk_overlay_show.toggled.connect(self._redraw_plots)
        of.addWidget(self.chk_overlay_show)

        self.sp_overlay_name = QtWidgets.QLineEdit()
        self.sp_overlay_name.setPlaceholderText("Имя варианта (необязательно)")
        of.addWidget(self.sp_overlay_name)

        btn_overlay_add = QtWidgets.QPushButton("➕ Зафиксировать как наложение")
        btn_overlay_add.setToolTip(
            "Сохранить кривые текущего 1D-расчёта как наложение, чтобы "
            "сравнить с последующими расчётами."
        )
        btn_overlay_add.clicked.connect(self._add_overlay_snapshot)
        of.addWidget(btn_overlay_add)

        btn_overlay_clear = QtWidgets.QPushButton("🗑 Очистить наложения")
        btn_overlay_clear.clicked.connect(self._clear_overlays)
        of.addWidget(btn_overlay_clear)

        self.lbl_overlay_count = QtWidgets.QLabel("Наложений: 0")
        self.lbl_overlay_count.setStyleSheet("color: #a8a29e; font-size: 10px;")
        of.addWidget(self.lbl_overlay_count)

        side_v.addWidget(gb_overlay)
        side_v.addStretch(1)
        h.addWidget(side)

        # Хранилище снимков для наложения (список dict с массивами кривых)
        self._overlays = []
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

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Поле течения (2D)» — заготовка квази-2D расчёта
    # ─────────────────────────────────────────────────────────────────────────
    def _build_field_2d_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Поле:"))
        self.cb_field_2d = QtWidgets.QComboBox()
        self.cb_field_2d.addItems([
            "M (число Маха)", "P (давление)", "T (температура)",
            "V (скорость)", "Угол потока, °",
        ])
        self._field_2d_keys = {
            0: ("M", "M"), 1: ("P_Pa", "P, Па"), 2: ("T_K", "T, К"),
            3: ("V_m_per_s", "V, м/с"), 4: ("flow_angle_deg", "угол, °"),
        }
        self.cb_field_2d.currentIndexChanged.connect(lambda *_: self._render_field_2d())
        ctrl.addWidget(self.cb_field_2d)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        self.lbl_field_2d_info = QtWidgets.QLabel(
            "Двумерный (осесимметричный) расчёт: квази-2D модель.\n"
            "Параметры (M, P, T, V) меняются по ДВУМ координатам — вдоль оси x\n"
            "и по радиусу r. Радиальное распределение числа Маха строится по\n"
            "источниковому (source-flow) приближению расходящегося потока, а\n"
            "T, P и V пересчитываются из M по изэнтропическим соотношениям.\n"
            "Полный метод характеристик (MOC) — следующий шаг (TODO).\n"
            "Выберите режим «Двумерный (2D)» во вкладке «Газодинамика (1D/2D)» и\n"
            "выполните расчёт, чтобы построить поле течения."
        )
        self.lbl_field_2d_info.setStyleSheet("color: #a8a29e; font-size: 10px;")
        self.lbl_field_2d_info.setWordWrap(True)
        v.addWidget(self.lbl_field_2d_info)

        self.canvas_field_2d = MplCanvas(width=7, height=4.5)
        v.addWidget(self.canvas_field_2d, 1)

        self._last_field_2d: Optional[Nozzle2DResult] = None
        return w

    def _update_field_2d_from_perf(self):
        """Выполняет квази-2D расчёт, если выбран режим 2D, иначе очищает."""
        self._last_field_2d = None
        is_2d = (getattr(self, "rb_dim_2d", None) is not None
                 and self.rb_dim_2d.isChecked())
        if not is_2d or self.perf is None:
            self._render_field_2d()
            return
        try:
            geom = self._build_calc_geometry(self.perf)
            if geom is None:
                self._render_field_2d()
                return
            n_r = int(self.sp_n_radial_2d.value()) if hasattr(self, "sp_n_radial_2d") else 21
            bl_on = self.chk_bl_2d.isChecked() if hasattr(self, "chk_bl_2d") else True
            bl_delta = float(self.sp_bl_delta_2d.value()) if hasattr(self, "sp_bl_delta_2d") else 0.12
            self._last_field_2d = solve_nozzle_2d(
                self.perf, geom, n_radial=n_r, method="quasi2d_stub",
                boundary_layer=bl_on, bl_delta_frac=bl_delta,
            )
        except Exception as e:
            self.statusBar().showMessage(f"2D-расчёт пропущен: {e}", 5000)
            self._last_field_2d = None
        self._render_field_2d()

    def _render_field_2d(self):
        c = getattr(self, "canvas_field_2d", None)
        if c is None:
            return
        c.fig.clear()
        res = self._last_field_2d
        if res is None:
            ax = c.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Нет данных 2D.\nВыберите режим «Двумерный (2D)» и выполните расчёт.",
                    ha='center', va='center', fontsize=11, color='#888')
            ax.set_axis_off()
            c.fig.tight_layout()
            c.draw()
            return
        idx = self.cb_field_2d.currentIndex()
        key, label = self._field_2d_keys.get(idx, ("M", "M"))
        try:
            vals = res.field_values(key)
        except Exception:
            vals = None
        ax = c.fig.add_subplot(111)
        if vals is None:
            ax.text(0.5, 0.5, f"Поле '{key}' недоступно", ha='center', va='center')
            ax.set_axis_off()
        else:
            pcm = ax.pcolormesh(res.x_grid, res.r_grid, vals, shading='auto', cmap='viridis')
            ax.plot(res.wall_x, res.wall_r, '-', color='#cc785c', lw=1.8)
            c.fig.colorbar(pcm, ax=ax, label=label)
            ax.set_xlabel("x, м")
            ax.set_ylabel("r, м")
            if res.metadata.get("is_stub"):
                tag = " (ЗАГОТОВКА, квази-2D)"
            else:
                tag = " (квази-2D, source-flow)"
            ax.set_title(f"Поле течения 2D — {label}{tag}")
            try:
                ax.set_aspect('equal', adjustable='datalim')
            except Exception:
                pass
        c.fig.tight_layout()
        c.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Геометрия сопла (Добровольский, гл. 2)»
    # ─────────────────────────────────────────────────────────────────────────
    def _build_geometry_tab(self) -> QtWidgets.QWidget:
        """Панель настройки ВСЕХ параметров геометрии сопла по Добровольскому
        (§2.3 коническое, §2.6 профилированное) + визуализация контура."""
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Левая колонка: параметры ──
        params = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(params)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(10)
        params.setMinimumWidth(360)
        params.setMaximumWidth(420)

        # Тип сопла
        gb_type = QtWidgets.QGroupBox("Тип сопла")
        ht = QtWidgets.QHBoxLayout(gb_type)
        self.rb_geom_conical = QtWidgets.QRadioButton("Коническое (§2.3)")
        self.rb_geom_profiled = QtWidgets.QRadioButton("Профилированное (§2.6)")
        self.rb_geom_rpa = QtWidgets.QRadioButton("RPA (bell)")
        self.rb_geom_profiled.setChecked(True)
        self.rb_geom_conical.toggled.connect(self._on_geom_type_changed)
        self.rb_geom_profiled.toggled.connect(self._on_geom_type_changed)
        self.rb_geom_rpa.toggled.connect(self._on_geom_type_changed)
        ht.addWidget(self.rb_geom_conical)
        ht.addWidget(self.rb_geom_profiled)
        ht.addWidget(self.rb_geom_rpa)
        pv.addWidget(gb_type)

        # Базовые размеры
        gb_base = QtWidgets.QGroupBox("Основные размеры")
        fb = QtWidgets.QFormLayout(gb_base)
        fb.setSpacing(6)

        self.sp_geom_Rthroat = QtWidgets.QDoubleSpinBox()
        self.sp_geom_Rthroat.setRange(0.0001, 100.0)
        self.sp_geom_Rthroat.setDecimals(4)
        self.sp_geom_Rthroat.setValue(0.0500)
        self.sp_geom_Rthroat.setSingleStep(0.005)
        self.sp_geom_Rthroat.setToolTip("R_кр — радиус критического сечения (горловины), м")

        self.sp_geom_AR = QtWidgets.QDoubleSpinBox()
        self.sp_geom_AR.setRange(1.001, 1000.0)
        self.sp_geom_AR.setDecimals(3)
        self.sp_geom_AR.setValue(16.000)
        self.sp_geom_AR.setSingleStep(1.0)
        self.sp_geom_AR.setToolTip("F_a/F_кр — геометрическая степень расширения")
        self.sp_geom_AR.valueChanged.connect(self._on_geom_ar_changed)

        self.sp_geom_Rcham_factor = QtWidgets.QDoubleSpinBox()
        self.sp_geom_Rcham_factor.setRange(1.05, 20.0)
        self.sp_geom_Rcham_factor.setDecimals(3)
        self.sp_geom_Rcham_factor.setValue(2.500)
        self.sp_geom_Rcham_factor.setSingleStep(0.1)
        self.sp_geom_Rcham_factor.setToolTip("R_камеры / R_кр")

        fb.addRow("R_кр (горловина), м:", self.sp_geom_Rthroat)
        fb.addRow("F_a/F_кр:", self.sp_geom_AR)
        fb.addRow("R_камеры / R_кр:", self.sp_geom_Rcham_factor)
        pv.addWidget(gb_base)

        # Углы
        gb_ang = QtWidgets.QGroupBox("Углы контура")
        fa = QtWidgets.QFormLayout(gb_ang)
        fa.setSpacing(6)

        self.sp_geom_theta_in = QtWidgets.QDoubleSpinBox()
        self.sp_geom_theta_in.setRange(10.0, 45.0)
        self.sp_geom_theta_in.setDecimals(2)
        self.sp_geom_theta_in.setValue(30.0)
        self.sp_geom_theta_in.setSingleStep(1.0)
        self.sp_geom_theta_in.setToolTip("θ_вх — полуугол дозвукового конуса (2θ_вх=45…80°)")

        self.sp_geom_theta_exit = QtWidgets.QDoubleSpinBox()
        self.sp_geom_theta_exit.setRange(3.0, 25.0)
        self.sp_geom_theta_exit.setDecimals(2)
        self.sp_geom_theta_exit.setValue(15.0)
        self.sp_geom_theta_exit.setSingleStep(0.5)
        self.sp_geom_theta_exit.setToolTip(
            "θ_a — угол контура на срезе.\n"
            "Конус: полуугол раствора (2θ_a=25…30° ⇒ 12.5…15°).\n"
            "Профиль: угол на срезе (по Рис. 2.14, если 'авто')."
        )

        self.sp_geom_theta_max = QtWidgets.QDoubleSpinBox()
        self.sp_geom_theta_max.setRange(5.0, 50.0)
        self.sp_geom_theta_max.setDecimals(2)
        self.sp_geom_theta_max.setValue(30.0)
        self.sp_geom_theta_max.setSingleStep(0.5)
        self.sp_geom_theta_max.setToolTip(
            "θ_m — угол контура в начале сверхзвуковой части (только профиль, §2.6)"
        )

        # «Авто» из Рис. 2.14 для профилированного
        self.chk_geom_auto_angles = QtWidgets.QCheckBox(
            "θ_m, θ_a, длина — авто из Рис. 2.14 (γ=1.23)"
        )
        self.chk_geom_auto_angles.setChecked(True)
        self.chk_geom_auto_angles.toggled.connect(self._on_geom_auto_toggled)

        self.sp_geom_len_ratio = QtWidgets.QDoubleSpinBox()
        self.sp_geom_len_ratio.setRange(0.5, 200.0)
        self.sp_geom_len_ratio.setDecimals(2)
        self.sp_geom_len_ratio.setValue(9.5)
        self.sp_geom_len_ratio.setSingleStep(0.5)
        self.sp_geom_len_ratio.setToolTip(
            "x̄_a = L_сверхзв / R_кр — относительная длина (только профиль)"
        )

        fa.addRow("θ_вх (дозвук), °:", self.sp_geom_theta_in)
        fa.addRow("θ_a (срез), °:", self.sp_geom_theta_exit)
        fa.addRow("θ_m (начало св/зв), °:", self.sp_geom_theta_max)
        fa.addRow(self.chk_geom_auto_angles)
        fa.addRow("x̄_a = L/R_кр:", self.sp_geom_len_ratio)
        pv.addWidget(gb_ang)

        # Скругления
        gb_round = QtWidgets.QGroupBox("Скругления (множители)")
        fr = QtWidgets.QFormLayout(gb_round)
        fr.setSpacing(6)

        self.sp_geom_Rsub = QtWidgets.QDoubleSpinBox()
        self.sp_geom_Rsub.setRange(0.1, 5.0)
        self.sp_geom_Rsub.setDecimals(3)
        self.sp_geom_Rsub.setValue(1.500)
        self.sp_geom_Rsub.setSingleStep(0.05)
        self.sp_geom_Rsub.setToolTip(
            "R_скр перед горловиной.\n"
            "Конус: ×D_кр (0.65…1.5). Профиль: ×R_кр (≈1.5·R_кр)."
        )

        self.sp_geom_rsup = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rsup.setRange(0.05, 3.0)
        self.sp_geom_rsup.setDecimals(3)
        self.sp_geom_rsup.setValue(0.450)
        self.sp_geom_rsup.setSingleStep(0.05)
        self.sp_geom_rsup.setToolTip("r_скр за горловиной ×R_кр (≈0.45·R_кр)")

        self.sp_geom_R1 = QtWidgets.QDoubleSpinBox()
        self.sp_geom_R1.setRange(0.5, 10.0)
        self.sp_geom_R1.setDecimals(3)
        self.sp_geom_R1.setValue(3.000)
        self.sp_geom_R1.setSingleStep(0.25)
        self.sp_geom_R1.setToolTip("R_1 — скругление на входе из камеры ×D_кр (2…4)")

        fr.addRow("R_скр × (D_кр/R_кр):", self.sp_geom_Rsub)
        fr.addRow("r_скр × R_кр:", self.sp_geom_rsup)
        fr.addRow("R_1 × D_кр:", self.sp_geom_R1)
        pv.addWidget(gb_round)
        self._geom_dobro_groups = [gb_ang, gb_round]

        # ── RPA Size & Geometry (параболический bell, нотация RPA) ──
        self.gb_geom_rpa = QtWidgets.QGroupBox("RPA Size & Geometry (bell)")
        frpa = QtWidgets.QFormLayout(self.gb_geom_rpa)
        frpa.setSpacing(6)

        self.sp_geom_rpa_b = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_b.setRange(10.0, 60.0); self.sp_geom_rpa_b.setDecimals(2)
        self.sp_geom_rpa_b.setValue(30.0); self.sp_geom_rpa_b.setSingleStep(1.0)
        self.sp_geom_rpa_b.setToolTip("Contraction angle b — угол сжатия конфузора, °")
        frpa.addRow("b (contraction), °:", self.sp_geom_rpa_b)

        self.sp_geom_rpa_R1Rt = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_R1Rt.setRange(0.2, 10.0); self.sp_geom_rpa_R1Rt.setDecimals(3)
        self.sp_geom_rpa_R1Rt.setValue(1.500); self.sp_geom_rpa_R1Rt.setSingleStep(0.1)
        self.sp_geom_rpa_R1Rt.setToolTip("R1/Rt — скругление сходящейся стороны горловины")
        frpa.addRow("R1/Rt:", self.sp_geom_rpa_R1Rt)

        self.sp_geom_rpa_R2 = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_R2.setRange(0.0, 1.0); self.sp_geom_rpa_R2.setDecimals(3)
        self.sp_geom_rpa_R2.setValue(0.500); self.sp_geom_rpa_R2.setSingleStep(0.05)
        self.sp_geom_rpa_R2.setToolTip("R2/R2max — относительный радиус входа в конфузор (0…1)")
        frpa.addRow("R2/R2max (0…1):", self.sp_geom_rpa_R2)

        self.sp_geom_rpa_RnRt = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_RnRt.setRange(0.1, 2.0); self.sp_geom_rpa_RnRt.setDecimals(3)
        self.sp_geom_rpa_RnRt.setValue(0.382); self.sp_geom_rpa_RnRt.setSingleStep(0.01)
        self.sp_geom_rpa_RnRt.setToolTip("Rn/Rt — скругление расходящейся стороны горловины (RPA=0.382)")
        frpa.addRow("Rn/Rt:", self.sp_geom_rpa_RnRt)

        self.chk_geom_rpa_auto = QtWidgets.QCheckBox("Tn, Te — авто (по ε, Le/Le15)")
        self.chk_geom_rpa_auto.setChecked(True)
        self.chk_geom_rpa_auto.toggled.connect(self._on_geom_rpa_auto_toggled)
        frpa.addRow(self.chk_geom_rpa_auto)

        self.sp_geom_rpa_Tn = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_Tn.setRange(5.0, 55.0); self.sp_geom_rpa_Tn.setDecimals(2)
        self.sp_geom_rpa_Tn.setValue(27.0); self.sp_geom_rpa_Tn.setSingleStep(0.5)
        self.sp_geom_rpa_Tn.setToolTip("Tn — начальный угол параболы, °")
        frpa.addRow("Tn (нач. парабола), °:", self.sp_geom_rpa_Tn)

        self.sp_geom_rpa_Te = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_Te.setRange(1.0, 24.0); self.sp_geom_rpa_Te.setDecimals(2)
        self.sp_geom_rpa_Te.setValue(10.0); self.sp_geom_rpa_Te.setSingleStep(0.5)
        self.sp_geom_rpa_Te.setToolTip("Te — конечный угол параболы, °")
        frpa.addRow("Te (кон. парабола), °:", self.sp_geom_rpa_Te)

        self.sp_geom_rpa_LeLe15 = QtWidgets.QDoubleSpinBox()
        self.sp_geom_rpa_LeLe15.setRange(50.0, 120.0); self.sp_geom_rpa_LeLe15.setDecimals(1)
        self.sp_geom_rpa_LeLe15.setValue(80.0); self.sp_geom_rpa_LeLe15.setSingleStep(1.0)
        self.sp_geom_rpa_LeLe15.setToolTip("Relative length Le/Le15, %")
        frpa.addRow("Le/Le15, %:", self.sp_geom_rpa_LeLe15)

        pv.addWidget(self.gb_geom_rpa)

        # Кнопки
        self.btn_geom_build = QtWidgets.QPushButton("▶  Построить контур")
        self.btn_geom_build.setObjectName("primary")
        self.btn_geom_build.setMinimumHeight(36)
        self.btn_geom_build.clicked.connect(self.on_build_geometry)
        pv.addWidget(self.btn_geom_build)

        self.btn_geom_from_perf = QtWidgets.QPushButton("⤵  Взять F_a/F_кр из расчёта")
        self.btn_geom_from_perf.setToolTip(
            "Подставить степень расширения и θ_a (ур. 2.23) из последнего\n"
            "газодинамического расчёта сопла."
        )
        self.btn_geom_from_perf.clicked.connect(self.on_geometry_from_perf)
        pv.addWidget(self.btn_geom_from_perf)

        self.btn_geom_export = QtWidgets.QPushButton("💾  Экспорт контура (CSV)")
        self.btn_geom_export.clicked.connect(self.on_export_geometry_csv)
        pv.addWidget(self.btn_geom_export)

        pv.addStretch(1)

        # ── Правая колонка: график + сводка ──
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)

        self.canvas_geometry = MplCanvas(width=7, height=4.5)
        rv.addWidget(self.canvas_geometry, 4)

        self.txt_geom_summary = QtWidgets.QPlainTextEdit()
        self.txt_geom_summary.setReadOnly(True)
        self.txt_geom_summary.setMaximumHeight(180)
        self.txt_geom_summary.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas','DejaVu Sans Mono',monospace; "
            "font-size: 10pt; }"
        )
        rv.addWidget(self.txt_geom_summary, 1)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidget(params)
        params_scroll.setWidgetResizable(True)
        params_scroll.setMinimumWidth(370)
        params_scroll.setMaximumWidth(430)
        splitter.addWidget(params_scroll)
        splitter.addWidget(right)
        splitter.setSizes([380, 900])
        root.addWidget(splitter)

        # последняя построенная геометрия
        self.last_geometry: Optional[NozzleGeometry] = None
        # синхронизировать активность виджетов
        self._on_geom_type_changed()
        self._on_geom_auto_toggled(self.chk_geom_auto_angles.isChecked())
        self._on_geom_rpa_auto_toggled()
        return w

    # ── обработчики состояния панели геометрии ───────────────────────────────
    def _on_geom_type_changed(self, *args):
        is_rpa = (getattr(self, "rb_geom_rpa", None) is not None
                  and self.rb_geom_rpa.isChecked())
        # Показываем RPA-группу только для RPA, добровольские группы — иначе
        if hasattr(self, "gb_geom_rpa"):
            self.gb_geom_rpa.setVisible(is_rpa)
        for gb in getattr(self, "_geom_dobro_groups", []):
            gb.setVisible(not is_rpa)
        if is_rpa:
            self._on_geom_rpa_auto_toggled(self.chk_geom_rpa_auto.isChecked())
            return

        is_conical = self.rb_geom_conical.isChecked()
        # θ_m, авто-углы и длина имеют смысл только для профилированного
        self.sp_geom_theta_max.setEnabled(not is_conical)
        self.chk_geom_auto_angles.setEnabled(not is_conical)
        if is_conical:
            self.sp_geom_len_ratio.setEnabled(False)
            self.sp_geom_theta_exit.setEnabled(True)
        else:
            self._on_geom_auto_toggled(self.chk_geom_auto_angles.isChecked())

    def _on_geom_rpa_auto_toggled(self, *args):
        auto = self.chk_geom_rpa_auto.isChecked()
        self.sp_geom_rpa_Tn.setEnabled(not auto)
        self.sp_geom_rpa_Te.setEnabled(not auto)

    def _on_geom_auto_toggled(self, checked: bool):
        if getattr(self, "rb_geom_rpa", None) is not None and self.rb_geom_rpa.isChecked():
            return
        if self.rb_geom_conical.isChecked():
            return
        # при «авто» углы θ_m/θ_a/длина берутся из Рис. 2.14 → блокируем поля
        self.sp_geom_theta_max.setEnabled(not checked)
        self.sp_geom_theta_exit.setEnabled(not checked)
        self.sp_geom_len_ratio.setEnabled(not checked)
        if checked:
            self._on_geom_ar_changed()

    def _on_geom_ar_changed(self, *args):
        # при «авто» подставить предлагаемые значения из Рис. 2.14
        if getattr(self, "rb_geom_rpa", None) is not None and self.rb_geom_rpa.isChecked():
            return
        if self.rb_geom_conical.isChecked() or not self.chk_geom_auto_angles.isChecked():
            return
        try:
            tm, ta, xa = optimal_angles_from_area_ratio(self.sp_geom_AR.value())
            self.sp_geom_theta_max.setValue(tm)
            self.sp_geom_theta_exit.setValue(ta)
            self.sp_geom_len_ratio.setValue(xa)
        except Exception:
            pass

    # ── обработчики панели профиля в основном расчёте ────────────────────────
    def _set_form_row_visible(self, form, widget, visible: bool):
        """Скрывает/показывает поле формы вместе с его меткой (QFormLayout)."""
        if widget is None:
            return
        widget.setVisible(visible)
        try:
            if isinstance(form, QtWidgets.QFormLayout):
                lbl = form.labelForField(widget)
                if lbl is not None:
                    lbl.setVisible(visible)
        except Exception:
            pass

    def _on_dim_mode_changed(self, *args):
        """Переключение 1D/2D режима газодинамического расчёта."""
        is_2d = (getattr(self, "rb_dim_2d", None) is not None
                 and self.rb_dim_2d.isChecked())
        if hasattr(self, "sp_n_radial_2d"):
            self.sp_n_radial_2d.setEnabled(is_2d)
        if hasattr(self, "chk_bl_2d"):
            self.chk_bl_2d.setEnabled(is_2d)
        if hasattr(self, "sp_bl_delta_2d"):
            self.sp_bl_delta_2d.setEnabled(is_2d and self.chk_bl_2d.isChecked())

    def _on_calc_geom_type_changed(self, *args):
        """Включает/выключает поля под выбранный тип сопла в панели расчёта."""
        is_rpa = (getattr(self, "rb_calc_rpa", None) is not None
                  and self.rb_calc_rpa.isChecked())
        is_conical = self.rb_calc_conical.isChecked()
        is_profiled = self.rb_calc_profiled.isChecked()

        formP = getattr(self, "_calc_formP", None)
        for wdg in getattr(self, "_calc_rpa_widgets", []):
            self._set_form_row_visible(formP, wdg, is_rpa)
        for wdg in getattr(self, "_calc_dobro_widgets", []):
            self._set_form_row_visible(formP, wdg, not is_rpa)

        if is_rpa:
            self._on_calc_geom_auto_toggled(self.chk_calc_rpa_auto.isChecked())
            return

        # Режим Добровольского (conical / profiled)
        self.sp_calc_theta_max.setEnabled(is_profiled)
        self.chk_calc_auto_angles.setEnabled(is_profiled)
        if is_conical:
            self.sp_calc_len_ratio.setEnabled(False)
            self.sp_calc_theta_exit.setEnabled(True)
        else:
            self._on_calc_geom_auto_toggled(self.chk_calc_auto_angles.isChecked())

    def _on_calc_geom_auto_toggled(self, checked: bool):
        # Режим RPA: авто Tn/Te блокирует ручные углы
        if getattr(self, "rb_calc_rpa", None) is not None and self.rb_calc_rpa.isChecked():
            auto = self.chk_calc_rpa_auto.isChecked()
            self.sp_calc_rpa_Tn.setEnabled(not auto)
            self.sp_calc_rpa_Te.setEnabled(not auto)
            return
        if self.rb_calc_conical.isChecked():
            return
        # при «авто» θ_m/θ_a/длина блокируются (берутся из Рис. 2.14 по F_a/F_кр)
        self.sp_calc_theta_max.setEnabled(not checked)
        self.sp_calc_theta_exit.setEnabled(not checked)
        self.sp_calc_len_ratio.setEnabled(not checked)

    @staticmethod
    def _length_to_m(v: float, unit: str) -> float:
        if unit == 'см':
            return v * 0.01
        if unit == 'мм':
            return v * 0.001
        return v

    def _build_calc_geometry(self, perf: "RocketPerformance") -> Optional["NozzleGeometry"]:
        """Строит геометрию сопла по выбранному в панели расчёта типу и
        параметрам, используя F_a/F_кр из результата расчёта (perf)."""
        try:
            ar = float(perf.stations[-1].Ae_At)
            if not (math.isfinite(ar) and ar > 1.0):
                return None
            R_throat = self._length_to_m(
                self.sp_calc_Rthroat.value(),
                self.cb_calc_Rthroat_unit.currentText(),
            )
            R_cham = self.sp_calc_Rcham.value() * R_throat

            if getattr(self, "rb_calc_rpa", None) is not None and self.rb_calc_rpa.isChecked():
                auto = self.chk_calc_rpa_auto.isChecked()
                return build_rpa_parabolic_nozzle(
                    R_throat, ar,
                    R_chamber_m=R_cham,
                    contraction_angle_deg=self.sp_calc_rpa_b.value(),
                    R1_over_Rt=self.sp_calc_rpa_R1Rt.value(),
                    Rn_over_Rt=self.sp_calc_rpa_RnRt.value(),
                    R2_over_R2max=self.sp_calc_rpa_R2.value(),
                    theta_n_deg=None if auto else self.sp_calc_rpa_Tn.value(),
                    theta_e_deg=None if auto else self.sp_calc_rpa_Te.value(),
                    length_fraction_pct=self.sp_calc_rpa_LeLe15.value(),
                )

            if self.rb_calc_conical.isChecked():
                return build_conical_nozzle(
                    R_throat, ar,
                    R_chamber_m=R_cham,
                    theta_exit_deg=self.sp_calc_theta_exit.value(),
                    theta_in_deg=self.sp_calc_theta_in.value(),
                    R_round_sub_factor=self.sp_calc_Rsub.value(),
                    R1_inlet_factor=self.sp_calc_R1.value(),
                    r_round_sup_factor=self.sp_calc_rsup.value(),
                )
            auto = self.chk_calc_auto_angles.isChecked()
            return build_profiled_nozzle(
                R_throat, ar,
                R_chamber_m=R_cham,
                theta_exit_deg=None if auto else self.sp_calc_theta_exit.value(),
                theta_max_deg=None if auto else self.sp_calc_theta_max.value(),
                length_ratio=None if auto else self.sp_calc_len_ratio.value(),
                theta_in_deg=self.sp_calc_theta_in.value(),
                R_round_sub_factor=self.sp_calc_Rsub.value(),
                r_round_sup_factor=self.sp_calc_rsup.value(),
                R1_inlet_factor=self.sp_calc_R1.value(),
            )
        except Exception:
            return None

    def on_geometry_from_perf(self):
        if self.perf is None:
            QtWidgets.QMessageBox.information(
                self, "Нет данных",
                "Сначала выполните газодинамический расчёт сопла "
                "(кнопка «Рассчитать сопло»)."
            )
            return
        try:
            ar = float(self.perf.stations[-1].Ae_At)
            if math.isfinite(ar) and ar > 1.0:
                self.sp_geom_AR.setValue(ar)
            self._on_geom_ar_changed()
            self.statusBar().showMessage(
                f"Степень расширения F_a/F_кр = {ar:.3f} взята из расчёта", 5000
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))

    def on_build_geometry(self):
        try:
            R_throat = float(self.sp_geom_Rthroat.value())
            ar = float(self.sp_geom_AR.value())
            R_cham = self.sp_geom_Rcham_factor.value() * R_throat

            if getattr(self, "rb_geom_rpa", None) is not None and self.rb_geom_rpa.isChecked():
                auto = self.chk_geom_rpa_auto.isChecked()
                geom = build_rpa_parabolic_nozzle(
                    R_throat, ar,
                    R_chamber_m=R_cham,
                    contraction_angle_deg=self.sp_geom_rpa_b.value(),
                    R1_over_Rt=self.sp_geom_rpa_R1Rt.value(),
                    Rn_over_Rt=self.sp_geom_rpa_RnRt.value(),
                    R2_over_R2max=self.sp_geom_rpa_R2.value(),
                    theta_n_deg=None if auto else self.sp_geom_rpa_Tn.value(),
                    theta_e_deg=None if auto else self.sp_geom_rpa_Te.value(),
                    length_fraction_pct=self.sp_geom_rpa_LeLe15.value(),
                )
            elif self.rb_geom_conical.isChecked():
                geom = build_conical_nozzle(
                    R_throat, ar,
                    R_chamber_m=R_cham,
                    theta_exit_deg=self.sp_geom_theta_exit.value(),
                    theta_in_deg=self.sp_geom_theta_in.value(),
                    R_round_sub_factor=self.sp_geom_Rsub.value(),
                    R1_inlet_factor=self.sp_geom_R1.value(),
                    r_round_sup_factor=self.sp_geom_rsup.value(),
                )
            else:
                auto = self.chk_geom_auto_angles.isChecked()
                geom = build_profiled_nozzle(
                    R_throat, ar,
                    R_chamber_m=R_cham,
                    theta_exit_deg=None if auto else self.sp_geom_theta_exit.value(),
                    theta_max_deg=None if auto else self.sp_geom_theta_max.value(),
                    length_ratio=None if auto else self.sp_geom_len_ratio.value(),
                    theta_in_deg=self.sp_geom_theta_in.value(),
                    R_round_sub_factor=self.sp_geom_Rsub.value(),
                    r_round_sup_factor=self.sp_geom_rsup.value(),
                    R1_inlet_factor=self.sp_geom_R1.value(),
                )
            self.last_geometry = geom
            self._render_geometry(geom)
            self._update_geometry_summary(geom)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка построения", str(e))

    def _render_geometry(self, geom: "NozzleGeometry"):
        c = self.canvas_geometry
        c.fig.clear()
        ax = c.fig.add_subplot(111)
        x, r = geom.as_xy_arrays()

        # дозвуковая / сверхзвуковая части — разными цветами
        x_throat = geom.length_subsonic_m
        col = '#cc785c'
        ax.plot(x, r, '-', color=col, lw=2.0)
        ax.plot(x, -r, '-', color=col, lw=2.0)
        ax.fill_between(x, -r, r, alpha=0.12, color=col)

        # горловина и срез
        ax.axvline(x_throat, color='#6b9bd1', ls='--', lw=1.0, alpha=0.8,
                   label=f"горловина R_кр={geom.R_throat_m*1e3:.1f} мм")
        ax.axvline(geom.length_total_m, color='#86b386', ls=':', lw=1.0, alpha=0.8,
                   label=f"срез R_a={geom.R_exit_m*1e3:.1f} мм")

        style = self._collect_style()
        if geom.method == "rpa_parabolic":
            title = "RPA параболическое сопло (bell)"
        elif geom.method == "conical":
            title = "Коническое сопло (§2.3)"
        else:
            title = "Профилированное сопло (§2.6)"
        style.title = (f"{title}  |  L={geom.length_total_m*1e3:.1f} мм  "
                       f"θ_a={geom.theta_exit_deg:.1f}°  φ_рас={geom.phi_dispersion:.4f}")
        style.xlabel = "Координата x, м"
        style.ylabel = "Радиус r, м"
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper left', fontsize=8)
        apply_plot_style(c.fig, ax, style)
        c.fig.tight_layout()
        c.draw()

    def _update_geometry_summary(self, geom: "NozzleGeometry"):
        if geom.method == "rpa_parabolic":
            self._update_geometry_summary_rpa(geom)
            return
        s = []
        s.append("═══ ГЕОМЕТРИЯ СОПЛА (Добровольский, гл. 2) ═══")
        s.append(f"Тип:                {'коническое (§2.3)' if geom.method=='conical' else 'профилированное (§2.6)'}")
        s.append(f"R_кр (горловина):   {geom.R_throat_m*1e3:.3f} мм")
        s.append(f"R_a (срез):         {geom.R_exit_m*1e3:.3f} мм")
        s.append(f"R_камеры:           {geom.R_chamber_m*1e3:.3f} мм")
        s.append(f"F_a/F_кр:           {geom.area_ratio:.4f}   (R_a/R_кр = {math.sqrt(geom.area_ratio):.4f})")
        s.append(f"θ_вх (дозвук):      {geom.theta_in_deg:.2f}°  (2θ_вх = {2*geom.theta_in_deg:.1f}°)")
        if geom.method != 'conical':
            s.append(f"θ_m (начало св/зв): {geom.theta_max_deg:.2f}°")
        s.append(f"θ_a (срез):         {geom.theta_exit_deg:.2f}°  (2θ_a = {2*geom.theta_exit_deg:.1f}°)")
        s.append(f"φ_рас:              {geom.phi_dispersion:.4f}   = (1+cos θ_a)/2")
        s.append(f"R_скр / r_скр / R_1: {geom.R_round_sub_m*1e3:.2f} / "
                 f"{geom.r_round_sup_m*1e3:.2f} / {geom.R1_inlet_m*1e3:.2f} мм")
        s.append(f"Длина дозвук./св.зв./полная: {geom.length_subsonic_m*1e3:.2f} / "
                 f"{geom.length_supersonic_m*1e3:.2f} / {geom.length_total_m*1e3:.2f} мм")
        s.append(f"Точек контура:      {len(geom.points)}")
        self.txt_geom_summary.setPlainText("\n".join(s))

    def _update_geometry_summary_rpa(self, geom: "NozzleGeometry"):
        md = geom.metadata or {}
        def g(key, default=float('nan')):
            return md.get(key, default)
        s = []
        s.append("═══ ГЕОМЕТРИЯ СОПЛА (RPA Size & Geometry, bell) ═══")
        s.append(f"Тип:                параболическое (Rao bell)")
        s.append(f"R_кр (Rt):          {geom.R_throat_m*1e3:.3f} мм")
        s.append(f"R_a (Re, срез):     {geom.R_exit_m*1e3:.3f} мм")
        s.append(f"R_камеры (Rc):      {geom.R_chamber_m*1e3:.3f} мм")
        s.append(f"ε = Ae/At:          {geom.area_ratio:.4f}   (Re/Rt = {math.sqrt(geom.area_ratio):.4f})")
        s.append(f"b (contraction):    {g('contraction_angle_deg'):.2f}°")
        s.append(f"R1/Rt:              {g('R1_over_Rt'):.3f}")
        s.append(f"R2/R2max:           {g('R2_over_R2max'):.3f}")
        s.append(f"Rn/Rt:              {g('Rn_over_Rt'):.3f}")
        s.append(f"Tn (нач. парабола): {g('theta_n_deg'):.2f}°")
        s.append(f"Te (кон. парабола): {g('theta_e_deg'):.2f}°   (θ_a={geom.theta_exit_deg:.2f}°)")
        s.append(f"Le/Le15:            {g('Le_over_Le15_pct'):.1f} %")
        le15 = g('Le15_m')
        if le15 == le15:  # not NaN
            s.append(f"Le15 (15°-конус):   {le15*1e3:.2f} мм")
        s.append(f"φ_рас:              {geom.phi_dispersion:.4f}   = (1+cos θ_a)/2")
        s.append(f"Длина дозвук./св.зв./полная: {geom.length_subsonic_m*1e3:.2f} / "
                 f"{geom.length_supersonic_m*1e3:.2f} / {geom.length_total_m*1e3:.2f} мм")
        s.append(f"Точек контура:      {len(geom.points)}")
        self.txt_geom_summary.setPlainText("\n".join(s))

    def on_export_geometry_csv(self):
        if getattr(self, "last_geometry", None) is None:
            QtWidgets.QMessageBox.information(
                self, "Нет данных", "Сначала постройте контур сопла."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить контур сопла", "nozzle_contour.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import csv
            geom = self.last_geometry
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f, delimiter=";")
                wr.writerow([f"# Геометрия сопла ({geom.method}), Добровольский гл.2"])
                wr.writerow([f"# R_кр={geom.R_throat_m:.6f} м, R_a={geom.R_exit_m:.6f} м, "
                             f"F_a/F_кр={geom.area_ratio:.4f}, "
                             f"theta_a={geom.theta_exit_deg:.3f} град, "
                             f"phi_рас={geom.phi_dispersion:.5f}"])
                wr.writerow(["x_m", "r_m"])
                for p in geom.points:
                    wr.writerow([f"{p.x_m:.6f}", f"{p.r_m:.6f}"])
            self.statusBar().showMessage(f"Контур сохранён: {path}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка экспорта", str(e))

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&Файл")
        self.act_export_csv = m_file.addAction("Экспорт CSV…")
        self.act_export_csv.triggered.connect(self.on_export_csv)
        self.act_export_csv.setEnabled(False)
        self.act_export_amesim = m_file.addAction("Экспорт Amesim (.data)…")
        self.act_export_amesim.triggered.connect(self.on_export_amesim)
        self.act_export_amesim.setEnabled(False)
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
            'density_sub': self.sp_density_sub.value(),
            'density_crit': self.sp_density_crit.value(),
            'density_sup': self.sp_density_sup.value(),
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
        self.act_export_csv.setEnabled(False)
        self.act_export_amesim.setEnabled(False)
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
        self.act_export_csv.setEnabled(True)
        self.act_export_amesim.setEnabled(True)
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
        self._update_field_2d_from_perf()

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

    # ─── Наложение графиков (overlay) для сравнения 1D-расчётов ───

    def _compute_curve_x(self, stations):
        """Координата x по длине сопла для текущих сечений (как в _redraw_plots)."""
        def length_to_m(v, unit):
            if unit == 'см':
                return v * 0.01
            if unit == 'мм':
                return v * 0.001
            return v
        return build_nozzle_geometry(
            stations,
            L_chamber=length_to_m(self.sp_L_chamber.value(), self.cb_L_chamber_unit.currentText()),
            L_conv=length_to_m(self.sp_L_conv.value(), self.cb_L_conv_unit.currentText()),
            L_div=length_to_m(self.sp_L_div.value(), self.cb_L_div_unit.currentText()),
        )

    def _snapshot_curves(self, perf):
        """Снимок кривых 1D-расчёта (x + P, T, V, M, ρ, γₛ) для наложения."""
        stations = perf.stations
        x = np.asarray(self._compute_curve_x(stations), dtype=float)
        return {
            "x": x,
            "P": np.array([s.P_Pa / 1e6 for s in stations]),
            "T": np.array([s.T_K for s in stations]),
            "V": np.array([s.V_m_per_s for s in stations]),
            "M": np.array([s.M for s in stations]),
            "rho": np.array([s.rho_kg_per_m3 for s in stations]),
            "gs": np.array([s.gamma_s for s in stations]),
        }

    def _add_overlay_snapshot(self):
        """Зафиксировать текущий 1D-расчёт как наложение для сравнения."""
        if getattr(self, "perf", None) is None:
            self.statusBar().showMessage(
                "Нет расчёта для фиксации наложения.", 4000
            )
            return
        if not hasattr(self, "_overlays"):
            self._overlays = []
        name = self.sp_overlay_name.text().strip()
        if not name:
            name = f"Вариант {len(self._overlays) + 1}"
        snap = self._snapshot_curves(self.perf)
        # цвет наложения из палитры (циклически)
        palette = ['#9aa0a6', '#d4a373', '#90be6d', '#577590',
                   '#f9c74f', '#bc6c25', '#8ecae6', '#e07a5f']
        snap["label"] = name
        snap["color"] = palette[len(self._overlays) % len(palette)]
        self._overlays.append(snap)
        self._update_overlay_count()
        self.sp_overlay_name.clear()
        self.statusBar().showMessage(f"Наложение «{name}» добавлено.", 4000)
        self._redraw_plots()

    def _clear_overlays(self):
        """Очистить все наложения."""
        self._overlays = []
        self._update_overlay_count()
        self._redraw_plots()

    def _update_overlay_count(self):
        if hasattr(self, "lbl_overlay_count"):
            n = len(getattr(self, "_overlays", []))
            self.lbl_overlay_count.setText(f"Наложений: {n}")

    def _draw_overlays(self, ax, key, *, twin=False):
        """Нарисовать наложенные кривые ``key`` на оси ``ax`` (фоном).

        Возвращает список handle'ов для легенды (по одному на наложение,
        только для основной оси, чтобы не дублировать в легенде).
        """
        handles = []
        if not getattr(self, "chk_overlay_show", None) or not self.chk_overlay_show.isChecked():
            return handles
        for ov in getattr(self, "_overlays", []):
            y = ov.get(key)
            if y is None:
                continue
            lw = max(0.8, self._collect_style().line_width * 0.7)
            line, = ax.plot(
                ov["x"], y, '-', color=ov["color"], lw=lw,
                alpha=0.55, zorder=1,
                label=(f"{ov['label']}" if not twin else None),
            )
            if not twin:
                handles.append(line)
        return handles

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

        ov_h = self._draw_overlays(ax1, "P")
        l1, = ax1.plot(x, P, 'o-' if style.show_markers else '-',
                       color='#cc785c', lw=style.line_width, ms=style.marker_size,
                       label='P, МПа', zorder=3)
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("Давление P, МПа")

        ax2 = ax1.twinx()
        self._draw_overlays(ax2, "T", twin=True)
        l2, = ax2.plot(x, T, 's--' if style.show_markers else '--',
                       color='#6ab0ff', lw=style.line_width, ms=style.marker_size,
                       label='T, К', zorder=3)
        ax2.set_ylabel("Температура T, К")
        ax2.legend(handles=[l1, l2] + ov_h, loc='best')

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
        ov_h = self._draw_overlays(ax1, "V")
        l1, = ax1.plot(x, V, 'o-' if style.show_markers else '-',
                       color='#82d27a', lw=style.line_width, ms=style.marker_size,
                       label='V, м/с', zorder=3)
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("Скорость потока V, м/с")
        ax2 = ax1.twinx()
        self._draw_overlays(ax2, "M", twin=True)
        l2, = ax2.plot(x, M, 'D--' if style.show_markers else '--',
                       color='#e6b800', lw=style.line_width, ms=style.marker_size,
                       label='M', zorder=3)
        ax2.set_ylabel("Число Маха M")
        # горизонталь M=1
        ax2.axhline(1.0, color='#a8a29e', lw=0.8, ls=':')
        ax2.legend(handles=[l1, l2] + ov_h, loc='best')
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
        ov_h = self._draw_overlays(ax1, "rho")
        l1, = ax1.plot(x, rho, 'o-' if style.show_markers else '-',
                       color='#cc785c', lw=style.line_width, ms=style.marker_size,
                       label='ρ, кг/м³', zorder=3)
        ax1.set_xlabel("Координата x, м")
        ax1.set_ylabel("Плотность ρ, кг/м³")
        ax2 = ax1.twinx()
        self._draw_overlays(ax2, "gs", twin=True)
        l2, = ax2.plot(x, gs, '^--' if style.show_markers else '--',
                       color='#c084fc', lw=style.line_width, ms=style.marker_size,
                       label='γₛ', zorder=3)
        ax2.set_ylabel("Изэнтр. показатель γₛ")
        ax2.legend(handles=[l1, l2] + ov_h, loc='best')
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

        # Вариант А: профиль по Добровольскому (выбранный тип + параметры)
        geom = None
        if getattr(self, "chk_use_dobro", None) is not None and self.chk_use_dobro.isChecked():
            geom = self._build_calc_geometry(self.perf)

        if geom is not None:
            self.last_geometry = geom  # синхронизируем с вкладкой геометрии
            gx, gr = geom.as_xy_arrays()
            r_max = float(np.max(gr))
            ax.plot(gx, gr, '-', color='#cc785c', lw=style.line_width * 1.2)
            ax.plot(gx, -gr, '-', color='#cc785c', lw=style.line_width * 1.2)
            ax.fill_between(gx, -gr, gr, alpha=0.15, color='#cc785c')

            # горловина и срез
            ax.axvline(geom.length_subsonic_m, color='#6b9bd1', ls='--', lw=0.9,
                       alpha=0.8, label="горловина")
            ax.axvline(geom.length_total_m, color='#86b386', ls=':', lw=0.9,
                       alpha=0.8, label="срез")
            ax.legend(loc='upper left', fontsize=8)

            style4 = self._collect_style()
            tname = ("Коническое (§2.3)" if geom.method == "conical"
                     else "Профилированное (§2.6)")
            style4.title = (f"Профиль сопла — {tname} | "
                            f"L={geom.length_total_m*1e3:.1f} мм, "
                            f"θ_a={geom.theta_exit_deg:.1f}°, "
                            f"φ_рас={geom.phi_dispersion:.4f}")
            style4.xlabel = "Координата x, м"
            style4.ylabel = "Радиус r, м"
            ax.set_aspect('equal', adjustable='datalim')
            apply_plot_style(c.fig, ax, style4)
            c.fig.tight_layout()
            c.draw()
            # также обновим вкладку «Геометрия сопла»
            try:
                self._render_geometry(geom)
                self._update_geometry_summary(geom)
            except Exception:
                pass
            return

        # Вариант Б (запасной): сглаженный профиль из сечений солвера
        r = nozzle_radius(stations)
        r_max = float(np.max(r))
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
            'density_sub': self.sp_density_sub.value(),
            'density_crit': self.sp_density_crit.value(),
            'density_sup': self.sp_density_sup.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'solver': 'cea' if self.rb_cea.isChecked() else 'own',
            'L_chamber': self.sp_L_chamber.value(),
            'L_conv': self.sp_L_conv.value(),
            'L_div': self.sp_L_div.value(),
            'gasdynamics': {
                'dim_mode': '2d' if self.rb_dim_2d.isChecked() else '1d',
                'n_radial_2d': self.sp_n_radial_2d.value(),
                'boundary_layer_2d': self.chk_bl_2d.isChecked(),
                'bl_delta_frac_2d': self.sp_bl_delta_2d.value(),
            },
            'geometry_profile': {
                'use_dobro': self.chk_use_dobro.isChecked(),
                'type': ('rpa' if self.rb_calc_rpa.isChecked()
                         else 'conical' if self.rb_calc_conical.isChecked()
                         else 'profiled'),
                'R_throat': self.sp_calc_Rthroat.value(),
                'R_throat_unit': self.cb_calc_Rthroat_unit.currentText(),
                'R_chamber_factor': self.sp_calc_Rcham.value(),
                'theta_in': self.sp_calc_theta_in.value(),
                'auto_angles': self.chk_calc_auto_angles.isChecked(),
                'theta_max': self.sp_calc_theta_max.value(),
                'theta_exit': self.sp_calc_theta_exit.value(),
                'len_ratio': self.sp_calc_len_ratio.value(),
                'R_sub': self.sp_calc_Rsub.value(),
                'r_sup': self.sp_calc_rsup.value(),
                'R1': self.sp_calc_R1.value(),
                # RPA-параметры
                'rpa_b': self.sp_calc_rpa_b.value(),
                'rpa_R1Rt': self.sp_calc_rpa_R1Rt.value(),
                'rpa_R2': self.sp_calc_rpa_R2.value(),
                'rpa_RnRt': self.sp_calc_rpa_RnRt.value(),
                'rpa_auto': self.chk_calc_rpa_auto.isChecked(),
                'rpa_Tn': self.sp_calc_rpa_Tn.value(),
                'rpa_Te': self.sp_calc_rpa_Te.value(),
                'rpa_LeLe15': self.sp_calc_rpa_LeLe15.value(),
            },
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
            self.sp_density_sub.setValue(cfg.get('density_sub', 1.0))
            self.sp_density_crit.setValue(cfg.get('density_crit', 1.0))
            self.sp_density_sup.setValue(cfg.get('density_sup', 1.0))
            self.chk_condensed.setChecked(cfg.get('include_condensed', True))
            if cfg.get('solver') == 'cea' and CANTERA_AVAILABLE:
                self.rb_cea.setChecked(True)
            else:
                self.rb_own.setChecked(True)
            self.sp_L_chamber.setValue(cfg.get('L_chamber', 0.1))
            self.sp_L_conv.setValue(cfg.get('L_conv', 0.05))
            self.sp_L_div.setValue(cfg.get('L_div', 0.2))
            gd = cfg.get('gasdynamics')
            if isinstance(gd, dict):
                if gd.get('dim_mode') == '2d':
                    self.rb_dim_2d.setChecked(True)
                else:
                    self.rb_dim_1d.setChecked(True)
                self.sp_n_radial_2d.setValue(int(gd.get('n_radial_2d', 21)))
                self.chk_bl_2d.setChecked(bool(gd.get('boundary_layer_2d', True)))
                self.sp_bl_delta_2d.setValue(float(gd.get('bl_delta_frac_2d', 0.12)))
                self._on_dim_mode_changed()
            gp = cfg.get('geometry_profile')
            if isinstance(gp, dict):
                self.chk_use_dobro.setChecked(gp.get('use_dobro', True))
                gtype = gp.get('type')
                if gtype == 'rpa':
                    self.rb_calc_rpa.setChecked(True)
                elif gtype == 'conical':
                    self.rb_calc_conical.setChecked(True)
                else:
                    self.rb_calc_profiled.setChecked(True)
                self.sp_calc_Rthroat.setValue(gp.get('R_throat', 0.05))
                self.cb_calc_Rthroat_unit.setCurrentText(gp.get('R_throat_unit', 'м'))
                self.sp_calc_Rcham.setValue(gp.get('R_chamber_factor', 2.5))
                self.sp_calc_theta_in.setValue(gp.get('theta_in', 30.0))
                self.chk_calc_auto_angles.setChecked(gp.get('auto_angles', True))
                self.sp_calc_theta_max.setValue(gp.get('theta_max', 30.0))
                self.sp_calc_theta_exit.setValue(gp.get('theta_exit', 15.0))
                self.sp_calc_len_ratio.setValue(gp.get('len_ratio', 9.5))
                self.sp_calc_Rsub.setValue(gp.get('R_sub', 1.5))
                self.sp_calc_rsup.setValue(gp.get('r_sup', 0.45))
                self.sp_calc_R1.setValue(gp.get('R1', 3.0))
                # RPA-параметры
                self.sp_calc_rpa_b.setValue(gp.get('rpa_b', 30.0))
                self.sp_calc_rpa_R1Rt.setValue(gp.get('rpa_R1Rt', 1.5))
                self.sp_calc_rpa_R2.setValue(gp.get('rpa_R2', 0.5))
                self.sp_calc_rpa_RnRt.setValue(gp.get('rpa_RnRt', 0.382))
                self.chk_calc_rpa_auto.setChecked(gp.get('rpa_auto', True))
                self.sp_calc_rpa_Tn.setValue(gp.get('rpa_Tn', 27.0))
                self.sp_calc_rpa_Te.setValue(gp.get('rpa_Te', 10.0))
                self.sp_calc_rpa_LeLe15.setValue(gp.get('rpa_LeLe15', 80.0))
                self._on_calc_geom_type_changed()
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
