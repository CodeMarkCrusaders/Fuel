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

# Plotly + Qt WebEngine — интерактивные графики 1D. Опционально: если чего-то
# нет, интерфейс корректно откатывается на matplotlib-холст (см. PlotlyCanvas).
PLOTLY_AVAILABLE = False
WEBENGINE_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None
    pio = None
try:
    from PyQt5 import QtWebEngineWidgets  # noqa: F401
    WEBENGINE_AVAILABLE = True
except ImportError:
    QtWebEngineWidgets = None

# Импорт решателей (всё через пакет fuel_equilibrium)
from ..rocket.nozzle_flow import (
    Propellant, StationResult, RocketPerformance,
    solve_rocket_nozzle, stoichiometric_OF,
)
from ..rocket.nozzle_geometry import (
    build_conical_nozzle, build_profiled_nozzle,
    build_geometry_from_performance, optimal_angles_from_area_ratio,
    dispersion_loss_coeff, NozzleGeometry,
    build_rpa_parabolic_nozzle, rao_reference_length_15deg, estimate_bell_angles,
)
from ..rocket.nozzle_flow_2d import solve_nozzle_2d, Nozzle2DResult
from ..rocket.analytic_sizing import (
    AnalyticSizingInput, AnalyticSizingResult, compute_analytic_sizing,
)
from ..io.reporting import print_nozzle_table
from ..core.nasa9_parser import parse_thermo_file
from ..core.equilibrium import find_thermo_db
from ..core.equilibrium_cache import clear_cache as clear_equilibrium_cache
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
    smooth: bool = False   # сглаживание кривых (сплайн) на 1D-графиках


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


# Готов ли встроенный интерактивный Plotly-холст (нужны и plotly, и WebEngine).
PLOTLY_CANVAS_READY = PLOTLY_AVAILABLE and WEBENGINE_AVAILABLE


class PlotlyCanvas(QtWidgets.QWidget):
    """Интерактивный холст на Plotly, встроенный в Qt через QWebEngineView.

    Принимает готовую ``plotly.graph_objects.Figure`` (метод :meth:`set_figure`)
    и рендерит её как самодостаточный HTML (offline, с подключённым plotly.js).
    Хранит последнюю фигуру в :attr:`figure`, чтобы её можно было сохранить.

    Работает только если доступны и ``plotly``, и ``PyQtWebEngine``
    (см. ``PLOTLY_CANVAS_READY``). Иначе использовать MplCanvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None  # последняя plotly-фигура (go.Figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._view = QtWebEngineWidgets.QWebEngineView(self)
        self._view.setContextMenuPolicy(Qt.NoContextMenu)
        lay.addWidget(self._view)
        # Заглушка до первого расчёта.
        self.show_message("Выполните расчёт сопла, чтобы построить графики.")

    def show_message(self, text: str, *, dark: bool = True):
        bg = '#1c1917' if dark else '#ffffff'
        fg = '#a8a29e' if dark else '#555555'
        html = (
            f"<html><body style='margin:0;background:{bg};"
            f"display:flex;align-items:center;justify-content:center;"
            f"height:100vh;font-family:sans-serif;color:{fg};'>"
            f"<div>{text}</div></body></html>"
        )
        self._view.setHtml(html)

    def set_figure(self, fig):
        """Отображает plotly-фигуру ``fig`` в виджете."""
        self.figure = fig
        # full_html=True + include_plotlyjs='inline' → автономная страница,
        # не требующая интернета (plotly.js встраивается прямо в HTML).
        html = pio.to_html(
            fig, full_html=True, include_plotlyjs='inline',
            config={
                'displaylogo': False,
                'responsive': True,
                # приближение колёсиком мыши (req: «приближать колёсиком»)
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {'format': 'png', 'scale': 2},
            },
        )
        self._view.setHtml(html)


class CheckableComboBox(QtWidgets.QComboBox):
    """Выпадающий список с множественным выбором (галочки в пунктах).

    Используется для выбора набора отображаемых графиков прямо в панели
    «Оформление графиков». В поле показывает сводку выбранного
    («Выбрано: N» или список подписей), список остаётся открытым при
    переключении галочек. Ключи пунктов хранятся в ``Qt.UserRole + 1``.
    """

    selectionChanged = pyqtSignal()
    KEY_ROLE = Qt.UserRole + 1

    def __init__(self, parent=None, placeholder: str = "Выбрать графики…"):
        super().__init__(parent)
        self._placeholder = placeholder
        self.setModel(QtGui.QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText(placeholder)
        # клик по полю открывает выпадающий список
        self.lineEdit().installEventFilter(self)
        self.view().installEventFilter(self)
        self.view().pressed.connect(self._on_item_pressed)
        self._update_text()

    # — построение списка —
    def addCheckItem(self, key: str, text: str, checked: bool = False):
        item = QtGui.QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
        item.setData(key, self.KEY_ROLE)
        self.model().appendRow(item)
        self._update_text()

    # — выбранные ключи —
    def checked_keys(self) -> list:
        keys = []
        for i in range(self.model().rowCount()):
            it = self.model().item(i)
            if it.data(Qt.CheckStateRole) == Qt.Checked:
                keys.append(it.data(self.KEY_ROLE))
        return keys

    def set_checked_keys(self, keys):
        keys = set(keys or [])
        for i in range(self.model().rowCount()):
            it = self.model().item(i)
            checked = it.data(self.KEY_ROLE) in keys
            it.setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
        self._update_text()
        self.selectionChanged.emit()

    # — взаимодействие —
    def _on_item_pressed(self, index):
        it = self.model().itemFromIndex(index)
        if it is None:
            return
        new_state = (Qt.Unchecked
                     if it.data(Qt.CheckStateRole) == Qt.Checked
                     else Qt.Checked)
        it.setData(new_state, Qt.CheckStateRole)
        self._update_text()
        self.selectionChanged.emit()

    def eventFilter(self, obj, event):
        if obj is self.lineEdit() and event.type() == QtCore.QEvent.MouseButtonRelease:
            self.showPopup()
            return True
        return super().eventFilter(obj, event)

    def hidePopup(self):
        # не закрываем список по клику на пункт — позволяем отметить несколько
        pass

    def _update_text(self):
        labels = []
        for i in range(self.model().rowCount()):
            it = self.model().item(i)
            if it.data(Qt.CheckStateRole) == Qt.Checked:
                labels.append(it.text())
        if not labels:
            text = ""
        elif len(labels) <= 2:
            text = ", ".join(labels)
        else:
            text = f"Выбрано: {len(labels)}"
        self._full_text = text
        self._apply_elided_text()

    def _apply_elided_text(self):
        """Показывает текст с многоточием (…), если он не помещается в поле."""
        le = self.lineEdit()
        full = getattr(self, "_full_text", "")
        if not full:
            le.setText("")
            le.setToolTip("")
            return
        fm = le.fontMetrics()
        avail = max(0, le.width() - 8)   # небольшой отступ под рамку
        elided = fm.elidedText(full, Qt.ElideRight, avail)
        le.setText(elided)
        # полный текст — в подсказке, если был обрезан
        le.setToolTip(full if elided != full else "")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # при изменении ширины пересчитываем многоточие
        self._apply_elided_text()


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

    def _solve_for_of(self, of_ratio: float):
        """Один прогон решателя сопла при заданном массовом O/F (Km)."""
        p = self.params
        # Внутри окислителя/горючего «масса» задаёт долю компонента (0.001..1).
        # Суммарное O/F (Km) задаётся отдельно и не зависит от этих долей.
        ox_components = p['ox_components']  # List[Dict{'name', 'mass', 'T'}]
        fu_components = p['fuel_components']
        if not ox_components or not fu_components:
            raise ValueError("Не заданы компоненты окислителя и/или горючего.")

        of_ratio = max(float(of_ratio), 1e-9)

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
        return perf

    def _find_optimum_of(self):
        """Поиск оптимального Km (массовое O/F), максимизирующего удельный импульс Isp.

        Стратегия: грубое сканирование сетки (геометрическое распределение точек
        вокруг стехиометрического Km0) + уточнение методом золотого сечения.
        Возвращает (best_perf, best_of).
        """
        p = self.params

        def isp_of(of_ratio):
            perf = self._solve_for_of(of_ratio)
            return perf, (perf.Isp_s if perf and perf.Isp_s is not None
                          and math.isfinite(perf.Isp_s) else -1.0)

        # Диапазон поиска: вокруг стехиометрии Km0, если она известна.
        km0 = p.get('of_stoich', float('nan'))
        if km0 is not None and math.isfinite(km0) and km0 > 0:
            of_lo = 0.2 * km0
            of_hi = 2.2 * km0
        else:
            of_lo, of_hi = 0.3, 20.0

        # Грубое сканирование (геометрическая сетка из 9 точек).
        n_grid = 9
        grid = [of_lo * (of_hi / of_lo) ** (i / (n_grid - 1)) for i in range(n_grid)]
        best_perf = None
        best_of = None
        best_isp = -1.0
        cache = {}
        for k, of in enumerate(grid):
            self.progress.emit(f"Поиск оптимума Km: сетка {k + 1}/{n_grid} (Km={of:.3f})...")
            try:
                perf, isp = isp_of(of)
            except Exception:
                continue
            cache[of] = (perf, isp)
            if isp > best_isp:
                best_isp, best_perf, best_of = isp, perf, of

        if best_of is None:
            # Сетка не дала валидных решений — fallback к Km0 или 1.0.
            fallback_of = km0 if (km0 and math.isfinite(km0) and km0 > 0) else 1.0
            return self._solve_for_of(fallback_of), fallback_of

        # Уточнение методом золотого сечения вокруг лучшей точки сетки.
        idx = grid.index(best_of)
        a = grid[max(0, idx - 1)]
        b = grid[min(n_grid - 1, idx + 1)]
        gr = (math.sqrt(5.0) - 1.0) / 2.0  # ≈0.618
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
            self.progress.emit(f"Уточнение оптимума Km: итерация {it + 1}/6...")
            if fc >= fd:
                b, d, fd, pd = d, c, fc, pc
                c = b - gr * (b - a)
                pc, fc = eval_of(c)
            else:
                a, c, fc, pc = c, d, fd, pd
                d = a + gr * (b - a)
                pd, fd = eval_of(d)

        # Выбор лучшего из всех просчитанных точек.
        for of, (perf, isp) in cache.items():
            if perf is not None and isp > best_isp:
                best_isp, best_perf, best_of = isp, perf, of

        self.progress.emit(f"Оптимум найден: Km = {best_of:.4f} (Isp = {best_isp:.2f} с).")
        return best_perf, best_of

    def run(self):
        try:
            p = self.params
            if not p.get('ox_components') or not p.get('fuel_components'):
                raise ValueError("Не заданы компоненты окислителя и/или горючего.")

            if p.get('optimize_of'):
                self.progress.emit("Поиск оптимального соотношения компонентов (max Isp)...")
                perf, _best_of = self._find_optimum_of()
            else:
                of_ratio = float(p.get('of_ratio', 1.0))
                if self.solver == 'cea':
                    self.progress.emit("Запуск CEA-решателя (Cantera)...")
                else:
                    self.progress.emit("Запуск собственного решателя (Gibbs)...")
                perf = self._solve_for_of(of_ratio)

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

        # ═══ Группа 4: Аналитический (инженерный) расчёт — АЛЬТЕРНАТИВА ═══
        # Инженерная методика РПА/Добровольского: по тяге и термодинамическим
        # данным напрямую определяются C*, расходы, площади и геометрия камеры.
        self.tabs_analytic = QtWidgets.QTabWidget()
        self.tabs_analytic.setObjectName("subtabs")
        self.tab_analytic = self._build_analytic_tab()
        self.tabs_analytic.addTab(self.tab_analytic, "Профиль сопла по тяге (РПА)")
        self.tabs.addTab(self.tabs_analytic, "Аналитический расчёт")

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

        # ─── Соотношение компонентов топлива ───
        # Три режима задания:
        #   Km     — массовое отношение расхода окислителя к расходу горючего (O/F);
        #   α      — отношение Km к стехиометрическому Km0 (α = Km/Km0);
        #   Оптимум — Km подбирается автоматически по максимуму удельного импульса Isp.
        self.cb_mix_mode = QtWidgets.QComboBox()
        self.cb_mix_mode.addItems([
            "Km (массовое O/F)",
            "α (Km/Km0)",
            "Оптимум (max Isp)",
        ])
        self.cb_mix_mode.setToolTip(
            "Способ задания соотношения компонентов топлива:\n"
            "  • Km — массовое отношение расхода окислителя к расходу горючего;\n"
            "  • α — отношение Km к стехиометрическому Km0 (α = Km/Km0);\n"
            "  • Оптимум — Km подбирается по максимуму удельного импульса Isp."
        )
        self.cb_mix_mode.currentIndexChanged.connect(self._on_mix_mode_changed)

        # Поле значения — пустое по умолчанию (QLineEdit вместо QDoubleSpinBox).
        self.ed_mix_value = QtWidgets.QLineEdit()
        self.ed_mix_value.setPlaceholderText("Km (O/F)")
        self.ed_mix_value.setValidator(
            QtGui.QDoubleValidator(0.0, 1e6, 6, self.ed_mix_value)
        )
        self.ed_mix_value.setToolTip(
            "Числовое значение соотношения компонентов согласно выбранному режиму.\n"
            "В режиме «Оптимум» поле не используется."
        )
        self.ed_mix_value.textChanged.connect(self._update_of_from_mixture)

        # Информационная метка: показывает Km0 / результирующий Km.
        self.lbl_of = QtWidgets.QLabel("Km0 = —")
        self.lbl_of.setStyleSheet("color: #cc785c; font-weight: bold;")

        of_layout = QtWidgets.QHBoxLayout()
        of_layout.addWidget(QtWidgets.QLabel("Соотношение:"))
        of_layout.addWidget(self.cb_mix_mode)
        of_layout.addWidget(self.ed_mix_value)
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
        # Давления — пустые поля по умолчанию (QLineEdit вместо QDoubleSpinBox).
        self.ed_Pc = QtWidgets.QLineEdit()
        self.ed_Pc.setPlaceholderText("давление в камере")
        self.ed_Pc.setValidator(QtGui.QDoubleValidator(1e-9, 1e9, 6, self.ed_Pc))
        # контейнер со списком единиц
        w_Pc = QtWidgets.QWidget()
        h_Pc = QtWidgets.QHBoxLayout(w_Pc)
        h_Pc.setContentsMargins(0, 0, 0, 0)
        h_Pc.addWidget(self.ed_Pc)
        self.cb_Pc_unit = QtWidgets.QComboBox()
        self.cb_Pc_unit.addItems(["Па", "кПа", "МПа", "бар", "атм"])
        self.cb_Pc_unit.setCurrentText("МПа")
        h_Pc.addWidget(self.cb_Pc_unit)

        self.ed_Pe = QtWidgets.QLineEdit()
        self.ed_Pe.setPlaceholderText("давление на срезе")
        self.ed_Pe.setValidator(QtGui.QDoubleValidator(1e-9, 1e9, 6, self.ed_Pe))
        w_Pe = QtWidgets.QWidget()
        h_Pe = QtWidgets.QHBoxLayout(w_Pe)
        h_Pe.setContentsMargins(0, 0, 0, 0)
        h_Pe.addWidget(self.ed_Pe)
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

        # Доп. расчётные сечения В КАМЕРЕ (между Injector и Nozzle inlet).
        # Injector ≡ Nozzle inlet (застойное состояние), поэтому эти сечения
        # распределяют длину камеры по оси x для отображения участка камеры.
        self._n_chamber_sections = 4
        self.sp_n_chamber = QtWidgets.QSpinBox()
        self.sp_n_chamber.setRange(0, 64)
        self.sp_n_chamber.setValue(self._n_chamber_sections)
        self.sp_n_chamber.setToolTip(
            "Число дополнительных расчётных сечений в камере —\n"
            "между Injector и Nozzle inlet. Состояние газа в камере\n"
            "застойное (P₀, T₀, V≈0), поэтому сечения распределяют\n"
            "длину камеры по оси x, делая участок камеры видимым на графиках."
        )
        self.sp_n_chamber.valueChanged.connect(self._on_chamber_sections_changed)

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
        form_gasd.addRow("Сечений в камере:", self.sp_n_chamber)
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

        # Размер камеры сгорания можно задать ОДНИМ из двух способов
        # (взаимоисключающе, как в RPA):
        #   • Длина камеры L_к (м) — напрямую;
        #   • Характеристическая длина L* = V_к / A_кр (м) — объёмная мера,
        #     из неё длина камеры выводится по площади горловины и камеры.
        w_mode = QtWidgets.QWidget()
        h_mode = QtWidgets.QHBoxLayout(w_mode)
        h_mode.setContentsMargins(0, 0, 0, 0)
        self.rb_chamber_len = QtWidgets.QRadioButton("Длина камеры")
        self.rb_chamber_lstar = QtWidgets.QRadioButton("Характеристическая L*")
        self.rb_chamber_len.setChecked(True)
        self.rb_chamber_len.setToolTip(
            "Задать размер камеры напрямую длиной L_к (м)."
        )
        self.rb_chamber_lstar.setToolTip(
            "Задать размер камеры характеристической длиной L* = V_к / A_кр (м).\n"
            "Длина камеры вычисляется по L*, площади горловины и площади камеры."
        )
        h_mode.addWidget(self.rb_chamber_len)
        h_mode.addWidget(self.rb_chamber_lstar)
        form4.addRow("Размер камеры:", w_mode)

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

        # Характеристическая длина L* (альтернатива длине камеры)
        self.sp_L_star = QtWidgets.QDoubleSpinBox()
        self.sp_L_star.setRange(0.000, 1000.0)
        self.sp_L_star.setDecimals(4)
        self.sp_L_star.setValue(1.000)   # типичная L* для O2/CH4 ~0.8…1.2 м
        self.sp_L_star.setSingleStep(0.05)
        w_L_star = QtWidgets.QWidget()
        h_L_star = QtWidgets.QHBoxLayout(w_L_star)
        h_L_star.setContentsMargins(0, 0, 0, 0)
        h_L_star.addWidget(self.sp_L_star)
        self.cb_L_star_unit = QtWidgets.QComboBox()
        self.cb_L_star_unit.addItems(["м", "см", "мм"])
        self.cb_L_star_unit.setCurrentText("м")
        h_L_star.addWidget(self.cb_L_star_unit)

        # Длины конфузора и дивергента БОЛЬШЕ НЕ ВВОДЯТСЯ вручную — они
        # вычисляются автоматически из геометрии (см. _auto_conv_div_lengths).
        # Спинбоксы оставлены скрытыми только для совместимости с сохранёнными
        # конфигурациями (загрузка/сохранение значений), в UI не отображаются.
        self.sp_L_conv = QtWidgets.QDoubleSpinBox()
        self.sp_L_conv.setRange(0.000, 1000.0)
        self.sp_L_conv.setDecimals(4)
        self.sp_L_conv.setValue(0.050)
        self.sp_L_conv.setVisible(False)
        self.cb_L_conv_unit = QtWidgets.QComboBox()
        self.cb_L_conv_unit.addItems(["м", "см", "мм"])
        self.cb_L_conv_unit.setCurrentText("м")
        self.cb_L_conv_unit.setVisible(False)

        self.sp_L_div = QtWidgets.QDoubleSpinBox()
        self.sp_L_div.setRange(0.000, 1000.0)
        self.sp_L_div.setDecimals(4)
        self.sp_L_div.setValue(0.200)
        self.sp_L_div.setVisible(False)
        self.cb_L_div_unit = QtWidgets.QComboBox()
        self.cb_L_div_unit.addItems(["м", "см", "мм"])
        self.cb_L_div_unit.setCurrentText("м")
        self.cb_L_div_unit.setVisible(False)

        self.lbl_L_chamber = QtWidgets.QLabel("Длина камеры:")
        self.lbl_L_star = QtWidgets.QLabel("Характеристическая L*:")
        form4.addRow(self.lbl_L_chamber, w_L_ch)
        form4.addRow(self.lbl_L_star, w_L_star)

        # Взаимоисключающее переключение «Длина камеры» / «L*»:
        # активным остаётся только один ввод, второй — заблокирован.
        self.rb_chamber_len.toggled.connect(self._on_chamber_size_mode_changed)
        self.rb_chamber_lstar.toggled.connect(self._on_chamber_size_mode_changed)
        self._on_chamber_size_mode_changed()

        self.form_input_geom.addWidget(gb_geom)

        # ─── Потери / реализуемые КПД (Estimated delivered performance) ───
        # Аналог панели RPA «Estimated delivered performance»:
        #   • Reaction efficiency  η_р  — КПД процессов в камере сгорания
        #     (неполнота сгорания, неравномерность смешения и т.п.);
        #   • Nozzle efficiency    η_с  — КПД сопла (потери на трение,
        #     рассеивание, неравновесность и т.п.);
        #   • Overall efficiency   η_общ = η_р · η_с — суммарный КПД, на который
        #     домножается идеальный (равновесный) удельный импульс/тяга.
        gb_loss = QtWidgets.QGroupBox("Потери (реализуемые КПД)")
        form_loss = QtWidgets.QFormLayout(gb_loss)
        form_loss.setSpacing(6)

        self.sp_eff_reaction = QtWidgets.QDoubleSpinBox()
        self.sp_eff_reaction.setRange(0.0, 1.0)
        self.sp_eff_reaction.setDecimals(4)
        self.sp_eff_reaction.setSingleStep(0.001)
        self.sp_eff_reaction.setValue(1.0)
        self.sp_eff_reaction.setToolTip(
            "КПД реакции (камеры сгорания) η_р: 0…1.\n"
            "Учитывает неполноту/неравномерность сгорания.\n"
            "1.0 — идеальный равновесный процесс."
        )

        self.sp_eff_nozzle = QtWidgets.QDoubleSpinBox()
        self.sp_eff_nozzle.setRange(0.0, 1.0)
        self.sp_eff_nozzle.setDecimals(4)
        self.sp_eff_nozzle.setSingleStep(0.001)
        self.sp_eff_nozzle.setValue(1.0)
        self.sp_eff_nozzle.setToolTip(
            "КПД сопла η_с: 0…1.\n"
            "Учитывает потери на трение, рассеивание, неравновесность.\n"
            "1.0 — идеальное изэнтропическое расширение."
        )

        # Суммарный КПД — вычисляемое поле (только чтение)
        self.sp_eff_overall = QtWidgets.QDoubleSpinBox()
        self.sp_eff_overall.setRange(0.0, 1.0)
        self.sp_eff_overall.setDecimals(4)
        self.sp_eff_overall.setValue(1.0)
        self.sp_eff_overall.setReadOnly(True)
        self.sp_eff_overall.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sp_eff_overall.setToolTip(
            "Суммарный КПД η_общ = η_р · η_с (вычисляется автоматически).\n"
            "На него домножаются идеальные Isp / C* / тяга."
        )

        form_loss.addRow("КПД реакции η_р:", self.sp_eff_reaction)
        form_loss.addRow("КПД сопла η_с:", self.sp_eff_nozzle)
        form_loss.addRow("Суммарный η_общ:", self.sp_eff_overall)

        self.sp_eff_reaction.valueChanged.connect(self._update_overall_efficiency)
        self.sp_eff_nozzle.valueChanged.connect(self._update_overall_efficiency)
        self._update_overall_efficiency()

        self.form_input_geom.addWidget(gb_loss)

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

        # Объединённая область графиков: один контейнер вместо двух подвкладок
        # «Графики (1D)» и «Поле течения (2D)». Сверху — единый селектор вида,
        # ниже — QStackedWidget (1D Plotly / поле 2D matplotlib).
        plot_widget = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_widget)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(4)

        # Каталог доступных для отображения величин:
        # (ключ из _section_series, подпись, единицы, цвет, признак log).
        self.PLOT_PARAM_DEFS = [
            ("P",     "Давление P",            "МПа",     "#cc785c", False),
            ("T",     "Температура T",         "К",       "#6ab0ff", False),
            ("V",     "Скорость потока V",     "м/с",     "#82d27a", False),
            ("M",     "Число Маха M",          "",        "#e6b800", False),
            ("rho",   "Плотность ρ",           "кг/м³",   "#cc785c", False),
            ("gs",    "Изэнтр. показатель γₛ", "",        "#c084fc", False),
            ("a",     "Скорость звука a",      "м/с",     "#4dd0e1", False),
            # — термодинамические величины —
            ("S",     "Энтропия S",            "Дж/(кг·К)", "#f472b6", False),
            ("H",     "Энтальпия H",           "МДж/кг",  "#fb923c", False),
            ("q_dyn", "Динам. давление q",     "МПа",     "#34d399", False),
            # — газодинамические функции —
            ("tau",   "τ(λ) = T/T₀",           "",        "#6ab0ff", False),
            ("pi",    "π(λ) = P/P₀",           "",        "#cc785c", False),
            ("eps",   "ε(λ) = ρ/ρ₀",           "",        "#c084fc", False),
            ("lam",   "λ(x) — скор. коэфф.",   "",        "#e6b800", False),
            ("q_gd",  "q(λ) — прив. расход",   "",        "#82d27a", False),
            ("y_gd",  "y(λ) — функц. имп.",    "",        "#4dd0e1", False),
        ]
        # по умолчанию показываем основной набор
        self._plot_default_keys = ["P", "T", "V", "M", "rho", "gs"]
        # показывать ли силуэт профиля сопла на 1D-графиках (req: вкл/выкл)
        self._show_profile_1d = False

        # — верхняя панель: выбор вида + (для 2D) выбор поля + сохранение —
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QtWidgets.QLabel("Вид:"))
        self.cb_plot_view = QtWidgets.QComboBox()
        self.cb_plot_view.addItems([
            "Графики параметров (1D)",
            "Поле течения (2D)",
        ])
        self.cb_plot_view.setToolTip(
            "«Графики параметров (1D)» — выбранные величины (P, T, V, M, …,\n"
            "а также газодинамические функции τ, π, ε, q, y, λ, энтропия,\n"
            "энтальпия, динамическое давление) по длине сопла (интерактивный\n"
            "Plotly: колёсико — приближение, ЛКМ/колесо — перемещение).\n"
            "«Поле течения (2D)» — цветовое поле параметра по сечению."
        )
        self.cb_plot_view.currentIndexChanged.connect(self._on_plot_view_changed)
        top_row.addWidget(self.cb_plot_view)

        top_row.addSpacing(16)
        self.lbl_field_2d_field = QtWidgets.QLabel("Поле:")
        top_row.addWidget(self.lbl_field_2d_field)
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
        top_row.addWidget(self.cb_field_2d)
        top_row.addStretch(1)

        self.btn_save_1d = QtWidgets.QPushButton("⬇ Сохранить выбранные")
        self.btn_save_1d.setToolTip(
            "Сохранить каждый выбранный график в отдельный файл "
            "(интерактивный HTML + PNG для Plotly, либо PNG).")
        self.btn_save_1d.clicked.connect(self._save_selected_1d_plots)
        top_row.addWidget(self.btn_save_1d)
        pv.addLayout(top_row)

        # — переключаемые страницы: 1D-графики / поле 2D —
        self.plot_stack = QtWidgets.QStackedWidget()

        # Страница 0 — интерактивные 1D-графики (Plotly или matplotlib).
        page_1d = QtWidgets.QWidget()
        p1 = QtWidgets.QVBoxLayout(page_1d)
        p1.setContentsMargins(0, 0, 0, 0)
        p1.setSpacing(4)
        self.use_plotly_1d = PLOTLY_CANVAS_READY
        if self.use_plotly_1d:
            self.canvas_1d = PlotlyCanvas(page_1d)
            self.toolbar_1d = None
            p1.addWidget(self.canvas_1d, 1)
        else:
            self.canvas_1d = MplCanvas(width=10, height=6)
            self.toolbar_1d = NavigationToolbar(self.canvas_1d, page_1d)
            p1.addWidget(self.toolbar_1d)
            p1.addWidget(self.canvas_1d, 1)
        self.plot_stack.addWidget(page_1d)

        # Страница 1 — поле течения 2D / газодинамические функции (mpl).
        self.tab_field_2d = self._build_field_2d_page()
        self.plot_stack.addWidget(self.tab_field_2d)

        pv.addWidget(self.plot_stack, 1)

        # Обратная совместимость: легаси-холсты (используются в наложениях).
        # Не отображаются, но методам не мешают.
        self.canvas_PT = MplCanvas(width=5, height=3.5)
        self.canvas_VM = MplCanvas(width=5, height=3.5)
        self.canvas_RHO = MplCanvas(width=5, height=3.5)

        h.addWidget(plot_widget, 1)

        # Справа — панель настройки стиля
        side = QtWidgets.QWidget()
        self._style_side_panel = side
        self._side_panel_width = 260
        side.setFixedWidth(self._side_panel_width)
        side_v = QtWidgets.QVBoxLayout(side)
        side_v.setContentsMargins(4, 4, 4, 4)

        # — слайдер настройки ширины боковой панели (по требованию) —
        width_row = QtWidgets.QHBoxLayout()
        width_row.setSpacing(6)
        width_row.addWidget(QtWidgets.QLabel("Ширина:"))
        self.sl_side_width = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_side_width.setRange(200, 520)
        self.sl_side_width.setValue(self._side_panel_width)
        self.sl_side_width.setToolTip("Ширина панели «Оформление графиков».")
        self.sl_side_width.valueChanged.connect(self._on_side_width_changed)
        width_row.addWidget(self.sl_side_width, 1)
        self.lbl_side_width = QtWidgets.QLabel(f"{self._side_panel_width}px")
        self.lbl_side_width.setStyleSheet("color: #a8a29e; font-size: 10px;")
        self.lbl_side_width.setMinimumWidth(40)
        width_row.addWidget(self.lbl_side_width)
        side_v.addLayout(width_row)

        gb_style = QtWidgets.QGroupBox("Оформление графиков")
        sf = QtWidgets.QFormLayout(gb_style)
        sf.setSpacing(4)

        # — выбор отображаемых графиков (выпадающий список с множ. выбором) —
        # Перенесено сюда из верхней панели по требованию.
        self.cb_plot_params = CheckableComboBox(placeholder="Показать графики…")
        for key, label, unit, _color, _log in self.PLOT_PARAM_DEFS:
            self.cb_plot_params.addCheckItem(
                key, label + (f", {unit}" if unit else ""),
                checked=(key in self._plot_default_keys),
            )
        self.cb_plot_params.selectionChanged.connect(self._redraw_plots)
        sf.addRow("Показать графики:", self.cb_plot_params)

        # — включение/выключение силуэта профиля сопла на 1D-графиках —
        self.chk_show_profile = QtWidgets.QCheckBox("Профиль сопла на графиках")
        self.chk_show_profile.setChecked(self._show_profile_1d)
        self.chk_show_profile.setToolTip(
            "Показывать силуэт контура сопла (r(x)) фоном на 1D-графиках.")
        self.chk_show_profile.toggled.connect(self._on_toggle_profile_1d)
        sf.addRow("", self.chk_show_profile)

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

        self.chk_markers = QtWidgets.QCheckBox("Отображать точки")
        self.chk_markers.setChecked(True)
        self.chk_markers.setToolTip(
            "Показывать маркеры в расчётных точках (сечениях) на 1D-графиках.")
        self.chk_markers.toggled.connect(self._redraw_plots)
        sf.addRow("", self.chk_markers)

        self.chk_smooth = QtWidgets.QCheckBox("Сглаживание графиков")
        self.chk_smooth.setChecked(False)
        self.chk_smooth.setToolTip(
            "Сглаживать кривые на 1D-графиках (сплайн-интерполяция).")
        self.chk_smooth.toggled.connect(self._redraw_plots)
        sf.addRow("", self.chk_smooth)

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
    # Вкладка «Аналитический расчёт профиля сопла» — АЛЬТЕРНАТИВА
    # (инженерная методика РПА/Добровольского: по тяге и термоданным)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_analytic_tab(self) -> QtWidgets.QWidget:
        """Альтернативный (инженерный) расчёт профиля сопла по заданной тяге.

        В отличие от основного термодинамического (равновесного) расчёта здесь
        размеры двигателя определяются «обратной» инженерной методикой:
        задаются тяга в пустоте, давления, соотношение компонентов и
        термодинамические показатели (Iуд, k, Rг, Tк) — и последовательно
        вычисляются C*, расходы, площади критики/среза и геометрия камеры.
        """
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Левая колонка: исходные данные ──
        params = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(params)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(10)
        params.setMinimumWidth(360)
        params.setMaximumWidth(440)

        intro = QtWidgets.QLabel(
            "Инженерная методика РПА / Добровольского.\n"
            "Альтернатива термодинамическому расчёту: размеры двигателя\n"
            "определяются по заданной тяге и термодинамическим данным."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa0a6; font-size:9pt;")
        pv.addWidget(intro)

        # ── Группа: тяга и давления ──
        gb_load = QtWidgets.QGroupBox("Тяга и давления")
        fl = QtWidgets.QFormLayout(gb_load)
        fl.setSpacing(6)

        self.sp_an_thrust = QtWidgets.QDoubleSpinBox()
        self.sp_an_thrust.setRange(1.0, 1.0e9)
        self.sp_an_thrust.setDecimals(1)
        self.sp_an_thrust.setValue(7_770_000.0)
        self.sp_an_thrust.setSingleStep(10_000.0)
        self.sp_an_thrust.setToolTip("Pн — тяга двигателя в пустоте, Н")

        self.sp_an_pk = QtWidgets.QDoubleSpinBox()
        self.sp_an_pk.setRange(0.001, 100.0)
        self.sp_an_pk.setDecimals(4)
        self.sp_an_pk.setValue(7.0)
        self.sp_an_pk.setSingleStep(0.1)
        self.sp_an_pk.setToolTip("pкс — давление в камере сгорания, МПа")

        self.sp_an_pa = QtWidgets.QDoubleSpinBox()
        self.sp_an_pa.setRange(0.00001, 10.0)
        self.sp_an_pa.setDecimals(5)
        self.sp_an_pa.setValue(0.0486)
        self.sp_an_pa.setSingleStep(0.001)
        self.sp_an_pa.setToolTip("pa — давление на срезе сопла, МПа")

        fl.addRow("Pн (тяга в пустоте), Н:", self.sp_an_thrust)
        fl.addRow("pк (камера), МПа:", self.sp_an_pk)
        fl.addRow("pa (срез), МПа:", self.sp_an_pa)
        pv.addWidget(gb_load)

        # ── Группа: компоненты и термодинамика ──
        gb_td = QtWidgets.QGroupBox("Термодинамические данные")
        ft = QtWidgets.QFormLayout(gb_td)
        ft.setSpacing(6)

        self.sp_an_Km = QtWidgets.QDoubleSpinBox()
        self.sp_an_Km.setRange(0.01, 100.0)
        self.sp_an_Km.setDecimals(4)
        self.sp_an_Km.setValue(2.27)
        self.sp_an_Km.setSingleStep(0.05)
        self.sp_an_Km.setToolTip("Km — действительное массовое соотношение окислитель/горючее")

        self.sp_an_isp = QtWidgets.QDoubleSpinBox()
        self.sp_an_isp.setRange(1.0, 100_000.0)
        self.sp_an_isp.setDecimals(4)
        self.sp_an_isp.setValue(3349.4838)
        self.sp_an_isp.setSingleStep(10.0)
        self.sp_an_isp.setToolTip("Iуд — удельный импульс в пустоте (из термодинам. расчёта), м/с")

        self.sp_an_k = QtWidgets.QDoubleSpinBox()
        self.sp_an_k.setRange(1.001, 2.0)
        self.sp_an_k.setDecimals(4)
        self.sp_an_k.setValue(1.1343)
        self.sp_an_k.setSingleStep(0.01)
        self.sp_an_k.setToolTip("k — показатель адиабаты (в камере и на срезе)")

        self.sp_an_Rg = QtWidgets.QDoubleSpinBox()
        self.sp_an_Rg.setRange(1.0, 5000.0)
        self.sp_an_Rg.setDecimals(3)
        self.sp_an_Rg.setValue(346.2)
        self.sp_an_Rg.setSingleStep(1.0)
        self.sp_an_Rg.setToolTip("Rг — газовая постоянная в камере, Дж/(кг·К)")

        self.sp_an_Tk = QtWidgets.QDoubleSpinBox()
        self.sp_an_Tk.setRange(100.0, 10_000.0)
        self.sp_an_Tk.setDecimals(2)
        self.sp_an_Tk.setValue(3692.99)
        self.sp_an_Tk.setSingleStep(10.0)
        self.sp_an_Tk.setToolTip("Tк — температура в камере сгорания, К")

        self.sp_an_alpha = QtWidgets.QDoubleSpinBox()
        self.sp_an_alpha.setRange(0.0, 100.0)
        self.sp_an_alpha.setDecimals(3)
        self.sp_an_alpha.setValue(0.81)
        self.sp_an_alpha.setSingleStep(0.01)
        self.sp_an_alpha.setToolTip("α — коэффициент избытка окислителя (справочно)")

        ft.addRow("Km (O/F массовое):", self.sp_an_Km)
        ft.addRow("Iуд (пустота), м/с:", self.sp_an_isp)
        ft.addRow("k (адиабата):", self.sp_an_k)
        ft.addRow("Rг, Дж/(кг·К):", self.sp_an_Rg)
        ft.addRow("Tк (камера), К:", self.sp_an_Tk)
        ft.addRow("α (справочно):", self.sp_an_alpha)
        pv.addWidget(gb_td)

        # ── Группа: коэффициенты потерь и геометрия камеры ──
        gb_loss = QtWidgets.QGroupBox("Потери и геометрия камеры")
        fll = QtWidgets.QFormLayout(gb_loss)
        fll.setSpacing(6)

        self.sp_an_phik = QtWidgets.QDoubleSpinBox()
        self.sp_an_phik.setRange(0.5, 1.0)
        self.sp_an_phik.setDecimals(4)
        self.sp_an_phik.setValue(0.99)
        self.sp_an_phik.setSingleStep(0.001)
        self.sp_an_phik.setToolTip("φк — коэффициент потерь в камере сгорания")

        self.sp_an_phic = QtWidgets.QDoubleSpinBox()
        self.sp_an_phic.setRange(0.5, 1.0)
        self.sp_an_phic.setDecimals(4)
        self.sp_an_phic.setValue(0.98)
        self.sp_an_phic.setSingleStep(0.001)
        self.sp_an_phic.setToolTip("φс — коэффициент потерь в сопле")

        self.sp_an_winj = QtWidgets.QDoubleSpinBox()
        self.sp_an_winj.setRange(1.0, 200.0)
        self.sp_an_winj.setDecimals(1)
        self.sp_an_winj.setValue(30.0)
        self.sp_an_winj.setSingleStep(1.0)
        self.sp_an_winj.setToolTip("Wср — средняя осевая скорость впрыска компонентов, м/с (20…40)")

        self.sp_an_rho = QtWidgets.QDoubleSpinBox()
        self.sp_an_rho.setRange(0.5, 10.0)
        self.sp_an_rho.setDecimals(2)
        self.sp_an_rho.setValue(2.0)
        self.sp_an_rho.setSingleStep(0.5)
        self.sp_an_rho.setToolTip("ρ — относительный радиус скругления входа")

        fll.addRow("φк (камера):", self.sp_an_phik)
        fll.addRow("φс (сопло):", self.sp_an_phic)
        fll.addRow("Wср (впрыск), м/с:", self.sp_an_winj)
        fll.addRow("ρ (скругление):", self.sp_an_rho)
        pv.addWidget(gb_loss)

        # ── Кнопки ──
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_an_compute = QtWidgets.QPushButton("Рассчитать")
        self.btn_an_compute.clicked.connect(self._on_analytic_compute)
        self.btn_an_from_main = QtWidgets.QPushButton("Из основного расчёта")
        self.btn_an_from_main.setToolTip(
            "Подставить Iуд, k, Rг, Tк, давления и Km из последнего\n"
            "термодинамического (равновесного) расчёта."
        )
        self.btn_an_from_main.clicked.connect(self._on_analytic_pull_from_main)
        btn_row.addWidget(self.btn_an_compute)
        btn_row.addWidget(self.btn_an_from_main)
        pv.addLayout(btn_row)
        pv.addStretch(1)

        root.addWidget(params)

        # ── Правая колонка: результаты ──
        self.txt_analytic = QtWidgets.QPlainTextEdit()
        self.txt_analytic.setReadOnly(True)
        self.txt_analytic.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas','DejaVu Sans Mono',monospace; "
            "font-size: 10.5pt; }"
        )
        self.txt_analytic.setPlainText(
            "Задайте исходные данные слева и нажмите «Рассчитать».\n\n"
            "Кнопка «Из основного расчёта» подставит термодинамические\n"
            "показатели (Iуд, k, Rг, Tк) и режим из последнего равновесного расчёта."
        )
        root.addWidget(self.txt_analytic, 1)
        return w

    def _on_analytic_pull_from_main(self):
        """Подставить параметры из последнего термодинамического расчёта."""
        perf = getattr(self, "perf", None)
        if perf is None:
            QtWidgets.QMessageBox.information(
                self, "Нет данных",
                "Сначала выполните основной (термодинамический) расчёт —\n"
                "затем его результаты можно подставить сюда."
            )
            return
        try:
            g0 = 9.80665
            # Удельный импульс в пустоте: RocketPerformance.Isp_vac_s [с] → м/с
            isp_vac_s = getattr(perf, "Isp_vac_s", None)
            if isp_vac_s:
                self.sp_an_isp.setValue(float(isp_vac_s) * g0)
            # Соотношение компонентов
            of = getattr(perf, "O_F", None)
            if of:
                self.sp_an_Km.setValue(float(of))
            # Параметры камеры берём из первого сечения (Injector / камера)
            stations = getattr(perf, "stations", None) or []
            chamber = None
            for st in stations:
                lbl = (getattr(st, "label", "") or "").lower()
                if "inject" in lbl or "chamber" in lbl or "камер" in lbl:
                    chamber = st
                    break
            if chamber is None and stations:
                chamber = stations[0]
            if chamber is not None:
                k = getattr(chamber, "gamma_eq", None) or getattr(chamber, "gamma_s", None)
                if k:
                    self.sp_an_k.setValue(float(k))
                Rg = getattr(chamber, "R_specific_J_per_kgK", None)
                if Rg:
                    self.sp_an_Rg.setValue(float(Rg))
                Tk = getattr(chamber, "T_K", None)
                if Tk:
                    self.sp_an_Tk.setValue(float(Tk))
                Pc = getattr(chamber, "P_Pa", None)
                if Pc:
                    self.sp_an_pk.setValue(float(Pc) / 1e6)
            # Давление на срезе — из последнего сверхзвукового сечения
            if stations:
                Pe = getattr(stations[-1], "P_Pa", None)
                if Pe:
                    self.sp_an_pa.setValue(float(Pe) / 1e6)
            # α — справочно
            alpha = getattr(perf, "alpha", None)
            if alpha:
                self.sp_an_alpha.setValue(float(alpha))
            QtWidgets.QMessageBox.information(
                self, "Готово",
                "Параметры подставлены из последнего термодинамического расчёта.\n"
                "Проверьте Iуд, k, Rг, Tк и давления перед расчётом."
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Ошибка", f"Не удалось подставить параметры:\n{exc}"
            )

    def _on_analytic_compute(self):
        """Выполнить аналитический расчёт профиля сопла и вывести результат."""
        try:
            inp = AnalyticSizingInput(
                thrust_vac_N=float(self.sp_an_thrust.value()),
                p_chamber_Pa=float(self.sp_an_pk.value()) * 1e6,
                p_exit_Pa=float(self.sp_an_pa.value()) * 1e6,
                Km=float(self.sp_an_Km.value()),
                Isp_vac_m_s=float(self.sp_an_isp.value()),
                k_adiabatic=float(self.sp_an_k.value()),
                R_gas_J_kgK=float(self.sp_an_Rg.value()),
                T_chamber_K=float(self.sp_an_Tk.value()),
                phi_k=float(self.sp_an_phik.value()),
                phi_c=float(self.sp_an_phic.value()),
                alpha=float(self.sp_an_alpha.value()),
                W_inj_mean_m_s=float(self.sp_an_winj.value()),
                rho_curvature=float(self.sp_an_rho.value()),
            )
            res = compute_analytic_sizing(inp)
            self.txt_analytic.setPlainText(self._format_analytic_result(inp, res))
            self._last_analytic_result = res
        except Exception as exc:  # noqa: BLE001
            self.txt_analytic.setPlainText(
                f"Ошибка расчёта:\n{exc}\n\n{traceback.format_exc()}"
            )

    @staticmethod
    def _format_analytic_result(inp: "AnalyticSizingInput",
                                r: "AnalyticSizingResult") -> str:
        """Текстовый отчёт по аналитическому расчёту профиля сопла."""
        def fnum(x, d=4):
            return f"{x:.{d}f}"
        L = []
        L.append("═" * 64)
        L.append("  АНАЛИТИЧЕСКИЙ РАСЧЁТ ПРОФИЛЯ СОПЛА (РПА / Добровольский)")
        L.append("  Альтернатива термодинамическому (равновесному) расчёту")
        L.append("═" * 64)
        L.append("")
        L.append("ИСХОДНЫЕ ДАННЫЕ")
        L.append("─" * 64)
        L.append(f"  Тяга в пустоте Pн ........... {inp.thrust_vac_N:,.1f} Н")
        L.append(f"  Давление в камере pк ........ {inp.p_chamber_Pa/1e6:.4f} МПа")
        L.append(f"  Давление на срезе pa ........ {inp.p_exit_Pa/1e6:.5f} МПа")
        L.append(f"  Соотношение компонентов Km .. {inp.Km:.4f}")
        L.append(f"  Удельный импульс Iуд ........ {inp.Isp_vac_m_s:.4f} м/с")
        L.append(f"  Показатель адиабаты k ....... {inp.k_adiabatic:.4f}")
        L.append(f"  Газовая постоянная Rг ....... {inp.R_gas_J_kgK:.3f} Дж/(кг·К)")
        L.append(f"  Температура в камере Tк ..... {inp.T_chamber_K:.2f} К")
        L.append(f"  φк = {inp.phi_k:.4f}   φс = {inp.phi_c:.4f}")
        L.append("")
        L.append("1. ЭНЕРГЕТИЧЕСКИЕ ПОКАЗАТЕЛИ КАМЕРЫ")
        L.append("─" * 64)
        L.append(f"  φуд = φк·φс ................. {fnum(r.phi_ud)}")
        L.append(f"  Характеристическая ск. C* .. {fnum(r.Cstar_m_s, 2)} м/с")
        L.append(f"  Ожидаемая C*ож = C*·φк ...... {fnum(r.Cstar_exp_m_s, 2)} м/с")
        L.append(f"  Ожидаемый Iуд.ож = Iуд·φуд .. {fnum(r.Isp_exp_m_s, 2)} м/с")
        L.append(f"  Коэф. тяги Kпт = Iуд.ож/C* .. {fnum(r.Kp_thrust)}")
        L.append(f"  Ожидаемый Kп.ож = Kпт·φс .... {fnum(r.Kp_thrust_exp)}")
        L.append("")
        L.append("2. РАСХОДЫ ТОПЛИВА")
        L.append("─" * 64)
        L.append(f"  Суммарный расход ṁ .......... {fnum(r.mdot_total_kg_s, 2)} кг/с")
        L.append(f"  Расход горючего ṁг .......... {fnum(r.mdot_fuel_kg_s, 2)} кг/с")
        L.append(f"  Расход окислителя ṁо ........ {fnum(r.mdot_ox_kg_s, 2)} кг/с")
        L.append("")
        L.append("3. ПЛОЩАДИ И ДИАМЕТРЫ (критика / срез)")
        L.append("─" * 64)
        L.append(f"  Относит. площадь среза F̄a .. {fnum(r.Fa_rel, 4)}")
        L.append(f"  Относит. диаметр среза D̄a .. {fnum(r.D_exit_rel, 4)}")
        L.append(f"  λ (приведённая ск.) ......... {fnum(r.lambda_chamber, 4)}")
        L.append(f"  εк0 = 1/f(λ) ................ {fnum(r.eps_k0, 4)}")
        L.append(f"  δк (потери на впрыск) ....... {fnum(r.delta_k, 4)}")
        L.append(f"  εк = εк0/δк ................. {fnum(r.eps_k, 4)}")
        L.append(f"  Fкр (1-е приб.) ............. {fnum(r.F_throat_1_m2, 4)} м²")
        L.append(f"  Dкр (1-е приб.) ............. {fnum(r.D_throat_1_m, 4)} м")
        L.append(f"  F̄к1 (отн. камера, 1 приб.) . {fnum(r.F_chamber_rel_1, 4)}")
        L.append(f"  Fкр (2-е приб., итог) ....... {fnum(r.F_throat_m2, 4)} м²")
        L.append(f"  Dкр (2-е приб., итог) ....... {fnum(r.D_throat_m, 4)} м")
        L.append(f"  Площадь среза Fa ............ {fnum(r.F_exit_m2, 4)} м²")
        L.append(f"  Диаметр среза Da ............ {fnum(r.D_exit_m, 4)} м")
        L.append("")
        L.append("4. ГЕОМЕТРИЯ КАМЕРЫ СГОРАНИЯ")
        L.append("─" * 64)
        L.append(f"  Приведённая длина Lпр ....... {fnum(r.L_reduced_m, 4)} м")
        L.append(f"  Условная длина Lк ........... {fnum(r.L_conditional_m, 4)} м")
        L.append(f"  Объём камеры Vк ............. {fnum(r.V_chamber_m3, 4)} м³")
        L.append(f"  Относит. площадь камеры F̄к2  {fnum(r.F_chamber_rel_2, 4)}")
        L.append(f"  Диаметр камеры Dк ........... {fnum(r.D_chamber_m, 4)} м")
        L.append(f"  Радиус скругления R1 ........ {fnum(r.R1_m, 4)} м")
        L.append(f"  Радиус скругления R2 ........ {fnum(r.R2_m, 4)} м")
        L.append(f"  Длина входной части Lвх ..... {fnum(r.L_inlet_m, 4)} м")
        L.append(f"  Объём входной части ΔVвх .... {fnum(r.dV_inlet_m3, 4)} м³")
        L.append(f"  Длина цил. участка Lц ....... {fnum(r.L_cyl_m, 4)} м")
        L.append("")
        L.append("─" * 64)
        notes = r.notes or {}
        if notes.get("method"):
            L.append(f"Метод: {notes['method']}")
        if notes.get("throat_formula"):
            L.append(f"Формула критики: {notes['throat_formula']}")
        L.append(
            "Примечание: абсолютные площади/объёмы зависят от эмпирических\n"
            "констант методики; энергетика, расходы и εк-цепочка совпадают\n"
            "с эталоном в пределах < 1–3 %."
        )
        return "\n".join(L)

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Поле течения (2D)» — заготовка квази-2D расчёта
    # ─────────────────────────────────────────────────────────────────────────
    def _build_field_2d_page(self) -> QtWidgets.QWidget:
        # Страница поля течения 2D / газодинамических функций внутри
        # объединённой области графиков. Селекторы вида и поля вынесены
        # в общую верхнюю панель (см. _build_plot_tab).
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        self.lbl_field_2d_info = QtWidgets.QLabel(
            "«Газодинамические функции (1D)» строятся всегда по результатам 1D-расчёта:\n"
            "τ(λ)=T/T₀, π(λ)=P/P₀, ε(λ)=ρ/ρ₀, расходная q(λ), удельный импульс y(λ)\n"
            "и скоростной коэффициент λ(x) по длине сопла.\n"
            "«Поле течения (2D)» — цветовое поле параметра по сечению (квази-2D, нужен\n"
            "режим «Двумерный (2D)» во вкладке «Газодинамика (1D/2D)»).\n"
            "Интерактивно: панель сверху — приближение (лупа) / перемещение (рука) /\n"
            "сброс (домик); наведите курсор — значение в точке покажется ниже."
        )
        self.lbl_field_2d_info.setStyleSheet("color: #a8a29e; font-size: 10px;")
        self.lbl_field_2d_info.setWordWrap(True)
        v.addWidget(self.lbl_field_2d_info)

        self.canvas_field_2d = MplCanvas(width=7, height=4.5)
        # Панель навигации matplotlib: масштабирование (zoom), панорамирование
        # (pan), сброс вида (home), сохранение. Даёт интерактивное приближение.
        self.toolbar_field_2d = NavigationToolbar(self.canvas_field_2d, w)
        v.addWidget(self.toolbar_field_2d)
        v.addWidget(self.canvas_field_2d, 1)

        # Строка считывания значения поля под курсором (x, r, величина).
        self.lbl_field_2d_cursor = QtWidgets.QLabel("Наведите курсор на поле…")
        self.lbl_field_2d_cursor.setStyleSheet(
            "color: #e7e5e4; background: #2a2724; border: 1px solid #44403c;"
            " border-radius: 4px; padding: 4px 8px; font-family: monospace;"
        )
        v.addWidget(self.lbl_field_2d_cursor)

        # Подключаем обработчик движения мыши для считывания значения у курсора.
        self.canvas_field_2d.mpl_connect(
            "motion_notify_event", self._on_field_2d_hover
        )
        self.canvas_field_2d.mpl_connect(
            "axes_leave_event", self._on_field_2d_leave
        )

        # Кэш данных текущего отрисованного поля (для интерполяции у курсора).
        self._field_2d_plot_cache: Optional[dict] = None
        self._field_2d_marker = None

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
        # Если 2D-поле успешно посчитано — автоматически переключаем единый
        # селектор вида на «Поле течения (2D)» (индекс 2); иначе остаёмся.
        if (self._last_field_2d is not None and hasattr(self, "cb_plot_view")
                and self.cb_plot_view.currentIndex() != 1):
            self.cb_plot_view.setCurrentIndex(1)
            return
        self._render_field_2d()

    def _field_view_is_2d(self) -> bool:
        """True, если выбран режим «Поле течения (2D)» в едином селекторе."""
        return (hasattr(self, "cb_plot_view")
                and self.cb_plot_view.currentIndex() == 1)

    def _on_plot_view_changed(self, *args):
        """Переключение единого вида: 1D-графики параметров / поле 2D."""
        idx = self.cb_plot_view.currentIndex() if hasattr(self, "cb_plot_view") else 0
        # Страница 0 — 1D-графики параметров; страница 1 — поле 2D (mpl).
        if hasattr(self, "plot_stack"):
            self.plot_stack.setCurrentIndex(0 if idx == 0 else 1)

        is_1d_params = (idx == 0)
        is_2d_field = (idx == 1)
        # Селектор поля (M/P/T/V) актуален только для 2D-поля.
        if hasattr(self, "cb_field_2d"):
            self.cb_field_2d.setVisible(is_2d_field)
        if hasattr(self, "lbl_field_2d_field"):
            self.lbl_field_2d_field.setVisible(is_2d_field)
        # Кнопка сохранения 1D и выбор параметров/профиля — только для 1D-режима.
        if hasattr(self, "btn_save_1d"):
            self.btn_save_1d.setVisible(is_1d_params)
        if hasattr(self, "cb_plot_params"):
            self.cb_plot_params.setEnabled(is_1d_params)
        if hasattr(self, "chk_show_profile"):
            self.chk_show_profile.setEnabled(is_1d_params)

        if is_1d_params:
            self._redraw_plots()
        else:
            self._render_field_2d()

    def _on_toggle_profile_1d(self, checked: bool):
        """Вкл/выкл силуэт профиля сопла на 1D-графиках."""
        self._show_profile_1d = bool(checked)
        self._redraw_plots()

    def _on_side_width_changed(self, value: int):
        """Изменение ширины боковой панели «Оформление графиков» слайдером."""
        self._side_panel_width = int(value)
        if getattr(self, "_style_side_panel", None) is not None:
            self._style_side_panel.setFixedWidth(self._side_panel_width)
        if hasattr(self, "lbl_side_width"):
            self.lbl_side_width.setText(f"{self._side_panel_width}px")
        # пересчёт многоточия в мультивыборе под новую ширину
        if hasattr(self, "cb_plot_params"):
            self.cb_plot_params._apply_elided_text()

    def _on_chamber_sections_changed(self, value: int):
        """Изменение числа расчётных сечений в камере (Injector→Nozzle inlet).

        Сечения камеры синтезируются в ``_section_series`` (застойное
        состояние камеры), поэтому пересчёт решателя не требуется — достаточно
        перерисовать графики.
        """
        self._n_chamber_sections = int(value)
        if getattr(self, "perf", None) is not None:
            self._redraw_plots()

    def _render_field_2d(self):
        c = getattr(self, "canvas_field_2d", None)
        if c is None:
            return
        c.fig.clear()
        self._field_2d_plot_cache = None
        self._field_2d_marker = None

        # На страницу поля 2D переходим только из режима «Поле течения (2D)».
        # Газодинамические функции (τ, π, ε, q, y, λ) теперь доступны прямо в
        # списке величин режима «Графики параметров (1D)».
        mode_2d = self._field_view_is_2d()
        if not mode_2d:
            ax = c.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Газодинамические функции (τ, π, ε, q, y, λ) теперь\n"
                    "доступны в списке величин режима\n"
                    "«Графики параметров (1D)».",
                    ha='center', va='center', fontsize=11, color='#888')
            ax.set_axis_off()
            c.fig.tight_layout()
            c.draw()
            return

        res = self._last_field_2d
        if res is None:
            ax = c.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Нет данных 2D.\nВыберите режим «Двумерный (2D)» и выполните расчёт.",
                    ha='center', va='center', fontsize=11, color='#888')
            ax.set_axis_off()
            c.fig.tight_layout()
            c.draw()
            if hasattr(self, "lbl_field_2d_cursor"):
                self.lbl_field_2d_cursor.setText("Нет данных 2D.")
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
            x_grid = np.asarray(res.x_grid, dtype=float)
            r_grid = np.asarray(res.r_grid, dtype=float)
            vals = np.asarray(vals, dtype=float)
            wall_x = np.asarray(res.wall_x, dtype=float)
            wall_r = np.asarray(res.wall_r, dtype=float)

            # ── Маскируем значения ВНЕ профиля сопла ──────────────────────────
            # Радиус стенки для каждого столбца сетки; узлы, где r чуть превышает
            # стенку (из-за дискретизации), исключаются из заливки, чтобы цвет
            # не выходил за границу контура.
            wall_r_col = wall_r.reshape(1, -1)
            outside = r_grid > (wall_r_col + 1e-12)
            vals_masked = np.ma.array(vals, mask=~np.isfinite(vals) | outside)

            # shading='gouraud' интерполирует цвет по УЗЛАМ сетки (а не по
            # ячейкам), поэтому заливка не «вылезает» за крайние узлы у стенки —
            # значения остаются строго внутри профиля.
            try:
                cmap = matplotlib.colormaps['viridis'].copy()
            except Exception:  # старые версии matplotlib
                cmap = matplotlib.cm.get_cmap('viridis').copy()
            cmap.set_bad(color=(0, 0, 0, 0))  # маскированные ячейки прозрачны
            pcm = ax.pcolormesh(
                x_grid, r_grid, vals_masked,
                shading='gouraud', cmap=cmap,
            )
            ax.plot(wall_x, wall_r, '-', color='#cc785c', lw=1.8)
            c.fig.colorbar(pcm, ax=ax, label=label)
            ax.set_xlabel("x, м")
            ax.set_ylabel("r, м")
            if res.metadata.get("is_stub"):
                tag = " (ЗАГОТОВКА, квази-2D)"
            else:
                tag = " (квази-2D, source-flow)"
            ax.set_title(f"Поле течения 2D — {label}{tag}")
            # Ограничиваем область отображения профилем (с небольшим полем).
            # adjustable='box' сохраняет равный масштаб осей и НЕ переопределяет
            # заданные пределы (в отличие от 'datalim'), поэтому вид остаётся
            # привязан к контуру сопла.
            try:
                xmin, xmax = float(wall_x.min()), float(wall_x.max())
                rmax = float(wall_r.max())
                pad = 0.03 * max(xmax - xmin, rmax, 1e-6)
                ax.set_xlim(xmin - pad, xmax + pad)
                ax.set_ylim(0.0, rmax + pad)
            except Exception:
                pass
            try:
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

            # Кэш для считывания значения под курсором (билинейная интерполяция).
            self._field_2d_plot_cache = {
                "ax": ax,
                "x_grid": x_grid,
                "r_grid": r_grid,
                "vals": vals,
                "wall_x": wall_x,
                "wall_r": wall_r,
                "label": label,
                "key": key,
            }
        c.fig.tight_layout()
        c.draw()
        if hasattr(self, "lbl_field_2d_cursor"):
            self.lbl_field_2d_cursor.setText("Наведите курсор на поле…")

    # ── Детальные газодинамические функции по длине сопла (по 1D-расчёту) ─────
    def _render_gasdynamic_functions_1d(self, c):
        """Строит 6 интерактивных графиков газодинамических функций по длине сопла.

        Источник данных — результаты 1D-расчёта (self.perf.stations). Графики:
          τ(λ) = T/T₀        — функция температуры
          π(λ) = P/P₀        — функция давления
          ε(λ) = ρ/ρ₀        — функция плотности
          q(λ)               — приведённый расход (нормирован на максимум = 1 в горловине)
          y(λ)               — функция удельного импульса = (1+λ²)/(2λ) · q-подобная
          λ(x)               — скоростной коэффициент по длине

        Все функции строятся от координаты x вдоль оси сопла (мм), плюс тонкой
        синей линией показан контур сопла r(x) (в относительном масштабе) для
        привязки к геометрии — как на эталонных графиках.
        """
        c.fig.clear()  # на случай прямого вызова (минуя _render_field_2d)
        perf = getattr(self, "perf", None)
        if perf is None or not getattr(perf, "stations", None):
            ax = c.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Нет данных 1D.\nВыполните газодинамический расчёт, чтобы\n"
                    "построить газодинамические функции по длине сопла.",
                    ha='center', va='center', fontsize=11, color='#888')
            ax.set_axis_off()
            c.fig.tight_layout()
            c.draw()
            if hasattr(self, "lbl_field_2d_cursor"):
                self.lbl_field_2d_cursor.setText("Нет данных 1D.")
            return

        all_stations = perf.stations

        # Параметры торможения (камера = первое сечение) — для нормировки τ, π, ε
        # и расчёта λ. Берём ДО фильтрации, т.к. это полные (stagnation) величины.
        T0 = float(getattr(all_stations[0], "T_K", 0.0)) or max(
            (float(getattr(s, "T_K", 0.0)) for s in all_stations), default=1.0)
        P0 = float(getattr(all_stations[0], "P_Pa", 0.0)) or max(
            (float(getattr(s, "P_Pa", 0.0)) for s in all_stations), default=1.0)
        rho0 = float(getattr(all_stations[0], "rho_kg_per_m3", 0.0)) or max(
            (float(getattr(s, "rho_kg_per_m3", 0.0)) for s in all_stations), default=1.0)
        gam0 = float(getattr(all_stations[0], "gamma_s", 0.0) or 0.0)
        k0 = gam0 if gam0 > 1.0 else 1.2
        a0 = float(getattr(all_stations[0], "a_m_per_s", 0.0)) or max(
            (float(getattr(s, "a_m_per_s", 0.0)) for s in all_stations), default=1.0)

        # ── Газодинамические функции от λ (изэнтропические соотношения) ───────
        # Считаем τ, π, ε, q, y АНАЛИТИЧЕСКИ из λ — это гарантирует гладкие,
        # физически согласованные кривые от начала камеры (λ→0 ⇒ τ,π,ε→1, q→0).
        def _tau_of_lambda(lm, k):
            k = max(k, 1.0001)
            return 1.0 - (k - 1.0) / (k + 1.0) * lm * lm        # T/T0
        def _pi_of_lambda(lm, k):
            k = max(k, 1.0001)
            base = np.clip(_tau_of_lambda(lm, k), 0.0, None)
            return base ** (k / (k - 1.0))                       # P/P0
        def _eps_of_lambda(lm, k):
            k = max(k, 1.0001)
            base = np.clip(_tau_of_lambda(lm, k), 0.0, None)
            return base ** (1.0 / (k - 1.0))                     # ρ/ρ0
        def _q_of_lambda(lm, k):
            k = max(k, 1.0001)
            base = np.clip(1.0 - (k - 1.0) / (k + 1.0) * lm * lm, 0.0, None)
            return (lm * ((k + 1.0) / 2.0) ** (1.0 / (k - 1.0))
                    * base ** (1.0 / (k - 1.0)))
        def _y_of_lambda(lm, k):
            with np.errstate(divide='ignore', invalid='ignore'):
                yy = np.where(lm > 1e-9,
                              (1.0 + lm * lm) / (2.0 * lm) * _q_of_lambda(lm, k),
                              0.0)
            return yy

        # q(λ) для обращения «площадь → λ»: q(λ) = A_кр/A = (R_кр/R)².
        def _lambda_from_area_ratio(area_ratio, k, supersonic):
            """Решает q(λ)=1/area_ratio относительно λ (бисекция).

            area_ratio = A/A_кр ≥ 1. supersonic=False → дозвуковая ветвь
            (0<λ<1), True → сверхзвуковая (1<λ<λ_max).
            """
            k = max(k, 1.0001)
            target = 1.0 / max(area_ratio, 1.0)   # q ∈ (0..1]
            lam_max = math.sqrt((k + 1.0) / (k - 1.0))
            if supersonic:
                lo, hi = 1.0, lam_max - 1e-6
            else:
                lo, hi = 1e-6, 1.0
            ql = float(_q_of_lambda(np.array([lo]), k)[0])
            qh = float(_q_of_lambda(np.array([hi]), k)[0])
            # q монотонна на каждой ветви: дозв. растёт, сверхзв. убывает
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                qm = float(_q_of_lambda(np.array([mid]), k)[0])
                if (qm < target) == (ql < target):
                    lo, ql = mid, qm
                else:
                    hi, qh = mid, qm
            return 0.5 * (lo + hi)

        # ── Реальный профиль сопла из геометрии (как на вкладке «Профиль») ────
        # Используем тот же контур, что и «Профилированное/Коническое» сопло,
        # вместо приближения √(Ae/At). λ(x) восстанавливается из A(x)/A_кр.
        geom = None
        try:
            geom = self._build_calc_geometry(perf)
        except Exception:
            geom = None

        if geom is not None and getattr(geom, "points", None):
            gx, gr = geom.as_xy_arrays()             # м, м (реальный контур)
            gx = np.asarray(gx, dtype=float)
            gr = np.asarray(gr, dtype=float)
            R_throat_m = float(getattr(geom, "R_throat_m", np.nanmin(gr)) or np.nanmin(gr))
            R_throat_m = max(R_throat_m, 1e-6)
            x_throat_m = float(getattr(geom, "length_subsonic_m", gx[int(np.nanargmin(gr))]))
            # сгущаем по x и убираем дубли
            x_u, iu = np.unique(gx, return_index=True)
            r_u = gr[iu]
            xs_m = np.linspace(float(x_u[0]), float(x_u[-1]), 280)
            try:
                from scipy.interpolate import PchipInterpolator  # type: ignore
                r_xs_m = PchipInterpolator(x_u, r_u)(xs_m)
            except Exception:
                r_xs_m = np.interp(xs_m, x_u, r_u)
            xs = xs_m * 1000.0                        # мм для оси графиков
            r_profile_mm = r_xs_m * 1000.0
            x_throat_mm = x_throat_m * 1000.0
            # λ(x) из площади: A/A_кр=(r/R_кр)², ветвь по положению относ. горловины
            area_ratio = np.clip((r_xs_m / R_throat_m) ** 2, 1.0, None)
            lam = np.empty_like(xs_m)
            for i in range(xs_m.size):
                supersonic = xs_m[i] > x_throat_m
                lam[i] = _lambda_from_area_ratio(float(area_ratio[i]), k0, supersonic)
            lam = np.clip(lam, 0.0, None)
        else:
            # ── Запасной вариант: контур и λ из сечений солвера ───────────────
            L_conv_auto, L_div_auto = self._auto_conv_div_lengths()
            try:
                x_all = np.asarray(build_nozzle_geometry(
                    all_stations, L_chamber=self._chamber_length_m(),
                    L_conv=L_conv_auto, L_div=L_div_auto), dtype=float)
            except Exception:
                x_all = np.arange(len(all_stations), dtype=float)
            x_m = x_all - float(np.nanmin(x_all))
            x_mm_nodes = x_m * 1000.0
            V = np.array([float(getattr(s, "V_m_per_s", 0.0)) for s in all_stations])
            a_cr = a0 * math.sqrt(2.0 / (k0 + 1.0)) if a0 > 0 else 1.0
            lam_nodes = np.where(a_cr > 0, V / a_cr, 0.0)
            try:
                r_nodes = np.asarray(nozzle_radius(all_stations), dtype=float)
            except Exception:
                Ae_At = np.array([float(getattr(s, "Ae_At", 1.0)) for s in all_stations])
                r_nodes = np.sqrt(np.clip(Ae_At, 1e-9, None))
            try:
                R_throat_m = self._length_to_m(self.sp_calc_Rthroat.value(),
                                               self.cb_calc_Rthroat_unit.currentText())
            except Exception:
                R_throat_m = 0.05
            R_throat_mm = max(R_throat_m, 1e-4) * 1000.0
            o0 = np.argsort(x_mm_nodes)
            x_u, iu = np.unique(x_mm_nodes[o0], return_index=True)
            lam_u = lam_nodes[o0][iu]
            r_u = (r_nodes[o0][iu]) * R_throat_mm
            if x_u.size >= 2:
                xs = np.linspace(float(x_u[0]), float(x_u[-1]), 240)
                try:
                    from scipy.interpolate import PchipInterpolator  # type: ignore
                    lam = PchipInterpolator(x_u, lam_u)(xs)
                    r_profile_mm = PchipInterpolator(x_u, r_u)(xs)
                except Exception:
                    lam = np.interp(xs, x_u, lam_u)
                    r_profile_mm = np.interp(xs, x_u, r_u)
            else:
                xs = x_u
                lam = lam_u
                r_profile_mm = r_u
            lam = np.clip(lam, 0.0, None)
            i_thr = int(np.nanargmin(r_profile_mm)) if r_profile_mm.size else 0
            x_throat_mm = float(xs[i_thr]) if xs.size else 0.0

        tau = _tau_of_lambda(lam, k0)
        pi = _pi_of_lambda(lam, k0)
        eps = _eps_of_lambda(lam, k0)
        q = _q_of_lambda(lam, k0)
        y = _y_of_lambda(lam, k0)
        if np.nanmax(np.abs(y)) > 0:
            y = y / np.nanmax(y)

        # На сгущённой сетке порядок уже прямой
        order = np.arange(xs.size)
        r_profile_mm = np.asarray(r_profile_mm, dtype=float)
        r_profile_mm = np.where(np.isfinite(r_profile_mm), r_profile_mm,
                                float(np.nanmedian(r_profile_mm)) if r_profile_mm.size else 1.0)
        # нормированная форма стенки [0..1] для ненавязчивого силуэта в панелях
        r_wall_max = float(np.nanmax(r_profile_mm)) if r_profile_mm.size else 1.0
        if not (r_wall_max > 0):
            r_wall_max = 1.0
        r_shape = r_profile_mm / r_wall_max
        # x_throat_mm уже определён выше (length_subsonic_m из геометрии либо
        # минимум контура в запасном варианте).

        # Кривые уже гладкие (аналитика из λ на частой сетке) — без доп. фильтра.
        def _smooth(arr):
            return np.asarray(arr, dtype=float)

        # ── Сетка 3×2: только газодинамические функции (профиль убран) ───────
        # Профиль сопла отображается на вкладках «Поле течения (2D)» и
        # «Геометрия сопла», поэтому здесь его не дублируем.
        gs = c.fig.add_gridspec(
            3, 2, height_ratios=[1.0, 1.0, 1.0],
            left=0.08, right=0.97, top=0.92, bottom=0.07,
            hspace=0.55, wspace=0.25,
        )
        axes_cache = []

        panels = [
            ("Функция температуры τ(λ)",   _smooth(tau), "τ(λ) = T/T0",  "#e03131"),
            ("Функция давления π(λ)",       _smooth(pi),  "π(λ) = P/P0",  "#1c3fce"),
            ("Функция плотности ε(λ)",      _smooth(eps), "ε(λ) = ρ/ρ0",  "#7b2cbf"),
            ("Расходная функция q(λ)",      _smooth(q),   "q(λ)",          "#2b8a3e"),
            ("Функция удельного импульса y(λ)", _smooth(y), "y(λ)",        "#f59f00"),
            ("Скоростной коэффициент λ(x)", _smooth(lam), "λ",             "#9c2a2a"),
        ]

        for idx, (title, ys, ylab, color) in enumerate(panels):
            row = idx // 2
            col = idx % 2
            ax = c.fig.add_subplot(gs[row, col])
            ys = np.asarray(ys, dtype=float)
            # ── Силуэт контура сопла (фон): полупрозрачная заливка r(x) ──
            ymin = np.nanmin(ys) if np.isfinite(np.nanmin(ys)) else 0.0
            ymax = np.nanmax(ys) if np.isfinite(np.nanmax(ys)) else 1.0
            if ymax <= ymin:
                ymax = ymin + 1.0
            pad = 0.08 * (ymax - ymin)
            y_lo, y_hi = ymin - pad, ymax + pad
            # стенка занимает нижние ~30% высоты панели — как ненавязчивый силуэт
            wall_top = y_lo + 0.30 * (y_hi - y_lo) * r_shape
            ax.fill_between(xs, y_lo, wall_top, color='#6ab0ff',
                            alpha=0.10, lw=0, zorder=0)
            ax.plot(xs, wall_top, '-', color='#6ab0ff', lw=0.7,
                    alpha=0.45, zorder=1)
            # Основная кривая функции
            ax.plot(xs, ys, '-', color=color, lw=2.2, zorder=3)
            # Вертикальная линия горловины
            ax.axvline(x_throat_mm, color='#555', ls=':', lw=1.0, zorder=2)
            if idx == 3:  # q(λ) — горизонталь q=1 (максимум в горловине)
                ax.axhline(1.0, color='#aaa', ls='--', lw=0.8, zorder=2)
            if idx == 5:  # λ(x) — горизонталь λ=1 (звуковая линия)
                ax.axhline(1.0, color='#aaa', ls='--', lw=0.8, zorder=2)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Координата вдоль оси сопла X, мм", fontsize=8)
            ax.set_ylabel(ylab, color=color, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            axes_cache.append((ax, xs, ys, title, ylab))

        c.fig.suptitle("Газодинамические функции по длине сопла (1D)",
                       fontsize=11, y=0.985)
        c.draw()

        # Кэш для считывания значений под курсором (мультипанельный режим)
        self._field_2d_plot_cache = {"gdf_axes": axes_cache, "mode": "gdf_1d"}
        if hasattr(self, "lbl_field_2d_cursor"):
            self.lbl_field_2d_cursor.setText(
                "Наведите курсор на график — координата X и значение функции покажутся ниже."
            )

    # ── Интерактивное считывание значения поля под курсором ──────────────────
    def _field_2d_value_at(self, x: float, r: float):
        """Возвращает (значение, r_стенки) поля в точке (x, r) внутри профиля.

        Использует билинейную интерполяцию по структурированной сетке (n_r, n_x),
        где строки — радиальные узлы (0 — ось, -1 — стенка). Если точка вне
        профиля (r > r_стенки) или вне диапазона x, возвращает (None, r_стенки).
        """
        cache = self._field_2d_plot_cache
        if cache is None:
            return None, None
        x_grid = cache["x_grid"]
        r_grid = cache["r_grid"]
        vals = cache["vals"]
        wall_x = cache["wall_x"]
        wall_r = cache["wall_r"]

        x_axis = x_grid[0, :]            # координаты x по столбцам (n_x,)
        if x_axis.size < 2:
            return None, None
        # вне диапазона x
        if x < x_axis.min() or x > x_axis.max():
            wr = float(np.interp(x, wall_x, wall_r)) if wall_x.size else None
            return None, wr

        # радиус стенки в точке x (для проверки «внутри профиля»)
        wr = float(np.interp(x, wall_x, wall_r))
        if r < 0.0 or r > wr + 1e-12:
            return None, wr

        # индекс столбца j и доля tx между x_axis[j], x_axis[j+1]
        j = int(np.clip(np.searchsorted(x_axis, x) - 1, 0, x_axis.size - 2))
        x0, x1 = x_axis[j], x_axis[j + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        tx = float(np.clip(tx, 0.0, 1.0))

        # радиальные координаты в этом столбце (могут отличаться по столбцам)
        def _interp_col(col):
            r_col = r_grid[:, col]
            v_col = vals[:, col]
            order = np.argsort(r_col)
            return float(np.interp(r, r_col[order], v_col[order]))

        v_left = _interp_col(j)
        v_right = _interp_col(j + 1)
        value = (1.0 - tx) * v_left + tx * v_right
        if not np.isfinite(value):
            return None, wr
        return value, wr

    def _on_field_2d_hover(self, event):
        cache = getattr(self, "_field_2d_plot_cache", None)
        lbl = getattr(self, "lbl_field_2d_cursor", None)
        if lbl is None:
            return
        if cache is None:
            return

        # ── Режим газодинамических функций (мультипанель 3×2) ──────────────────
        if cache.get("mode") == "gdf_1d":
            if event.inaxes is None or event.xdata is None:
                return
            x_cur = float(event.xdata)
            for ax, xs, ys, title, ylab in cache.get("gdf_axes", []):
                if event.inaxes is ax:
                    try:
                        val = float(np.interp(x_cur, xs, ys))
                    except Exception:
                        val = float("nan")
                    lbl.setText(
                        f"{title}:   X = {x_cur:.1f} мм    →    {ylab} = {val:.5g}"
                    )
                    return
            return

        if event.inaxes is not cache.get("ax"):
            return
        if event.xdata is None or event.ydata is None:
            return
        x, r = float(event.xdata), float(event.ydata)
        value, wr = self._field_2d_value_at(x, r)
        c = self.canvas_field_2d

        # удаляем прежний маркер
        if self._field_2d_marker is not None:
            try:
                self._field_2d_marker.remove()
            except Exception:
                pass
            self._field_2d_marker = None

        if value is None:
            extra = " (вне профиля)" if wr is not None and r > wr else ""
            lbl.setText(f"x = {x:.4g} м,  r = {r:.4g} м{extra}")
            c.draw_idle()
            return

        # форматирование значения в зависимости от величины поля
        label = cache.get("label", "")
        key = cache.get("key", "")
        if key == "P_Pa":
            vtxt = f"{value/1e6:.4g} МПа ({value:.4g} Па)"
        elif key == "T_K":
            vtxt = f"{value:.5g} К"
        elif key == "V_m_per_s":
            vtxt = f"{value:.5g} м/с"
        elif key == "flow_angle_deg":
            vtxt = f"{value:.4g}°"
        else:
            vtxt = f"{value:.5g}"
        lbl.setText(f"x = {x:.4g} м,  r = {r:.4g} м    →    {label} = {vtxt}")

        # маркер точки на поле
        try:
            (self._field_2d_marker,) = cache["ax"].plot(
                [x], [r], marker='o', ms=6, mfc='none',
                mec='#f5f5f4', mew=1.4, zorder=5,
            )
        except Exception:
            self._field_2d_marker = None
        c.draw_idle()

    def _on_field_2d_leave(self, event):
        lbl = getattr(self, "lbl_field_2d_cursor", None)
        if self._field_2d_marker is not None:
            try:
                self._field_2d_marker.remove()
            except Exception:
                pass
            self._field_2d_marker = None
            try:
                self.canvas_field_2d.draw_idle()
            except Exception:
                pass
        if lbl is not None:
            lbl.setText("Наведите курсор на поле…")

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

    def _on_chamber_size_mode_changed(self, *args):
        """Взаимоисключающее переключение «Длина камеры» / «L*».

        Активным оставляем только выбранный способ задания размера камеры,
        второй ввод блокируется (и визуально приглушается), чтобы исключить
        двусмысленный одновременный ввод длины и характеристической длины.
        """
        use_len = (getattr(self, "rb_chamber_len", None) is not None
                   and self.rb_chamber_len.isChecked())
        # Длина камеры
        for wdg in (getattr(self, "sp_L_chamber", None),
                    getattr(self, "cb_L_chamber_unit", None),
                    getattr(self, "lbl_L_chamber", None)):
            if wdg is not None:
                wdg.setEnabled(use_len)
        # Характеристическая длина L*
        for wdg in (getattr(self, "sp_L_star", None),
                    getattr(self, "cb_L_star_unit", None),
                    getattr(self, "lbl_L_star", None)):
            if wdg is not None:
                wdg.setEnabled(not use_len)

    def _update_overall_efficiency(self, *args):
        """Пересчёт суммарного КПД η_общ = η_р · η_с (поле только для чтения)."""
        try:
            eta_r = float(self.sp_eff_reaction.value())
            eta_n = float(self.sp_eff_nozzle.value())
        except Exception:
            return
        if hasattr(self, "sp_eff_overall"):
            self.sp_eff_overall.setValue(eta_r * eta_n)

    def _chamber_length_m(self) -> float:
        """Длина цилиндрической части камеры в метрах.

        Если выбран ввод «Длина камеры» — берём её напрямую.
        Если выбран ввод «Характеристическая L*» — выводим длину камеры из
        характеристической длины:

            L* = V_к / A_кр   (определение характеристической длины),

        для цилиндрической камеры объёмом V_к = A_к · L_к:

            L_к = L* · A_кр / A_к = L* · (R_кр / R_к)² = L* / (R_к/R_кр)².

        Отношение R_к/R_кр берём из поля «R_камеры / R_кр» панели профиля
        (sp_calc_Rcham); при недоступности — нейтральное 1.0.
        """
        def _len_to_m(v, unit):
            if unit == 'см':
                return v * 0.01
            if unit == 'мм':
                return v * 0.001
            return v

        use_lstar = (getattr(self, "rb_chamber_lstar", None) is not None
                     and self.rb_chamber_lstar.isChecked())
        if not use_lstar:
            return _len_to_m(self.sp_L_chamber.value(),
                             self.cb_L_chamber_unit.currentText())

        L_star = _len_to_m(self.sp_L_star.value(),
                           self.cb_L_star_unit.currentText())
        rcham_ratio = 1.0
        sp = getattr(self, "sp_calc_Rcham", None)
        if sp is not None:
            try:
                rcham_ratio = float(sp.value())
            except Exception:
                rcham_ratio = 1.0
        rcham_ratio = max(rcham_ratio, 1.0)
        return L_star / (rcham_ratio * rcham_ratio)

    def _auto_conv_div_lengths(self) -> tuple:
        """Автоматически вычисляет длины конфузора и дивергента (в метрах).

        Раньше эти величины задавались вручную (поля «Конфузор»/«Дивергент»).
        Теперь они выводятся из геометрии сопла:

          • Конфузор (дозвук): конус от радиуса камеры R_к к радиусу горловины
            R_кр с полууглом θ_вх →  L_conv = (R_к − R_кр) / tg θ_вх.
          • Дивергент (сверхзвук): конус от R_кр к радиусу среза R_a с полууглом
            θ_a →  L_div = (R_a − R_кр) / tg θ_a, где R_a = R_кр·√(F_a/F_кр).

        Радиус горловины и отношение R_к/R_кр берутся из панели профиля
        (sp_calc_Rthroat, sp_calc_Rcham), полуугол входа — из sp_calc_theta_in,
        полуугол среза — из sp_calc_theta_exit (или 15° по умолчанию). Степень
        расширения F_a/F_кр — из результата расчёта (self.perf), иначе ~6.
        """
        # Радиус горловины (м)
        try:
            R_throat = self._length_to_m(
                self.sp_calc_Rthroat.value(),
                self.cb_calc_Rthroat_unit.currentText(),
            )
        except Exception:
            R_throat = 0.05
        R_throat = max(R_throat, 1e-4)

        # Отношение радиусов камеры и горловины
        try:
            rcham_ratio = max(float(self.sp_calc_Rcham.value()), 1.05)
        except Exception:
            rcham_ratio = 3.0
        R_cham = rcham_ratio * R_throat

        # Степень расширения F_a/F_кр (из расчёта, иначе оценка)
        area_ratio = 6.0
        perf = getattr(self, "perf", None)
        if perf is not None and getattr(perf, "stations", None):
            try:
                ar = float(perf.stations[-1].Ae_At)
                if math.isfinite(ar) and ar > 1.0:
                    area_ratio = ar
            except Exception:
                pass
        R_exit = R_throat * math.sqrt(area_ratio)

        # Полуугол входа (дозвуковой конус)
        try:
            theta_in = math.radians(max(5.0, float(self.sp_calc_theta_in.value())))
        except Exception:
            theta_in = math.radians(30.0)
        # Полуугол среза (сверхзвуковой конус)
        theta_exit = math.radians(15.0)
        sp_te = getattr(self, "sp_calc_theta_exit", None)
        if sp_te is not None:
            try:
                theta_exit = math.radians(max(3.0, float(sp_te.value())))
            except Exception:
                pass

        L_conv = (R_cham - R_throat) / math.tan(theta_in) if theta_in > 0 else 0.0
        L_div = (R_exit - R_throat) / math.tan(theta_exit) if theta_exit > 0 else 0.0
        return max(L_conv, 1e-4), max(L_div, 1e-4)

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
            # Сбрасываем кэш равновесных составов: термоданные могли измениться,
            # старые результаты больше не валидны.
            clear_equilibrium_cache()
            
            # Обновить mixture_widget с загруженной базой
            self.mixture_widget.species_db = self.species_db
            # Обновить species_db в обоих списках компонентов
            self.mixture_widget.oxidizer_list.species_db = self.species_db
            self.mixture_widget.fuel_list.species_db = self.species_db
            
            # Инициализировать стандартной смесью (внутренние доли 1.0/1.0).
            # По стандарту поле соотношения остаётся пустым — пользователь
            # задаёт Km/α сам либо выбирает режим «Оптимум».
            self.mixture_widget.set_mixture({
                'ox_components': [{'name': 'O2(L)', 'mass': 1.000, 'T': 0}],
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

    @staticmethod
    def _get_float_field(widget) -> Optional[float]:
        """Прочитать число из QLineEdit. Возвращает None, если пусто/некорректно/≤0."""
        try:
            txt = widget.text().strip().replace(',', '.')
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

    def _mix_mode(self) -> str:
        """Текущий режим задания соотношения: 'km' | 'alpha' | 'optimum'."""
        idx = self.cb_mix_mode.currentIndex()
        return {0: 'km', 1: 'alpha', 2: 'optimum'}.get(idx, 'km')

    def _get_mix_value(self) -> Optional[float]:
        """Числовое значение поля соотношения (Km или α), либо None."""
        return self._get_float_field(self.ed_mix_value)

    def _compute_km0(self) -> float:
        """Стехиометрическое соотношение Km0 для текущей смеси.

        Возвращает NaN, если база веществ не загружена либо компоненты не заданы.
        """
        if not self.species_db:
            return float('nan')
        mixture = self.mixture_widget.get_mixture()
        ox_names = [c['name'] for c in mixture.get('ox_components', []) if c.get('name')]
        fu_names = [c['name'] for c in mixture.get('fuel_components', []) if c.get('name')]
        if not ox_names or not fu_names:
            return float('nan')
        try:
            oxidizers = [self.species_db[n] for n in ox_names if n in self.species_db]
            fuels = [self.species_db[n] for n in fu_names if n in self.species_db]
            if not oxidizers or not fuels:
                return float('nan')
            return stoichiometric_OF(oxidizers, fuels)
        except Exception:
            return float('nan')

    def _on_mix_mode_changed(self):
        """Реакция на смену режима соотношения компонентов."""
        mode = self._mix_mode()
        if mode == 'optimum':
            self.ed_mix_value.setEnabled(False)
            self.ed_mix_value.setPlaceholderText("подбирается автоматически")
        else:
            self.ed_mix_value.setEnabled(True)
            self.ed_mix_value.setPlaceholderText(
                "Km (O/F)" if mode == 'km' else "α (Km/Km0)"
            )
        self._update_of_from_mixture()

    def _update_of_from_mixture(self):
        """Обновить информационную подпись (Km0 / результирующий Km / α)."""
        mode = self._mix_mode()
        km0 = self._compute_km0()
        km0_str = f"{km0:.4f}" if math.isfinite(km0) else "—"

        if mode == 'optimum':
            self.lbl_of.setText(f"Km0 = {km0_str}  (Km → max Isp)")
            return

        val = self._get_mix_value()
        if mode == 'km':
            if val is None:
                self.lbl_of.setText(f"Km0 = {km0_str}")
            elif math.isfinite(km0) and km0 > 0:
                self.lbl_of.setText(
                    f"Km = {val:.4f}, α = {val / km0:.4f}  (Km0 = {km0_str})"
                )
            else:
                self.lbl_of.setText(f"Km = {val:.4f}  (Km0 = {km0_str})")
        else:  # alpha
            if val is None:
                self.lbl_of.setText(f"Km0 = {km0_str}")
            elif math.isfinite(km0) and km0 > 0:
                self.lbl_of.setText(
                    f"α = {val:.4f}, Km = {val * km0:.4f}  (Km0 = {km0_str})"
                )
            else:
                self.lbl_of.setText(f"α = {val:.4f}  (Km0 = —, нужна база веществ)")

    def _resolve_of_ratio(self) -> Optional[float]:
        """Итоговое массовое O/F (Km) для расчёта.

        Возвращает None в режиме «Оптимум» либо если значение не задано.
        """
        mode = self._mix_mode()
        if mode == 'optimum':
            return None
        val = self._get_mix_value()
        if val is None:
            return None
        if mode == 'km':
            return val
        # alpha-режим: Km = α · Km0
        km0 = self._compute_km0()
        if not (math.isfinite(km0) and km0 > 0):
            return None
        return val * km0

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

        # ─── Валидация обязательных (пустых по умолчанию) полей ───
        mix_mode = self._mix_mode()
        optimize_of = (mix_mode == 'optimum')
        of_ratio = self._resolve_of_ratio()  # None в режиме «Оптимум» или если пусто

        P_chamber = self._get_float_field(self.ed_Pc)
        P_exit = self._get_float_field(self.ed_Pe)

        missing = []
        if P_chamber is None:
            missing.append("давление в камере (Pк)")
        if P_exit is None:
            missing.append("давление на срезе (Pс)")
        if not optimize_of and of_ratio is None:
            if mix_mode == 'km':
                missing.append("соотношение компонентов Km")
            else:  # alpha
                km0 = self._compute_km0()
                if not (math.isfinite(km0) and km0 > 0):
                    missing.append("α (требуется загруженная база для расчёта Km0)")
                else:
                    missing.append("соотношение компонентов α")

        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Заполните поля",
                "Не заданы обязательные параметры:\n  • " + "\n  • ".join(missing)
            )
            return

        params = {
            'ox_components': mixture['ox_components'],
            'fuel_components': mixture['fuel_components'],
            'of_ratio': of_ratio if of_ratio is not None else 1.0,
            'optimize_of': optimize_of,
            'of_stoich': self._compute_km0(),
            'P_chamber': pv_to_pa(P_chamber, self.cb_Pc_unit.currentText()),
            'P_exit': pv_to_pa(P_exit, self.cb_Pe_unit.currentText()),
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

        # ── Реализуемые характеристики с учётом потерь (КПД) ──
        # η_р действует на C* (камера), η_с — на C_F (сопло), η_общ — на Isp.
        eta_r = float(getattr(self, "sp_eff_reaction", None).value()
                      if getattr(self, "sp_eff_reaction", None) is not None else 1.0)
        eta_n = float(getattr(self, "sp_eff_nozzle", None).value()
                      if getattr(self, "sp_eff_nozzle", None) is not None else 1.0)
        eta_o = eta_r * eta_n
        if eta_o < 1.0 - 1e-9:
            s.append("─" * 70)
            s.append("  Реализуемые характеристики с учётом потерь:")
            s.append(f"  (η_реакц={eta_r:.4f}, η_сопла={eta_n:.4f}, η_общ={eta_o:.4f})")
            s.append("─" * 70)
            s.append(f"  Isp_дел (срез,  P_amb=0):  {perf.Isp_s*eta_o:8.4f} с")
            s.append(f"  Isp_дел (вакуум):          {perf.Isp_vac_s*eta_o:8.4f} с")
            s.append(f"  C*_дел:                    {perf.Cstar_m_per_s*eta_r:8.4f} м/с")
            s.append(f"  CF_дел:                    {perf.CF*eta_n:8.4f}")
            s.append(f"  Ve_дел:                    {perf.stations[-1].V_m_per_s*eta_o:8.4f} м/с")
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
        if hasattr(self, "chk_smooth"):
            s.smooth = self.chk_smooth.isChecked()
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
            L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1],
        )

    def _snapshot_curves(self, perf):
        """Снимок кривых 1D-расчёта (x + P, T, V, M, ρ, γₛ) для наложения.

        Берёт значения из единого очищенного источника ``_section_series``,
        чтобы наложения были согласованы с основными графиками и без «иголок».
        """
        ser = self._section_series(perf)
        if ser:
            return {
                "x": ser["x_m"],
                "P": ser["P_Pa"] / 1e6,
                "T": ser["T_K"],
                "V": ser["V"],
                "M": ser["M"],
                "rho": ser["rho"],
                "gs": ser["gamma_s"],
                "a": ser.get("a"),
                "S": ser.get("S"),
                "H": (ser.get("H") / 1e6 if ser.get("H") is not None else None),
                "q_dyn": (ser.get("q_dyn") / 1e6
                          if ser.get("q_dyn") is not None else None),
                "tau": ser.get("tau"),
                "pi": ser.get("pi"),
                "eps": ser.get("eps"),
                "lam": ser.get("lam"),
                "q_gd": ser.get("q_gd"),
                "y_gd": ser.get("y_gd"),
            }
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

    # ─────────────────────────────────────────────────────────────────────────
    #  ЕДИНЫЙ ИСТОЧНИК ДАННЫХ ПО СЕЧЕНИЯМ.
    #  Все параметры газа по сечениям считаются/чистятся здесь (в одном месте)
    #  и отдаются как словарь массивов. Любые графики (вкладка «Графики (1D)»,
    #  газодинамические функции) и наложения берут данные именно отсюда.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _hampel_filter(y, window: int = 2, n_sigma: float = 3.0):
        """Удаляет одиночные выбросы (медианный фильтр Хампеля).

        Точка считается выбросом, если отклоняется от скользящей медианы более
        чем на n_sigma·(1.4826·MAD). Выбросы заменяются медианой окна. Сохраняет
        физический тренд, убирая «иголки» от несошедшегося SP-решателя.
        """
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

    def _section_series(self, perf=None) -> dict:
        """Единый расчёт параметров газа по сечениям сопла (с очисткой шума).

        Возвращает словарь numpy-массивов, отсортированных по координате x (м),
        с убранными дубликатами по x и подавленными одиночными выбросами в
        «шумных» величинах (γₛ, скорость звука, число Маха) — это убирает
        «иголки», возникающие из-за неполной сходимости SP-задачи в отдельных
        сечениях.

        Ключи: x_m, P_Pa, T_K, rho, V, a, M, gamma_s, Ae_At, label,
        i_throat (индекс горловины), x_throat_m.
        """
        if perf is None:
            perf = getattr(self, "perf", None)
        if perf is None or not getattr(perf, "stations", None):
            return {}

        stations = list(perf.stations)
        L_chamber = self._chamber_length_m()
        L_conv, L_div = self._auto_conv_div_lengths()
        x = np.asarray(build_nozzle_geometry(
            stations,
            L_chamber=L_chamber,
            L_conv=L_conv,
            L_div=L_div,
        ), dtype=float)

        P = np.array([float(s.P_Pa) for s in stations])
        T = np.array([float(s.T_K) for s in stations])
        rho = np.array([float(getattr(s, "rho_kg_per_m3", 0.0)) for s in stations])
        V = np.array([float(getattr(s, "V_m_per_s", 0.0)) for s in stations])
        a = np.array([float(getattr(s, "a_m_per_s", 0.0)) for s in stations])
        gs = np.array([float(getattr(s, "gamma_s", 0.0)) for s in stations])
        Ae = np.array([float(getattr(s, "Ae_At", float('inf'))) for s in stations])
        labels = [getattr(s, "label", "") for s in stations]

        # ── Доп. расчётные сечения В КАМЕРЕ (между Injector и Nozzle inlet) ──
        # Injector и Nozzle inlet термодинамически совпадают (застойное
        # состояние камеры: одинаковые P₀, T₀, ρ₀, V≈0). Дополнительные сечения
        # в камере распределяют длину камеры по оси x, делая участок камеры
        # видимым на графиках. Их состояние копируется с инжектора (камеры).
        n_cham = int(getattr(self, "_n_chamber_sections", 0) or 0)
        if n_cham > 0:
            i_inj = None
            for i, lab in enumerate(labels):
                if str(lab).strip().lower() == "injector":
                    i_inj = i
                    break
            if i_inj is None:
                i_inj = int(np.argmin(x)) if x.size else 0
            xs_extra = np.linspace(0.0, L_chamber, n_cham + 2)[1:-1]
            P0 = P[i_inj]; T0 = T[i_inj]; rho0 = rho[i_inj]
            V0 = V[i_inj]; a0 = a[i_inj]; gs0 = gs[i_inj]; Ae0 = Ae[i_inj]
            for k, xc in enumerate(xs_extra, start=1):
                x = np.append(x, float(xc))
                P = np.append(P, P0); T = np.append(T, T0)
                rho = np.append(rho, rho0); V = np.append(V, V0)
                a = np.append(a, a0); gs = np.append(gs, gs0)
                Ae = np.append(Ae, Ae0)
                labels.append(f"Chamber {k}")

        # 1) сортировка по x и удаление дубликатов координаты
        order = np.argsort(x, kind="stable")
        x = x[order]; P = P[order]; T = T[order]; rho = rho[order]
        V = V[order]; a = a[order]; gs = gs[order]; Ae = Ae[order]
        labels = [labels[i] for i in order]
        _, iu = np.unique(np.round(x, 9), return_index=True)
        iu = np.sort(iu)
        x = x[iu]; P = P[iu]; T = T[iu]; rho = rho[iu]
        V = V[iu]; a = a[iu]; gs = gs[iu]; Ae = Ae[iu]
        labels = [labels[i] for i in iu]

        # 2) подавление одиночных выбросов в «шумных» величинах решателя.
        #    γₛ и a физически гладкие — «иголки» это численный шум SP-задачи.
        #    Для γₛ берём чуть шире окно и строже порог: показатель меняется
        #    плавно, поэтому даже умеренные «дрожания» — это шум, а не физика.
        gs = self._hampel_filter(gs, window=3, n_sigma=2.0)
        gs = self._hampel_filter(gs, window=2, n_sigma=2.0)
        a = self._hampel_filter(a, window=2, n_sigma=2.5)
        # число Маха пересчитываем из (уже очищенной) скорости звука —
        # это убирает «иглы» по M, согласовав его с V и a.
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(a > 0, V / a, 0.0)
        M = self._hampel_filter(M, window=2, n_sigma=3.0)

        # горловина — ближайшее к M=1 сечение (минимум |M-1|)
        try:
            i_throat = int(np.nanargmin(np.abs(M - 1.0)))
        except Exception:
            i_throat = 0

        # относительный радиус контура r/r_throat = sqrt(Ae/At) — для силуэта
        # профиля сопла на графиках. В камере (Ae→∞ из инжектора) ограничиваем
        # величину радиусом камеры (макс. конечное значение профиля).
        with np.errstate(invalid='ignore'):
            r_rel = np.sqrt(np.clip(Ae, 0.0, None))
        finite = r_rel[np.isfinite(r_rel)]
        r_cap = float(np.nanmax(finite)) if finite.size else 1.0
        r_rel = np.where(np.isfinite(r_rel), r_rel, r_cap)

        # ── Дополнительные термодинамические величины из исходных сечений ──────
        # Энтропия S и энтальпия H берутся напрямую из StationResult (если есть),
        # с той же сортировкой/дедупликацией, что и остальные массивы.
        S = np.array([float(getattr(s, "S_J_per_kgK", float('nan'))) for s in stations])
        H = np.array([float(getattr(s, "H_J_per_kg", float('nan'))) for s in stations])
        if n_cham > 0:
            S0 = S[i_inj] if S.size else float('nan')
            H0 = H[i_inj] if H.size else float('nan')
            for _ in range(int(n_cham)):
                S = np.append(S, S0)
                H = np.append(H, H0)
        # применяем тот же порядок сортировки/дедупликации (order → iu)
        if S.size == order.size:
            S = S[order][iu]
        else:
            S = np.full_like(x, float('nan'))
        if H.size == order.size:
            H = H[order][iu]
        else:
            H = np.full_like(x, float('nan'))

        # ── Газодинамические функции (изэнтропические соотношения) ────────────
        # Параметры торможения (камера = сечение инжектора / минимум x).
        try:
            i0 = int(np.argmin(x)) if x.size else 0
        except Exception:
            i0 = 0
        T0 = float(T[i0]) if T.size and T[i0] > 0 else (
            float(np.nanmax(T)) if T.size else 1.0)
        P0 = float(P[i0]) if P.size and P[i0] > 0 else (
            float(np.nanmax(P)) if P.size else 1.0)
        rho0 = float(rho[i0]) if rho.size and rho[i0] > 0 else (
            float(np.nanmax(rho)) if rho.size else 1.0)
        # τ = T/T₀, π = P/P₀, ε = ρ/ρ₀ — прямо из величин по сечениям.
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = T / T0 if T0 else np.zeros_like(T)
            pi = P / P0 if P0 else np.zeros_like(P)
            eps = rho / rho0 if rho0 else np.zeros_like(rho)
        # λ — скоростной коэффициент: λ² = ((k+1)/2·M²) / (1 + (k-1)/2·M²).
        k = np.where(gs > 1.0, gs, 1.2)
        with np.errstate(divide='ignore', invalid='ignore'):
            lam2 = ((k + 1.0) / 2.0 * M * M) / (1.0 + (k - 1.0) / 2.0 * M * M)
        lam = np.sqrt(np.clip(lam2, 0.0, None))
        # q(λ) — приведённый расход; y(λ) — функция удельного импульса.
        with np.errstate(divide='ignore', invalid='ignore'):
            base = np.clip(1.0 - (k - 1.0) / (k + 1.0) * lam * lam, 0.0, None)
            q_gd = (lam * ((k + 1.0) / 2.0) ** (1.0 / (k - 1.0))
                    * base ** (1.0 / (k - 1.0)))
            y_gd = np.where(lam > 1e-9,
                            (1.0 + lam * lam) / (2.0 * lam) * q_gd, 0.0)
        # Динамическое давление q_dyn = ½·ρ·V².
        q_dyn = 0.5 * rho * V * V

        return {
            "x_m": x, "P_Pa": P, "T_K": T, "rho": rho,
            "V": V, "a": a, "M": M, "gamma_s": gs, "Ae_At": Ae,
            "r_rel": r_rel,
            "S": S, "H": H,
            "tau": tau, "pi": pi, "eps": eps,
            "lam": lam, "q_gd": q_gd, "y_gd": y_gd,
            "q_dyn": q_dyn,
            "label": labels, "i_throat": i_throat,
            "x_throat_m": float(x[i_throat]) if x.size else 0.0,
        }

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

        # ── Единый источник данных по сечениям (очищенный, отсортированный) ──
        ser = self._section_series(self.perf)
        if not ser:
            return

        # Если активен НЕ режим «Графики параметров (1D)», перенаправляем на
        # отрисовку поля/газодинамических функций и выходим.
        if hasattr(self, "cb_plot_view") and self.cb_plot_view.currentIndex() != 0:
            self._render_field_2d()
            return

        # Холст «Графики параметров (1D)»: только ВЫБРАННЫЕ пользователем величины.
        self._draw_selected_1d(ser, style)

        # Синхронизируем геометрию (для вкладок «Геометрия»/«Поле 2D»),
        # если включён расчёт профиля по Добровольскому.
        try:
            if (getattr(self, "chk_use_dobro", None) is not None
                    and self.chk_use_dobro.isChecked()):
                geom = self._build_calc_geometry(self.perf)
                if geom is not None:
                    self.last_geometry = geom
                    self._render_geometry(geom)
                    self._update_geometry_summary(geom)
        except Exception:
            pass

    # ── Каталог величин для графиков 1D (значения берём из _section_series) ──
    def _plot_param_value(self, key: str, ser: dict):
        """Возвращает массив значений величины ``key`` из единого источника."""
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
        # ── термодинамические величины ──
        if key == "S":
            return ser.get("S")
        if key == "H":
            v = ser.get("H")
            return v / 1e6 if v is not None else None   # Дж/кг → МДж/кг
        if key == "q_dyn":
            v = ser.get("q_dyn")
            return v / 1e6 if v is not None else None   # Па → МПа
        # ── газодинамические функции ──
        if key == "tau":
            return ser.get("tau")
        if key == "pi":
            return ser.get("pi")
        if key == "eps":
            return ser.get("eps")
        if key == "lam":
            return ser.get("lam")
        if key == "q_gd":
            return ser.get("q_gd")
        if key == "y_gd":
            return ser.get("y_gd")
        return None

    def _selected_plot_keys(self):
        """Ключи величин, отмеченных пользователем (в порядке каталога)."""
        cb = getattr(self, "cb_plot_params", None)
        if cb is None:
            return list(self._plot_default_keys)
        checked = set(cb.checked_keys())
        return [k for (k, *_rest) in self.PLOT_PARAM_DEFS if k in checked]

    def _draw_selected_1d(self, ser, style):
        """Рисует выбранные величины на ``canvas_1d``.

        Диспетчер: если активен интерактивный Plotly-холст — строит plotly-
        фигуру, иначе откатывается на matplotlib. Данные в обоих случаях берутся
        из единого ``_section_series`` (расчёт в одном месте), графики лишь
        отображают их.
        """
        if getattr(self, "use_plotly_1d", False) and isinstance(
                getattr(self, "canvas_1d", None), PlotlyCanvas):
            self._draw_selected_1d_plotly(ser, style)
        else:
            self._draw_selected_1d_mpl(ser, style)

    # ── Построение plotly-фигуры из единого источника данных ─────────────────
    def _build_plotly_1d_figure(self, ser, style, keys):
        """Собирает ``go.Figure`` с подграфиками выбранных величин ``keys``.

        Используется и для отображения на холсте, и для сохранения в файлы —
        чтобы экранный и сохранённый вид были идентичны.
        """
        x = ser["x_m"]
        x_thr = ser.get("x_throat_m", None)
        defs = {k: (lbl, unit, color) for (k, lbl, unit, color, _log)
                in self.PLOT_PARAM_DEFS}

        dark = style.dark_plot
        paper_bg = '#1c1917' if dark else '#ffffff'
        plot_bg = '#262624' if dark else '#ffffff'
        fg = '#fafaf9' if dark else '#000000'
        grid_color = 'rgba(150,150,147,0.30)' if dark else 'rgba(120,120,120,0.30)'

        n = len(keys)
        ncols = 1 if n == 1 else 2
        nrows = (n + ncols - 1) // ncols
        titles = [defs[k][0] + (f", {defs[k][1]}" if defs[k][1] else "")
                  for k in keys]
        fig = make_subplots(
            rows=max(nrows, 1), cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.12 if nrows > 1 else 0.0,
            horizontal_spacing=0.09 if ncols > 1 else 0.0,
        )

        mode = 'lines+markers' if style.show_markers else 'lines'
        ms = max(3, int(style.marker_size))
        lw = max(1.0, float(style.line_width))
        # сглаживание кривых — сплайн-форма линии Plotly (по требованию).
        line_shape = 'spline' if getattr(style, "smooth", False) else 'linear'
        for i, key in enumerate(keys):
            row = i // ncols + 1
            col = i % ncols + 1
            label, unit, color = defs[key]
            y = self._plot_param_value(key, ser)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode=mode, name=label,
                    line=dict(color=color, width=lw, shape=line_shape),
                    marker=dict(size=ms, color=color),
                    showlegend=False,
                    hovertemplate=(f"{label}<br>x=%{{x:.4g}} м<br>"
                                   f"%{{y:.4g}} {unit}<extra></extra>"),
                ),
                row=row, col=col,
            )
            # наложения для сравнения вариантов (если ключ совпадает)
            self._add_plotly_overlays(fig, key, row, col)
            # линия M = 1 (звуковая)
            if key == "M":
                fig.add_hline(row=row, col=col, y=1.0,
                              line=dict(color='#a8a29e', width=1, dash='dot'))
            # вертикаль горловины
            if x_thr is not None:
                fig.add_vline(row=row, col=col, x=x_thr,
                              line=dict(color='#888', width=1, dash='dot'))
            fig.update_xaxes(title_text="x, м", row=row, col=col,
                             gridcolor=grid_color, zeroline=False,
                             color=fg, linecolor=fg, mirror=True,
                             ticks='inside', showline=True)
            fig.update_yaxes(title_text=(unit if unit else label),
                             row=row, col=col, gridcolor=grid_color,
                             zeroline=False, color=fg, linecolor=fg,
                             mirror=True, ticks='inside', showline=True)

        fig.update_layout(
            paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
            font=dict(family=style.font_family, size=style.font_size_tick,
                      color=fg),
            margin=dict(l=60, r=20, t=40, b=50),
            hovermode='closest',
            showlegend=False,
            # перемещение по графику при зажатой ЛКМ (req: «перемещаться,
            # зажимая левую кнопку мыши»); колёсико — приближение (scrollZoom).
            dragmode='pan',
        )
        # цвет/размер заголовков подграфиков
        for ann in fig.layout.annotations:
            ann.font.update(color=fg, size=style.font_size_axis)

        # Силуэт профиля сопла (по требованию — включаемый фон r(x)).
        if getattr(self, "_show_profile_1d", False) and "r_rel" in ser:
            self._add_plotly_profile(fig, ser, keys, ncols)
        return fig

    def _add_plotly_profile(self, fig, ser, keys, ncols):
        """Рисует силуэт контура сопла r(x) фоном в нижней части подграфиков.

        Силуэт (относительный радиус r/r_throat из ``ser['r_rel']``) рисуется
        прямо на осях каждого подграфика (row/col), масштабированный в нижние
        ~28 % диапазона значений параметра — как ориентир формы сопла под
        кривыми. Заливается полупрозрачным цветом. Не создаёт дополнительных
        осей (надёжно для произвольного числа подграфиков).
        """
        import numpy as _np
        x = _np.asarray(ser["x_m"], dtype=float)
        r = _np.asarray(ser["r_rel"], dtype=float)
        if x.size == 0 or r.size == 0 or not _np.any(_np.isfinite(r)):
            return
        rmax = float(_np.nanmax(r)) or 1.0
        r_unit = r / rmax  # в [0..1]
        fill_color = 'rgba(106,176,255,0.12)'
        line_color = 'rgba(106,176,255,0.45)'
        for i, key in enumerate(keys):
            row = i // ncols + 1
            col = i % ncols + 1
            y = self._plot_param_value(key, ser)
            if y is None:
                continue
            y = _np.asarray(y, dtype=float)
            finite = y[_np.isfinite(y)]
            if finite.size == 0:
                continue
            ymin = float(_np.nanmin(finite))
            ymax = float(_np.nanmax(finite))
            span = (ymax - ymin) or (abs(ymax) or 1.0)
            base = ymin
            top = ymin + 0.28 * span * r_unit  # силуэт в нижней полосе
            # базовая линия (низ) — для заливки tonexty
            fig.add_trace(
                go.Scatter(
                    x=x, y=_np.full_like(x, base), mode='lines',
                    line=dict(width=0, color=fill_color),
                    hoverinfo='skip', showlegend=False,
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=top, mode='lines',
                    line=dict(width=1.2, color=line_color),
                    fill='tonexty', fillcolor=fill_color,
                    name='Профиль сопла', hoverinfo='skip', showlegend=False,
                ),
                row=row, col=col,
            )
        return fig

    def _add_plotly_overlays(self, fig, key, row, col):
        """Добавляет наложенные кривые сравнения (если включены) в plotly."""
        if (not getattr(self, "chk_overlay_show", None)
                or not self.chk_overlay_show.isChecked()):
            return
        for ov in getattr(self, "_overlays", []):
            y = ov.get(key)
            if y is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ov["x"], y=y, mode='lines', name=ov.get("label", ""),
                    line=dict(color=ov["color"], width=1.4, dash='solid'),
                    opacity=0.55, showlegend=False,
                    hovertemplate=(f"{ov.get('label','')}<br>"
                                   f"x=%{{x:.4g}}<br>%{{y:.4g}}<extra></extra>"),
                ),
                row=row, col=col,
            )

    def _draw_selected_1d_plotly(self, ser, style):
        """Строит интерактивный Plotly-холст из выбранных величин."""
        c = getattr(self, "canvas_1d", None)
        if c is None:
            return
        keys = self._selected_plot_keys()
        if not keys:
            c.show_message("Выберите параметры для отображения",
                           dark=style.dark_plot)
            return
        fig = self._build_plotly_1d_figure(ser, style, keys)
        c.set_figure(fig)

    @staticmethod
    def _smooth_xy(x, y, n_out: int = 300):
        """Возвращает сглаженную кривую (x, y) через PCHIP-интерполяцию.

        Используется для matplotlib-фолбэка при включённом сглаживании.
        Если scipy недоступен или точек слишком мало — возвращает исходные.
        """
        try:
            xa = np.asarray(x, dtype=float)
            ya = np.asarray(y, dtype=float)
            m = np.isfinite(xa) & np.isfinite(ya)
            xa, ya = xa[m], ya[m]
            if xa.size < 3:
                return x, y
            xu, iu = np.unique(xa, return_index=True)
            yu = ya[iu]
            if xu.size < 3:
                return x, y
            xs = np.linspace(float(xu[0]), float(xu[-1]), int(n_out))
            try:
                from scipy.interpolate import PchipInterpolator  # type: ignore
                ys = PchipInterpolator(xu, yu)(xs)
            except Exception:
                ys = np.interp(xs, xu, yu)
            return xs, ys
        except Exception:
            return x, y

    def _draw_selected_1d_mpl(self, ser, style):
        """Резервная отрисовка на matplotlib (если Plotly недоступен)."""
        c = getattr(self, "canvas_1d", None)
        if c is None:
            return
        c.fig.clear()
        x = ser["x_m"]

        keys = self._selected_plot_keys()
        defs = {k: (lbl, unit, color) for (k, lbl, unit, color, _log)
                in self.PLOT_PARAM_DEFS}

        self._apply_fig_facecolor(c.fig, style)
        if not keys:
            ax = c.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Выберите параметры для отображения",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#a8a29e')
            ax.set_axis_off()
            c.draw()
            return

        n = len(keys)
        ncols = 1 if n == 1 else 2
        nrows = (n + ncols - 1) // ncols
        x_thr = ser.get("x_throat_m", None)

        marker = 'o' if style.show_markers else None
        smooth = getattr(style, "smooth", False)
        for i, key in enumerate(keys):
            label, unit, color = defs[key]
            y = self._plot_param_value(key, ser)
            ax = c.fig.add_subplot(nrows, ncols, i + 1)
            # маркеры — на исходных точках; линия — сглаженная (если включено).
            xl, yl = x, y
            if smooth and y is not None:
                xl, yl = self._smooth_xy(x, y)
            ax.plot(xl, yl, '-', color=color, lw=style.line_width, zorder=3)
            if marker is not None:
                ax.plot(x, y, linestyle='none', marker=marker, color=color,
                        ms=style.marker_size, zorder=4)
            # наложения для сравнения вариантов (если ключ совпадает)
            self._draw_overlays(ax, key)
            if key == "M":
                ax.axhline(1.0, color='#a8a29e', lw=0.8, ls=':')
            if x_thr is not None:
                ax.axvline(x_thr, color='#888', ls=':', lw=0.8, alpha=0.7)
            ttl = label + (f", {unit}" if unit else "")
            self._style_subplot(ax, style, title=ttl,
                                 xlabel="Координата x, м",
                                 ylabel=(unit if unit else label))
        try:
            c.fig.tight_layout(pad=1.2)
        except Exception:
            pass
        c.draw()

    def _apply_fig_facecolor(self, fig, style):
        bg = '#1c1917' if style.dark_plot else '#ffffff'
        fig.patch.set_facecolor(bg)

    def _style_subplot(self, ax, style, *, title="", xlabel="", ylabel=""):
        """Лёгкая стилизация одиночного подграфика (тёмный/светлый фон)."""
        fg = '#fafaf9' if style.dark_plot else '#000000'
        bg = '#262624' if style.dark_plot else '#ffffff'
        ax.set_facecolor(bg)
        if title:
            ax.set_title(title, color=fg, fontsize=style.font_size_axis,
                         fontfamily=style.font_family)
        if xlabel:
            ax.set_xlabel(xlabel, color=fg, fontsize=style.font_size_axis,
                          fontfamily=style.font_family)
        if ylabel:
            ax.set_ylabel(ylabel, color=fg, fontsize=style.font_size_axis,
                          fontfamily=style.font_family)
        ax.tick_params(axis='both', which='major',
                       labelsize=style.font_size_tick,
                       direction=style.tick_direction,
                       length=5, width=1.0, color=fg, labelcolor=fg)
        ax.tick_params(axis='both', which='minor',
                       direction=style.tick_direction,
                       length=3, width=0.7, color=fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
            spine.set_linewidth(style.spine_linewidth)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if style.grid_major:
            ax.grid(True, which='major', alpha=0.3, color=fg)
        if style.grid_minor:
            ax.grid(True, which='minor', alpha=0.15, color=fg)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(bg)
            leg.get_frame().set_edgecolor(fg)
            for txt in leg.get_texts():
                txt.set_color(fg)

    def _save_selected_1d_plots(self):
        """Сохраняет каждый выбранный график 1D в отдельный файл.

        Если активен Plotly — сохраняет интерактивный HTML (всегда) и PNG
        (если установлен kaleido). Иначе — PNG через matplotlib.
        """
        if self.perf is None:
            QtWidgets.QMessageBox.information(
                self, "Нет данных", "Сначала выполните расчёт сопла.")
            return
        keys = self._selected_plot_keys()
        if not keys:
            QtWidgets.QMessageBox.information(
                self, "Нет выбора", "Отметьте хотя бы один параметр.")
            return
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Папка для сохранения графиков")
        if not dir_path:
            return
        style = self._collect_style()
        ser = self._section_series(self.perf)
        if not ser:
            return

        if getattr(self, "use_plotly_1d", False):
            self._save_selected_1d_plotly(ser, style, keys, dir_path)
        else:
            self._save_selected_1d_mpl(ser, style, keys, dir_path)

    def _save_selected_1d_plotly(self, ser, style, keys, dir_path):
        """Сохраняет каждый график как интерактивный HTML (+ PNG, если можно)."""
        defs = {k: (lbl, unit, color) for (k, lbl, unit, color, _log)
                in self.PLOT_PARAM_DEFS}
        saved_html = 0
        saved_png = 0
        png_error = None
        for key in keys:
            fig = self._build_plotly_1d_figure(ser, style, [key])
            # уберём общий заголовок-аннотацию подграфика: для одиночного
            # файла достаточно заголовка оси/легенды (он уже задаёт смысл).
            html_path = os.path.join(dir_path, f"nozzle_1d_{key}.html")
            try:
                pio.write_html(fig, html_path, full_html=True,
                               include_plotlyjs='inline',
                               config={'displaylogo': False})
                saved_html += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка сохранения", f"{html_path}\n{e}")
                return
            # PNG — опционально (нужен kaleido)
            png_path = os.path.join(dir_path, f"nozzle_1d_{key}.png")
            try:
                fig.write_image(png_path, width=900, height=560, scale=2)
                saved_png += 1
            except Exception as e:  # kaleido отсутствует/ошибка — не критично
                png_error = str(e)
        msg = f"Сохранено HTML: {saved_html}"
        if saved_png:
            msg += f", PNG: {saved_png}"
        elif png_error:
            msg += " (PNG пропущен: установите kaleido для экспорта в PNG)"
        msg += f" → {dir_path}"
        self.statusBar().showMessage(msg)

    def _save_selected_1d_mpl(self, ser, style, keys, dir_path):
        """Резервное сохранение в PNG через matplotlib."""
        x = ser["x_m"]
        x_thr = ser.get("x_throat_m", None)
        defs = {k: (lbl, unit, color) for (k, lbl, unit, color, _log)
                in self.PLOT_PARAM_DEFS}
        from matplotlib.figure import Figure as _Fig
        saved = 0
        marker = 'o' if style.show_markers else None
        for key in keys:
            label, unit, color = defs[key]
            y = self._plot_param_value(key, ser)
            fig = _Fig(figsize=(7, 4.5), dpi=200)
            ax = fig.add_subplot(111)
            ax.plot(x, y, '-', marker=marker, color=color,
                    lw=style.line_width, ms=style.marker_size)
            if key == "M":
                ax.axhline(1.0, color='#a8a29e', lw=0.8, ls=':')
            if x_thr is not None:
                ax.axvline(x_thr, color='#888', ls=':', lw=0.8, alpha=0.7)
            ttl = label + (f", {unit}" if unit else "")
            self._style_subplot(ax, style, title=ttl,
                                 xlabel="Координата x, м",
                                 ylabel=(unit if unit else label))
            self._apply_fig_facecolor(fig, style)
            fig.tight_layout()
            path = os.path.join(dir_path, f"nozzle_1d_{key}.png")
            try:
                fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(),
                            bbox_inches='tight')
                saved += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка сохранения", f"{path}\n{e}")
                return
        self.statusBar().showMessage(
            f"Сохранено графиков: {saved} → {dir_path}")

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
            L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1],
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
        # Графики 1D: если активен Plotly-холст — сохраняем его последнюю
        # фигуру (PNG через kaleido / HTML), иначе matplotlib-холст.
        if getattr(self, "use_plotly_1d", False) and isinstance(
                getattr(self, "canvas_1d", None), PlotlyCanvas):
            fig = getattr(self.canvas_1d, "figure", None)
            if fig is not None:
                p_png = os.path.join(dir_path, "nozzle_1d.png")
                p_html = os.path.join(dir_path, "nozzle_1d.html")
                try:
                    pio.write_html(fig, p_html, full_html=True,
                                   include_plotlyjs='inline')
                except Exception:
                    pass
                try:
                    fig.write_image(p_png, width=1100, height=700, scale=2)
                except Exception:
                    pass  # нет kaleido — HTML всё равно сохранён
        else:
            path = os.path.join(dir_path, "nozzle_1d.png")
            try:
                self.canvas_1d.fig.savefig(
                    path, dpi=200,
                    facecolor=self.canvas_1d.fig.get_facecolor(),
                    bbox_inches='tight')
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка сохранения", f"{path}\n{e}")
                return

        # Состав продуктов сгорания — всегда matplotlib-холст.
        path = os.path.join(dir_path, "nozzle_species.png")
        try:
            self.canvas_species.fig.savefig(
                path, dpi=200,
                facecolor=self.canvas_species.fig.get_facecolor(),
                bbox_inches='tight')
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Ошибка сохранения", f"{path}\n{e}")
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
            L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1],
        )

        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f, delimiter=';')
                # Заголовок
                w.writerow(['# RPA-Style Rocket Nozzle Calculator — Export'])
                ox_desc, fu_desc = self._get_mixture_summary()
                w.writerow([f'# Окислитель: {ox_desc}'])
                w.writerow([f'# Горючее: {fu_desc}'])
                w.writerow([f'# Pc = {self.perf.stations[0].P_Pa / 1e6:.4f} МПа, '
                            f'Pe = {self.perf.stations[-1].P_Pa / 1e6:.4f} МПа, '
                            f'Km = {self.perf.O_F:.4f}'])
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
            L_chamber=self._chamber_length_m(),
            L_conv=self._auto_conv_div_lengths()[0],
            L_div=self._auto_conv_div_lengths()[1],
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
                        f"Pc={self.perf.stations[0].P_Pa / 1e6:.4f} MPa, "
                        f"Pe={self.perf.stations[-1].P_Pa / 1e6:.4f} MPa\n")
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
            'mix_mode': self._mix_mode(),
            'mix_value': self.ed_mix_value.text().strip(),
            'Pc_field': self.ed_Pc.text().strip(),
            'Pe_field': self.ed_Pe.text().strip(),
            'Pc_unit': self.cb_Pc_unit.currentText(),
            'Pe_unit': self.cb_Pe_unit.currentText(),
            'n_inter': self.sp_n_inter.value(),
            'n_chamber': self.sp_n_chamber.value(),
            'density_sub': self.sp_density_sub.value(),
            'density_crit': self.sp_density_crit.value(),
            'density_sup': self.sp_density_sup.value(),
            'include_condensed': self.chk_condensed.isChecked(),
            'solver': 'cea' if self.rb_cea.isChecked() else 'own',
            'L_chamber': self.sp_L_chamber.value(),
            'L_conv': self.sp_L_conv.value(),
            'L_div': self.sp_L_div.value(),
            'chamber_size_mode': ('lstar' if self.rb_chamber_lstar.isChecked()
                                  else 'length'),
            'L_star': self.sp_L_star.value(),
            'L_star_unit': self.cb_L_star_unit.currentText(),
            'losses': {
                'reaction_eff': self.sp_eff_reaction.value(),
                'nozzle_eff': self.sp_eff_nozzle.value(),
            },
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
            # ─── Соотношение компонентов (режим + значение) ───
            mode_map = {'km': 0, 'alpha': 1, 'optimum': 2}
            self.cb_mix_mode.setCurrentIndex(
                mode_map.get(cfg.get('mix_mode', 'km'), 0)
            )
            # Обратная совместимость со старым форматом ('of_ratio').
            if 'mix_value' in cfg:
                self.ed_mix_value.setText(str(cfg.get('mix_value', '')))
            elif 'of_ratio' in cfg:
                self.cb_mix_mode.setCurrentIndex(0)  # старый формат = Km
                self.ed_mix_value.setText(f"{cfg.get('of_ratio'):.4f}")
            else:
                self.ed_mix_value.clear()

            # ─── Давления (поле + единица) ───
            if 'Pc_field' in cfg:
                self.ed_Pc.setText(str(cfg.get('Pc_field', '')))
                self.cb_Pc_unit.setCurrentText(cfg.get('Pc_unit', 'МПа'))
            elif 'Pc_MPa' in cfg:
                self.ed_Pc.setText(f"{cfg.get('Pc_MPa'):.6f}")
                self.cb_Pc_unit.setCurrentText('МПа')
            else:
                self.ed_Pc.clear()

            if 'Pe_field' in cfg:
                self.ed_Pe.setText(str(cfg.get('Pe_field', '')))
                self.cb_Pe_unit.setCurrentText(cfg.get('Pe_unit', 'МПа'))
            elif 'Pe_MPa' in cfg:
                self.ed_Pe.setText(f"{cfg.get('Pe_MPa'):.6f}")
                self.cb_Pe_unit.setCurrentText('МПа')
            else:
                self.ed_Pe.clear()

            self._on_mix_mode_changed()
            self.sp_n_inter.setValue(cfg.get('n_inter', 8))
            self.sp_n_chamber.setValue(cfg.get('n_chamber', 4))
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
            # Характеристическая длина L* и режим задания размера камеры
            self.sp_L_star.setValue(float(cfg.get('L_star', 1.0)))
            self.cb_L_star_unit.setCurrentText(cfg.get('L_star_unit', 'м'))
            if cfg.get('chamber_size_mode') == 'lstar':
                self.rb_chamber_lstar.setChecked(True)
            else:
                self.rb_chamber_len.setChecked(True)
            self._on_chamber_size_mode_changed()
            # Потери (КПД)
            losses = cfg.get('losses')
            if isinstance(losses, dict):
                self.sp_eff_reaction.setValue(float(losses.get('reaction_eff', 1.0)))
                self.sp_eff_nozzle.setValue(float(losses.get('nozzle_eff', 1.0)))
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
