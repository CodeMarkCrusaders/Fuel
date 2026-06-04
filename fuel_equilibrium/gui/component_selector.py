"""
Диалоговое окно для выбора компонентов топлива (окислитель/горючее), как в RPA.
"""

from typing import Dict, List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from ..core.nasa9_parser import Species


class ComponentSelectorDialog(QtWidgets.QDialog):
    """Диалог выбора компонентов топлива с фильтрацией и поиском."""
    
    def __init__(self, species_db: Dict[str, Species], parent=None, 
                 mode: str = "oxidizer", selected_components: List[str] = None):
        """
        Args:
            species_db: словарь вид {name -> Species}
            parent: родительское окно
            mode: "oxidizer" или "fuel" (для фильтрации)
            selected_components: список уже выбранных компонентов
        """
        super().__init__(parent)
        self.species_db = species_db
        self.mode = mode
        self.selected = selected_components or []
        
        self.setWindowTitle(f"Выбор компонента ({'Окислитель' if mode == 'oxidizer' else 'Горючее'})")
        self.resize(600, 500)
        
        self._build_ui()
        self._load_components()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Поиск
        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("Поиск:")
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Введите имя компонента или элемент...")
        self.search_input.textChanged.connect(self._filter_components)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Фильтр по типу
        filter_layout = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Состояние:")
        
        self.filter_group = QtWidgets.QButtonGroup()
        self.rb_all = QtWidgets.QRadioButton("Все")
        self.rb_all.setChecked(True)
        self.rb_gas = QtWidgets.QRadioButton("Газ")
        self.rb_liquid = QtWidgets.QRadioButton("Жидкость")
        self.rb_solid = QtWidgets.QRadioButton("Твёрдое")
        
        self.filter_group.addButton(self.rb_all, 0)
        self.filter_group.addButton(self.rb_gas, 1)
        self.filter_group.addButton(self.rb_liquid, 2)
        self.filter_group.addButton(self.rb_solid, 3)
        self.filter_group.buttonClicked.connect(self._filter_components)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.rb_all)
        filter_layout.addWidget(self.rb_gas)
        filter_layout.addWidget(self.rb_liquid)
        filter_layout.addWidget(self.rb_solid)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Таблица компонентов
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Компонент", "Формула", "Мол. масса (г/моль)", "Состояние"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 80)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_details)
        layout.addWidget(self.table)
        
        # Детали компонента
        self.details_text = QtWidgets.QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(100)
        layout.addWidget(QtWidgets.QLabel("Информация:"))
        layout.addWidget(self.details_text)
        
        # Кнопки
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_select = QtWidgets.QPushButton("✓ Выбрать")
        self.btn_select.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("✗ Отмена")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_select)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)
    
    def _load_components(self):
        """Загрузить все компоненты из базы."""
        self.all_components = list(self.species_db.items())
        self._filter_components()
    
    def _filter_components(self):
        """Фильтровать компоненты по поиску и типу."""
        search_text = self.search_input.text().lower()
        
        # Определить фильтр по типу
        phase_filter = None
        if self.rb_gas.isChecked():
            phase_filter = 0
        elif self.rb_liquid.isChecked():
            phase_filter = 2
        elif self.rb_solid.isChecked():
            phase_filter = 1
        
        self.table.setRowCount(0)
        
        for name, species in self.all_components:
            # Поиск по имени или элементам
            if search_text and search_text not in name.lower():
                # Проверить поиск по элементам
                elements_str = ', '.join(species.elements.keys())
                if search_text not in elements_str.lower():
                    continue
            
            # Фильтр по типу
            if phase_filter is not None and species.phase != phase_filter:
                continue
            
            # Добавить строку
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Имя
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, name)  # Сохранить имя
            self.table.setItem(row, 0, name_item)
            
            # Формула (элементы)
            formula = ', '.join(f"{el}{int(cnt) if cnt != 1 else ''}" 
                              for el, cnt in sorted(species.elements.items()))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(formula))
            
            # Молярная масса
            mw_item = QtWidgets.QTableWidgetItem(f"{species.mol_weight:.2f}")
            mw_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 2, mw_item)
            
            # Состояние
            phase_names = {0: "газ", 1: "твёрдое", 2: "жидкость"}
            phase_item = QtWidgets.QTableWidgetItem(phase_names.get(species.phase, "?"))
            phase_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.table.setItem(row, 3, phase_item)
    
    def _update_details(self):
        """Обновить информацию о выбранном компоненте."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.details_text.clear()
            return
        
        row = rows[0].row()
        name = self.table.item(row, 0).data(Qt.UserRole)
        species = self.species_db[name]
        
        details = f"""
Компонент: {name}
Описание: {species.description}
Молярная масса: {species.mol_weight:.4f} г/моль
Теплота образования (298K): {species.hf298:.2f} Дж/моль
Элементный состав: {', '.join(f"{el}:{cnt}" for el, cnt in species.elements.items())}
Состояние: {['газ', 'твёрдое', 'жидкость'][species.phase]}
Температурные интервалы: {species.n_intervals}
        """.strip()
        
        self.details_text.setPlainText(details)
    
    def get_selected_component(self) -> Optional[str]:
        """Вернуть имя выбранного компонента."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return self.table.item(row, 0).data(Qt.UserRole)


class ComponentListWidget(QtWidgets.QWidget):
    """Виджет для управления списком компонентов (как в RPA)."""
    
    components_changed = pyqtSignal(list)  # Передаёт список (имена, массы, температуры)
    
    def __init__(self, species_db: Dict[str, Species], parent=None,
                 mode: str = "oxidizer"):
        """
        Args:
            species_db: словарь {name -> Species}
            parent: родительское окно
            mode: "oxidizer" или "fuel"
        """
        super().__init__(parent)
        self.species_db = species_db
        self.mode = mode
        self.components = []  # List[Dict{'name', 'mass', 'T'}]
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Таблица компонентов
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Компонент", "Масса (отн.)", "T (K)", ""])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 30)
        self.table.setMaximumHeight(150)
        # Двойной клик для добавления компонента
        self.table.doubleClicked.connect(self._add_component)
        layout.addWidget(self.table)
        
        # Кнопки
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Добавить")
        self.btn_add.clicked.connect(self._add_component)
        self.btn_remove = QtWidgets.QPushButton("− Удалить")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_normalize = QtWidgets.QPushButton("⚖ Нормализовать")
        self.btn_normalize.setToolTip("Нормализовать массы компонент в текущем списке так, чтобы их сумма = 1 (в пределах окислителя или горючего)")
        self.btn_normalize.clicked.connect(self._normalize_masses)
        button_layout.addWidget(self.btn_add)
        button_layout.addWidget(self.btn_remove)
        button_layout.addWidget(self.btn_normalize)
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _add_component(self):
        """Открыть диалог выбора компонента."""
        if self.species_db is None:
            QtWidgets.QMessageBox.warning(
                self, "Ошибка",
                "База компонентов ещё не загружена.\n"
                "Подождите несколько секунд и попробуйте снова."
            )
            return
        
        dialog = ComponentSelectorDialog(
            self.species_db,
            parent=self,
            mode=self.mode,
            selected_components=[c['name'] for c in self.components]
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            comp_name = dialog.get_selected_component()
            if comp_name:
                # Добавить в список
                self.components.append({
                    'name': comp_name,
                    'mass': 1.0,
                    'T': 0.0
                })
                self._refresh_table()
                self.components_changed.emit(self.components)
    
    def _refresh_table(self):
        """Обновить таблицу компонентов."""
        self.table.setRowCount(len(self.components))
        
        for row, comp in enumerate(self.components):
            # Имя (читаемо)
            name_item = QtWidgets.QTableWidgetItem(comp['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            # Масса (относительная, редактируемо)
            mass_spin = QtWidgets.QDoubleSpinBox()
            mass_spin.setRange(0.001, 1.0)
            mass_spin.setDecimals(3)
            mass_spin.setValue(comp['mass'])
            mass_spin.setSingleStep(0.001)
            mass_spin.setToolTip("Массовая доля компонента внутри текущей группы (окислитель или горючее), диапазон 0.001–1. O/F задаётся отдельно в основной панели.")
            mass_spin.valueChanged.connect(
                lambda val, r=row: self._update_mass(r, val)
            )
            self.table.setCellWidget(row, 1, mass_spin)
            
            # Температура (редактируемо)
            T_spin = QtWidgets.QDoubleSpinBox()
            T_spin.setRange(0.0, 5000.0)
            T_spin.setDecimals(2)
            T_spin.setValue(comp['T'])
            T_spin.setSingleStep(10.0)
            T_spin.setSpecialValueText("авто")
            T_spin.valueChanged.connect(
                lambda val, r=row: self._update_T(r, val)
            )
            self.table.setCellWidget(row, 2, T_spin)
            
            # Удаление
            btn_remove = QtWidgets.QPushButton("×")
            btn_remove.setMaximumWidth(30)
            btn_remove.clicked.connect(lambda _, r=row: self._remove_component(r))
            self.table.setCellWidget(row, 3, btn_remove)
    
    def _update_mass(self, row: int, value: float):
        """Обновить массу компонента."""
        if row < len(self.components):
            self.components[row]['mass'] = value
            self.components_changed.emit(self.components)
    
    def _update_T(self, row: int, value: float):
        """Обновить температуру компонента."""
        if row < len(self.components):
            self.components[row]['T'] = value
            self.components_changed.emit(self.components)
    
    def _remove_component(self, row: int):
        """Удалить компонент."""
        if 0 <= row < len(self.components):
            self.components.pop(row)
            self._refresh_table()
            self.components_changed.emit(self.components)
    
    def _remove_selected(self):
        """Удалить выбранный компонент."""
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._remove_component(rows[0].row())
    
    def _normalize_masses(self):
        """Нормализовать массы так, чтобы их сумма = 1."""
        total = sum(c['mass'] for c in self.components)
        if total > 1e-9:
            for c in self.components:
                c['mass'] /= total
            self._refresh_table()
            self.components_changed.emit(self.components)
    
    def set_components(self, components: List[Dict]):
        """Установить список компонентов."""
        self.components = [dict(c) for c in components]
        self._refresh_table()
    
    def get_components(self) -> List[Dict]:
        """Получить список компонентов."""
        return [dict(c) for c in self.components]


class MixturePropellantWidget(QtWidgets.QWidget):
    """Виджет для управления смесью окислителя и горючего с множественными компонентами."""
    
    mixture_changed = pyqtSignal(dict)  # {'ox_components', 'fuel_components'}
    
    def __init__(self, species_db: Dict[str, Species], parent=None):
        super().__init__(parent)
        self.species_db = species_db
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Окислитель
        gb_ox = QtWidgets.QGroupBox("Окислитель")
        ox_layout = QtWidgets.QVBoxLayout(gb_ox)
        self.oxidizer_list = ComponentListWidget(self.species_db, mode="oxidizer")
        self.oxidizer_list.components_changed.connect(self._on_mixture_changed)
        ox_layout.addWidget(self.oxidizer_list)
        layout.addWidget(gb_ox)
        
        # Горючее
        gb_fuel = QtWidgets.QGroupBox("Горючее")
        fuel_layout = QtWidgets.QVBoxLayout(gb_fuel)
        self.fuel_list = ComponentListWidget(self.species_db, mode="fuel")
        self.fuel_list.components_changed.connect(self._on_mixture_changed)
        fuel_layout.addWidget(self.fuel_list)
        layout.addWidget(gb_fuel)
        
        layout.addStretch()
    
    def _on_mixture_changed(self):
        """Сигнал об изменении смеси."""
        self.mixture_changed.emit({
            'ox_components': self.oxidizer_list.get_components(),
            'fuel_components': self.fuel_list.get_components(),
        })
    
    def get_mixture(self) -> Dict:
        """Получить текущую смесь."""
        return {
            'ox_components': self.oxidizer_list.get_components(),
            'fuel_components': self.fuel_list.get_components(),
        }
    
    def set_mixture(self, mixture: Dict):
        """Установить смесь."""
        self.oxidizer_list.set_components(mixture.get('ox_components', []))
        self.fuel_list.set_components(mixture.get('fuel_components', []))
