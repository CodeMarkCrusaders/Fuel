#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диалоги выбора компонентов топлива (окислитель/горючее) на Dear PyGui.
Классификация веществ перенесена из component_selector.py без изменений.
"""

from typing import Dict, List, Optional

import dearpygui.dearpygui as dpg

from ..core.nasa9_parser import Species
from ..io.action_logger import ActionLogger


def _get_slot_children(item_tag: str, slot: int = 1) -> List[int]:
    """Возвращает детей item_tag из нужного slot для разных версий DPG."""
    try:
        info = dpg.get_item_info(item_tag) or {}
    except Exception:
        return []
    children = info.get("children", {})
    if isinstance(children, dict):
        return list(children.get(slot, []) or [])
    if isinstance(children, (list, tuple)) and len(children) > slot:
        return list(children[slot] or [])
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Классификация веществ (перенесено из component_selector.py)
# ═══════════════════════════════════════════════════════════════════════════
_OXIDIZER_ELEMENTS = {"F", "O", "CL", "BR", "I"}
_FUEL_ELEMENTS = {
    "H", "C", "B", "AL", "BE", "LI", "MG", "NA", "K", "SI", "S", "P",
    "ZR", "TI", "FE", "CU", "ZN", "W", "MO", "CR", "NI", "MN", "CA",
    "BA", "SR",
}


def is_ion(name: str, species: Species) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if n.endswith("+") or n.endswith("-"):
        return True
    if "++" in n or "--" in n:
        return True
    try:
        els = {str(k).upper() for k in species.elements.keys()}
    except Exception:
        els = set()
    if "E" in els:
        return True
    return False


def classify_role(name: str, species: Species) -> str:
    try:
        els = {str(k).upper(): float(v) for k, v in species.elements.items()}
    except Exception:
        els = {}
    els.pop("E", None)
    ox_atoms = sum(c for el, c in els.items() if el in _OXIDIZER_ELEMENTS)
    fuel_atoms = sum(c for el, c in els.items() if el in _FUEL_ELEMENTS)
    fuel_only_elements = {el for el in els if el in _FUEL_ELEMENTS}
    if ox_atoms <= 0 and fuel_atoms <= 0:
        return "both"
    if ox_atoms > 0 and fuel_atoms <= 0:
        return "oxidizer"
    if fuel_atoms > 0 and ox_atoms <= 0:
        return "fuel"
    if fuel_only_elements == {"H"} and ox_atoms >= fuel_atoms:
        return "oxidizer"
    if ox_atoms >= 2.0 * fuel_atoms:
        return "oxidizer"
    return "fuel"


def allowed_for_mode(name: str, species: Species, mode: str) -> bool:
    if is_ion(name, species):
        return False
    role = classify_role(name, species)
    if role == "both":
        return True
    return role == mode


# ═══════════════════════════════════════════════════════════════════════════
# Виджет списка компонентов (ComponentListWidget) — DPG
# ═══════════════════════════════════════════════════════════════════════════

class ComponentListWidgetDPG:
    """Список компонентов с массами/температурами, DPG-версия.

    Показывает таблицу компонентов группы (окислитель/горючее), позволяет
    добавлять (через модальный диалог), удалять и нормализовать массы.
    """

    def __init__(self, parent_tag: str, species_db, mode: str = "oxidizer",
                 on_change=None):
        self.species_db = species_db
        self.mode = mode
        self.components: List[Dict] = []  # [{'name','mass','T'}]
        self._on_change = on_change  # callback(components_list)
        self._parent = parent_tag
        self._table_tag = f"clw_table_{mode}_{id(self)}"
        self._build()

    def _build(self):
        with dpg.group(parent=self._parent, horizontal=True):
            dpg.add_button(label="+ Добавить",
                           callback=self._add_component)
            dpg.add_button(label="− Удалить",
                           callback=self._remove_selected)
            dpg.add_button(label="⚖ Нормализовать",
                           callback=self._normalize_masses)

        with dpg.table(tag=self._table_tag,
                       parent=self._parent,
                       header_row=True, reorderable=False, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp,
                       height=150):
            dpg.add_table_column(label="Компонент", width_fixed=True,
                                 init_width_or_weight=160)
            dpg.add_table_column(label="Масса (отн.)")
            dpg.add_table_column(label="T (K)")
            dpg.add_table_column(label="", width_fixed=True,
                                 init_width_or_weight=30)

    def _refresh_table(self):
        """Перестроить строки таблицы."""
        # Удаляем все дочерние элементы таблицы (строки)
        for child in _get_slot_children(self._table_tag, slot=1):
            dpg.delete_item(child)

        for i, comp in enumerate(self.components):
            with dpg.table_row(parent=self._table_tag):
                dpg.add_text(comp["name"])
                dpg.add_input_float(default_value=float(comp["mass"]),
                                    width=-1, min_value=0.001, min_clamped=True,
                                    step=0.01, format="%.3f",
                                    tag=f"{self._table_tag}_mass_{i}",
                                    callback=self._make_mass_cb(i))
                dpg.add_input_float(default_value=float(comp["T"]),
                                    width=-1, min_value=0.0, min_clamped=False,
                                    step=10.0, format="%.1f",
                                    tag=f"{self._table_tag}_T_{i}",
                                    callback=self._make_T_cb(i))
                dpg.add_button(label="×",
                               callback=self._make_remove_cb(i))

    def _make_mass_cb(self, i):
        def _cb(s, a, *_):
            if i < len(self.components):
                self.components[i]["mass"] = float(a)
                self._notify()
        return _cb

    def _make_T_cb(self, i):
        def _cb(s, a, *_):
            if i < len(self.components):
                self.components[i]["T"] = float(a)
                self._notify()
        return _cb

    def _make_remove_cb(self, i):
        def _cb(*_):
            if 0 <= i < len(self.components):
                self.components.pop(i)
                self._refresh_table()
                self._notify()
        return _cb

    def _notify(self):
        if self._on_change:
            self._on_change(self.get_components())

    def _add_component(self, *_):
        """Открыть диалог выбора компонента."""
        if not self.species_db:
            return
        ActionLogger.info("Открыт диалог выбора компонента",
                          mode="окислитель" if self.mode == "oxidizer" else "горючее")
        self._selector_dialog = ComponentSelectorDialogDPG(
            self.species_db, mode=self.mode,
            selected=[c["name"] for c in self.components],
            callback=self._on_component_selected,
        )

    def _on_component_selected(self, name: Optional[str]):
        """Колбэк, вызываемый после закрытия диалога выбора компонента."""
        if name:
            ActionLogger.info("Компонент добавлен", name=name, mode=self.mode)
            self.components.append({"name": name, "mass": 1.0, "T": 0.0})
            self._refresh_table()
            self._notify()
        else:
            ActionLogger.info("Выбор компонента отменён", mode=self.mode)

    def _remove_selected(self, *_):
        # В DPG выбор строки через клик; здесь удаляем последнюю как fallback.
        if self.components:
            removed = self.components[-1]["name"]
            ActionLogger.info("Компонент удалён", name=removed, mode=self.mode)
            self.components.pop()
            self._refresh_table()
            self._notify()

    def _normalize_masses(self, *_):
        ActionLogger.info("Нормализация масс компонентов", mode=self.mode)
        total = sum(c["mass"] for c in self.components)
        if total > 1e-9:
            for c in self.components:
                c["mass"] /= total
            self._refresh_table()
            self._notify()

    def set_components(self, components: List[Dict]):
        self.components = [dict(c) for c in components]
        self._refresh_table()

    def get_components(self) -> List[Dict]:
        return [dict(c) for c in self.components]


# ═══════════════════════════════════════════════════════════════════════════
# Модальный диалог выбора компонента (ComponentSelectorDialog) — DPG
# ═══════════════════════════════════════════════════════════════════════════

class ComponentSelectorDialogDPG:
    """Модальное окно выбора компонента с поиском/фильтром, DPG-версия."""

    def __init__(self, species_db: Dict[str, Species], mode: str = "oxidizer",
                 selected: List[str] = None, callback=None):
        self.species_db = species_db
        self.mode = mode
        self.selected = selected or []
        self._callback = callback
        self._result: Optional[str] = None
        self._modal_id: Optional[int] = None

        # Уникальные теги на экземпляр диалога (иначе между окнами бывают коллизии).
        self._tag_prefix = f"__cs_{id(self)}"
        self._tag_modal = f"{self._tag_prefix}_modal"
        self._tag_search = f"{self._tag_prefix}_search"
        self._tag_filter = f"{self._tag_prefix}_filter"
        self._tag_table = f"{self._tag_prefix}_table"
        self._tag_details = f"{self._tag_prefix}_details"

        self._build_modal()

    def _build_modal(self):
        # Создаём всплывающее окно поверх вьюпорта.
        title = (f"Выбор компонента ("
                 f"{'Окислитель' if self.mode == 'oxidizer' else 'Горючее'})")
        if dpg.does_item_exist(self._tag_modal):
            dpg.delete_item(self._tag_modal)

        with dpg.window(tag=self._tag_modal, label=title,
                        modal=True, no_close=True, width=620, height=500,
                        pos=[100, 80]) as self._modal_id:
            dpg.add_input_text(label="Поиск",
                               hint="Имя или элемент...",
                               tag=self._tag_search,
                               callback=self._filter)
            with dpg.group(horizontal=True):
                dpg.add_text("Состояние:")
                dpg.add_radio_button(
                    ["Все", "Газ", "Жидкость", "Твёрдое"],
                    tag=self._tag_filter, horizontal=True,
                    default_value="Все", callback=self._filter)
            dpg.add_text(
                f"Показаны вещества для "
                f"{'окислителя' if self.mode == 'oxidizer' else 'горючего'}.",
                color=(168, 162, 158))
            with dpg.table(tag=self._tag_table, header_row=True,
                           resizable=True, height=300,
                           policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="Компонент", width_fixed=True,
                                     init_width_or_weight=200)
                dpg.add_table_column(label="Формула", width_fixed=True,
                                     init_width_or_weight=120)
                dpg.add_table_column(label="М (г/моль)")
                dpg.add_table_column(label="Сост.")
            dpg.add_text("", tag=self._tag_details, wrap=580)
            with dpg.group(horizontal=True):
                dpg.add_button(label="✓ Выбрать",
                               callback=lambda *_: self._accept())
                dpg.add_button(label="✗ Отмена",
                               callback=lambda *_: self._cancel())
        self._populate()

    def _populate(self):
        """Заполнить таблицу отфильтрованными компонентами."""
        table = self._tag_table
        for child in _get_slot_children(table, slot=1):
            dpg.delete_item(child)
        search = (dpg.get_value(self._tag_search) or "").lower().strip()
        fstate = dpg.get_value(self._tag_filter) or "Все"
        state_map = {"Все": None, "Газ": "gas",
                     "Жидкость": "liquid", "Твёрдое": "solid"}
        state_filter = state_map.get(fstate)
        for name, sp in self.species_db.items():
            if not allowed_for_mode(name, sp, self.mode):
                continue
            if search and search not in name.lower():
                els = ", ".join(sp.elements.keys()).lower()
                if search not in els:
                    continue
            if state_filter and sp.aggregate_state != state_filter:
                continue
            with dpg.table_row(parent=table):
                dpg.add_selectable(label=name,
                                   callback=self._make_select_cb(name))
                formula = ", ".join(
                    f"{el}{int(c) if c != 1 else ''}"
                    for el, c in sorted(sp.elements.items()))
                dpg.add_text(formula)
                dpg.add_text(f"{sp.mol_weight:.2f}")
                dpg.add_text(sp.aggregate_state_ru)

    def _make_select_cb(self, name):
        def _cb(s, a, *_):
            if a:
                sp = self.species_db.get(name)
                if sp:
                    txt = (f"Компонент: {name}\n"
                           f"Описание: {sp.description}\n"
                           f"М: {sp.mol_weight:.4f} г/моль\n"
                           f"Hf298: {sp.hf298:.2f} Дж/моль\n"
                           f"Сост.: {sp.aggregate_state_ru}")
                    dpg.set_value(self._tag_details, txt)
                self._result = name
                ActionLogger.debug("Компонент выделен в диалоге", name=name, mode=self.mode)
        return _cb

    def _filter(self, *args):
        self._populate()

    def _accept(self, *_):
        if not self._result:
            ActionLogger.warning("Компонент не выбран в диалоге", mode=self.mode)
            if dpg.does_item_exist(self._tag_details):
                dpg.set_value(self._tag_details, "Сначала выберите компонент в таблице.")
            return

        ActionLogger.info("Подтверждён выбор компонента", name=self._result, mode=self.mode)
        if self._modal_id is not None:
            dpg.delete_item(self._modal_id)
            self._modal_id = None
        if self._callback:
            self._callback(self._result)

    def _cancel(self, *_):
        ActionLogger.info("Диалог выбора компонента закрыт без выбора", mode=self.mode)
        if self._modal_id is not None:
            dpg.delete_item(self._modal_id)
            self._modal_id = None
        if self._callback:
            self._callback(None)


# ═══════════════════════════════════════════════════════════════════════════
# Виджет смеси (MixturePropellantWidget) — DPG
# ═══════════════════════════════════════════════════════════════════════════

class MixturePropellantWidgetDPG:
    """Виджет управления смесью окислитель + горючее (DPG)."""

    def __init__(self, parent_tag: str, species_db=None,
                 on_change=None):
        self.species_db = species_db
        self._on_change = on_change
        self._parent = parent_tag
        self.oxidizer_list: Optional[ComponentListWidgetDPG] = None
        self.fuel_list: Optional[ComponentListWidgetDPG] = None
        self._build()

    def _build(self):
        with dpg.collapsing_header(parent=self._parent,
                                   label="Окислитель", default_open=True):
            ox_tag = f"{self._parent}_ox_group"
            with dpg.group(tag=ox_tag):
                pass
            self.oxidizer_list = ComponentListWidgetDPG(
                ox_tag, self.species_db, mode="oxidizer",
                on_change=self._notify)
        with dpg.collapsing_header(parent=self._parent,
                                   label="Горючее", default_open=True):
            fu_tag = f"{self._parent}_fu_group"
            with dpg.group(tag=fu_tag):
                pass
            self.fuel_list = ComponentListWidgetDPG(
                fu_tag, self.species_db, mode="fuel",
                on_change=self._notify)

    def _notify(self, *args):
        if self._on_change:
            self._on_change(self.get_mixture())

    def get_mixture(self) -> Dict:
        return {
            "ox_components": (self.oxidizer_list.get_components()
                              if self.oxidizer_list else []),
            "fuel_components": (self.fuel_list.get_components()
                                if self.fuel_list else []),
        }

    def set_mixture(self, mixture: Dict):
        if self.oxidizer_list:
            self.oxidizer_list.set_components(mixture.get("ox_components", []))
        if self.fuel_list:
            self.fuel_list.set_components(mixture.get("fuel_components", []))