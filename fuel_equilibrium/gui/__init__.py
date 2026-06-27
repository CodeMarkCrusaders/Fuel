"""
fuel_equilibrium.gui — графический интерфейс (Dear PyGui).

Содержит:
    * app                       — главное окно RPA-style калькулятора сопла
    * component_selector_dpg    — диалог/виджет выбора компонент топлива (DPG)
    * component_selector        — legacy-модуль (PyQt5, оставлен для совместимости)

Запуск:  ``python -m fuel_equilibrium.gui.app``
"""

__all__ = ["main"]


def main():
    """Точка входа GUI. Импорт ленивый — чтобы импорт самого пакета
    fuel_equilibrium не тянул за собой Dear PyGui и matplotlib."""
    from .app import main as _main
    return _main()