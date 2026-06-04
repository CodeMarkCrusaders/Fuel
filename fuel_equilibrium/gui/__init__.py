"""
fuel_equilibrium.gui — графический интерфейс (PyQt5).

Содержит:
    * app                 — главное окно RPA-style калькулятора сопла
    * component_selector  — диалог/виджет выбора компонент топлива

Запуск:  ``python -m fuel_equilibrium.gui.app``
"""

__all__ = ["main"]


def main():
    """Точка входа GUI. Импорт ленивый — чтобы импорт самого пакета
    fuel_equilibrium не тянул за собой PyQt5 и matplotlib."""
    from .app import main as _main
    return _main()
