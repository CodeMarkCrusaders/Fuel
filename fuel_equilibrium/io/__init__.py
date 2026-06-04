"""
fuel_equilibrium.io — I/O-слой: логирование итераций, batch-API через CSV,
форматирование отчётов.

ВАЖНО: имя пакета совпадает с модулем stdlib ``io``. Это не вызывает конфликта
потому, что Python 3 использует absolute imports по умолчанию: ``import io`` в
сторонних файлах по-прежнему резолвится в стандартную библиотеку, а доступ к
этому подпакету идёт исключительно через ``fuel_equilibrium.io``.

Доступ к печатным функциям (``print_result`` / ``print_nozzle_table``)
организован ЛЕНИВО через ``__getattr__``. Это нужно, потому что
``reporting`` импортирует ``EquilibriumResult`` и ``RocketPerformance`` из
``core`` и ``rocket``, а ``core.gibbs_solver`` сам импортирует
``iteration_logger`` из этого же пакета — без ленивой загрузки получился бы
циклический импорт.
"""

from .iteration_logger import IterationLogger, NullLogger

__all__ = [
    "IterationLogger",
    "NullLogger",
    "print_result",
    "print_nozzle_table",
]


def __getattr__(name):
    # PEP 562: ленивые атрибуты модуля
    if name in ("print_result", "print_nozzle_table"):
        from . import reporting
        return getattr(reporting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
