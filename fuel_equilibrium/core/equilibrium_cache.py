# Кэш равновесных составов.
#
# Решение задачи минимизации Гиббса (TP/HP/SP) — самая дорогая операция в
# расчёте. При построении сопла один и тот же набор веществ с одним и тем же
# элементным балансом многократно решается при близких/совпадающих (T|H|S, P):
#   * поиск горловины брентом многократно бьёт по одинаковым P;
#   * срез и промежуточные сечения иногда повторяют давления;
#   * поиск оптимального Km повторяет камеру/горловину для одинаковых смесей.
#
# Кэш ключуется по физически значимым входам (тип задачи, набор веществ,
# элементный баланс, целевой параметр, давление, учёт конденсата) и хранит
# готовый EquilibriumResult. Возвращается глубокая копия, т.к. вызывающий код
# мутирует результат (problem_type, label у станций и т.п.).

import threading
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

# Округление ключа: количество значащих знаков для float-параметров.
# Слишком грубо — рискуем вернуть «не тот» состав; слишком точно — мало
# попаданий. 1e-6 по относительной величине — разумный компромисс.
_REL_TOL = 1e-6

# Максимальный размер кэша (число записей). При превышении — простая FIFO-очистка.
_MAX_ENTRIES = 4096


def _round_rel(x: float) -> float:
    """Округляет float к фиксированному числу значащих цифр для устойчивого ключа."""
    if x is None:
        return None
    if not np.isfinite(x):
        return float(x)
    if x == 0.0:
        return 0.0
    import math
    digits = 9  # ~ -log10(_REL_TOL) + запас
    exp = math.floor(math.log10(abs(x)))
    factor = 10 ** (digits - 1 - exp)
    return round(x * factor) / factor


def _elements_key(elements: Dict[str, float]) -> Tuple:
    return tuple(sorted((k, _round_rel(float(v))) for k, v in elements.items()))


def _species_key(species_list: List, include_condensed: bool) -> Tuple:
    # имена однозначно определяют набор веществ (и их термоданные из общей БД).
    names = tuple(sorted(sp.name for sp in species_list))
    return (names, bool(include_condensed))


# ─────────────────────────────────────────────────────────────────────────────
# Кэш
# ─────────────────────────────────────────────────────────────────────────────

class EquilibriumCache:
    """Потокобезопасный кэш равновесных составов с учётом статистики."""

    def __init__(self, max_entries: int = _MAX_ENTRIES):
        self._store: Dict[Tuple, object] = {}
        self._order: List[Tuple] = []
        self._lock = threading.Lock()
        self._max = max_entries
        self.hits = 0
        self.misses = 0
        self.enabled = True

    # — построение ключа ————————————————————————————————————————————————
    @staticmethod
    def make_key(
        problem_type: str,
        species_list: List,
        elements: Dict[str, float],
        target: float,
        P: float,
        include_condensed: bool,
    ) -> Tuple:
        return (
            problem_type,
            _species_key(species_list, include_condensed),
            _elements_key(elements),
            _round_rel(float(target)),
            _round_rel(float(P)),
        )

    # — доступ ——————————————————————————————————————————————————————————
    def get(self, key: Tuple):
        if not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            return deepcopy(entry)

    def put(self, key: Tuple, result) -> None:
        if not self.enabled:
            return
        with self._lock:
            if key not in self._store:
                self._order.append(key)
                # простая FIFO-очистка при переполнении
                if len(self._order) > self._max:
                    old = self._order.pop(0)
                    self._store.pop(old, None)
            self._store[key] = deepcopy(result)

    # — управление ——————————————————————————————————————————————————————
    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = self.hits + self.misses
            rate = (self.hits / total) if total else 0.0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'entries': len(self._store),
                'hit_rate': rate,
            }

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)


# Глобальный кэш по умолчанию (общий на процесс).
_GLOBAL_CACHE = EquilibriumCache()


def get_global_cache() -> EquilibriumCache:
    return _GLOBAL_CACHE


def clear_cache() -> None:
    _GLOBAL_CACHE.clear()


def cache_stats() -> Dict[str, int]:
    return _GLOBAL_CACHE.stats()


def set_cache_enabled(flag: bool) -> None:
    _GLOBAL_CACHE.set_enabled(flag)
