# Логгер итераций для расчётов химического равновесия.
# Пишет в текстовый файл подробную информацию обо всех шагах решателя:
# - входные данные (T, P, H, S, реагенты)
# - значения целевой функции на каждой итерации
# - текущий состав смеси
# - невязки балансов
# - сходимость внешнего цикла (для HP/SP-задач)

import os
import sys
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict


class IterationLogger:
    """Журнал итераций решателя.

    Ничего не делает, если path=None (тихий режим).
    Иначе открывает файл и пишет в него все события.
    """

    def __init__(self, path: Optional[str] = None, also_stdout: bool = False):
        self.path = path
        self.also_stdout = also_stdout
        self._fp = None
        self._t_start = time.time()

        if path:
            # автоматически создадим папку, если нужно
            d = os.path.dirname(os.path.abspath(path))
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            self._fp = open(path, 'w', encoding='utf-8')
            self._write_header()

    @property
    def enabled(self) -> bool:
        return self._fp is not None

    def _write_header(self):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._raw_write('=' * 78 + '\n')
        self._raw_write(f'  ЛОГ ИТЕРАЦИЙ — расчёт химического равновесия\n')
        self._raw_write(f'  старт: {ts}\n')
        self._raw_write('=' * 78 + '\n\n')

    def _raw_write(self, text: str):
        if self._fp is not None:
            self._fp.write(text)
            self._fp.flush()
        if self.also_stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

    # ───── публичные методы ─────

    def log(self, message: str = ''):
        """Простая строка в журнал."""
        if not self.enabled and not self.also_stdout:
            return
        dt = time.time() - self._t_start
        self._raw_write(f'[{dt:8.3f} с]  {message}\n')

    def section(self, title: str):
        """Раздел в журнале (визуально выделен)."""
        if not self.enabled and not self.also_stdout:
            return
        self._raw_write('\n' + '-' * 78 + '\n')
        self._raw_write(f'  {title}\n')
        self._raw_write('-' * 78 + '\n')

    def header_problem(
        self,
        problem_type: str,
        reactants: str,
        elements: Dict[str, float],
        T: Optional[float] = None,
        P: Optional[float] = None,
        H: Optional[float] = None,
        S: Optional[float] = None,
    ):
        """Шапка с условиями задачи."""
        if not self.enabled and not self.also_stdout:
            return
        self.section(f'ПОСТАНОВКА ЗАДАЧИ — тип: {problem_type}')
        self._raw_write(f'  Реагенты: {reactants}\n')
        self._raw_write('  Элементный баланс (моль):\n')
        for el, n in sorted(elements.items()):
            self._raw_write(f'      {el}: {n:.6f}\n')
        if T is not None:
            self._raw_write(f'  T = {T:.4f} К\n')
        if P is not None:
            self._raw_write(f'  P = {P:.4f} Па  ({P/101325:.6f} атм)\n')
        if H is not None:
            self._raw_write(f'  H_target = {H:.4f} Дж\n')
        if S is not None:
            self._raw_write(f'  S_target = {S:.4f} Дж/К\n')
        self._raw_write('\n')

    def log_species_list(self, names: List[str], n_gas: int, n_cond: int):
        """Список веществ-кандидатов."""
        if not self.enabled and not self.also_stdout:
            return
        self.section(f'ВЕЩЕСТВА-КАНДИДАТЫ  (газов: {n_gas}, конденсата: {n_cond})')
        for i, name in enumerate(names):
            tag = 'газ' if i < n_gas else 'конд.'
            self._raw_write(f'    [{i:3d}] {name:<25s} {tag}\n')
        self._raw_write('\n')

    def inner_iter(
        self,
        outer: int,
        inner: int,
        gibbs: float,
        n_vec: np.ndarray,
        names: List[str],
        residual: float = None,
        top_k: int = 10,
    ):
        """Запись одной итерации SLSQP/trust-constr."""
        if not self.enabled and not self.also_stdout:
            return
        prefix = f'[outer {outer:3d}] ' if outer >= 0 else ''
        self._raw_write(
            f'  {prefix}iter {inner:5d}:  G/RT = {gibbs:.8e}'
        )
        if residual is not None:
            self._raw_write(f',  невязка = {residual:.3e}')
        self._raw_write('\n')

        # топ компонентов
        idx = np.argsort(-n_vec)[:top_k]
        for k in idx:
            ni = n_vec[k]
            if ni > 1e-15:
                self._raw_write(f'        n[{names[k]:<20s}] = {ni:.6e}\n')

    def outer_iter(
        self,
        outer: int,
        T: float,
        target_name: str,
        target_value: float,
        current_value: float,
        residual_T: float,
    ):
        """Шаг внешнего цикла по T для HP/SP-задач."""
        if not self.enabled and not self.also_stdout:
            return
        self._raw_write(
            f'  >> внешний шаг {outer:3d}:  T = {T:10.4f} К,  '
            f'{target_name} = {current_value:.6e}  '
            f'(target {target_value:.6e},  ΔT = {residual_T:.4e})\n'
        )

    def result_summary(
        self,
        converged: bool,
        iterations: int,
        T: float,
        P: float,
        H: Optional[float] = None,
        S: Optional[float] = None,
        residual: float = None,
        gibbs: float = None,
    ):
        if not self.enabled and not self.also_stdout:
            return
        self.section('ИТОГ')
        self._raw_write(f'  Сходимость: {"ДА" if converged else "НЕТ"}\n')
        self._raw_write(f'  Всего итераций (внутр.): {iterations}\n')
        self._raw_write(f'  T = {T:.4f} К\n')
        self._raw_write(f'  P = {P:.4f} Па\n')
        if H is not None:
            self._raw_write(f'  H = {H:.4f} Дж\n')
        if S is not None:
            self._raw_write(f'  S = {S:.4f} Дж/К\n')
        if gibbs is not None:
            self._raw_write(f'  G/RT финальное = {gibbs:.6e}\n')
        if residual is not None:
            self._raw_write(f'  Невязка элементов = {residual:.3e}\n')

    def log_composition(self, names, moles, total_moles, top_k: int = 30):
        """Записать финальный состав."""
        if not self.enabled and not self.also_stdout:
            return
        self.section(f'СОСТАВ СМЕСИ  (всего молей газа: {total_moles:.6e})')
        idx = np.argsort(-np.asarray(moles))[:top_k]
        self._raw_write(f'  {"Компонент":<25s} {"моль":>16s} {"мол.доля":>16s}\n')
        for k in idx:
            ni = moles[k]
            if ni > 1e-12:
                xi = ni / total_moles if total_moles > 0 else 0.0
                self._raw_write(f'  {names[k]:<25s} {ni:>16.6e} {xi:>16.6e}\n')
        self._raw_write('\n')

    def close(self):
        if self._fp is not None:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            dt = time.time() - self._t_start
            self._raw_write(f'\n{"=" * 78}\n')
            self._raw_write(f'  завершено: {ts}  (длительность: {dt:.2f} с)\n')
            self._raw_write(f'{"=" * 78}\n')
            self._fp.close()
            self._fp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# одиночка-заглушка для случая, когда лог не нужен
class NullLogger(IterationLogger):
    """Логгер, который ничего не пишет (для умолчания)."""

    def __init__(self):
        self.path = None
        self.also_stdout = False
        self._fp = None
        self._t_start = time.time()

    @property
    def enabled(self):
        return False

    def _raw_write(self, text):
        pass

    def close(self):
        pass
