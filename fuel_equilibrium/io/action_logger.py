#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ActionLogger — журнал действий пользователя в GUI.

Пишет в файл временные метки и описания ключевых событий:
запуск расчёта, экспорт, сохранение/загрузка конфигурации, ошибки.

Файл лога:  ~/rpa_action.log  (ротация при превышении 1 МБ).
Если файл недоступен — логгирование бесшумно отключается.
"""

import os
import logging
import logging.handlers
import threading
from typing import Optional, List, Tuple

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "rpa_action.log")
_MAX_BYTES = 1_048_576  # 1 МБ
_BACKUP_COUNT = 1

_LOGGER: Optional[logging.Logger] = None
_PENDING_RECORDS: List[Tuple[int, str]] = []
_LOCK = threading.RLock()


def _get_logger() -> logging.Logger:
    """Ленивая инициализация логгера (один раз при первом вызове)."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    _LOGGER = logging.getLogger("RocketNozzleGUI")
    _LOGGER.setLevel(logging.INFO)

    # Предотвращаем дублирование хендлеров при пересоздании
    if _LOGGER.handlers:
        return _LOGGER

    try:
        handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except (OSError, PermissionError):
        # Если не удалось создать файл — заглушка
        handler = logging.NullHandler()

    formatter = logging.Formatter(
        "[%(asctime)s]  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    return _LOGGER


class ActionLogger:
    """Статический интерфейс для логирования действий пользователя.

    Пример использования::

        from ..io.action_logger import ActionLogger

        ActionLogger.info("Расчёт запущен", solver="CEA")
        ActionLogger.warning("База NASA-9 не загружена")
        ActionLogger.error("Ошибка экспорта", path="...", detail=str(e))
    """

    @staticmethod
    def _log(level: int, action: str, **details) -> None:
        """Положить сообщение в буфер; запись в файл — через flush()."""
        if details:
            parts = [f"{k}={v!r}" for k, v in details.items()]
            msg = f"{action}  ({'; '.join(parts)})"
        else:
            msg = action
        with _LOCK:
            _PENDING_RECORDS.append((level, msg))

    # ── публичные методы ──────────────────────────────────────────────────

    @staticmethod
    def info(action: str, **details) -> None:
        """Информационное сообщение."""
        ActionLogger._log(logging.INFO, action, **details)

    @staticmethod
    def warning(action: str, **details) -> None:
        """Предупреждение (например, нехватка данных)."""
        ActionLogger._log(logging.WARNING, action, **details)

    @staticmethod
    def error(action: str, **details) -> None:
        """Ошибка (исключение, сбой расчёта)."""
        ActionLogger._log(logging.ERROR, action, **details)

    @staticmethod
    def log_path() -> str:
        """Вернуть путь к текущему файлу лога."""
        return _LOG_FILE

    @staticmethod
    def flush(reason: str = "") -> None:
        """Сбросить буфер действий в файл.

        Запись откладывается до ключевых точек:
        - после завершения расчёта;
        - при закрытии программы.
        """
        with _LOCK:
            if not _PENDING_RECORDS:
                return
            records = list(_PENDING_RECORDS)
            _PENDING_RECORDS.clear()

        logger = _get_logger()
        if reason:
            logger.info("Сброс буфера журнала (%s)", reason)
        for level, msg in records:
            logger.log(level, msg)

        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    @staticmethod
    def shutdown(reason: str = "") -> None:
        """Сбросить буфер и завершить подсистему logging."""
        ActionLogger.flush(reason=reason)
        logging.shutdown()
