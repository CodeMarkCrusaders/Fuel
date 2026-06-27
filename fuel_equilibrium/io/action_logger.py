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
from typing import Optional

_LOG_DIR = os.path.expanduser("~")
_LOG_FILE = os.path.join(_LOG_DIR, "rpa_action.log")
_MAX_BYTES = 1_048_576  # 1 МБ
_BACKUP_COUNT = 1

_LOGGER: Optional[logging.Logger] = None


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
        """Записать сообщение в лог с дополнительными полями."""
        logger = _get_logger()
        if details:
            parts = [f"{k}={v!r}" for k, v in details.items()]
            msg = f"{action}  ({'; '.join(parts)})"
        else:
            msg = action
        logger.log(level, msg)

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
