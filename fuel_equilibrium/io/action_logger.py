#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ActionLogger — централизованное логирование GUI и приложения.

Что делает:
- пишет лог в ротируемый файл;
- хранит последние записи в памяти (ring buffer);
- отдаёт записи для отображения прямо в интерфейсе;
- поддерживает фильтрацию по уровню и поиску.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

_LOG_DIR = os.path.join(os.path.expanduser("~"), ".fuel_equilibrium", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")
_MAX_BYTES = 2_097_152  # 2 МБ
_BACKUP_COUNT = 3
_MEMORY_LIMIT = 5000


@dataclass(frozen=True)
class LogEntry:
    """Структурированная запись журнала."""

    timestamp: datetime
    level: str
    action: str
    details: Dict[str, object]
    message: str


class _InMemoryLogHandler(logging.Handler):
    """Хендлер Python logging, складывающий записи в память."""

    def __init__(self, sink: "_LogSink"):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = getattr(record, "action_payload", {}) or {}
            action = str(payload.get("action") or record.getMessage())
            details = payload.get("details") or {}
            if not isinstance(details, dict):
                details = {"value": str(details)}

            self._sink.append(
                LogEntry(
                    timestamp=datetime.fromtimestamp(record.created),
                    level=record.levelname,
                    action=action,
                    details=details,
                    message=record.getMessage(),
                )
            )
        except Exception:
            self.handleError(record)


class _LogSink:
    """Потокобезопасный ring buffer для логов."""

    def __init__(self, maxlen: int):
        self._items: Deque[LogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, entry: LogEntry) -> None:
        with self._lock:
            self._items.append(entry)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def get_all(self) -> List[LogEntry]:
        with self._lock:
            return list(self._items)


_LOGGER: Optional[logging.Logger] = None
_SINK = _LogSink(_MEMORY_LIMIT)


def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def _details_to_text(details: Dict[str, object]) -> str:
    if not details:
        return ""
    try:
        return " | " + json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return " | " + "; ".join(f"{k}={v!r}" for k, v in details.items())


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("RocketNozzleGUI")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # На случай повторной инициализации очищаем старые хендлеры.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    memory_handler = _InMemoryLogHandler(_SINK)
    memory_handler.setLevel(logging.DEBUG)
    logger.addHandler(memory_handler)

    try:
        _ensure_log_dir()
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        logger.addHandler(logging.NullHandler())

    _LOGGER = logger
    return logger


class ActionLogger:
    """Статический API для логирования действий GUI и чтения журнала."""

    @staticmethod
    def _log(level: int, action: str, **details) -> None:
        logger = _get_logger()
        details_text = _details_to_text(details)
        message = f"{action}{details_text}"
        logger.log(level, message, extra={"action_payload": {"action": action, "details": details}})

    @staticmethod
    def debug(action: str, **details) -> None:
        ActionLogger._log(logging.DEBUG, action, **details)

    @staticmethod
    def info(action: str, **details) -> None:
        ActionLogger._log(logging.INFO, action, **details)

    @staticmethod
    def warning(action: str, **details) -> None:
        ActionLogger._log(logging.WARNING, action, **details)

    @staticmethod
    def error(action: str, **details) -> None:
        ActionLogger._log(logging.ERROR, action, **details)

    @staticmethod
    def exception(action: str, **details) -> None:
        ActionLogger._log(logging.ERROR, action, **details)

    @staticmethod
    def log_path() -> str:
        _get_logger()
        return _LOG_FILE

    @staticmethod
    def get_entries(limit: int = 300, min_level: str = "INFO", contains: str = "") -> List[LogEntry]:
        """Вернуть записи журнала из памяти (фильтр по уровню и подстроке)."""
        _get_logger()
        level_no = logging._nameToLevel.get((min_level or "INFO").upper(), logging.INFO)
        needle = (contains or "").strip().lower()

        data = _SINK.get_all()
        filtered: List[LogEntry] = []
        for entry in data:
            entry_level_no = logging._nameToLevel.get(entry.level, logging.INFO)
            if entry_level_no < level_no:
                continue
            if needle:
                text = f"{entry.action} {entry.message} {entry.details}".lower()
                if needle not in text:
                    continue
            filtered.append(entry)

        if limit > 0:
            filtered = filtered[-limit:]
        return filtered

    @staticmethod
    def render_entries(limit: int = 300, min_level: str = "INFO", contains: str = "") -> str:
        """Вернуть записи журнала одной строкой для текстового виджета."""
        entries = ActionLogger.get_entries(limit=limit, min_level=min_level, contains=contains)
        if not entries:
            return "Логи пока пусты."

        lines: List[str] = []
        for e in entries:
            details_text = _details_to_text(e.details)
            lines.append(
                f"[{e.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {e.level:<7} {e.action}{details_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def clear_memory() -> None:
        _SINK.clear()

    @staticmethod
    def flush(reason: str = "") -> None:
        """Принудительно сбросить хендлеры logging."""
        logger = _get_logger()
        if reason:
            logger.debug("flush() | %s", reason)
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    @staticmethod
    def shutdown(reason: str = "") -> None:
        """Сбросить хендлеры и завершить подсистему logging."""
        ActionLogger.flush(reason=reason)
        logging.shutdown()
