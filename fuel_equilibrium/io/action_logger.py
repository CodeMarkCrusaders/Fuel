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
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

_MAX_BYTES = 2_097_152  # 2 МБ
_BACKUP_COUNT = 3
_MEMORY_LIMIT = 5000

_LOGGER: Optional[logging.Logger] = None
_ACTIVE_LOG_FILE: Optional[str] = None
_LOGGER_INIT_ERROR: Optional[str] = None


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


_SINK = _LogSink(_MEMORY_LIMIT)


def _details_to_text(details: Dict[str, object]) -> str:
    if not details:
        return ""
    try:
        return " | " + json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return " | " + "; ".join(f"{k}={v!r}" for k, v in details.items())


def _candidate_log_paths() -> List[str]:
    """Набор путей для лог-файла (по приоритету, с fallback)."""
    candidates: List[str] = []

    env_dir = os.environ.get("FUEL_EQUILIBRIUM_LOG_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, "app.log"))

    # Предпочитаем локальный logs рядом с рабочим каталогом запуска.
    candidates.append(os.path.join(os.getcwd(), "logs", "app.log"))

    # Путь относительно пакета (удобно для запуска из исходников).
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(project_dir, "logs", "app.log"))

    # Пользовательский профиль.
    candidates.append(os.path.join(os.path.expanduser("~"), ".fuel_equilibrium", "logs", "app.log"))

    # Последний fallback: временный каталог.
    candidates.append(os.path.join(tempfile.gettempdir(), "fuel_equilibrium", "logs", "app.log"))

    # Удаляем дубликаты с сохранением порядка.
    unique: List[str] = []
    seen = set()
    for p in candidates:
        norm = os.path.abspath(os.path.normpath(p))
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _try_create_file_handler() -> Optional[logging.Handler]:
    """Пытается создать файловый хендлер по fallback-путям."""
    global _ACTIVE_LOG_FILE, _LOGGER_INIT_ERROR

    last_error: Optional[str] = None
    for path in _candidate_log_paths():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _ACTIVE_LOG_FILE = path
            _LOGGER_INIT_ERROR = None
            return handler
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

    _ACTIVE_LOG_FILE = None
    _LOGGER_INIT_ERROR = last_error or "Не удалось создать RotatingFileHandler"
    return None


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

    file_handler = _try_create_file_handler()
    if file_handler is not None:
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
        logger.error("Не удалось инициализировать файловый лог", reason=_LOGGER_INIT_ERROR)

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
        if _ACTIVE_LOG_FILE:
            return _ACTIVE_LOG_FILE
        # Если файловый лог не поднялся, показываем первый ожидаемый путь.
        return _candidate_log_paths()[0]

    @staticmethod
    def log_init_error() -> str:
        _get_logger()
        return _LOGGER_INIT_ERROR or ""

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
            ActionLogger.debug("flush", reason=reason)
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
