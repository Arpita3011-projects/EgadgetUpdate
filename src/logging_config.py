"""
logging_config.py
─────────────────
Centralized application logging to logs/application.log.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_LOG_CONFIGURED = False


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure root application logger (idempotent).

    Returns
    -------
    logging.Logger
        Named logger ``egadget``.
    """
    global _LOG_CONFIGURED

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir = os.path.join(root_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'application.log')

    logger = logging.getLogger('egadget')
    logger.setLevel(log_level)

    if not _LOG_CONFIGURED:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        _LOG_CONFIGURED = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the ``egadget`` namespace."""
    setup_logging()
    if name:
        return logging.getLogger(f'egadget.{name}')
    return logging.getLogger('egadget')
