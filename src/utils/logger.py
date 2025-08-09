"""Centralized logging configuration for the application."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a configured logger.
    
    Args:
        name: Logger name. Defaults to root logger if None.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL env var or INFO.
        format_string: Custom format string. Uses default if None.
    
    Returns:
        Configured logger instance.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger if name is None; named loggers will propagate to root
    is_root = name is None
    logger = logging.getLogger() if is_root else logging.getLogger(name)
    
    if is_root:
        # Idempotent root configuration: attach handlers only once
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)

            # Generate filename with current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            log_file = logs_dir / f"{current_date}.log"

            # Set up file handler
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)

            # Also keep console output
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(console_handler)

        logger.setLevel(getattr(logging, level, logging.INFO))
    else:
        # Named loggers: do not attach handlers; send everything to root
        # Ensure no stray handlers are attached to avoid duplicate records
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.propagate = True
        logger.setLevel(getattr(logging, level, logging.INFO))
    
    return logger


# Default application logger
logger = setup_logger(__name__)
