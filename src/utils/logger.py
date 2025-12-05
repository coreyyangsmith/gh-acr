"""Centralized logging configuration for the application.

This module provides a unified logging setup that ensures consistent log
formatting, file output, and console display across all application modules.

Features
--------
- **Dual output**: Logs go to both console and daily log files
- **Configurable levels**: Set via LOG_LEVEL environment variable
- **Automatic file rotation**: New log file each day in `logs/` directory
- **Hierarchical logging**: Named loggers propagate to root
- **Idempotent setup**: Safe to call multiple times

Usage Patterns
--------------
1. Root logger setup (typically in startup.py)::

    from src.utils.logger import setup_logger
    logger = setup_logger()  # Configures root logger
    logger.info("Application started")

2. Module-specific logger::

    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)  # Named logger
    logger.debug("Processing item %d", item_id)

3. Custom configuration::

    logger = setup_logger(
        name="my_module",
        level="DEBUG",
        format_string="%(levelname)s: %(message)s"
    )

Environment Variables
---------------------
- LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  Default: INFO

Log File Location
-----------------
Log files are written to `./logs/YYYY-MM-DD.log` where YYYY-MM-DD is the
current date. The logs directory is created automatically if it doesn't exist.

Example Output
--------------
Console and file output format::

    2025-01-15 10:30:45,123 - src.agents.llm_base - INFO - Loading model gpt-4o-mini
    2025-01-15 10:30:46,456 - src.cli.runner - INFO - Processing scenario 123
"""

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

    This function configures either the root logger or a named logger with
    appropriate handlers for console and file output. It is safe to call
    multiple times - handlers are only added once to the root logger.

    Parameters
    ----------
    name
        Logger name. If None, configures and returns the root logger.
        For module loggers, pass __name__ to get hierarchical naming.
    level
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Defaults to LOG_LEVEL environment variable or INFO.
    format_string
        Custom format string for log messages.
        Default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use.

    Examples
    --------
    >>> # Configure root logger (typically done once at startup)
    >>> root_logger = setup_logger()
    >>> root_logger.info("Application started")

    >>> # Get a module-specific logger
    >>> module_logger = setup_logger(__name__)
    >>> module_logger.debug("Processing item %d", 42)

    >>> # Custom configuration
    >>> debug_logger = setup_logger("debug", level="DEBUG")
    >>> debug_logger.debug("Verbose output enabled")

    Notes
    -----
    - The root logger setup is idempotent - calling setup_logger() multiple
      times will not add duplicate handlers.
    - Named loggers propagate to root by default, so they inherit the root
      logger's handlers and formatting.
    - Log files are stored in ./logs/ with daily rotation (YYYY-MM-DD.log).
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Determine if we're configuring root or a named logger
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

            # File handler for persistent logs
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)

            # Console handler for immediate feedback
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(console_handler)

        logger.setLevel(getattr(logging, level, logging.INFO))
    else:
        # Named loggers: propagate to root, no direct handlers
        # Remove any stray handlers to avoid duplicate records
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.propagate = True
        logger.setLevel(getattr(logging, level, logging.INFO))

    return logger


# Default application logger instance
# This is pre-configured for immediate use in modules that import it
logger = setup_logger(__name__)


__all__ = [
    "setup_logger",
    "logger",
]
