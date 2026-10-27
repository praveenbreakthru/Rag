"""
Logging configuration for Sakthi Finance RAG system.
Outputs readable console logs via Rich and persistent structured logs to file.
"""

import logging
import os
from pathlib import Path
from rich.logging import RichHandler
from ragcli.config import settings


def setup_logging():
    log_dir = Path(settings.log_file).parent
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handlers = [
        RichHandler(rich_tracebacks=True, show_time=True, show_path=False),
        logging.FileHandler(settings.log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
