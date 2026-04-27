# Shared logging and filesystem utilities.
import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    # Return a configured logger for the given module name.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return logging.getLogger(name)


def ensure_dirs(*paths: Path) -> None:
    # Create all given directories if they don't already exist.
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
