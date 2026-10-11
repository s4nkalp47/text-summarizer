import os
from pathlib import Path

import yaml
from box import ConfigBox

from textSummarizer.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Load a YAML file into a ConfigBox, which allows attribute access (cfg.a.b)."""
    with open(path_to_yaml, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if content is None:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")
    logger.info(f"Loaded YAML file: {path_to_yaml}")
    return ConfigBox(content)


def create_directories(paths: list[Path], verbose: bool = True) -> None:
    """Create each directory (and its parents). Existing directories are left alone."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")
