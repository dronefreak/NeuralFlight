"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Dictionary with configuration
    """
    # Try multiple possible config locations
    possible_paths = [
        Path("config") / f"{config_name}.yaml",
        Path(__file__).parent.parent.parent.parent / "config" / f"{config_name}.yaml",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"Could not find config file: {config_name}.yaml")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find project root
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()
