"""
FCLGA GraphTransformer - Configuration Module

This module handles configuration management for the FCLGA GraphTransformer pipeline.
Loads defaults from YAML, allows CLI overrides.
"""

__version__ = "1.0.0"

from pathlib import Path

import yaml

# Import path management
# Import domain constants
from . import constants, paths


def load_config(config_path=None):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses defaults.yaml

    Returns:
        dict: Configuration dictionary with all parameters
    """
    if config_path is None:
        config_path = Path(__file__).parent / "defaults.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_default(section, key, default=None):
    """
    Get a default value from the YAML config.

    Args:
        section: Config section (e.g., 'model', 'training')
        key: Parameter name
        default: Fallback value if not found

    Returns:
        Configuration value or default
    """
    try:
        config = load_config()
        return config.get(section, {}).get(key, default)
    except Exception:
        return default


__all__ = [
    "load_config",
    "get_default",
    "paths",
    "constants",
]
