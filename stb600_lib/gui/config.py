"""
GUI Config Module - JSON Configuration Management
=================================================

Handles loading, saving configuration with shared settings and per-color ROIs.
Structure:
  - "shared": settings that apply to all colors
  - "rois": per-color ROI settings (blue, red, yellow)
"""

import os
import json

# Default config file path
DEFAULT_CONFIG_PATH = "pipeline_config.json"

# Default config structure: shared settings + per-color ROIs
DEFAULT_CONFIG = {
    "shared": {
        "remove_green": {"h_low": 35, "h_high": 85},
        "binarize": {"thresh_value": 0},
        "opening": {"kernel_size": 5, "iterations": 1},
        "big_contours": {
            "red_min": 5000, "red_max": 80000,
            "yellow_min": 80000, "yellow_max": 160000,
            "blue_min": 160000, "blue_max": 250000,
        },
        "inner_parts": {"min_area": 50, "max_area": 10000},
        "labeled": {"extent_threshold": 0.5, "small_area_max": 2100, "medium_area_max": 5600},
    },
    "rois": {
        "blue": {"num_rois": 3, "offset_pct": 0, "height_pct": 77},
        "red": {"num_rois": 3, "offset_pct": 0, "height_pct": 77},
        "yellow": {"num_rois": 3, "offset_pct": 0, "height_pct": 100},
    },
}


def load_config(path=DEFAULT_CONFIG_PATH):
    """
    Load configuration from JSON file.

    Parameters
    ----------
    path : str
        Path to config file.

    Returns
    -------
    dict
        Configuration dictionary with shared settings and per-color ROIs.
    """
    if not os.path.exists(path):
        # Create default config file
        save_config(DEFAULT_CONFIG, path)
        return _deep_copy(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Ensure structure exists
        if "shared" not in config:
            config["shared"] = _deep_copy(DEFAULT_CONFIG["shared"])
        if "rois" not in config:
            config["rois"] = _deep_copy(DEFAULT_CONFIG["rois"])

        # Ensure all colors have ROI settings
        for color in ["blue", "red", "yellow"]:
            if color not in config["rois"]:
                config["rois"][color] = _deep_copy(DEFAULT_CONFIG["rois"].get(color, DEFAULT_CONFIG["rois"]["blue"]))

        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return _deep_copy(DEFAULT_CONFIG)


def save_config(config, path=DEFAULT_CONFIG_PATH):
    """
    Save configuration to JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    path : str
        Path to config file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


def get_color_config(config, color):
    """
    Get combined configuration for a specific color.

    Merges shared settings with color-specific ROI settings.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    color : str
        Color name ('blue', 'red', 'yellow').

    Returns
    -------
    dict
        Combined configuration with all settings for the color.
    """
    color = color.lower()
    result = {}

    # Copy shared settings
    shared = config.get("shared", DEFAULT_CONFIG["shared"])
    for section, values in shared.items():
        result[section] = values.copy() if isinstance(values, dict) else values

    # Add color-specific ROIs
    rois = config.get("rois", DEFAULT_CONFIG["rois"])
    if color in rois:
        result["rois"] = rois[color].copy()
    else:
        result["rois"] = DEFAULT_CONFIG["rois"].get(color, DEFAULT_CONFIG["rois"]["blue"]).copy()

    return result


def update_color_config(config, color, section, values):
    """
    Update configuration for a section.

    ROI settings are saved per-color, all other settings go to shared.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    color : str
        Color name (used only for ROI updates).
    section : str
        Section name (e.g., 'remove_green', 'binarize', 'rois', etc.).
    values : dict
        Values to update in the section.
    """
    color = color.lower()

    if section == "rois":
        # ROIs are per-color
        if "rois" not in config:
            config["rois"] = _deep_copy(DEFAULT_CONFIG["rois"])
        if color not in config["rois"]:
            config["rois"][color] = {}
        config["rois"][color].update(values)
    else:
        # All other settings are shared
        if "shared" not in config:
            config["shared"] = {}
        if section not in config["shared"]:
            config["shared"][section] = {}
        config["shared"][section].update(values)


def _deep_copy(obj):
    """Create a deep copy of nested dicts."""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    return obj
