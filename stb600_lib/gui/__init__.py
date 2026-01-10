"""
GUI Module - Interactive Tkinter Applications
=============================================

Provides interactive GUI tools for parameter tuning and real-time processing.

Classes:
    - BaseTab: Base class for all pipeline tab views
    - PipelineApp: Main multi-tab pipeline application
    - ResultViewerApp: Simple result viewer with automatic pipeline
    - VideoProcessorApp: Video/camera processor with object tracking
    - Various Tab classes for each processing step

Functions:
    - run_pipeline_gui: Quick launcher for the pipeline GUI
    - run_result_viewer: Quick launcher for the result viewer
    - run_video_processor: Quick launcher for video processor
    - load_config, save_config: Config file management
"""

from .base import BaseTab, cv_bgr_to_tk, resize_for_view
from .app import PipelineApp, run_pipeline_gui
from .result_viewer import ResultViewerApp, run_result_viewer
from .video_processor import VideoProcessorApp, run_video_processor
from .config import load_config, save_config, DEFAULT_CONFIG_PATH
from .tabs import (
    InputTab,
    RemoveGreenTab,
    BinarizeTab,
    OpeningTab,
    BigContoursTab,
    CroppedPartTab,
    InnerColorPartsTab,
    LabeledPartsTab,
    ROIsTab,
)

__all__ = [
    # Base utilities
    "BaseTab",
    "cv_bgr_to_tk",
    "resize_for_view",
    # Main app
    "PipelineApp",
    "run_pipeline_gui",
    # Result viewer
    "ResultViewerApp",
    "run_result_viewer",
    # Video processor
    "VideoProcessorApp",
    "run_video_processor",
    # Config
    "load_config",
    "save_config",
    "DEFAULT_CONFIG_PATH",
    # Tabs
    "InputTab",
    "RemoveGreenTab",
    "BinarizeTab",
    "OpeningTab",
    "BigContoursTab",
    "CroppedPartTab",
    "InnerColorPartsTab",
    "LabeledPartsTab",
    "ROIsTab",
]
