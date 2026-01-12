"""
IO Module - Image/Video/Camera Loading
======================================

Functions for loading images, videos, and camera streams using OpenCV.
"""

import os
import cv2
import numpy as np


def load_source_cv2(source=None):
    """
    Load an image, a video, or a real-time camera stream using OpenCV.

    Parameters
    ----------
    source : str, int, or None
        - str (path): if ends with .jpg, .png, .jpeg, etc. -> image
                      if ends with .mp4, .avi, .mov, etc. -> video
        - int: camera index (0, 1, ...) for real-time camera
        - None: by default uses camera 0

    Returns
    -------
    source_type : str
        One of "image", "video", or "camera"
    resource : np.ndarray or cv2.VideoCapture
        - np.ndarray if image
        - cv2.VideoCapture if video or camera

    Raises
    ------
    RuntimeError
        If camera or video cannot be opened.
    FileNotFoundError
        If file path does not exist.
    ValueError
        If file extension is unknown or image cannot be read.
    TypeError
        If source parameter type is invalid.
    """
    # Default: open camera 0
    if source is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open default camera (index 0).")
        return "camera", cap

    # If source is an integer -> camera index
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {source}.")
        return "camera", cap

    # If source is a string -> assume file path
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File does not exist: {source}")

        extension = os.path.splitext(source)[1].lower()

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

        if extension in image_extensions:
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Could not read image: {source}")
            return "image", img

        elif extension in video_extensions:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {source}")
            return "video", cap

        else:
            raise ValueError(f"Unknown file extension: {extension}")

    raise TypeError("Parameter 'source' must be None, int, or str.")
