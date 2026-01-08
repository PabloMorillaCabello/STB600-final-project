"""
GUI Base Module - Base Classes and Utilities
============================================

Provides base classes and utility functions for the GUI components.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# Display size limits
MAX_WIDTH = 800
MAX_HEIGHT = 800


def resize_for_view(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Resize an image to fit within display dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    max_width : int
        Maximum display width.
    max_height : int
        Maximum display height.

    Returns
    -------
    np.ndarray
        Resized image (or original if already fits).
    """
    h, w = img.shape[:2]
    if w <= max_width and h <= max_height:
        return img
    scale_w = max_width / float(w)
    scale_h = max_height / float(h)
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cv_bgr_to_tk(img_bgr):
    """
    Convert OpenCV BGR image to Tkinter-compatible PhotoImage.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image from OpenCV.

    Returns
    -------
    ImageTk.PhotoImage
        Tkinter-compatible image.
    """
    img_bgr = resize_for_view(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(im_pil)


class BaseTab(ttk.Frame):
    """
    Base class for all pipeline tab views.

    Provides a standard layout with a title, controls panel, and image display area.

    Parameters
    ----------
    parent : ttk.Notebook
        Parent notebook widget.
    app : PipelineApp
        Reference to the main application.
    title : str
        Title displayed at the top of the tab.
    """

    def __init__(self, parent, app, title=""):
        super().__init__(parent)
        self.app = app

        # Layout configuration
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        # Title label
        self.title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Controls frame (left side)
        self.controls = ttk.Frame(self)
        self.controls.grid(row=1, column=0, sticky="nsw", padx=5, pady=5)
        self.controls.columnconfigure(0, weight=1)

        # Image display label (right side)
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Reference to keep Tkinter image alive
        self.tk_img = None

    def set_image(self, img_bgr):
        """
        Display a BGR image in the tab's image area.

        Parameters
        ----------
        img_bgr : np.ndarray
            BGR image to display.
        """
        if img_bgr is None:
            return
        self.tk_img = cv_bgr_to_tk(img_bgr)
        self.image_label.image = self.tk_img
        self.image_label.config(image=self.tk_img)

    def update_image(self):
        """
        Update the displayed image. Override in subclasses.
        """
        pass
