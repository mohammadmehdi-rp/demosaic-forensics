"""
Utility functions: image loading, visualization helpers, etc.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """
    Load image in BGR format using OpenCV.

    Raises FileNotFoundError if the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {p}")
    return img
