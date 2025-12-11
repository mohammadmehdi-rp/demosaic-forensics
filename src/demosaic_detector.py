"""
Global demosaicing-trace detector.

Implements a Gallagher & Chen-style method that:
- takes an RGB/BGR image,
- extracts the green channel,
- applies a high-pass filter,
- builds the diagonal variance signal,
- computes the spectral peak ratio at f = 0.5.
"""

import numpy as np
import cv2
from typing import Tuple


def _highpass_green(img: np.ndarray) -> np.ndarray:
    """
    Apply 3x3 high-pass filter to the 'green' information of an image.

    - If image has 3 channels (BGR), use the green channel.
    - If image has 4 channels (BGRA), drop alpha and use green.
    - If image is grayscale, use it directly.
    """
    if img.ndim == 2:  # grayscale
        G = img.astype(np.float64)

    elif img.ndim == 3 and img.shape[2] == 3:  # BGR
        G = img[..., 1].astype(np.float64)

    elif img.ndim == 3 and img.shape[2] == 4:  # BGRA
        # Ignore alpha, use green channel
        G = img[..., 1].astype(np.float64)

        # (alternative would be: img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        #  then take green from that, but it's equivalent for our purposes.)

    else:
        raise ValueError(
            "Unsupported image shape for _highpass_green: "
            f"{img.shape}"
        )

    kernel = np.array(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]],
        dtype=np.float64,
    )

    filtered = cv2.filter2D(
        G,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT,
    )

    O = np.abs(filtered)
    return O


def _diagonal_variance_signal(O: np.ndarray) -> np.ndarray:
    """
    Build the 1D 'diagonal variance' signal m(d) from a 2D image O,
    where each entry is the mean absolute value over one diagonal x + y = d.

    Parameters
    ----------
    O : np.ndarray
        2D array (H, W), e.g. high-pass absolute image.

    Returns
    -------
    np.ndarray
        1D array m of length H + W - 1 (float64).
    """
    H, W = O.shape
    num_diagonals = H + W - 1
    m = np.zeros(num_diagonals, dtype=np.float64)
    counts = np.zeros(num_diagonals, dtype=np.int32)

    # Loop over all pixels once and accumulate by d = x + y
    # y: row index (0..H-1), x: column index (0..W-1)
    for y in range(H):
        for x in range(W):
            d = x + y
            m[d] += O[y, x]
            counts[d] += 1

    # Avoid division by zero (shouldn't happen but just in case)
    valid = counts > 0
    m[valid] /= counts[valid]

    return m


def compute_global_score(img_bgr: np.ndarray) -> float:
    """
    Compute the global demosaicing score s for a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (as read by cv2).

    Returns
    -------
    float
        Global peak-ratio score s. Larger means stronger demosaicing trace.
    """
    # 1) High-pass filtered absolute image on green channel
    O = _highpass_green(img_bgr)

    # 2) Diagonal 'variance' signal m(d)
    m = _diagonal_variance_signal(O)

    # 3) FFT of m
    M = np.fft.fft(m)
    S = np.abs(M)
    N = S.size

    if N < 4:
        # Too short to be meaningful; return a very small score
        return 0.0

    # 4) Peak at normalized frequency 0.5 -> index N//2
    peak = S[N // 2]

    # 5) Median of spectrum excluding DC component (index 0)
    # (you can also exclude index N//2 itself from the median if you want)
    spectrum_for_median = S[1:]  # ignore DC
    median_ref = np.median(spectrum_for_median)

    # 6) Ratio
    eps = 1e-8
    score = float(peak / (median_ref + eps))
    return score


def classify_image(score: float, threshold: float = 25.224) -> str:
    """
    Classify image based on global score.

    Parameters
    ----------
    score : float
        Global demosaicing score s.
    threshold : float
        Decision threshold; larger => more likely demosaiced.

    Returns
    -------
    str
        'camera-like' if score > threshold else 'suspicious'.
    """
    return "camera-like" if score > threshold else "suspicious"


if __name__ == "__main__":
    # Simple manual test: change the path to an image on your machine
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m src.demosaic_detector path/to/image.jpg")
        sys.exit(0)

    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {img_path}")

    s = compute_global_score(img)
    label = classify_image(s)
    print(f"Score: {s:.3f}, label: {label}")
    
