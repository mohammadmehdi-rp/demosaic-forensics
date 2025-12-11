"""
Local demosaicing-trace analysis for forgery localization.

Produces a coarse map of demosaicing strength, then upsamples it to the
full image size. High values => strong demosaicing trace (likely genuine),
low values => weak trace (suspicious / possibly manipulated).
"""

from typing import List

import cv2
import numpy as np

from .demosaic_detector import _highpass_green, _diagonal_variance_signal


def _score_from_patch(patch: np.ndarray) -> float:
    """
    Compute Gallagher-style score s for a *high-pass absolute* patch.

    Parameters
    ----------
    patch : np.ndarray
        2D array (H, W) of high-pass absolute values.

    Returns
    -------
    float
        Local score s for this patch.
    """
    m = _diagonal_variance_signal(patch)
    N = m.size
    if N < 4:
        return 0.0

    M = np.fft.fft(m)
    S = np.abs(M)

    peak = S[N // 2]
    spectrum_for_median = S[1:]  # ignore DC
    median_ref = np.median(spectrum_for_median)
    eps = 1e-8
    return float(peak / (median_ref + eps))


def _make_starts(length: int, window_size: int, step: int) -> List[int]:
    """
    Compute start indices for sliding windows, making sure we cover the
    whole dimension and include the end region.
    """
    if length <= window_size:
        return [0]

    starts = list(range(0, length - window_size + 1, step))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def compute_local_map(
    img_bgr: np.ndarray,
    window_size: int = 64,
    step: int = 16,
) -> np.ndarray:
    """
    Compute local demosaicing score map for a BGR (or grayscale) image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR or grayscale format.
    window_size : int
        Size of the square window used for local analysis.
    step : int
        Step between windows (in pixels). Larger step -> faster, coarser map.

    Returns
    -------
    np.ndarray
        2D float array of size (H, W) with local scores.
        Higher values => stronger demosaicing trace.
    """
    # High-pass absolute image on green info
    O = _highpass_green(img_bgr)
    H, W = O.shape

    y_starts = _make_starts(H, window_size, step)
    x_starts = _make_starts(W, window_size, step)

    map_small = np.zeros((len(y_starts), len(x_starts)), dtype=np.float64)

    for iy, y0 in enumerate(y_starts):
        for ix, x0 in enumerate(x_starts):
            patch = O[y0 : y0 + window_size, x0 : x0 + window_size]
            s_local = _score_from_patch(patch)
            map_small[iy, ix] = s_local

    # Upsample coarse grid to full resolution
    local_map = cv2.resize(
        map_small,
        (W, H),
        interpolation=cv2.INTER_CUBIC,
    )

    return local_map

def suspicious_map_and_mask(
    img_bgr: np.ndarray,
    window_size: int = 64,
    step: int = 16,
    threshold_quantile: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute suspiciousness map and a binary mask of most suspicious regions.

    Steps:
    - compute local demosaicing strength map,
    - normalize and invert (low strength -> high suspiciousness),
    - threshold suspiciousness at a given quantile,
    - clean up the mask.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image (BGR or grayscale).
    window_size : int
        Local analysis window size.
    step : int
        Step between windows.
    threshold_quantile : float
        Quantile in [0,1]; e.g. 0.85 means we mark the top 15% most
        suspicious pixels as manipulated.

    Returns
    -------
    suspicious : np.ndarray
        2D float32 array in [0,1], high means more suspicious.
    mask : np.ndarray
        2D uint8 array in {0,1}, binary suspicious-region mask.
    """
    local_map = compute_local_map(img_bgr, window_size=window_size, step=step)

    # normalize local_map and invert to get suspiciousness
    local_min = float(local_map.min())
    local_ptp = float(local_map.max() - local_min)
    if local_ptp < 1e-8:
        suspicious = np.zeros_like(local_map, dtype=np.float32)
    else:
        norm = (local_map - local_min) / local_ptp
        suspicious = 1.0 - norm  # low demosaicing => high suspiciousness
        suspicious = suspicious.astype(np.float32)

    # threshold at quantile
    flat = suspicious.flatten()
    t = np.quantile(flat, threshold_quantile)
    mask = (suspicious >= t).astype(np.uint8)

    # simple morphological cleanup (remove tiny specks, smooth edges)
    mask_255 = (mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = (mask_255 > 0).astype(np.uint8)

    return suspicious, mask
