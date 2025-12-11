"""
Gradio interface for the demosaicing-pattern inconsistency detector.
Uses a File input so we can handle 16-bit TIFFs with OpenCV instead of Pillow.
"""

import gradio as gr
import numpy as np
import cv2
from typing import Any

import tifffile 

from .demosaic_detector import compute_global_score, classify_image
from .local_detector import suspicious_map_and_mask


def _file_to_path(file_obj: Any) -> str:
    """
    Extract filesystem path from the various file formats Gradio can give us.
    """
    # Gradio v4 sometimes passes a dict with a 'name' field
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]

    # Or a file-like with .name
    if hasattr(file_obj, "name"):
        return file_obj.name

    # Or just a plain string path
    if isinstance(file_obj, str):
        return file_obj

    raise ValueError(f"Unsupported file object type: {type(file_obj)}")


def analyze_file(file_obj: Any):
    """
    Main Gradio callback.

    Input: file uploaded by the user (TIFF/PNG/JPG/etc).
    We read it with OpenCV (cv2.imread) so 16-bit TIFFs are supported.
    """
    if file_obj is None:
        return {"error": "No file uploaded"}, None, None, None

    try:
        path = _file_to_path(file_obj)
    except Exception as e:
        return {"error": f"Could not get file path: {e}"}, None, None, None

     # --- Read image: try OpenCV first, then fall back to tifffile ---
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            img = tifffile.imread(path)
        except Exception as e:
            return {
                "error": (
                    "Cannot read image with OpenCV or tifffile:\n"
                    f"{path}\nException: {e}"
                )
            }, None, None, None

    # This 'img' is used for all analysis (BGR or grayscale)
    img_for_algo = img

    # Prepare a nice 8-bit RGB version for display
    if img.ndim == 2:  # grayscale
        disp = img.astype(np.float32)
    else:
        # Convert BGR to RGB
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Handle higher bit-depths: normalize to 0-255 uint8
    if disp.dtype != np.uint8:
        min_val = disp.min()
        max_val = disp.max()
        if max_val > min_val:
            disp = (255.0 * (disp - min_val) / (max_val - min_val)).astype(
                np.uint8
            )
        else:
            disp = np.zeros_like(disp, dtype=np.uint8)

    # Ensure 3-channel RGB for display
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)

    # ---- Global score & classification ----
    score = compute_global_score(img_for_algo)
    label = classify_image(score)

    # ---- Local suspiciousness & mask ----
    suspicious, mask = suspicious_map_and_mask(img_for_algo)

    # suspicious is in [0,1]; convert to uint8 and smooth a bit
    susp_uint8 = (255.0 * suspicious).astype(np.uint8)
    susp_uint8 = cv2.GaussianBlur(susp_uint8, (0, 0), 1.0)

    heatmap_bgr = cv2.applyColorMap(susp_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay mask in red on top of original
    mask_color = np.zeros_like(disp)
    mask_color[..., 0] = 255  # red channel
    mask_expanded = (mask[:, :, None] * mask_color).astype(np.uint8)
    overlay_mask = cv2.addWeighted(disp, 0.7, mask_expanded, 0.3, 0)

    return (
        {"label": label, "score": float(score)},
        disp,
        heatmap_rgb,
        overlay_mask,
    )


def main():
    print("[gradio_app] Creating interface...")
    demo = gr.Interface(
        fn=analyze_file,
        inputs=gr.File(
            label="Input image (.tif, .png, .jpg, ...)",
            file_types=["image"],
        ),
        outputs=[
            gr.JSON(label="Global decision"),
            gr.Image(label="Original (normalized)"),
            gr.Image(label="Suspiciousness heatmap"),
            gr.Image(label="Binary mask overlay"),
        ],
        title="Demosaicing Pattern Inconsistency Detector",
        description=(
            "Upload a camera or forged image (TIFF/PNG/JPG). "
            "The app computes a global demosaicing score and "
            "highlights locally suspicious regions."
        ),
    )

    print("[gradio_app] Launching Gradio app...")
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    print("[gradio_app] __main__ reached, calling main()")
    main()
