# experiments/visualize_local.py

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.local_detector import suspicious_map_and_mask  # noqa: E402


DATA_ROOT = ROOT / "data" / "dataset" / "dev-dataset-forged"


def process_image(img_path: Path, out_dir: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not read {img_path}")
        return

    # ensure 3-channel RGB for display
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    suspicious, mask = suspicious_map_and_mask(img)

    # suspicious in [0,1] -> uint8 for colormap
    susp_uint8 = (255.0 * suspicious).astype(np.uint8)
    susp_uint8 = cv2.GaussianBlur(susp_uint8, (0, 0), 1.0)

    heatmap_bgr = cv2.applyColorMap(susp_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # overlay heatmap
    overlay_heat = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # overlay binary mask: red regions on top of original
    mask_color = np.zeros_like(img_rgb)
    mask_color[..., 0] = 255  # red channel
    mask_expanded = (mask[:, :, None] * mask_color).astype(np.uint8)
    overlay_mask = cv2.addWeighted(img_rgb, 0.7, mask_expanded, 0.3, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = img_path.stem

    cv2.imwrite(
        str(out_dir / f"{base}_orig.png"),
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / f"{base}_heatmap.png"),
        cv2.cvtColor(overlay_heat, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / f"{base}_mask.png"),
        cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR),
    )

    print(f"Saved visualizations for {img_path.name}")


def main():
    out_dir = ROOT / "experiments" / "results" / "local_maps"
    
    exts = ("tif", "tiff", "jpg", "jpeg")
    paths = []
    for ext in exts:
        paths.extend(sorted(DATA_ROOT.glob(f"dev_*.{ext}")))
    paths = sorted(paths, key=lambda p: p.stem)

    if not paths:
        print(f"No forged images found in {DATA_ROOT}")
        return

    # You can limit to a subset if there are many
    for img_path in paths[:20]:
        process_image(img_path, out_dir)


if __name__ == "__main__":
    main()
