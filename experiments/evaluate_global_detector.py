import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.demosaic_detector import compute_global_score, classify_image  # noqa


DATA_ROOT = ROOT / "data" / "dataset"

def iter_images():
    """
    Yield (path, label_int, label_str) for all images.

    label_int: 1 for camera, 0 for forged
    """
    # camera images
    for cam_dir in sorted(DATA_ROOT.glob("flat-camera-*")):
        if not cam_dir.is_dir():
            continue
        for img_path in sorted(cam_dir.glob("*.tif")):
            yield img_path, 1, "camera"

    # forged images: all dev_*.(tif|tiff|jpg|jpeg)
    forged_dir = DATA_ROOT / "dev-dataset-forged"
    if forged_dir.is_dir():
        exts = ("tif", "tiff", "jpg", "jpeg")
        paths = []
        for ext in exts:
            paths.extend(sorted(forged_dir.glob(f"dev_*.{ext}")))
        paths = sorted(paths, key=lambda p: p.stem)

        for img_path in paths:
            yield img_path, 0, "forged"



def main(threshold: float):
    paths = []
    labels = []
    scores = []
    preds = []

    for img_path, label_int, label_str in iter_images():
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: could not read {img_path}")
            continue

        s = compute_global_score(img)
        label_pred_str = classify_image(s, threshold=threshold)
        pred_int = 1 if label_pred_str == "camera-like" else 0

        paths.append(str(img_path))
        labels.append(label_int)
        scores.append(s)
        preds.append(pred_int)

        print(f"{img_path.name:20s}  "
              f"score={s:7.3f}  "
              f"true={'camera' if label_int==1 else 'forged':7s}  "
              f"pred={'camera' if pred_int==1 else 'forged':7s}")

    if not paths:
        print("No images processed – check DATA_ROOT.")
        return

    labels = np.array(labels, dtype=int)
    preds = np.array(preds, dtype=int)

    # confusion matrix
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (preds == labels).mean()
    # camera = positive class
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n=== Evaluation with fixed threshold "
          f"{threshold:.3f} ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% (camera class)")
    print(f"Recall:    {recall*100:.2f}% (camera class)")
    print(f"F1-score:  {f1*100:.2f}% (camera class)")
    print("Confusion matrix (camera=1, forged=0):")
    print(f"  TP={tp}, FN={fn}")
    print(f"  FP={fp}, TN={tn}")

    # Save a tiny summary text file for the report
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "global_eval_summary.txt").open("w") as f:
        f.write(f"Threshold: {threshold:.3f}\n")
        f.write(f"Accuracy:  {acc*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}% (camera)\n")
        f.write(f"Recall:    {recall*100:.2f}% (camera)\n")
        f.write(f"F1-score:  {f1*100:.2f}% (camera)\n")
        f.write(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.224, 
        help="Global demosaicing score threshold",
    )
    args = parser.parse_args()
    main(args.threshold)
