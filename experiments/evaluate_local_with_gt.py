"""
Evaluate local tampering localization using the ground-truth maps in
data/dataset/dev-dataset-maps.

For each forged image dev_XXXX.(tif/jpg) and corresponding map dev_XXXX.bmp:
- run suspicious_map_and_mask() to get the predicted mask,
- compare with the ground-truth mask,
- compute IoU, precision, recall, F1,
- save per-image results to CSV and print dataset averages.
"""

import sys
from pathlib import Path
import csv

import numpy as np
import cv2

# Make src/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.local_detector import suspicious_map_and_mask  # noqa: E402


DATA_ROOT = ROOT / "data" / "dataset"
FORGED_DIR = DATA_ROOT / "dev-dataset-forged"
GT_DIR = DATA_ROOT / "dev-dataset-maps"


def find_forged_image(stem: str) -> Path | None:
    """
    Given a stem like 'dev_0001', find the corresponding forged image
    in dev-dataset-forged with any supported extension.
    """
    exts = ("tif", "tiff", "jpg", "jpeg", "png", "bmp")
    candidates: list[Path] = []
    for ext in exts:
        candidates.extend(FORGED_DIR.glob(f"{stem}.{ext}"))

    if not candidates:
        return None

    # If multiple exist, pick the first in sorted order
    return sorted(candidates)[0]


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute IoU, precision, recall, F1 and confusion matrix entries
    for a single image.

    pred_mask, gt_mask: binary arrays (0/1) of the same shape.
    """
    pred = pred_mask.astype(bool).ravel()
    gt = gt_mask.astype(bool).ravel()

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    denom_iou = tp + fp + fn
    iou = tp / (denom_iou + eps)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def main():
    if not GT_DIR.is_dir():
        print(f"Ground-truth directory not found: {GT_DIR}")
        return

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "local_eval_with_gt.csv"

    per_image: list[dict] = []
    all_tp = all_tn = all_fp = all_fn = 0

    gt_paths = sorted(GT_DIR.glob("dev_*.bmp"))
    if not gt_paths:
        print(f"No ground-truth maps found in {GT_DIR}")
        return
    
        # Limit evaluation to at most 100 images
    max_images = 100
    gt_paths = gt_paths[:max_images]
    print(f"Evaluating on {len(gt_paths)} image/GT pairs (max {max_images}).")

    for gt_path in gt_paths:
        stem = gt_path.stem  # e.g. 'dev_0001'
        img_path = find_forged_image(stem)
        if img_path is None:
            print(f"[WARN] No forged image found for {stem}; skipping.")
            continue

        # Load forged image (original resolution)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read forged image {img_path}; skipping.")
            continue

        # Run our local detector
        suspicious, pred_mask = suspicious_map_and_mask(img)
        pred_mask = (pred_mask > 0).astype(np.uint8)

        # Load ground-truth mask
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[WARN] Could not read ground-truth map {gt_path}; skipping.")
            continue
        gt_mask = (gt > 0).astype(np.uint8)

        # If sizes differ, resize prediction to ground-truth size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compute metrics
        m = compute_metrics(pred_mask, gt_mask)
        all_tp += m["tp"]
        all_tn += m["tn"]
        all_fp += m["fp"]
        all_fn += m["fn"]

        per_image.append(
            {
                "image": stem,
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "iou": m["iou"],
            }
        )

        print(
            f"{stem}: IoU={m['iou']:.3f}, "
            f"P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}"
        )

    if not per_image:
        print("No image/GT pairs evaluated.")
        return

    # Save per-image CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "tp",
                "tn",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "iou",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image)

    # Compute dataset-level averages (macro)
    precisions = [r["precision"] for r in per_image]
    recalls = [r["recall"] for r in per_image]
    f1s = [r["f1"] for r in per_image]
    ious = [r["iou"] for r in per_image]

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    macro_iou = float(np.mean(ious))

    # Also aggregated micro metrics
    eps = 1e-8
    micro_precision = all_tp / (all_tp + all_fp + eps)
    micro_recall = all_tp / (all_tp + all_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    micro_iou = all_tp / (all_tp + all_fp + all_fn + eps)

    print("\n=== Local localization evaluation (macro averages) ===")
    print(f"Mean IoU:      {macro_iou:.3f}")
    print(f"Mean Precision:{macro_precision:.3f}")
    print(f"Mean Recall:   {macro_recall:.3f}")
    print(f"Mean F1-score: {macro_f1:.3f}")

    print("\n=== Local localization evaluation (micro over all pixels) ===")
    print(f"IoU:           {micro_iou:.3f}")
    print(f"Precision:     {micro_precision:.3f}")
    print(f"Recall:        {micro_recall:.3f}")
    print(f"F1-score:      {micro_f1:.3f}")

    print(f"\nPer-image results saved to: {out_csv}")


if __name__ == "__main__":
    main()
