import sys
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Make sure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.demosaic_detector import compute_global_score  # noqa


DATA_ROOT = ROOT / "data" / "dataset"


def iter_camera_images():
    """
    Yield (path, label) for genuine camera images from flat-camera-* folders.
    """
    for cam_dir in sorted(DATA_ROOT.glob("flat-camera-*")):
        if not cam_dir.is_dir():
            continue
        for img_path in sorted(cam_dir.glob("*.tif")):
            yield img_path, "camera"

def iter_forged_images():
    """
    Yield (path, label) for forged images from dev-dataset-forged.
    Now supports both .tif and .jpg.
    """
    forged_dir = DATA_ROOT / "dev-dataset-forged"
    if not forged_dir.is_dir():
        return

    exts = ("tif", "tiff", "jpg", "jpeg")
    paths = []
    for ext in exts:
        paths.extend(sorted(forged_dir.glob(f"dev_*.{ext}")))
    # sort by stem (dev_0001, dev_0002, …) so order is stable
    paths = sorted(paths, key=lambda p: p.stem)

    for img_path in paths:
        yield img_path, "forged"


def main():
    results = []

    # 1) process camera images
    for img_path, label in list(iter_camera_images()):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: could not read {img_path}")
            continue

        score = compute_global_score(img)
        results.append({"path": str(img_path), "label": label, "score": score})
        print(f"[camera] {img_path.name}: {score:.3f}")

    # 2) process forged images
    for img_path, label in list(iter_forged_images()):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: could not read {img_path}")
            continue

        score = compute_global_score(img)
        results.append({"path": str(img_path), "label": label, "score": score})
        print(f"[forged] {img_path.name}: {score:.3f}")

    if not results:
        print("No images processed. Check DATA_ROOT and folder structure.")
        return

    # 3) save to CSV
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "global_scores.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved scores to {out_csv}")

    # 4) simple stats & threshold search
    scores = np.array([r["score"] for r in results], dtype=float)
    labels = np.array([1 if r["label"] == "camera" else 0 for r in results],
                      dtype=int)

    cam_scores = scores[labels == 1]
    forg_scores = scores[labels == 0]

    print("\n=== Summary statistics ===")
    print(f"Camera images: n={len(cam_scores)}, "
          f"mean={cam_scores.mean():.3f}, "
          f"std={cam_scores.std():.3f}, "
          f"min={cam_scores.min():.3f}, "
          f"max={cam_scores.max():.3f}")

    if len(forg_scores) > 0:
        print(f"Forged images: n={len(forg_scores)}, "
              f"mean={forg_scores.mean():.3f}, "
              f"std={forg_scores.std():.3f}, "
              f"min={forg_scores.min():.3f}, "
              f"max={forg_scores.max():.3f}")
    else:
        print("No forged images found; skipping threshold tuning.")
        return

    # 5) search for best threshold between min and max scores
    t_min = float(scores.min())
    t_max = float(scores.max())

    thresholds = np.linspace(t_min, t_max, 200)
    best_acc = -1.0
    best_t = None

    for t in thresholds:
        preds = (scores > t).astype(int)  # 1 = camera, 0 = forged
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t

    print("\n=== Threshold tuning ===")
    print(f"Best threshold (by simple accuracy): {best_t:.3f}")
    print(f"Accuracy at best threshold: {best_acc*100:.2f}%")

    # Optional: confusion matrix at best threshold
    preds = (scores > best_t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    print("Confusion matrix (label: camera=1, forged=0):")
    print(f"  TP={tp}, FN={fn}")
    print(f"  FP={fp}, TN={tn}")

    plt.figure()
    plt.hist(cam_scores, bins=20, alpha=0.5, label="camera")
    plt.hist(forg_scores, bins=20, alpha=0.5, label="forged")
    plt.axvline(best_t, linestyle="--", label=f"threshold={best_t:.2f}")
    plt.xlabel("Global demosaicing score s")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Score distribution: camera vs forged")
    plt.show()

if __name__ == "__main__":
    main()
