# Demosaicing Pattern Inconsistency Detector

Implementation of a **demosaicing-based image forensics pipeline** for detecting forged images and localizing manipulated regions.  

The project follows the methods described in the demosaicing-trace works referenced in the course (Link2) and is tailored to the provided **flat-camera** and **dev-dataset** image sets.

Author: **Mohammadmehdi Rajabpourshirazy**  
Academic project – Image Forensics / Multimedia Security

---

## 1. Overview

Digital cameras use a **Color Filter Array (CFA)** and a **demosaicing algorithm** to reconstruct full-color images.  
This process leaves a characteristic **periodic spatial pattern** in the image statistics, especially in the green channel.

This repository contains:

- A **global detector** that estimates the strength of demosaicing traces and classifies images as *camera-like* or *suspicious*.
- A **local detector** that finds **spatial inconsistencies** in demosaicing patterns and produces heatmaps and binary masks of suspicious regions.
- A set of **evaluation scripts** for:
  - global camera vs forged classification,
  - local localization against the ground-truth tampering masks.
- An interactive **Gradio web app** for visual exploration.

---

## 2. Repository Structure

```text
demosaic-forensics/
│
├── data/
│   └── dataset/
│       ├── flat-camera-1/
│       ├── flat-camera-2/
│       ├── flat-camera-3/
│       ├── flat-camera-4/
│       ├── dev-dataset-forged/   # forged images (TIFF + JPG)
│       └── dev-dataset-maps/     # ground-truth tampering masks (BMP)
│
├── experiments/
│   ├── tune_global_detector.py       # grid search for global threshold
│   ├── evaluate_global_detector.py   # global metrics & score CSV
│   ├── visualize_local.py            # local heatmaps & overlays for forged images
│   └── evaluate_local_with_gt.py     # local IoU / precision / recall vs GT maps
│   └── results/
│       ├── figures/                  # e.g. score histogram
│       ├── local_maps/               # local heatmaps & overlays
│       ├── global_scores.csv
│       └── global_eval_summary.txt
│
├── src/
│   ├── demosaic_detector.py   # global demosaicing score & classifier
│   ├── local_detector.py      # sliding-window local suspiciousness maps
│   ├── utils.py               # shared utilities
│   └── gradio_app.py          # Gradio web UI
│
├── requirements.txt
└── README.md
````

> **Note:** The dataset itself is **not included** in this repository for size/licensing reasons.
> Place the provided folders (`flat-camera-*`, `dev-dataset-forged`, `dev-dataset-maps`) under `data/dataset/` as shown above.

---

## 3. Installation

Tested with **Python 3.10** on Windows.

```bash
# Clone the repository
git clone <your-repo-url> demosaic-forensics
cd demosaic-forensics

# (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Required main packages:

* `numpy`
* `opencv-python`
* `tifffile` (robust TIFF reading, including 16-bit)
* `matplotlib`
* `gradio`
* `Pillow`, `scipy`, `tqdm` (utility deps)

---

## 4. Global Detector

### 4.1 Method Summary

For each image:

1. Extract **green channel** (or grayscale intensity).
2. Apply a 3×3 high-pass filter (Laplacian-like), take the absolute value.
3. Build the **diagonal variance signal** ( m(d) ) by averaging residuals along diagonals ( x + y = d ).
4. Compute the discrete Fourier transform ( M(\omega) ) of ( m(d) ).
5. Define the **global demosaicing score**

[
s = \frac{|M(\pi)|}{\text{median}(|M(\omega)|,\ \omega \neq 0)} ,
]

where frequency ( \pi ) corresponds to normalized frequency 0.5 (period 2).
6. Compare ( s ) with a tuned threshold ( t ):

* ( s > t )  → **camera-like**
* ( s \le t ) → **suspicious / non-demosaiced**

Implementation: `src/demosaic_detector.py`.

### 4.2 Tuning the Threshold

```bash
python experiments/tune_global_detector.py
```

This will:

* Compute scores for all images in `flat-camera-*` and `dev-dataset-forged`.
* Search over a grid of thresholds.
* Save:

  * `experiments/results/global_scores.csv`
  * `experiments/results/figures/Figure_1.png` (score histogram camera vs forged).

The tuned threshold used in this project is:

* **t = 25.224**

### 4.3 Evaluation

```bash
python experiments/evaluate_global_detector.py
```

Outputs:

* `experiments/results/global_eval_summary.txt` with metrics.
* Reuses `global_scores.csv` if present.

**Results (camera treated as positive class):**

* **Accuracy:** 78.42 %
* **Precision (camera):** 60.76 %
* **Recall (camera):** 99.50 %
* **F1-score (camera):** 75.45 %
* **Confusion matrix:**

  * TP = 398, FN = 2, FP = 257, TN = 543

Interpretation: the detector **almost never flags genuine camera images as suspicious** (very high recall), at the cost of some forged images still being classified as camera-like.

---

## 5. Local Detector & Localization

### 5.1 Method Summary

To localize potential tampering:

1. Compute the same high-pass residual image as in the global method.
2. Slide a window (e.g. 64×64) over the image with a step (e.g. 16 pixels).
3. For each window, compute the diagonal variance signal and local demosaicing score ( s(x,y) ).
4. Build a coarse local score map and upsample it to the full resolution.
5. Convert to a **suspiciousness map**:

   * normalize scores to ([0,1]),
   * invert: low demosaicing → high suspiciousness.
6. Threshold at a chosen **quantile** (e.g. top 15% most suspicious pixels) to get a binary mask.
7. Apply simple morphological open/close to remove tiny specks and smooth the regions.

Implementation: `src/local_detector.py` (`compute_local_map`, `suspicious_map_and_mask`).

### 5.2 Qualitative Visualization

```bash
python experiments/visualize_local.py
```

This script:

* Iterates over forged images in `dev-dataset-forged` (TIFF + JPG).
* Produces, for each selected image, three PNGs in `experiments/results/local_maps/`:

  * `<name>_orig.png` – normalized original image
  * `<name>_heatmap.png` – suspiciousness heatmap overlay
  * `<name>_mask.png` – binary suspicious region overlay

Use these for the report/presentation (success + failure examples).

### 5.3 Quantitative Localization with Ground Truth

```bash
python experiments/evaluate_local_with_gt.py
```

This script:

* Loads up to 100 ground-truth maps from `dev-dataset-maps/dev_XXXX.bmp`.

* Finds the corresponding forged image `dev_XXXX.(tif/jpg)` in `dev-dataset-forged`.

* Runs `suspicious_map_and_mask(img)` to get a predicted mask.

* Resizes the prediction to the GT size if needed.

* Computes per-image metrics:

  * **IoU** (Intersection over Union)
  * **precision**
  * **recall**
  * **F1-score**

* Saves all per-image results to:

  * `experiments/results/local_eval_with_gt.csv`

* Prints:

  * **Macro averages** (mean over images)
  * **Micro averages** (over all pixels)

Example macro results (for one configuration on 100 images):

* **Mean IoU:** 0.45
* **Mean Precision:** 0.68
* **Mean Recall:** 0.59
* **Mean F1-score:** 0.55

These numbers summarize how well the binary suspicious masks align with the true tampered regions.

---

## 6. Gradio Web Interface

The project includes a simple web interface built with **Gradio**.

### 6.1 Launching the App

From the project root:

```bash
python -m src.gradio_app
```

A local URL (e.g. `http://127.0.0.1:7860`) will be printed; open it in a browser.

### 6.2 Interface Description

**Input**

* `File` upload (`.tif`, `.tiff`, `.jpg`, `.jpeg`, `.png`)
* Internally uses `cv2.imread` with `tifffile` fallback → supports 16-bit TIFFs and 3/4-channel images.

**Outputs**

1. **Global decision (JSON)**

   * `label`: `"camera-like"` or `"suspicious"`
   * `score`: global demosaicing score ( s )
2. **Original (normalized)** – 8-bit RGB visualization of the input.
3. **Suspiciousness heatmap** – continuous map overlaid with colormap.
4. **Binary mask overlay** – red regions marking most suspicious pixels.

This tool is primarily intended for qualitative analysis and demonstration.

---

## 7. Reproducibility Checklist

1. Place the dataset under `data/dataset/` with the expected folder names.

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run global tuning and evaluation:

   ```bash
   python experiments/tune_global_detector.py
   python experiments/evaluate_global_detector.py
   ```

4. Generate local visualizations:

   ```bash
   python experiments/visualize_local.py
   ```

5. Evaluate local localization (optionally, up to 100 image/GT pairs):

   ```bash
   python experiments/evaluate_local_with_gt.py
   ```

6. Launch the Gradio app:

   ```bash
   python -m src.gradio_app
   ```

---

## 8. Known Limitations

* The method assumes a **Bayer-like CFA** and may not generalize to sensors with different patterns or aggressive in-camera processing.
* Highly textured scenes, very flat regions, or strong post-processing (compression, resizing) can weaken or distort the demosaicing pattern, leading to:

  * false positives (large suspicious masks),
  * partial localization of subtle tampering.
* Global threshold is optimized on the provided dataset; other datasets may require re-tuning.

---

## 9. Acknowledgements

* Project supervised as part of an academic course in image forensics / multimedia security.
* Demosaicing-trace method based on the works referenced in the original project description (link1 / link2).
* Dataset and ground-truth tampering maps provided by the course instructors.

---

## 10. License

This repository is intended for **research and educational** use.
If you plan to reuse or redistribute the code or dataset, please check the licensing terms of the original papers and datasets and adapt this section accordingly.

