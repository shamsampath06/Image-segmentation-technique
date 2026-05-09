"""
evaluate.py
-----------
Evaluation metrics for semantic image segmentation.

Metrics implemented:
  • Pixel Accuracy — fraction of correctly classified pixels
  • IoU (Intersection over Union / Jaccard Index) — overlap measure
  • Dice Coefficient (F1-Score) — harmonic mean of precision & recall
  • Adaptive Threshold — auto-select best threshold per prediction
"""

import numpy as np
from typing import Tuple, Dict


# ─── Adaptive Threshold ───────────────────────────────────────────────────────

def adaptive_threshold(mask: np.ndarray) -> float:
    """
    Automatically determine the best binarization threshold for a prediction mask.
    Uses Otsu-like approach: finds threshold that maximizes inter-class variance.
    Falls back to mean of prediction values.

    Parameters
    ----------
    mask : np.ndarray  — predicted probability map, values in [0, 1]

    Returns
    -------
    threshold : float
    """
    flat = mask.squeeze().flatten().astype(np.float32)
    if flat.max() < 0.05:
        # Model predicts near-zero everywhere — use very low threshold
        return float(flat.mean() + 0.5 * flat.std())

    # Otsu-like: sweep thresholds and pick max variance
    best_t  = 0.5
    best_var = -1.0
    for t in np.linspace(0.05, 0.95, 50):
        bg = flat[flat <= t]
        fg = flat[flat >  t]
        if len(bg) == 0 or len(fg) == 0:
            continue
        n_bg = len(bg) / len(flat)
        n_fg = len(fg) / len(flat)
        var = n_bg * n_fg * (bg.mean() - fg.mean()) ** 2
        if var > best_var:
            best_var = var
            best_t   = t
    return float(best_t)


def confidence_score(mask: np.ndarray) -> float:
    """
    Model confidence: mean distance of predictions from 0.5 (uncertainty boundary).
    1.0 = fully confident on every pixel, 0.0 = all uncertainty.
    """
    flat = mask.squeeze().flatten().astype(np.float32)
    return float(np.mean(np.abs(flat - 0.5)) * 2.0)


# ─── Core Metric Functions ────────────────────────────────────────────────────

def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Pixel Accuracy = correct pixels / total pixels.

    Parameters
    ----------
    y_true    : np.ndarray  shape (H, W) or (H, W, 1) — ground truth binary mask
    y_pred    : np.ndarray  shape (H, W) or (H, W, 1) — predicted probability map
    threshold : float       — binarization threshold for y_pred

    Returns
    -------
    accuracy : float in [0, 1]
    """
    y_true = y_true.squeeze()
    y_pred = (y_pred.squeeze() > threshold).astype(np.uint8)
    y_true = (y_true.squeeze() > 0.5).astype(np.uint8)
    correct = np.sum(y_true == y_pred)
    total   = y_true.size
    return float(correct) / float(total)


def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Intersection over Union (Jaccard Index).

    IoU = |Intersection| / |Union|
        = TP / (TP + FP + FN)
    """
    y_true = (y_true.squeeze() > 0.5).astype(np.float32)
    y_pred = (y_pred.squeeze() > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union        = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def dice_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Dice Coefficient (F1-Score for segmentation).

    Dice = 2 * |Intersection| / (|Y_true| + |Y_pred|)
    """
    y_true = (y_true.squeeze() > 0.5).astype(np.float32)
    y_pred = (y_pred.squeeze() > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    ))


def confusion_matrix_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN from flattened binary arrays.

    Returns
    -------
    (TP, FP, FN, TN) : tuple of ints
    """
    y_true = (y_true.squeeze().flatten() > 0.5).astype(np.uint8)
    y_pred = (y_pred.squeeze().flatten() > threshold).astype(np.uint8)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    return TP, FP, FN, TN


# ─── Batch Evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    threshold: float = 0.5,
    use_adaptive: bool = True
) -> Dict[str, float]:
    """
    Compute mean metrics over a batch of predictions.

    Parameters
    ----------
    y_true_batch   : np.ndarray  shape (N, H, W, 1)
    y_pred_batch   : np.ndarray  shape (N, H, W, 1)
    threshold      : float
    use_adaptive   : bool — if True, auto-selects best threshold per image

    Returns
    -------
    dict with keys: 'accuracy', 'iou', 'dice', '*_std'
    """
    accuracies, ious, dices = [], [], []

    for i in range(len(y_true_batch)):
        t = adaptive_threshold(y_pred_batch[i]) if use_adaptive else threshold
        accuracies.append(pixel_accuracy(y_true_batch[i], y_pred_batch[i], t))
        ious.append(iou_score(y_true_batch[i], y_pred_batch[i], t))
        dices.append(dice_coefficient(y_true_batch[i], y_pred_batch[i], t))

    return {
        "accuracy": float(np.mean(accuracies)),
        "iou":      float(np.mean(ious)),
        "dice":     float(np.mean(dices)),
        "accuracy_std": float(np.std(accuracies)),
        "iou_std":      float(np.std(ious)),
        "dice_std":     float(np.std(dices)),
    }


def area_percentage(mask: np.ndarray, threshold: float = None) -> Dict[str, float]:
    """
    Calculate the percentage area of each region in a predicted mask.

    Parameters
    ----------
    mask      : np.ndarray  — predicted probability mask (H, W) or (H, W, 1)
    threshold : float or None — if None, uses adaptive threshold

    Returns
    -------
    dict with 'foreground_pct', 'background_pct', 'total_px', pixel counts
    """
    if threshold is None:
        threshold = adaptive_threshold(mask)
    binary   = (mask.squeeze() > threshold).astype(np.uint8)
    total    = binary.size
    fg_count = int(np.sum(binary == 1))
    bg_count = total - fg_count
    return {
        "foreground_pct": round(100.0 * fg_count / total, 2),
        "background_pct": round(100.0 * bg_count / total, 2),
        "foreground_px":  fg_count,
        "background_px":  bg_count,
        "total_px":       total,
        "threshold":      round(threshold, 3),
    }


if __name__ == "__main__":
    rng    = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(5, 128, 128, 1)).astype(np.float32)
    y_pred = rng.uniform(0, 1,   size=(5, 128, 128, 1)).astype(np.float32)

    results = evaluate_batch(y_true, y_pred)
    print("Batch Evaluation Results:")
    for k, v in results.items():
        print(f"  {k:<18}: {v:.4f}")
