"""
visualize.py
------------
Visualization utilities for semantic image segmentation results.

Features:
  • Color-coded segmentation overlays (foreground/background)
  • Raw probability heatmap visualization
  • Side-by-side comparison: Original | Ground Truth | Predicted
  • Before/After blend for interactive comparison
  • Save outputs to output/ folder
  • Area percentage bar charts
  • Training history plots (loss/accuracy curves)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, List, Dict

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: (30,  30, 180, 130),   # background → semi-transparent blue
    1: (0,  220, 100, 200),   # foreground → semi-transparent green
}

REGION_LABELS = {0: "Background", 1: "Foreground (Object)"}


# ─── Helper: normalize image to uint8 ────────────────────────────────────────

def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or any range to uint8 [0,255] safely."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if img.max() <= 1.0 and img.min() >= 0.0:
        img = img * 255.0
    elif img.max() > 1.0:
        # Already in [0,255] range but float
        pass
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── Mask Colorization ────────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a binary probability mask to an RGBA color-coded image.

    Parameters
    ----------
    mask      : np.ndarray  shape (H, W) or (H, W, 1) — predicted probs in [0,1]
    threshold : float

    Returns
    -------
    color_mask : np.ndarray  shape (H, W, 4), uint8 RGBA
    """
    binary = (mask.squeeze() > threshold).astype(np.uint8)
    H, W   = binary.shape
    color  = np.zeros((H, W, 4), dtype=np.uint8)

    for cls_idx, rgba in CLASS_COLORS.items():
        region = (binary == cls_idx)
        color[region] = rgba

    return color


def binary_mask_image(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Return a clean black-and-white uint8 image from a probability mask.
    White = foreground (255), Black = background (0). Returns 3-channel RGB.
    """
    squeezed = mask.squeeze().astype(np.float32)
    binary   = (squeezed > threshold).astype(np.uint8) * 255
    # Stack into 3 channels to prevent any 2D Streamlit rendering bugs (black frame issues)
    rgb_binary = np.stack([binary, binary, binary], axis=-1)
    return rgb_binary


def prediction_heatmap(mask: np.ndarray, colormap: str = "plasma") -> np.ndarray:
    """
    Render a raw probability mask as a false-color heatmap (uint8 RGB).

    Parameters
    ----------
    mask     : np.ndarray  — probability mask [0,1]
    colormap : str         — matplotlib colormap name

    Returns
    -------
    heatmap_rgb : np.ndarray shape (H, W, 3), uint8
    """
    squeezed = mask.squeeze().astype(np.float32)
    # Normalize in case values are very small
    vmin, vmax = squeezed.min(), squeezed.max()
    if vmax - vmin < 1e-6:
        normalized = np.zeros_like(squeezed)
    else:
        normalized = (squeezed - vmin) / (vmax - vmin)

    cmap   = cm.get_cmap(colormap)
    rgba   = cmap(normalized)                 # (H, W, 4) float [0,1]
    rgb    = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.50,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Blend a color-coded mask onto the original RGB image.

    Parameters
    ----------
    image : np.ndarray  shape (H, W, 3), float32 [0,1] or uint8 [0,255]
    mask  : np.ndarray  shape (H, W) or (H, W, 1)
    alpha : float       — mask opacity (0=transparent, 1=opaque)

    Returns
    -------
    blended : np.ndarray  shape (H, W, 3), uint8 [0,255]
    """
    img_u8 = to_uint8(image)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=-1)

    color_mask  = colorize_mask(mask, threshold)           # RGBA uint8
    overlay_rgb = color_mask[..., :3].astype(np.float32)   # RGB
    blend_alpha = color_mask[..., 3:4].astype(np.float32) / 255.0  # per-pixel alpha

    bg   = img_u8.astype(np.float32)
    blended = bg * (1.0 - blend_alpha * alpha) + overlay_rgb * (blend_alpha * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def before_after_blend(
    image: np.ndarray,
    mask: np.ndarray,
    split: float = 0.5,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Horizontal split: left=original, right=overlay. split in [0,1].

    Returns
    -------
    combined : np.ndarray  shape (H, W, 3), uint8
    """
    img_u8   = to_uint8(image)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=-1)
    overlay  = overlay_mask_on_image(image, mask, threshold=threshold)
    H, W     = img_u8.shape[:2]
    split_px = int(W * split)

    combined = img_u8.copy()
    combined[:, split_px:] = overlay[:, split_px:]

    # Draw a white split line
    line = min(split_px, W - 1)
    combined[:, line, :] = 255
    return combined


# ─── Comparison Plot ──────────────────────────────────────────────────────────

def plot_comparison(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Segmentation Results",
    threshold: float = 0.5
) -> plt.Figure:
    """
    Create a side-by-side comparison figure:
        Original | Ground Truth | Predicted Overlay
    """
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="white")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    img_u8 = to_uint8(image)

    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image", fontsize=11, color="white")
    axes[0].axis("off")

    col = 1
    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask, threshold=0.5)
        axes[col].imshow(gt_overlay)
        axes[col].set_title("Ground Truth Mask", fontsize=11, color="white")
        axes[col].axis("off")
        col += 1

    pred_overlay = overlay_mask_on_image(image, pred_mask, threshold=threshold)
    axes[col].imshow(pred_overlay)

    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[k][:3]) / 255, label=REGION_LABELS[k])
        for k in CLASS_COLORS
    ]
    axes[col].legend(handles=patches, loc="lower right", fontsize=8,
                     facecolor="#2a2a4a", labelcolor="white")
    axes[col].set_title("Predicted Segmentation", fontsize=11, color="white")
    axes[col].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [visualize] Saved comparison → {save_path}")

    return fig


# ─── Area Bar Chart ───────────────────────────────────────────────────────────

def plot_area_chart(area_info: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Draw a horizontal bar chart of region area percentages.
    """
    labels = ["🟢 Foreground", "🔵 Background"]
    values = [area_info["foreground_pct"], area_info["background_pct"]]
    colors = ["#00dc64", "#4361ee"]

    fig, ax = plt.subplots(figsize=(5, 1.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(labels, values, color=colors, edgecolor="none",
                   height=0.45, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9, fontweight="bold",
            color="white"
        )

    ax.set_xlim(0, 120)
    ax.set_xlabel("Area (%)", fontsize=8, color="white")
    ax.set_title("Region Area Distribution", fontsize=10, fontweight="bold",
                 color="white", pad=6)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ─── Training History Plot ────────────────────────────────────────────────────

def plot_training_history(history_path: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Load training history JSON and plot loss & accuracy curves.
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Training History", fontsize=14, fontweight="bold", color="white")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(alpha=0.15, color="white")

    # Loss
    axes[0].plot(epochs, history["loss"],     "o-",  label="Train Loss",
                 color="#f72585", linewidth=2.5, markersize=5)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss",
                 color="#f72585", alpha=0.55, linewidth=2, markersize=5)
    axes[0].set_title("Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#2a2a4a", labelcolor="white")

    # Accuracy
    acc_key     = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    axes[1].plot(epochs, history[acc_key],     "o-",  label="Train Acc",
                 color="#4cc9f0", linewidth=2.5, markersize=5)
    axes[1].plot(epochs, history[val_acc_key], "s--", label="Val Acc",
                 color="#4cc9f0", alpha=0.55, linewidth=2, markersize=5)
    axes[1].set_title("Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(facecolor="#2a2a4a", labelcolor="white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [visualize] Saved training history → {save_path}")

    return fig


# ─── Save Segmented Output ────────────────────────────────────────────────────

def save_segmented_output(
    image: np.ndarray,
    pred_mask: np.ndarray,
    filename: str,
    threshold: float = 0.5
) -> str:
    """Save the overlay (original + predicted mask) as a PNG to output/."""
    blended   = overlay_mask_on_image(image, pred_mask, threshold=threshold)
    save_path = os.path.join(OUTPUT_DIR, f"{filename}_segmented.png")
    Image.fromarray(blended).save(save_path)
    return save_path


# ─── U-Net Architecture Diagram ──────────────────────────────────────────────

def draw_unet_diagram(save_path: str = "docs/unet_architecture.png") -> None:
    """Draw a simplified schematic of the U-Net architecture using matplotlib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#0f0c29")

    enc_color  = "#4361ee"
    bot_color  = "#7209b7"
    dec_color  = "#3a86ff"
    skip_color = "#f72585"
    out_color  = "#4cc9f0"

    def draw_box(x, y, w, h, color, label, fontsize=8):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor="white", facecolor=color, alpha=0.88,
                              zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold", zorder=4)

    # Encoder
    enc_pos    = [(0.4, 5.2), (0.4, 3.9), (0.4, 2.6), (0.4, 1.3)]
    enc_labels = ["Enc 1\nConv 64", "Enc 2\nConv 128", "Enc 3\nConv 256", "Enc 4\nConv 512"]
    for (x, y), lbl in zip(enc_pos, enc_labels):
        draw_box(x, y, 1.4, 0.9, enc_color, lbl)

    # Bottleneck
    draw_box(5.0, 0.4, 1.8, 0.9, bot_color, "Bottleneck\nConv 1024", fontsize=9)

    # Decoder
    dec_pos    = [(9.0, 5.2), (9.0, 3.9), (9.0, 2.6), (9.0, 1.3)]
    dec_labels = ["Dec 4\nUp 512", "Dec 3\nUp 256", "Dec 2\nUp 128", "Dec 1\nUp 64"]
    for (x, y), lbl in zip(dec_pos, dec_labels):
        draw_box(x, y, 1.4, 0.9, dec_color, lbl)

    # Output
    draw_box(12.0, 5.2, 1.6, 0.9, out_color, "Output\nSigmoid", fontsize=9)

    # Skip connections
    for (ex, ey), (dx, dy) in zip(enc_pos, dec_pos):
        mid_y = ey + 0.45
        ax.annotate("", xy=(dx, dy + 0.45), xytext=(ex + 1.4, ey + 0.45),
                    arrowprops=dict(arrowstyle="->", color=skip_color, lw=1.8,
                                   connectionstyle="arc3,rad=-0.25"),
                    zorder=2)

    # Title
    ax.text(8.0, 6.6, "U-Net Encoder-Decoder Architecture", ha="center",
            color="white", fontsize=15, fontweight="bold")

    # Legend
    legend_items = [
        (enc_color,  "Encoder Block (Conv + MaxPool)"),
        (bot_color,  "Bottleneck"),
        (dec_color,  "Decoder Block (UpSample + Conv)"),
        (skip_color, "Skip Connections"),
        (out_color,  "Output Layer (Sigmoid)"),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((12.2, 3.8 - i * 0.55), 0.35, 0.35, color=col, zorder=3))
        ax.text(12.65, 3.97 - i * 0.55, lbl, color="white", fontsize=8, va="center")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [visualize] U-Net architecture diagram saved → {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    draw_unet_diagram()
    print("Architecture diagram generated.")
