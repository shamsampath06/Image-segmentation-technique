"""
main.py
-------
Command-Line Training & Evaluation Pipeline
============================================
Usage:
  python main.py              — full pipeline (download → train → evaluate → save)
  python main.py --skip-train — skip training, use existing model
  python main.py --epochs N   — set number of training epochs (default: 10)
  python main.py --samples N  — max dataset samples (default: 200)

Outputs:
  model/unet_model.h5         — saved model weights
  training/history.json       — training loss/accuracy per epoch
  training/training_curves.png
  output/<name>_segmented.png — predicted overlays for test images
  training/metrics.json       — final evaluation metrics
  docs/report.html            — auto-generated HTML report
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# ── Project modules ────────────────────────────────────────────────────────────
import dataset_loader
import unet as unet_module
import evaluate as eval_module
import visualize


# ─── Directory Setup ──────────────────────────────────────────────────────────
os.makedirs("model",    exist_ok=True)
os.makedirs("training", exist_ok=True)
os.makedirs("output",   exist_ok=True)
os.makedirs("docs",     exist_ok=True)

MODEL_PATH   = os.path.join("model",    "unet_model.h5")
HISTORY_PATH = os.path.join("training", "history.json")
METRICS_PATH = os.path.join("training", "metrics.json")
CURVES_PATH  = os.path.join("training", "training_curves.png")


# ─── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="U-Net Image Segmentation Training Pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load existing model")
    parser.add_argument("--epochs",  type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Max dataset samples to use (default: 200)")
    parser.add_argument("--batch",   type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--img-size",type=int, default=128,
                        help="Image resize dimension (default: 128)")
    return parser.parse_args()


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(max_samples: int, img_size: int):
    """Download and load dataset, return train/val/test splits."""
    # Override module constants if user passed custom values
    dataset_loader.TARGET_SIZE  = (img_size, img_size)
    dataset_loader.MAX_SAMPLES  = max_samples

    print("=" * 60)
    print("  STEP 1 — Loading Dataset")
    print("=" * 60)
    images, masks, stems = dataset_loader.load_dataset()

    # Split: 70% train | 15% val | 15% test
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        images, masks, stems, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=42
    )

    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)}   samples")
    print(f"  Test  : {len(X_test)}  samples\n")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test, s_test)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, epochs: int, batch_size: int, img_size: int):
    """Build U-Net, train, save weights and history."""
    import tensorflow as tf

    print("=" * 60)
    print("  STEP 2 — Building U-Net Model")
    print("=" * 60)
    model = unet_module.build_unet(input_shape=(img_size, img_size, 3))
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print("\n" + "=" * 60)
    print(f"  STEP 3 — Training ({epochs} epochs, batch={batch_size})")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\n  [main] Training history saved → {HISTORY_PATH}")

    # Plot training curves
    visualize.plot_training_history(HISTORY_PATH, save_path=CURVES_PATH)

    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, s_test):
    """Run predictions on test set, compute metrics, save outputs."""
    import tensorflow as tf

    print("=" * 60)
    print("  STEP 4 — Evaluating on Test Set")
    print("=" * 60)

    y_pred = model.predict(X_test, batch_size=4, verbose=1)
    metrics = eval_module.evaluate_batch(y_test, y_pred)

    print("\n  ┌─────────────────────────────┐")
    print(  "  │     EVALUATION RESULTS      │")
    print(  "  ├─────────────────────────────┤")
    print(f"  │  Pixel Accuracy : {metrics['accuracy']:.4f}       │")
    print(f"  │  IoU Score      : {metrics['iou']:.4f}       │")
    print(f"  │  Dice Score     : {metrics['dice']:.4f}       │")
    print(  "  └─────────────────────────────┘\n")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [main] Metrics saved → {METRICS_PATH}")

    return y_pred, metrics


# ─── Save Outputs ─────────────────────────────────────────────────────────────
def save_outputs(X_test, y_test, y_pred, s_test, n_save: int = 10):
    """Save comparison images and segmented outputs for up to n_save samples."""
    print("=" * 60)
    print(f"  STEP 5 — Saving Output Visualizations (up to {n_save})")
    print("=" * 60)

    n_save = min(n_save, len(X_test))
    saved_paths = []

    for i in range(n_save):
        stem = s_test[i]

        # Full comparison plot
        comp_path = os.path.join("output", f"{stem}_comparison.png")
        visualize.plot_comparison(
            image=X_test[i], gt_mask=y_test[i], pred_mask=y_pred[i],
            save_path=comp_path, title=f"Segmentation — {stem}"
        )
        plt_close_all()

        # Segmented overlay only
        seg_path = visualize.save_segmented_output(X_test[i], y_pred[i], stem)
        saved_paths.append(seg_path)

    print(f"\n  [main] {n_save} outputs saved in output/")
    return saved_paths


def plt_close_all():
    """Helper: close all matplotlib figures to free memory."""
    import matplotlib.pyplot as plt
    plt.close("all")


# ─── Report Generation ────────────────────────────────────────────────────────
def generate_report():
    """Call the report generator module to create docs/report.html."""
    print("=" * 60)
    print("  STEP 6 — Generating HTML Report")
    print("=" * 60)
    try:
        import docs.report_generator as rg
        rg.generate_report()
    except Exception as e:
        print(f"  [warn] Report generation failed: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "█" * 60)
    print("  IMAGE SEGMENTATION — U-Net Training Pipeline")
    print("█" * 60 + "\n")

    # 1. Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test, s_test) = \
        load_data(args.samples, args.img_size)

    # 2. Train or load model
    if args.skip_train and os.path.exists(MODEL_PATH):
        import tensorflow as tf
        print(f"  [main] Loading existing model from {MODEL_PATH}")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "bce_dice_loss": unet_module.bce_dice_loss,
                "dice_loss":     unet_module.dice_loss,
            }
        )
    else:
        model = train_model(X_train, y_train, X_val, y_val,
                            epochs=args.epochs,
                            batch_size=args.batch,
                            img_size=args.img_size)

    # 3. Evaluate
    y_pred, metrics = evaluate_model(model, X_test, y_test, s_test)

    # 4. Save outputs
    save_outputs(X_test, y_test, y_pred, s_test)

    # 5. Architecture diagram
    print("  [main] Drawing U-Net architecture diagram...")
    visualize.draw_unet_diagram("docs/unet_architecture.png")

    # 6. Generate report
    generate_report()

    print("\n" + "█" * 60)
    print("  ✓  PIPELINE COMPLETE")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Outputs : output/")
    print(f"  Report  : docs/report.html")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
