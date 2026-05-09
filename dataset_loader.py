"""
dataset_loader.py
-----------------
Handles automatic downloading, extraction, and preprocessing of the
Oxford-IIIT Pet Dataset for semantic image segmentation.

Steps:
  1. Download images and annotation trimaps from Oxford servers
  2. Extract a random subset (up to MAX_SAMPLES images)
  3. Resize to TARGET_SIZE and normalize
  4. Convert trimaps to binary masks (foreground=1, background=0)
  5. Return numpy arrays ready for model training
"""

import os
import io
import random
import tarfile
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
TARGET_SIZE   = (128, 128)   # resize all images/masks to this
MAX_SAMPLES   = 200          # maximum images to use (keep training fast)
RANDOM_SEED   = 42

DATASET_DIR   = "dataset"
IMAGES_DIR    = os.path.join(DATASET_DIR, "images")
MASKS_DIR     = os.path.join(DATASET_DIR, "masks")

# Oxford-IIIT Pet Dataset URLs
IMAGES_URL    = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL    = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _download_file(url: str, dest_path: str, desc: str = "Downloading") -> None:
    """Stream-download a file with a progress bar."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_dataset() -> None:
    """
    Download and extract the Oxford-IIIT Pet Dataset.
    Only downloads if the dataset directories are not already present.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR,  exist_ok=True)

    # Check if already downloaded
    existing_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))]
    existing_masks  = [f for f in os.listdir(MASKS_DIR)  if f.endswith(".png")]

    if len(existing_images) >= MAX_SAMPLES and len(existing_masks) >= MAX_SAMPLES:
        print(f"[dataset_loader] Dataset already present ({len(existing_images)} images). Skipping download.")
        return

    print("[dataset_loader] Downloading Oxford-IIIT Pet Dataset...")
    print("  NOTE: Full archive is ~800 MB but only a small subset will be kept.\n")

    tmp_images_tar = os.path.join(DATASET_DIR, "_images.tar.gz")
    tmp_annots_tar = os.path.join(DATASET_DIR, "_annotations.tar.gz")

    # Download archives
    _download_file(IMAGES_URL, tmp_images_tar, desc="Images archive")
    _download_file(ANNOTS_URL, tmp_annots_tar, desc="Annotations archive")

    # ── Extract images ────────────────────────────────────────────────────────
    print("\n[dataset_loader] Extracting images...")
    with tarfile.open(tmp_images_tar, "r:gz") as tar:
        # Collect jpg members only
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".jpg")]
        random.seed(RANDOM_SEED)
        random.shuffle(members)
        selected = members[:MAX_SAMPLES]

        for member in tqdm(selected, desc="Extracting images"):
            f = tar.extractfile(member)
            if f is None:
                continue
            img_name  = os.path.basename(member.name)
            save_path = os.path.join(IMAGES_DIR, img_name)
            with open(save_path, "wb") as out:
                out.write(f.read())

    # ── Extract corresponding trimaps ─────────────────────────────────────────
    print("[dataset_loader] Extracting annotation trimaps...")
    selected_stems = {os.path.splitext(os.path.basename(m.name))[0] for m in selected}

    with tarfile.open(tmp_annots_tar, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if "trimaps" in m.name and m.name.lower().endswith(".png")]
        for member in tqdm(members, desc="Extracting masks"):
            stem = os.path.splitext(os.path.basename(member.name))[0]
            if stem not in selected_stems:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            save_path = os.path.join(MASKS_DIR, os.path.basename(member.name))
            with open(save_path, "wb") as out:
                out.write(f.read())

    # Remove archives to save disk space
    os.remove(tmp_images_tar)
    os.remove(tmp_annots_tar)
    print(f"\n[dataset_loader] Dataset ready: {len(selected)} image/mask pairs.\n")


def _preprocess_image(path: str) -> np.ndarray:
    """Load, resize, and normalize an RGB image → float32 in [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _preprocess_mask(path: str) -> np.ndarray:
    """
    Load and resize an Oxford trimap mask.

    Oxford trimaps use values:
      1 = foreground (pet)
      2 = background
      3 = uncertain / border region

    We convert to binary: foreground=1, everything else=0.
    Output shape: (H, W, 1), dtype float32
    """
    mask = Image.open(path).convert("L")   # grayscale
    mask = mask.resize(TARGET_SIZE, Image.NEAREST)
    arr  = np.array(mask, dtype=np.uint8)

    # Binary mask: pixel==1 → foreground
    binary = (arr == 1).astype(np.float32)
    return binary[..., np.newaxis]          # shape (H, W, 1)


def load_dataset():
    """
    Main entry point: download if needed, then load all image/mask pairs.

    Returns
    -------
    images : np.ndarray  shape (N, H, W, 3),  float32, [0,1]
    masks  : np.ndarray  shape (N, H, W, 1),  float32, {0,1}
    stems  : list[str]   base filenames (without extension) for reference
    """
    download_dataset()

    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])
    mask_files  = {os.path.splitext(f)[0]: f for f in os.listdir(MASKS_DIR) if f.endswith(".png")}

    images, masks, stems = [], [], []

    for img_file in tqdm(image_files, desc="Loading dataset"):
        stem = os.path.splitext(img_file)[0]
        if stem not in mask_files:
            continue                        # skip if no matching mask

        img_path  = os.path.join(IMAGES_DIR, img_file)
        mask_path = os.path.join(MASKS_DIR,  mask_files[stem])

        try:
            images.append(_preprocess_image(img_path))
            masks.append(_preprocess_mask(mask_path))
            stems.append(stem)
        except Exception as e:
            print(f"  [warn] Skipping {img_file}: {e}")

    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks,  dtype=np.float32)
    print(f"[dataset_loader] Loaded {len(images)} pairs | shape: {images.shape}, {masks.shape}")
    return images, masks, stems


if __name__ == "__main__":
    imgs, msks, _ = load_dataset()
    print(f"Images: {imgs.shape}  Masks: {msks.shape}")
    print(f"Image range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"Mask  unique values: {np.unique(msks)}")
