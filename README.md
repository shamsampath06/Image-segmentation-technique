# Image Segmentation Techniques — Project

## Overview
A complete mini project demonstrating **U-Net based semantic image segmentation** with:
- Auto-download of the Oxford-IIIT Pet Dataset
- U-Net deep learning model (TensorFlow/Keras)
- Interactive Streamlit web application
- Real-time webcam segmentation (OpenCV)
- Area analysis, comparison, and auto-generated report

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py
```
This will:
- Download & preprocess dataset (~200 images from Oxford Pet)
- Train U-Net for up to 10 epochs
- Save model to `model/unet_model.h5`
- Save segmented outputs to `output/`
- Generate `docs/report.html`

### 3. Launch Web App
```bash
streamlit run app.py
```

---

## Command Line Options

```bash
python main.py --help

Options:
  --skip-train     Skip training, load existing model
  --epochs N       Number of training epochs (default: 10)
  --samples N      Max dataset images to use (default: 200)
  --batch N        Batch size (default: 8)
  --img-size N     Image resize dimension (default: 128)
```

---

## Project Structure

```
Image Segmentation Techniques/
├── dataset/
│   ├── images/          ← downloaded & normalized images
│   └── masks/           ← binary segmentation masks
├── model/
│   └── unet_model.h5    ← trained model weights
├── training/
│   ├── history.json     ← training loss/accuracy log
│   ├── metrics.json     ← final evaluation metrics
│   └── training_curves.png
├── output/              ← segmented output images
├── docs/
│   ├── report.html      ← auto-generated academic report
│   ├── report_generator.py
│   └── unet_architecture.png
├── app.py               ← Streamlit web application
├── main.py              ← CLI training pipeline
├── unet.py              ← U-Net model (TF/Keras)
├── dataset_loader.py    ← auto-download + preprocess
├── evaluate.py          ← IoU, Dice, Accuracy metrics
├── visualize.py         ← color overlays, charts, diagrams
└── requirements.txt
```

---

## Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Home & Dashboard | Metrics overview, training curves, output gallery |
| 📤 Upload & Segment | Upload any image, segment with U-Net, download result |
| 📷 Real-Time Webcam | Live webcam segmentation via OpenCV |
| 🔍 Comparison View | U-Net vs simple threshold baseline |
| 📖 About & Architecture | U-Net diagram, tech stack, usage guide |

---

## Evaluation Metrics

- **Pixel Accuracy** — correct pixels / total pixels
- **IoU (Jaccard Index)** — intersection / union of predicted and true masks
- **Dice Coefficient** — 2 × intersection / (|predicted| + |true|)

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Deep Learning Model | TensorFlow 2.x / Keras |
| Image Processing | OpenCV, Pillow |
| Web Application | Streamlit |
| Visualization | Matplotlib |
| Data Processing | NumPy, scikit-learn |
| Dataset | Oxford-IIIT Pet Dataset |

---

## Notes

- First run downloads dataset (~800 MB compressed, ~8 MB extracted subset)
- Training on CPU: ~15–25 minutes for 10 epochs with 200 images at 128×128
- Webcam feature requires physical webcam connection
- Model saved as `.h5` format for easy reload

---

*Mini Project — Usage of Image Segmentation Techniques for Object and Region Identification in Images*
