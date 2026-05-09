"""
docs/report_generator.py
-------------------------
Auto-generates a comprehensive HTML academic report for the
Image Segmentation mini project.

Sections:
  1. Abstract
  2. Introduction
  3. Problem Statement
  4. Objectives
  5. Methodology
  6. U-Net Architecture (with embedded diagram)
  7. Implementation Details
  8. Results (embedded output images)
  9. Evaluation Metrics
  10. Conclusion
  11. Future Scope
"""

import os
import json
import base64
import glob
from datetime import datetime

REPORT_PATH  = os.path.join("docs", "report.html")
METRICS_PATH = os.path.join("training", "metrics.json")
HISTORY_PATH = os.path.join("training", "history.json")
CURVES_PATH  = os.path.join("training", "training_curves.png")
ARCH_IMG     = os.path.join("docs",     "unet_architecture.png")
OUTPUT_DIR   = "output"


def _img_to_b64(path: str) -> str:
    """Encode an image file to base64 string for HTML embedding."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _embed_img(path: str, caption: str = "", width: str = "100%") -> str:
    """Return an <img> tag with base64-encoded image or empty string."""
    b64 = _img_to_b64(path)
    if not b64:
        return f'<p style="color:#888; font-style:italic;">[ {caption} — image not yet generated ]</p>'
    ext  = os.path.splitext(path)[1].lstrip(".").replace("jpg", "jpeg")
    mime = f"image/{ext}"
    html = f'<figure style="text-align:center; margin:16px 0">'
    html += f'<img src="data:{mime};base64,{b64}" width="{width}" style="border-radius:10px; border:1px solid #334; max-width:100%;">'
    if caption:
        html += f'<figcaption style="color:#aaa; font-size:0.85rem; margin-top:6px;">{caption}</figcaption>'
    html += '</figure>'
    return html


def generate_report() -> None:
    """Generate the full HTML report and save to docs/report.html."""
    os.makedirs("docs", exist_ok=True)

    # Load metrics if available
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    # Load training history if available
    history = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)

    # Gather output images (max 6)
    output_imgs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_comparison.png")))[:6]

    # Metrics table rows
    metric_rows = ""
    if metrics:
        rows = [
            ("Pixel Accuracy",    f"{metrics.get('accuracy', 0)*100:.2f}%", f"±{metrics.get('accuracy_std',0)*100:.2f}%"),
            ("IoU (Jaccard)",     f"{metrics.get('iou',      0)*100:.2f}%", f"±{metrics.get('iou_std',     0)*100:.2f}%"),
            ("Dice Coefficient",  f"{metrics.get('dice',     0)*100:.2f}%", f"±{metrics.get('dice_std',    0)*100:.2f}%"),
        ]
        for name, val, std in rows:
            metric_rows += f"<tr><td>{name}</td><td><b>{val}</b></td><td>{std}</td></tr>"
    else:
        metric_rows = "<tr><td colspan='3' style='color:#888;'>Metrics not yet available. Run training first.</td></tr>"

    # Output image gallery
    gallery_html = ""
    if output_imgs:
        gallery_html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0;">'
        for p in output_imgs:
            name = os.path.basename(p).replace("_comparison.png", "")
            gallery_html += f'<div>{_embed_img(p, caption=name, width="100%")}</div>'
        gallery_html += '</div>'
    else:
        gallery_html = '<p style="color:#888; font-style:italic;">Output images will appear here after running python main.py</p>'

    # Training curves
    curves_html = _embed_img(CURVES_PATH, "Training Loss & Accuracy Curves", "90%")

    # Architecture diagram
    arch_html = _embed_img(ARCH_IMG, "U-Net Encoder-Decoder Architecture", "90%")

    # Epoch count
    n_epochs = len(history.get("loss", [])) if history else "—"

    # Timestamp
    ts = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Image Segmentation — Project Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Source+Code+Pro:wght@400;600&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}

    body {{
      font-family: 'Inter', sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      line-height: 1.7;
      font-size: 16px;
    }}

    /* ── Cover Page ── */
    .cover {{
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 60px 40px;
      position: relative;
      overflow: hidden;
    }}
    .cover::before {{
      content: '';
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at 30% 40%, rgba(114,9,183,0.25) 0%, transparent 60%),
                  radial-gradient(ellipse at 70% 60%, rgba(76,201,240,0.18) 0%, transparent 55%);
    }}
    .cover-content {{ position: relative; z-index: 1; max-width: 800px; }}
    .cover-badge {{
      display: inline-block;
      padding: 6px 20px;
      border-radius: 99px;
      font-size: 0.8rem;
      font-weight: 700;
      background: rgba(76,201,240,0.12);
      color: #4cc9f0;
      border: 1px solid rgba(76,201,240,0.3);
      margin-bottom: 24px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }}
    h1.title {{
      font-size: 2.6rem;
      font-weight: 900;
      color: #fff;
      text-shadow: 0 0 40px rgba(76,201,240,0.4);
      line-height: 1.2;
      margin-bottom: 16px;
    }}
    .subtitle {{
      font-size: 1.05rem;
      color: rgba(255,255,255,0.55);
      margin-bottom: 32px;
    }}
    .cover-meta {{
      display: flex;
      gap: 24px;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 32px;
    }}
    .cover-meta-item {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px;
      padding: 12px 20px;
      font-size: 0.85rem;
      color: rgba(255,255,255,0.7);
    }}
    .cover-meta-item strong {{ display: block; color: #4cc9f0; font-size: 0.8rem; margin-bottom: 2px; }}

    /* ── Layout ── */
    .container {{
      max-width: 900px;
      margin: 0 auto;
      padding: 60px 24px;
    }}

    /* ── TOC ── */
    .toc {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 32px 40px;
      margin-bottom: 48px;
    }}
    .toc h2 {{ color: #4cc9f0; margin-bottom: 16px; font-size: 1.1rem; }}
    .toc ol {{ padding-left: 24px; }}
    .toc li {{ margin: 6px 0; }}
    .toc a {{ color: #a8b0c0; text-decoration: none; transition: color 0.2s; }}
    .toc a:hover {{ color: #4cc9f0; }}

    /* ── Section ── */
    .section {{
      margin-bottom: 56px;
      scroll-margin-top: 24px;
    }}
    h2.section-h {{
      font-size: 1.65rem;
      font-weight: 800;
      color: #fff;
      border-left: 4px solid #4cc9f0;
      padding-left: 16px;
      margin-bottom: 20px;
    }}
    h3 {{
      font-size: 1.1rem;
      font-weight: 700;
      color: #7dd3fc;
      margin: 20px 0 8px;
    }}
    p {{ margin-bottom: 14px; color: #a8b0c0; }}

    /* ── Callout boxes ── */
    .callout {{
      background: rgba(76,201,240,0.06);
      border-left: 4px solid #4cc9f0;
      border-radius: 0 10px 10px 0;
      padding: 16px 20px;
      margin: 20px 0;
      font-size: 0.95rem;
    }}

    /* ── Metrics table ── */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 0.95rem;
    }}
    thead tr {{ background: rgba(76,201,240,0.12); }}
    thead th {{
      text-align: left;
      padding: 12px 16px;
      color: #4cc9f0;
      font-weight: 700;
      border-bottom: 2px solid rgba(76,201,240,0.25);
    }}
    tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
    tbody td {{
      padding: 11px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      color: #c9d1d9;
    }}

    /* ── Code blocks ── */
    pre, code {{
      font-family: 'Source Code Pro', monospace;
      background: rgba(255,255,255,0.05);
      border-radius: 6px;
      font-size: 0.88rem;
    }}
    code {{ padding: 2px 6px; color: #f97316; }}
    pre {{
      padding: 18px 22px;
      overflow-x: auto;
      border: 1px solid rgba(255,255,255,0.08);
      margin: 16px 0;
      color: #c9d1d9;
    }}

    /* ── Ordered/Unordered lists ── */
    ol, ul {{ padding-left: 24px; margin-bottom: 14px; }}
    li {{ margin: 5px 0; color: #a8b0c0; }}

    /* ── Badges ── */
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .badge {{
      display: inline-block;
      padding: 4px 14px;
      border-radius: 99px;
      font-size: 0.78rem;
      font-weight: 700;
    }}
    .badge-blue   {{ background: rgba(76,201,240,0.12); color: #4cc9f0; border: 1px solid rgba(76,201,240,0.3); }}
    .badge-purple {{ background: rgba(114,9,183,0.2);   color: #c084fc; border: 1px solid rgba(114,9,183,0.4); }}
    .badge-green  {{ background: rgba(0,220,100,0.1);   color: #4ade80; border: 1px solid rgba(0,220,100,0.3); }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      padding: 40px 24px;
      color: rgba(255,255,255,0.25);
      font-size: 0.8rem;
      border-top: 1px solid rgba(255,255,255,0.06);
    }}

    @media print {{
      .cover {{ min-height: auto; padding: 40px; }}
      h1.title {{ font-size: 1.8rem; }}
    }}
  </style>
</head>
<body>

<!-- ═══════════ COVER PAGE ═══════════ -->
<div class="cover">
  <div class="cover-content">
    <div class="cover-badge">Mini Project Report</div>
    <h1 class="title">Usage of Image Segmentation Techniques for Object and Region Identification in Images</h1>
    <div class="subtitle">U-Net Based Semantic Image Segmentation with Deep Learning</div>
    <div class="cover-meta">
      <div class="cover-meta-item"><strong>Model</strong>U-Net (Encoder-Decoder)</div>
      <div class="cover-meta-item"><strong>Dataset</strong>Oxford-IIIT Pet Dataset</div>
      <div class="cover-meta-item"><strong>Framework</strong>TensorFlow / Keras</div>
      <div class="cover-meta-item"><strong>Generated</strong>{ts}</div>
    </div>
  </div>
</div>

<!-- ═══════════ CONTENT ═══════════ -->
<div class="container">

  <!-- Table of Contents -->
  <div class="toc">
    <h2>📋 Table of Contents</h2>
    <ol>
      <li><a href="#abstract">Abstract</a></li>
      <li><a href="#intro">Introduction</a></li>
      <li><a href="#problem">Problem Statement</a></li>
      <li><a href="#objectives">Objectives</a></li>
      <li><a href="#methodology">Methodology</a></li>
      <li><a href="#architecture">U-Net Architecture</a></li>
      <li><a href="#implementation">Implementation Details</a></li>
      <li><a href="#results">Results & Visualizations</a></li>
      <li><a href="#metrics">Evaluation Metrics</a></li>
      <li><a href="#conclusion">Conclusion</a></li>
      <li><a href="#future">Future Scope</a></li>
    </ol>
  </div>

  <!-- 1. Abstract -->
  <div class="section" id="abstract">
    <h2 class="section-h">1. Abstract</h2>
    <p>
      Image segmentation is a fundamental task in computer vision that involves partitioning a digital image
      into multiple segments (groups of pixels) to simplify and/or change the representation of an image
      into something more meaningful and easier to analyze. This project presents a complete implementation
      of semantic image segmentation using the U-Net deep learning architecture.
    </p>
    <p>
      We apply this technique to the Oxford-IIIT Pet Dataset, training a convolutional neural network to
      accurately delineate pets (foreground) from background regions. The trained model achieves measurable
      performance in terms of pixel accuracy, Intersection over Union (IoU), and Dice coefficient. An
      interactive Streamlit web application is built on top of the model, enabling real-time segmentation
      through webcam input, image uploads, area analysis, and method comparison.
    </p>
    <div class="callout">
      <strong>Key Contribution:</strong> End-to-end pipeline from dataset download to model deployment,
      featuring real-time segmentation, quantitative area analysis, and an interactive dashboard — all
      in a student-accessible mini-project format.
    </div>
  </div>

  <!-- 2. Introduction -->
  <div class="section" id="intro">
    <h2 class="section-h">2. Introduction</h2>
    <p>
      The human visual system effortlessly segments scenes into distinct objects and regions. Replicating
      this capability in machines is essential for numerous real-world applications: autonomous vehicles
      must identify roads, pedestrians, and obstacles; medical imaging systems must delineate tumors or
      organs; agricultural drones must distinguish crop from weeds.
    </p>
    <p>
      Traditional segmentation approaches relied on handcrafted features such as color histograms, gradient
      maps, or graph cuts. While these methods work under limited conditions, they fail to generalize across
      diverse real-world images. Deep learning has revolutionized this field: convolutional neural networks
      (CNNs) learn hierarchical features automatically from data, achieving state-of-the-art performance.
    </p>
    <p>
      The <strong>U-Net</strong> architecture, introduced by Ronneberger et al. (2015), was originally
      designed for biomedical image segmentation. Its elegant encoder-decoder structure with skip connections
      allows it to produce precise pixel-level predictions even with limited training data — making it ideal
      for educational mini-projects.
    </p>
    <div class="badge-row">
      <span class="badge badge-blue">Deep Learning</span>
      <span class="badge badge-blue">Computer Vision</span>
      <span class="badge badge-purple">Semantic Segmentation</span>
      <span class="badge badge-green">TensorFlow</span>
      <span class="badge badge-green">Streamlit</span>
    </div>
  </div>

  <!-- 3. Problem Statement -->
  <div class="section" id="problem">
    <h2 class="section-h">3. Problem Statement</h2>
    <p>
      Given an RGB image <em>I</em> of dimensions H × W × 3, the goal of semantic segmentation is to
      produce a label map <em>M</em> of dimensions H × W, where each pixel <em>M(i, j)</em> is assigned
      a class label <em>c ∈ {{0, 1, ..., C-1}}</em>.
    </p>
    <p>
      In the binary case (this project): each pixel is either <strong>foreground</strong> (the pet, class=1)
      or <strong>background</strong> (class=0). The model must learn to distinguish these regions purely
      from visual appearance, without any explicit feature engineering.
    </p>
    <div class="callout">
      <strong>Challenge:</strong> Pets vary widely in appearance (37 breeds), pose, lighting, and scale.
      The model must learn robust features that generalize across these variations, while remaining
      computationally feasible on standard hardware.
    </div>
  </div>

  <!-- 4. Objectives -->
  <div class="section" id="objectives">
    <h2 class="section-h">4. Objectives</h2>
    <ol>
      <li>Automatically download and preprocess a semantic segmentation dataset</li>
      <li>Implement a complete U-Net model using TensorFlow/Keras</li>
      <li>Train the model and record loss/accuracy metrics per epoch</li>
      <li>Evaluate segmentation quality using IoU, Dice coefficient, and pixel accuracy</li>
      <li>Visualize results with color-coded overlays and side-by-side comparisons</li>
      <li>Build a Streamlit web application with upload, real-time webcam, and dashboard features</li>
      <li>Implement area calculation showing percentage of each segmented region</li>
      <li>Auto-generate this comprehensive HTML report</li>
    </ol>
  </div>

  <!-- 5. Methodology -->
  <div class="section" id="methodology">
    <h2 class="section-h">5. Methodology</h2>

    <h3>5.1 Dataset</h3>
    <p>
      The <strong>Oxford-IIIT Pet Dataset</strong> contains 7,349 images of 37 pet breeds with
      pixel-level trimap annotations. A random subset of up to 200 images is used for this mini-project
      to enable fast training on standard hardware.
    </p>
    <p>
      Trimaps use three values: <code>1 = foreground</code>, <code>2 = background</code>,
      <code>3 = uncertain/border</code>. We binarize to foreground=1, all else=0.
    </p>

    <h3>5.2 Preprocessing Pipeline</h3>
    <ul>
      <li>Resize images and masks to <strong>128 × 128</strong> pixels</li>
      <li>Normalize pixel values to <code>[0, 1]</code> by dividing by 255</li>
      <li>Binarize masks: foreground=1, background=0</li>
      <li>Split dataset: <strong>70%</strong> train / <strong>15%</strong> validation / <strong>15%</strong> test</li>
    </ul>

    <h3>5.3 Loss Function</h3>
    <p>
      We use a combined <strong>Binary Cross-Entropy + Dice Loss</strong> which addresses class imbalance
      (background pixels typically outnumber foreground pixels):
    </p>
    <pre>L_total = L_BCE + L_Dice

L_BCE  = −[y · log(p) + (1−y) · log(1−p)]
L_Dice = 1 − (2·|X∩Y| + ε) / (|X| + |Y| + ε)</pre>

    <h3>5.4 Training Configuration</h3>
    <table>
      <thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Optimizer</td><td>Adam (lr = 1e-4)</td></tr>
        <tr><td>Epochs</td><td>Up to {n_epochs} (with early stopping)</td></tr>
        <tr><td>Batch Size</td><td>8</td></tr>
        <tr><td>Input Size</td><td>128 × 128 × 3</td></tr>
        <tr><td>Loss Function</td><td>BCE + Dice Loss</td></tr>
        <tr><td>Early Stopping</td><td>Patience = 3</td></tr>
        <tr><td>LR Schedule</td><td>ReduceLROnPlateau (factor=0.5)</td></tr>
      </tbody>
    </table>
  </div>

  <!-- 6. Architecture -->
  <div class="section" id="architecture">
    <h2 class="section-h">6. U-Net Architecture</h2>
    <p>
      U-Net consists of two symmetric paths: an <strong>encoder</strong> (contracting path) that captures
      context, and a <strong>decoder</strong> (expanding path) that enables precise localization.
      Skip connections concatenate feature maps from encoder to decoder at each resolution level,
      preserving fine spatial detail.
    </p>
    {arch_html}
    <h3>Architecture Layers</h3>
    <table>
      <thead><tr><th>Component</th><th>Filters</th><th>Output Shape</th><th>Operation</th></tr></thead>
      <tbody>
        <tr><td>Input</td><td>—</td><td>128 × 128 × 3</td><td>RGB image</td></tr>
        <tr><td>Enc Block 1</td><td>64</td><td>64 × 64 × 64</td><td>Conv→BN→ReLU ×2, MaxPool</td></tr>
        <tr><td>Enc Block 2</td><td>128</td><td>32 × 32 × 128</td><td>Conv→BN→ReLU ×2, MaxPool</td></tr>
        <tr><td>Enc Block 3</td><td>256</td><td>16 × 16 × 256</td><td>Conv→BN→ReLU ×2, MaxPool</td></tr>
        <tr><td>Enc Block 4</td><td>512</td><td>8 × 8 × 512</td><td>Conv→BN→ReLU ×2, MaxPool</td></tr>
        <tr><td>Bottleneck</td><td>1024</td><td>8 × 8 × 1024</td><td>Conv→BN→ReLU ×2</td></tr>
        <tr><td>Dec Block 4</td><td>512</td><td>16 × 16 × 512</td><td>UpSample, Concat, Conv ×2</td></tr>
        <tr><td>Dec Block 3</td><td>256</td><td>32 × 32 × 256</td><td>UpSample, Concat, Conv ×2</td></tr>
        <tr><td>Dec Block 2</td><td>128</td><td>64 × 64 × 128</td><td>UpSample, Concat, Conv ×2</td></tr>
        <tr><td>Dec Block 1</td><td>64</td><td>128 × 128 × 64</td><td>UpSample, Concat, Conv ×2</td></tr>
        <tr><td>Output</td><td>1</td><td>128 × 128 × 1</td><td>Conv 1×1, Sigmoid</td></tr>
      </tbody>
    </table>
  </div>

  <!-- 7. Implementation -->
  <div class="section" id="implementation">
    <h2 class="section-h">7. Implementation Details</h2>
    <h3>Project Structure</h3>
    <pre>Image Segmentation Techniques/
├── dataset/
│   ├── images/         ← normalized RGB images (128×128)
│   └── masks/          ← binary segmentation masks
├── model/
│   └── unet_model.h5   ← trained model weights
├── training/
│   ├── history.json    ← epoch-wise loss/accuracy
│   ├── metrics.json    ← final evaluation metrics
│   └── training_curves.png
├── output/             ← segmented output images
├── docs/
│   └── report.html     ← this report
├── dataset_loader.py   ← auto-download + preprocess
├── unet.py             ← U-Net model definition
├── evaluate.py         ← IoU, Dice, Accuracy metrics
├── visualize.py        ← color overlays, charts, diagrams
├── main.py             ← CLI training pipeline
├── app.py              ← Streamlit web application
└── requirements.txt</pre>

    <h3>Running the Project</h3>
    <pre># Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train the model (downloads dataset automatically)
python main.py

# Step 3: Launch the web app
streamlit run app.py

# Optional: Skip training, use existing model
python main.py --skip-train

# Optional: Adjust training parameters
python main.py --epochs 5 --samples 100 --batch 4</pre>
  </div>

  <!-- 8. Results -->
  <div class="section" id="results">
    <h2 class="section-h">8. Results & Visualizations</h2>
    <h3>Training Curves</h3>
    {curves_html}
    <h3>Segmentation Outputs</h3>
    <p>
      The following images show side-by-side comparisons of original input images,
      ground truth masks, and U-Net predicted segmentations:
    </p>
    {gallery_html}
  </div>

  <!-- 9. Metrics -->
  <div class="section" id="metrics">
    <h2 class="section-h">9. Evaluation Metrics</h2>
    <p>Three standard segmentation metrics are computed on the held-out test set:</p>

    <table>
      <thead>
        <tr><th>Metric</th><th>Value (Test Set)</th><th>Std Dev</th></tr>
      </thead>
      <tbody>
        {metric_rows}
      </tbody>
    </table>

    <h3>Metric Definitions</h3>
    <ul>
      <li><strong>Pixel Accuracy</strong>: Fraction of correctly classified pixels across all classes</li>
      <li><strong>IoU (Jaccard Index)</strong>: |Intersection| / |Union| of predicted and ground truth masks</li>
      <li><strong>Dice Coefficient</strong>: 2·|Intersection| / (|Predicted| + |Ground Truth|) — sensitive to small regions</li>
    </ul>
    <div class="callout">
      <strong>Interpretation:</strong> An IoU above 0.60 is considered good for binary segmentation on a small
      dataset. Higher IoU indicates better overlap between predicted and ground truth masks.
    </div>
  </div>

  <!-- 10. Conclusion -->
  <div class="section" id="conclusion">
    <h2 class="section-h">10. Conclusion</h2>
    <p>
      This project successfully demonstrates the application of U-Net deep learning architecture for
      semantic image segmentation. The model learns to distinguish foreground objects from background
      regions with meaningful accuracy, even when trained on a small 200-image subset.
    </p>
    <p>
      Key achievements:
    </p>
    <ul>
      <li>Automated dataset acquisition, preprocessing, and model training pipeline</li>
      <li>U-Net with skip connections achieving measurable IoU on Oxford Pet dataset</li>
      <li>Interactive Streamlit dashboard with 5 feature-rich pages</li>
      <li>Real-time webcam segmentation using OpenCV integration</li>
      <li>Quantitative area analysis (foreground/background % per image)</li>
      <li>Method comparison (simple threshold baseline vs U-Net)</li>
      <li>Comprehensive auto-generated documentation</li>
    </ul>
    <p>
      The project demonstrates the power of deep learning for pixel-wise classification tasks and provides
      a solid foundation for more advanced segmentation work.
    </p>
  </div>

  <!-- 11. Future Scope -->
  <div class="section" id="future">
    <h2 class="section-h">11. Future Scope</h2>
    <ul>
      <li><strong>Multi-Class Segmentation</strong>: Extend to classify multiple object categories simultaneously using softmax output</li>
      <li><strong>Larger Datasets</strong>: Train on full Oxford Pet Dataset or COCO for improved generalization</li>
      <li><strong>Advanced Architectures</strong>: Implement DeepLab v3+, SegFormer, or Mask R-CNN for comparison</li>
      <li><strong>Data Augmentation</strong>: Add random flips, crops, color jitter using tf.data pipelines</li>
      <li><strong>TensorFlow Lite Export</strong>: Convert model for mobile/edge deployment</li>
      <li><strong>Instance Segmentation</strong>: Distinguish individual object instances (not just classes)</li>
      <li><strong>Video Segmentation</strong>: Track segmented regions across video frames using temporal consistency</li>
      <li><strong>Transfer Learning</strong>: Use EfficientNet or ResNet backbone in the encoder for better feature extraction</li>
    </ul>
  </div>

  <!-- References -->
  <div class="section">
    <h2 class="section-h">References</h2>
    <ol>
      <li>Ronneberger, O., Fischer, P., & Brox, T. (2015). <em>U-Net: Convolutional Networks for Biomedical Image Segmentation</em>. MICCAI.</li>
      <li>Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). <em>Cats and Dogs</em>. IEEE CVPR.</li>
      <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). <em>Deep Learning</em>. MIT Press.</li>
      <li>Abadi, M. et al. (2016). <em>TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems</em>.</li>
    </ol>
  </div>

</div>

<footer>
  <p>Generated automatically by report_generator.py | {ts}</p>
  <p>Image Segmentation Mini Project — U-Net Semantic Segmentation</p>
</footer>

</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report_generator] HTML report saved → {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
    print("Report generated successfully!")
