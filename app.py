"""
app.py
------
SegmentAI - Ultimate Image Segmentation Dashboard
==================================================
U-Net Deep Learning - Oxford-IIIT Pet Dataset - TensorFlow

Features:
  Home Dashboard      : live metrics + training curves + output gallery
  Upload & Segment    : upload -> segment -> area analysis + download
  Real-Time Webcam    : live OpenCV feed with segmentation overlay
  Comparison View     : U-Net vs simple threshold baseline
  Image History       : last 5 processed images (session cache)
  About & Architecture: architecture diagram + tech details

Run:
    streamlit run app.py
"""

import os
import io
import json
import time
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title  = "SegmentAI - Image Segmentation Dashboard",
    page_icon   = ":microscope:",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Paths
MODEL_PATH   = os.path.join("model",    "unet_model.h5")
HISTORY_PATH = os.path.join("training", "history.json")
METRICS_PATH = os.path.join("training", "metrics.json")
OUTPUT_DIR   = "output"
ARCH_IMG     = os.path.join("docs",     "unet_architecture.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("training", exist_ok=True)

# ── Auto-download model if running on cloud (Render) ──────────────────────────
def _download_model():
    """Download model weights from Google Drive if not present locally."""
    if os.path.exists(MODEL_PATH):
        return

    # 1) Hardcoded fallback – always works without any env var configuration
    file_id = "18MG2wqxiP0Bn98f2kQf8bQCOB2FNeV4N"
    url = f"https://drive.google.com/uc?id={file_id}"

    # 2) Allow override via environment variable (optional)
    env_url = os.environ.get("MODEL_DOWNLOAD_URL", "").strip()
    if env_url:
        url = env_url

    try:
        st.info("⬇️ Downloading model weights… please wait (~360 MB)")
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        st.success("✅ Model downloaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Model download failed: {e}")

_download_model()

# Session State Init
if "image_history" not in st.session_state:
    st.session_state.image_history = []

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #0d1b2a 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0820 0%, #131030 100%) !important;
    border-right: 1px solid rgba(76,201,240,0.15) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(76,201,240,0.2);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(76,201,240,0.18);
    border-color: rgba(76,201,240,0.5);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #4cc9f0, #7209b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55) !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #fff !important;
    border-left: 4px solid #4cc9f0;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    line-height: 1.4;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(114,9,183,0.35), rgba(76,201,240,0.18));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 36px 48px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: #fff !important;
    text-shadow: 0 0 40px rgba(76,201,240,0.4);
    margin-bottom: 8px;
}
.hero-sub {
    color: rgba(255,255,255,0.65) !important;
    font-size: 1.05rem;
}

.info-box {
    background: rgba(76,201,240,0.07);
    border: 1px solid rgba(76,201,240,0.25);
    border-radius: 12px;
    padding: 14px 20px;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.92rem;
    line-height: 1.7;
    margin: 10px 0;
}
.warn-box {
    background: rgba(255,149,0,0.08);
    border: 1px solid rgba(255,149,0,0.3);
    border-radius: 12px;
    padding: 14px 20px;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.92rem;
    line-height: 1.7;
    margin: 10px 0;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 4px;
}
.badge-blue   { background: rgba(76,201,240,0.12); color: #4cc9f0 !important; border: 1px solid rgba(76,201,240,0.4); }
.badge-green  { background: rgba(0,220,100,0.12);  color: #00dc64 !important; border: 1px solid rgba(0,220,100,0.4); }
.badge-purple { background: rgba(114,9,183,0.25);  color: #c084fc !important; border: 1px solid rgba(114,9,183,0.5); }
.badge-orange { background: rgba(255,149,0,0.12);  color: #ff9500 !important; border: 1px solid rgba(255,149,0,0.4); }

.perf-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.perf-label { font-size: 0.8rem; color: rgba(255,255,255,0.5) !important; margin-bottom: 2px; }
.perf-value { font-size: 1.1rem; font-weight: 700; color: #4cc9f0 !important; }

.compact-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(76,201,240,0.2);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
}
.compact-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: #4cc9f0;
}
.compact-lbl {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
}

.img-caption {
    font-size: 0.88rem;
    font-weight: 600;
    color: rgba(255,255,255,0.75) !important;
    margin-top: 8px;
    letter-spacing: 0.3px;
    text-align: center;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    border: 3px dashed #4cc9f0 !important;
    border-radius: 14px !important;
    background: rgba(76,201,240,0.2) !important;
    padding: 30px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border: 3px dashed #ffffff !important;
    background: rgba(76,201,240,0.3) !important;
}
[data-testid="stFileUploadDropzone"] span, [data-testid="stFileUploadDropzone"] small, [data-testid="stFileUploadDropzone"] div {
    color: #ffffff !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: #4cc9f0 !important;
    color: #0f0c29 !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 8px 16px !important;
    margin-top: 10px !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background: #ffffff !important;
    color: #0f0c29 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4361ee, #7209b7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}

button[data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    margin-right: 4px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-bottom: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(76,201,240,0.2) !important;
    color: #4cc9f0 !important;
    border: 1px solid rgba(76,201,240,0.5) !important;
    border-bottom: 3px solid #4cc9f0 !important;
}

table {
    width: 100%;
    color: #ffffff !important;
    background-color: transparent !important;
}
thead tr th, th {
    background-color: rgba(76,201,240,0.2) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    padding: 12px;
}
tbody tr:nth-child(even), tr:nth-child(even) { 
    background-color: rgba(255,255,255,0.05) !important; 
}
tbody tr td, td {
    color: #ffffff !important; 
    padding: 12px;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #00dc64, #3a86ff) !important;
    color: #ffffff !important;
    border: 2px solid rgba(255,255,255,0.5) !important;
    box-shadow: 0 4px 15px rgba(0,220,100,0.4) !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    padding: 10px 20px !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,220,100,0.6) !important;
}

.stImage img {
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: #1a1040 !important;
    border: 1px solid rgba(76,201,240,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {
    color: #ffffff !important;
}
div[data-baseweb="popover"] {
    background-color: #1a1040 !important;
}
ul[data-baseweb="menu"] {
    background-color: #1a1040 !important;
    border: 1px solid rgba(76,201,240,0.5) !important;
}
li[role="option"] {
    background-color: #1a1040 !important;
    color: #ffffff !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"], li[role="option"][aria-highlighted="true"] {
    background-color: #4cc9f0 !important;
    color: #000000 !important;
}

hr { border-color: rgba(255,255,255,0.08) !important; }

p, span, label, div { color: #e2e8f0; }
h1, h2, h3, h4, h5 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# Model Loader
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        import unet as unet_module
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "bce_dice_loss": unet_module.bce_dice_loss,
                "dice_loss":     unet_module.dice_loss,
            }
        )
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


def predict(model, img_array: np.ndarray) -> np.ndarray:
    x    = img_array[np.newaxis, ...]
    pred = model.predict(x, verbose=0)
    return pred[0]


@st.cache_resource
def load_deeplab():
    try:
        import torch
        import torchvision
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load DeepLabV3: {e}")
        return None

def predict_deeplab(model, img_array: np.ndarray) -> np.ndarray:
    import torch
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img_array).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    binary_mask = (output_predictions > 0).astype(np.float32)
    return binary_mask


def preprocess_upload(pil_img: Image.Image, size: int = 128) -> np.ndarray:
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:20px 0 12px;'>
            <div style='font-size:2.5rem; margin-bottom:4px;'>&#128300;</div>
            <div style='font-size:1.25rem; font-weight:900; color:#4cc9f0;'>SegmentAI</div>
            <div style='font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:4px;'>
                U-Net Image Segmentation
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
        <div style='font-size:0.7rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:1px; color:rgba(255,255,255,0.4);
                    padding: 0 0 8px 4px;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Home & Dashboard",
         "Upload & Segment",
         "Real-Time Webcam",
         "Comparison View",
         "Image History",
         "About & Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
        <div style='font-size:0.7rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:1px; color:rgba(255,255,255,0.4);
                    padding: 0 0 8px 4px;'>Model Status</div>
    """, unsafe_allow_html=True)

    model_ok = os.path.exists(MODEL_PATH)
    if model_ok:
        st.markdown('<span class="badge badge-green">&#10003; Model Ready</span>', unsafe_allow_html=True)
        sz = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.markdown(f'<div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-top:6px;">Size: {sz:.1f} MB</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">&#9888; Not Trained</span>', unsafe_allow_html=True)

    n_hist = len(st.session_state.image_history)
    if n_hist > 0:
        st.markdown(f'<div style="margin-top:10px;"><span class="badge badge-blue">{n_hist} in history</span></div>',
                    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:rgba(255,255,255,0.8); font-size:0.8rem; text-align:center; padding-bottom:10px;'>
        Mini Project &middot; U-Net &middot; TensorFlow<br>
        <b>Best performance on pet images (cats &amp; dogs)</b>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE: HOME & DASHBOARD
# =============================================================================
if page == "Home & Dashboard":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>&#128300; Image Segmentation Dashboard</div>
        <div class='hero-sub'>U-Net Deep Learning &nbsp;&middot;&nbsp; Oxford-IIIT Pet Dataset &nbsp;&middot;&nbsp; Real-Time Capable</div>
        <div style='color:#4cc9f0; font-size:1.1rem; font-weight:700; margin-top:12px;'>Best performance on pet images (cats &amp; dogs)</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    output_images = [f for f in os.listdir(OUTPUT_DIR)
                     if f.endswith(".png") and "_segmented" in f]
    n_processed = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    def metric_card(value, label):
        return f"""<div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>"""

    with col1:
        st.markdown(metric_card(str(n_processed), "Images Processed"), unsafe_allow_html=True)
    with col2:
        v = f"{metrics.get('accuracy', 0)*100:.1f}%" if metrics else "N/A"
        st.markdown(metric_card(v, "Pixel Accuracy"), unsafe_allow_html=True)
    with col3:
        v = f"{metrics.get('iou', 0)*100:.1f}%" if metrics else "N/A"
        st.markdown(metric_card(v, "IoU Score"), unsafe_allow_html=True)
    with col4:
        v = f"{metrics.get('dice', 0)*100:.1f}%" if metrics else "N/A"
        st.markdown(metric_card(v, "Dice Coefficient"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training Curves
    if os.path.exists(HISTORY_PATH):
        st.markdown("<div class='section-title'>&#128200; Training History</div>", unsafe_allow_html=True)
        curves_path = os.path.join("training", "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, use_column_width=True)
        else:
            import visualize
            fig = visualize.plot_training_history(HISTORY_PATH)
            st.pyplot(fig)
            plt.close("all")

    # Output Gallery
    all_outputs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    if all_outputs:
        st.markdown("<div class='section-title'>&#128444; Recent Outputs</div>", unsafe_allow_html=True)
        preview = all_outputs[:6]
        cols    = st.columns(min(len(preview), 3), gap="medium")
        for i, fname in enumerate(preview):
            with cols[i % 3]:
                try:
                    img = Image.open(os.path.join(OUTPUT_DIR, fname))
                    st.image(img, use_column_width=True,
                             caption=fname.replace("_comparison.png","").replace("_segmented.png","").replace("_", " ").title())
                except Exception:
                    pass
    else:
        st.markdown("""
        <div class='info-box'>
            No output images yet. Run <b>python main.py</b> or use the <b>Upload &amp; Segment</b> page.
        </div>""", unsafe_allow_html=True)

    # Detailed Metrics
    if metrics:
        st.markdown("<div class='section-title'>&#128202; Detailed Metrics</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.table({
                "Metric":  ["Pixel Accuracy", "IoU (Jaccard)", "Dice Coefficient"],
                "Value":   [f"{metrics['accuracy']:.4f}",
                            f"{metrics['iou']:.4f}",
                            f"{metrics['dice']:.4f}"],
                "Std Dev": [f"+/-{metrics.get('accuracy_std',0):.4f}",
                            f"+/-{metrics.get('iou_std',0):.4f}",
                            f"+/-{metrics.get('dice_std',0):.4f}"],
            })

        with col_b:
            names  = ["Accuracy", "IoU", "Dice"]
            values = [metrics["accuracy"], metrics["iou"], metrics["dice"]]
            colors = ["#4cc9f0", "#7209b7", "#3a86ff"]
            fig_m, ax_m = plt.subplots(figsize=(5, 2.6))
            fig_m.patch.set_facecolor("#1a1a2e")
            ax_m.set_facecolor("#1a1a2e")
            bars = ax_m.barh(names, values, color=colors, height=0.45, alpha=0.9)
            for b, v in zip(bars, values):
                ax_m.text(v + 0.01, b.get_y() + b.get_height()/2,
                          f"{v:.3f}", va="center", color="white", fontsize=9)
            ax_m.set_xlim(0, 1.18)
            ax_m.spines[["top","right","left","bottom"]].set_visible(False)
            ax_m.tick_params(colors="white", labelsize=9)
            st.pyplot(fig_m)
            plt.close("all")


# =============================================================================
# PAGE: UPLOAD & SEGMENT
# =============================================================================
elif page == "Upload & Segment":
    import visualize
    import evaluate as eval_module

    st.markdown("<div class='section-title'>&#128228; Upload &amp; Segment</div>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("""
        <div class='warn-box'>
            Model not found. Please run <code>python main.py</code> first to train.
        </div>""", unsafe_allow_html=True)
        st.stop()

    ctrl1, ctrl2 = st.columns([2, 1], gap="large")
    with ctrl1:
        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Upload any image to get a U-Net segmentation prediction"
        )
    with ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice_upload = st.selectbox("Active Model", ["U-Net (default)", "DeepLabV3 (Pretrained - Multiple Objects)"])
        threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.05, max_value=0.95, value=0.30, step=0.05,
            help="Pixel confidence above this = foreground. Lower = more foreground detected."
        )
        st.markdown(
            f'<div class="info-box" style="padding:8px 12px; font-size:0.82rem;">'
            f'Threshold: <b style="color:#4cc9f0;">{threshold:.2f}</b></div>',
            unsafe_allow_html=True)

    if uploaded is not None:
        pil_img = Image.open(uploaded)

        t0 = time.time()
        model_name = model_choice_upload.split(' ')[0]
        with st.spinner(f"Running {model_name} segmentation..."):
            img_arr = preprocess_upload(pil_img, size=128)
            if "DeepLab" in model_choice_upload:
                dl_model = load_deeplab()
                if dl_model:
                    pred_mask = predict_deeplab(dl_model, img_arr)
                else:
                    pred_mask = np.zeros((128,128), dtype=np.float32)
            else:
                pred_mask = predict(model, img_arr)
        elapsed = time.time() - t0

        auto_t     = eval_module.adaptive_threshold(pred_mask)
        confidence = eval_module.confidence_score(pred_mask)

        st.success("Segmentation successful!")
        st.warning("Please note: Best performance is observed on pet images (cats & dogs).")

        # Performance Panel
        st.markdown("<div class='section-title'>&#9889; Performance Panel</div>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3, gap="medium")
        with p1:
            st.markdown(f"""<div class='perf-item'>
                <div class='perf-label'>Processing Time</div>
                <div class='perf-value'>{elapsed*1000:.0f} ms</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class='perf-item'>
                <div class='perf-label'>Model Confidence</div>
                <div class='perf-value'>{confidence*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            st.markdown(f"""<div class='perf-item'>
                <div class='perf-label'>Threshold Used</div>
                <div class='perf-value'>{threshold:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Segmentation Results", "Probability Heatmap", "Before / After"])

        with tab1:
            st.markdown("<div class='section-title'>Segmentation Results</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3, gap="medium")

            with c1:
                orig_disp = (img_arr * 255).astype(np.uint8)
                st.image(orig_disp, use_column_width=True)
                st.markdown("<div class='img-caption'>Original Image</div>", unsafe_allow_html=True)

            with c2:
                overlay = visualize.overlay_mask_on_image(img_arr, pred_mask, threshold=threshold)
                st.image(overlay, use_column_width=True)
                st.markdown("<div class='img-caption'>Predicted Segmentation</div>", unsafe_allow_html=True)

            with c3:
                bw_mask = (pred_mask > threshold).astype(np.uint8) * 255
                if bw_mask.ndim == 3 and bw_mask.shape[-1] == 1:
                    bw_mask = bw_mask.squeeze(-1)
                st.image(bw_mask, use_column_width=True, clamp=False, output_format="PNG")
                st.markdown("<div class='img-caption'>Binary Mask (White=Object)</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='section-title'>Probability Heatmap</div>", unsafe_allow_html=True)
            st.markdown("""<div class='info-box'>
                This heatmap shows the raw prediction probabilities from U-Net.
                Bright colors = high probability of foreground.
            </div>""", unsafe_allow_html=True)
            cmap_choice = st.selectbox("Colormap", ["plasma", "viridis", "hot", "jet", "inferno"])
            h1, h2, h3 = st.columns(3, gap="medium")
            with h1:
                st.image((img_arr * 255).astype(np.uint8), use_column_width=True,
                         caption="Original Image")
            with h2:
                heatmap = visualize.prediction_heatmap(pred_mask, colormap=cmap_choice)
                st.image(heatmap, use_column_width=True, caption=f"Prediction Heatmap ({cmap_choice})")
            with h3:
                hm_overlay = (img_arr * 0.45 * 255 + heatmap * 0.55).clip(0, 255).astype(np.uint8)
                st.image(hm_overlay, use_column_width=True, caption="Blended Heatmap + Image")

        with tab3:
            st.markdown("<div class='section-title'>Before / After Comparison</div>", unsafe_allow_html=True)
            split_pct = st.slider("Split Position (%)", 10, 90, 50, 5,
                                  help="Adjust to reveal the segmentation")
            ba_img = visualize.before_after_blend(img_arr, pred_mask,
                                                  split=split_pct/100.0, threshold=threshold)
            st.image(ba_img, use_column_width=True,
                     caption=f"Original (Left, {split_pct}%) vs Segmented (Right)")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Area Analysis
        st.markdown("<div class='section-title'>&#128208; Area Analysis</div>", unsafe_allow_html=True)
        area_info = eval_module.area_percentage(pred_mask, threshold=threshold)
        
        col_c, col_d = st.columns([1, 1], gap="large")
        with col_c:
            a1, a2, a3 = st.columns(3, gap="small")
            with a1:
                st.markdown(f"""<div class='compact-card'>
                    <div class='compact-val' style='color:#00dc64;'>{area_info['foreground_pct']:.1f}%</div>
                    <div class='compact-lbl'>Foreground</div>
                </div>""", unsafe_allow_html=True)
            with a2:
                st.markdown(f"""<div class='compact-card'>
                    <div class='compact-val' style='color:#4361ee;'>{area_info['background_pct']:.1f}%</div>
                    <div class='compact-lbl'>Background</div>
                </div>""", unsafe_allow_html=True)
            with a3:
                st.markdown(f"""<div class='compact-card'>
                    <div class='compact-val' style='color:#e2e8f0; font-size:1.2rem;'>{area_info['total_px']:,}</div>
                    <div class='compact-lbl'>Total Pixels</div>
                </div>""", unsafe_allow_html=True)
                
        with col_d:
            fig_area, ax_area = plt.subplots(figsize=(5, 1.2))
            fig_area.patch.set_facecolor("#1a1a2e")
            ax_area.set_facecolor("#1a1a2e")
            labels = ["Foreground", "Background"]
            pcts = [area_info["foreground_pct"], area_info["background_pct"]]
            colors = ["#00dc64", "#4361ee"]
            bars = ax_area.barh(labels, pcts, color=colors, height=0.4)
            for bar, pct in zip(bars, pcts):
                ax_area.text(pct + 2, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%", 
                             va='center', color='white', fontweight='bold', fontsize=10)
            ax_area.set_xlim(0, 110)
            ax_area.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax_area.tick_params(colors="white", labelsize=10, left=False, bottom=False)
            ax_area.set_xticks([])
            st.pyplot(fig_area)
            plt.close("all")

        # Downloads
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>&#128190; Save &amp; Download</div>", unsafe_allow_html=True)

        save_name = os.path.splitext(uploaded.name)[0]
        save_path = visualize.save_segmented_output(img_arr, pred_mask, save_name, threshold)
        st.success(f"Segmented output saved to: {save_path}")

        dl1, dl2, dl3 = st.columns(3, gap="medium")
        with dl1:
            st.download_button(
                "Download Overlay",
                data=pil_to_bytes(Image.fromarray(overlay)),
                file_name=f"{save_name}_overlay.png",
                mime="image/png",
            )
        with dl2:
            st.download_button(
                "Download Binary Mask",
                data=pil_to_bytes(Image.fromarray(bw_mask)),
                file_name=f"{save_name}_mask.png",
                mime="image/png",
            )
        with dl3:
            st.download_button(
                "Download Original (128px)",
                data=pil_to_bytes(Image.fromarray((img_arr * 255).astype(np.uint8))),
                file_name=f"{save_name}_original.png",
                mime="image/png",
            )

        # Add to Image History
        entry = {
            "name":      uploaded.name,
            "timestamp": time.strftime("%H:%M:%S"),
            "overlay":   overlay.copy(),
            "mask":      bw_mask.copy(),
            "original":  (img_arr * 255).astype(np.uint8),
            "fg_pct":    area_info["foreground_pct"],
            "bg_pct":    area_info["background_pct"],
            "conf":      round(confidence * 100, 1),
            "time_ms":   round(elapsed * 1000),
        }
        existing = [h for h in st.session_state.image_history if h["name"] != uploaded.name]
        st.session_state.image_history = ([entry] + existing)[:5]


# =============================================================================
# PAGE: REAL-TIME WEBCAM
# =============================================================================
elif page == "Real-Time Webcam":
    import visualize
    import evaluate as eval_module

    st.markdown("<div class='section-title'>&#128247; Real-Time Webcam Segmentation</div>",
                unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("<div class='warn-box'>Model not found. Run <code>python main.py</code> first.</div>",
                    unsafe_allow_html=True)
        st.stop()

    st.markdown("""<div class='info-box'>
        Live segmentation using your webcam via OpenCV.
        Click <b>Start Webcam</b> to begin. The model runs every N frames for smooth performance.
    </div>""", unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3, gap="medium")
    with wc1:
        threshold_rt = st.slider("Threshold", 0.05, 0.95, 0.30, 0.05, key="rt_thresh")
    with wc2:
        seg_interval = st.slider("Segment every N frames", 1, 10, 3, key="rt_interval")
    with wc3:
        model_choice = st.selectbox("Active Model", 
            ["U-Net (default)", "DeepLabV3 (Pretrained - Multiple Objects)", "Edge detection mode (fun feature)", "Fast mode (lightweight)"]
        )
        
    show_split = st.checkbox("Before/After split view", value=False)

    col_l, col_r = st.columns(2, gap="medium")
    start_btn = col_l.button("Start Webcam", type="primary")
    stop_btn  = col_r.button("Stop")

    frame_display = st.empty()
    stats_display = st.empty()

    if start_btn:
        try:
            import cv2
        except ImportError:
            st.error("OpenCV not installed. Run: pip install opencv-python")
            st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No webcam detected. Connect a webcam and try again.")
            st.stop()

        frame_count = 0
        cached_pred = None
        fps_times   = []

        try:
            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                t_frame     = time.time()

                frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_small = cv2.resize(frame_rgb, (128, 128))
                img_arr     = frame_small.astype(np.float32) / 255.0

                if frame_count % seg_interval == 0 or cached_pred is None:
                    if "U-Net" in model_choice:
                        cached_pred = predict(model, img_arr)
                    elif "DeepLab" in model_choice:
                        dl_model = load_deeplab()
                        if dl_model:
                            cached_pred = predict_deeplab(dl_model, img_arr)
                        else:
                            cached_pred = np.zeros((128,128), dtype=np.float32)
                    elif "fast" in model_choice.lower() or "lightweight" in model_choice.lower():
                        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
                        # Fast adaptive thresholding acting as lightweight segmentation
                        _, th = cv2.threshold(gray, int(threshold_rt * 255), 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        cached_pred = (th / 255.0).astype(np.float32)
                    elif "edge" in model_choice.lower():
                        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        cached_pred = (edges / 255.0).astype(np.float32)

                if show_split:
                    disp128 = visualize.before_after_blend(img_arr, cached_pred,
                                                           split=0.5, threshold=threshold_rt)
                else:
                    disp128 = visualize.overlay_mask_on_image(img_arr, cached_pred,
                                                              threshold=threshold_rt)

                display  = cv2.resize(disp128,  (384, 384), interpolation=cv2.INTER_NEAREST)
                orig_d   = cv2.resize(frame_small, (384, 384))
                combined = np.concatenate([orig_d, display], axis=1)

                frame_display.image(combined, caption="Original (Left) | Segmented (Right)",
                                    use_column_width=True)

                fps_times.append(1.0 / max(time.time() - t_frame, 0.001))
                if len(fps_times) > 20: fps_times.pop(0)
                fps  = np.mean(fps_times)
                area = eval_module.area_percentage(cached_pred, threshold=threshold_rt)

                stats_display.markdown(f"""
                <div style='display:flex; gap:16px; margin-top:6px;'>
                    <div class='perf-item' style='flex:1'>
                        <div class='perf-label'>FPS</div>
                        <div class='perf-value'>{fps:.1f}</div>
                    </div>
                    <div class='perf-item' style='flex:1'>
                        <div class='perf-label'>Foreground</div>
                        <div class='perf-value'>{area['foreground_pct']:.1f}%</div>
                    </div>
                    <div class='perf-item' style='flex:1'>
                        <div class='perf-label'>Background</div>
                        <div class='perf-value'>{area['background_pct']:.1f}%</div>
                    </div>
                    <div class='perf-item' style='flex:1'>
                        <div class='perf-label'>Frame</div>
                        <div class='perf-value'>#{frame_count}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                time.sleep(0.03)
        finally:
            cap.release()
            frame_display.success("Webcam session ended.")


# =============================================================================
# PAGE: COMPARISON VIEW
# =============================================================================
elif page == "Comparison View":
    import visualize
    import evaluate as eval_module

    st.markdown("<div class='section-title'>&#128270; Method Comparison: U-Net vs Threshold Baseline</div>",
                unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("<div class='warn-box'>Model not found. Run <code>python main.py</code> first.</div>",
                    unsafe_allow_html=True)
        st.stop()

    st.markdown("""<div class='info-box'>
        Upload an image to compare Simple Threshold baseline vs U-Net deep learning side-by-side.
    </div>""", unsafe_allow_html=True)

    cmp1, cmp2 = st.columns([2, 1], gap="large")
    with cmp1:
        uploaded_cmp = st.file_uploader("Upload image for comparison",
                                        type=["jpg","jpeg","png"], key="cmp_uploader")
    with cmp2:
        st.markdown("<br>", unsafe_allow_html=True)
        threshold_cmp = st.slider("U-Net Threshold", 0.05, 0.95, 0.30, 0.05, key="cmp_thresh")
        simple_t      = st.slider("Simple Threshold", 0.1, 0.9, 0.5, 0.05, key="simple_thresh")

    if uploaded_cmp:
        pil_img = Image.open(uploaded_cmp)
        img_arr = preprocess_upload(pil_img, size=128)

        with st.spinner("Running U-Net..."):
            t0 = time.time()
            pred_mask = predict(model, img_arr)
            unet_ms   = (time.time() - t0) * 1000

        t1 = time.time()
        gray     = np.mean(img_arr, axis=-1, keepdims=True)
        gray_nm  = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
        simple_mask = (gray_nm > simple_t).astype(np.float32)
        simple_ms = (time.time() - t1) * 1000

        auto_t = eval_module.adaptive_threshold(pred_mask)

        st.markdown("<hr>", unsafe_allow_html=True)

        img_u8 = (img_arr * 255).astype(np.uint8)
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.image(img_u8, use_column_width=True, caption="Original Image")

        with col2:
            s_overlay = visualize.overlay_mask_on_image(img_arr, simple_mask, threshold=simple_t)
            st.image(s_overlay, use_column_width=True, caption=f"Simple Threshold (t={simple_t:.2f})")
            s_area = eval_module.area_percentage(simple_mask, threshold=simple_t)
            st.markdown(f"""<div class='info-box'>
                FG: <b>{s_area['foreground_pct']:.1f}%</b> &middot; BG: <b>{s_area['background_pct']:.1f}%</b>
                &middot; Time: <b>{simple_ms:.1f}ms</b></div>""", unsafe_allow_html=True)

        with col3:
            u_overlay = visualize.overlay_mask_on_image(img_arr, pred_mask, threshold=threshold_cmp)
            st.image(u_overlay, use_column_width=True, caption=f"U-Net (t={threshold_cmp:.2f})")
            u_area = eval_module.area_percentage(pred_mask, threshold=threshold_cmp)
            st.markdown(f"""<div class='info-box'>
                FG: <b>{u_area['foreground_pct']:.1f}%</b> &middot; BG: <b>{u_area['background_pct']:.1f}%</b>
                &middot; Time: <b>{unet_ms:.1f}ms</b></div>""", unsafe_allow_html=True)

        # Binary masks
        st.markdown("<div class='section-title'>Binary Masks</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2, gap="large")
        with m1:
            st.image(visualize.binary_mask_image(simple_mask, threshold=simple_t),
                     use_column_width=True, caption="Simple Threshold Binary Mask")
        with m2:
            st.image(visualize.binary_mask_image(pred_mask, threshold=threshold_cmp),
                     use_column_width=True, caption="U-Net Binary Mask")

        # Load metrics for accurate table if they exist
        sys_metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f: sys_metrics = json.load(f)
            
        unet_acc = f"{sys_metrics.get('accuracy', 0)*100:.1f}%" if sys_metrics else "N/A"
        unet_iou = f"{sys_metrics.get('iou', 0)*100:.1f}%" if sys_metrics else "N/A"

        # Comparison table
        st.markdown("<div class='section-title'>Method Comparison Table</div>", unsafe_allow_html=True)
        st.table({
            "Method":          ["Simple Thresholding", "U-Net (Deep Learning)", "DeepLab / MobileNet (Reference)"],
            "Accuracy":        ["~55.0%", unet_acc, "92.3% (Est)"],
            "IoU Score":       ["~30.0%", unet_iou, "82.5% (Est)"],
            "Processing Speed":[f"{simple_ms:.1f} ms", f"{unet_ms:.1f} ms", "~120.5 ms"],
            "Complexity":      ["Very Low (O(1))", "High (~31M Params)", "Medium (~3.2M Params)"],
        })


# =============================================================================
# PAGE: IMAGE HISTORY
# =============================================================================
elif page == "Image History":
    st.markdown("<div class='section-title'>&#128336; Image History (Last 5 Segmentations)</div>",
                unsafe_allow_html=True)

    history = st.session_state.image_history

    if not history:
        st.markdown("""<div class='info-box'>
            No images processed yet in this session.<br>
            Go to <b>Upload &amp; Segment</b> to start.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='info-box'>
            Showing <b>{len(history)}</b> recently processed image(s) this session.
        </div>""", unsafe_allow_html=True)

        for idx, entry in enumerate(history):
            with st.expander(
                f"{entry['name']}  |  {entry['timestamp']}  |  "
                f"FG: {entry['fg_pct']:.1f}%  |  BG: {entry['bg_pct']:.1f}%  |  "
                f"{entry['time_ms']} ms",
                expanded=(idx == 0)
            ):
                h1, h2, h3 = st.columns(3, gap="medium")
                with h1:
                    st.image(entry["original"], use_column_width=True, caption="Original Image")
                with h2:
                    st.image(entry["overlay"],  use_column_width=True, caption="Segmentation Overlay")
                with h3:
                    st.image(entry["mask"],     use_column_width=True, caption="Binary Mask")

                s1, s2, s3, s4 = st.columns(4, gap="small")
                with s1:
                    st.markdown(f"""<div class='perf-item'>
                        <div class='perf-label'>Foreground</div>
                        <div class='perf-value'>{entry['fg_pct']:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""<div class='perf-item'>
                        <div class='perf-label'>Background</div>
                        <div class='perf-value'>{entry['bg_pct']:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""<div class='perf-item'>
                        <div class='perf-label'>Confidence</div>
                        <div class='perf-value'>{entry['conf']:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""<div class='perf-item'>
                        <div class='perf-label'>Time</div>
                        <div class='perf-value'>{entry['time_ms']} ms</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Clear History", type="secondary"):
            st.session_state.image_history = []
            st.rerun()


# =============================================================================
# PAGE: ABOUT & ARCHITECTURE
# =============================================================================
elif page == "About & Architecture":
    st.markdown("<div class='section-title'>&#128214; Project Documentation &amp; Model Architecture</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='hero-banner' style='padding:28px 36px; text-align:left;'>
        <h2 style='margin-top:0; color:#4cc9f0; font-weight:900;'>&#127988; SegmentAI — Image Segmentation Dashboard</h2>
        <p style='color:rgba(255,255,255,0.85); font-size:1.05rem; line-height:1.8;'>
            <b>Problem:</b> Accurate semantic segmentation is hard due to lighting, poses, and occlusions.<br>
            <b>Objective:</b> Professional AI dashboard isolating pets and objects using deep learning.<br>
            <b>Dataset:</b> Oxford-IIIT Pet Dataset — 37 breeds, ~7,000 labeled images.
        </p>
        <div style='display:flex; gap:10px; flex-wrap:wrap; margin-top:8px;'>
            <span class='badge badge-blue'>TensorFlow 2.x</span>
            <span class='badge badge-purple'>U-Net</span>
            <span class='badge badge-green'>DeepLabV3</span>
            <span class='badge badge-orange'>OpenCV</span>
            <span class='badge badge-blue'>PyTorch</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Upload & Segment ──
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(76,201,240,0.12),rgba(114,9,183,0.08));
                border:1px solid rgba(76,201,240,0.35);border-radius:18px;padding:22px 26px;margin:20px 0 8px;'>
        <h3 style='color:#4cc9f0;font-weight:900;margin:0 0 4px;'>&#128228; Section 1 — Upload &amp; Segment</h3>
        <p style='color:rgba(255,255,255,0.55);font-size:0.88rem;margin:0;'>2 selectable models for image upload segmentation</p>
    </div>""", unsafe_allow_html=True)

    u1, u2 = st.columns(2, gap="large")
    with u1:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#4cc9f0;font-size:1.05rem;'>&#129302; Model 1 — U-Net (Custom Trained)</b><br><br>
            <b>Type:</b> Convolutional encoder-decoder with skip connections<br>
            <b>Trained on:</b> Oxford-IIIT Pet Dataset (cats &amp; dogs)<br>
            <b>Architecture:</b> Encoder (64→128→256→512) + Bottleneck (1024) + Decoder with skip connections<br>
            <b>Output:</b> 128×128 sigmoid probability mask<br>
            <b>Loss:</b> Binary Cross-Entropy + Dice Loss<br>
            <b>Params:</b> ~31M | <b>Size:</b> ~360 MB | <b>Speed:</b> ~50ms<br>
            <b>Best for:</b> Cats &amp; Dogs
        </div>""", unsafe_allow_html=True)
    with u2:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#c084fc;font-size:1.05rem;'>&#127758; Model 2 — DeepLabV3 (Pretrained, PyTorch)</b><br><br>
            <b>Type:</b> Atrous Convolution + ASPP (Atrous Spatial Pyramid Pooling)<br>
            <b>Backbone:</b> ResNet-50 pretrained on COCO dataset<br>
            <b>Key Feature:</b> Multi-scale context via dilated convolutions<br>
            <b>Output:</b> 21-class semantic segmentation map<br>
            <b>No training needed</b> — pretrained weights from torchvision<br>
            <b>Params:</b> ~39M | <b>Speed:</b> ~120ms<br>
            <b>Detects:</b> Person, Car, Cat, Dog, Chair, Road + 15 more
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='text-align:center;margin-top:8px;'>
        <b>Upload &amp; Segment Pipeline:</b><br>
        &#128229; Upload &rarr; &#9881; Resize 128&times;128 &rarr; &#129302; Model Inference &rarr; &#127912; Mask Generation &rarr; &#128202; Area Analysis &rarr; &#128190; Download
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Real-Time Webcam ──
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(0,220,100,0.1),rgba(58,134,255,0.08));
                border:1px solid rgba(0,220,100,0.3);border-radius:18px;padding:22px 26px;margin:20px 0 8px;'>
        <h3 style='color:#00dc64;font-weight:900;margin:0 0 4px;'>&#128247; Section 2 — Real-Time Webcam</h3>
        <p style='color:rgba(255,255,255,0.55);font-size:0.88rem;margin:0;'>4 selectable models for live camera segmentation</p>
    </div>""", unsafe_allow_html=True)

    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#4cc9f0;'>&#129302; Model 1 — U-Net (default)</b><br>
            Same trained U-Net runs on every Nth webcam frame.<br>
            <b>Speed:</b> ~50ms/frame | Best for pet images (cats &amp; dogs)
        </div>
        <div class='info-box' style='margin-top:8px;'>
            <b style='color:#c084fc;'>&#127758; Model 2 — DeepLabV3 (Pretrained)</b><br>
            Pretrained COCO model applied live. Multi-object: people, cars, chairs, pets.<br>
            <b>Speed:</b> ~120ms/frame | Use N=5+ frames for smooth output
        </div>""", unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#00dc64;'>&#9889; Model 3 — Fast Mode (Otsu Thresholding)</b><br>
            Classical OpenCV: grayscale + adaptive binary threshold.<br>
            <b>Algorithm:</b> cv2.THRESH_BINARY_INV + THRESH_OTSU<br>
            <b>Speed:</b> &lt;5ms | No neural network — pure classical CV
        </div>
        <div class='info-box' style='margin-top:8px;'>
            <b style='color:#ff9500;'>&#128165; Model 4 — Canny Edge Detection</b><br>
            Computes image gradients and marks strong edges as foreground.<br>
            <b>Algorithm:</b> cv2.Canny(gray, low=50, high=150)<br>
            <b>Speed:</b> &lt;3ms | Detects boundaries only (not semantic)
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='text-align:center;margin-top:8px;'>
        <b>Webcam Pipeline:</b><br>
        &#127909; Capture Frame &rarr; &#128260; Resize 128&times;128 &rarr; &#129302; Segment every N frames &rarr; &#128444; Overlay Mask &rarr; &#128225; Display 384&times;384 &rarr; &#128202; Live FPS
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Comparison View ──
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(255,149,0,0.1),rgba(255,60,60,0.06));
                border:1px solid rgba(255,149,0,0.3);border-radius:18px;padding:22px 26px;margin:20px 0 8px;'>
        <h3 style='color:#ff9500;font-weight:900;margin:0 0 4px;'>&#128270; Section 3 — Comparison View</h3>
        <p style='color:rgba(255,255,255,0.55);font-size:0.88rem;margin:0;'>U-Net Deep Learning vs Classical Thresholding — side by side</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#ff9500;font-size:1.0rem;'>&#128200; Method A — Simple Threshold (Baseline)</b><br><br>
            Classical non-AI pixel intensity method. No neural network.<br><br>
            <b>Steps:</b><br>
            &nbsp;&nbsp;1. Convert RGB &rarr; Grayscale (mean of channels)<br>
            &nbsp;&nbsp;2. Normalize to 0&ndash;1 range<br>
            &nbsp;&nbsp;3. Pixel &gt; threshold &rarr; Foreground (white)<br>
            &nbsp;&nbsp;4. Pixel &le; threshold &rarr; Background (black)<br><br>
            <b>Accuracy:</b> ~55% | <b>IoU:</b> ~30% | <b>Speed:</b> &lt;1ms<br>
            <b>Limitation:</b> Fails on complex textures and similar colors
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#4cc9f0;font-size:1.0rem;'>&#129302; Method B — U-Net Deep Learning</b><br><br>
            Trained CNN that understands semantic meaning — not just pixel color.<br><br>
            <b>Advantages:</b><br>
            &nbsp;&nbsp;• Understands object shape and context<br>
            &nbsp;&nbsp;• Robust to lighting variation and texture<br>
            &nbsp;&nbsp;• Skip connections preserve fine edge detail<br>
            &nbsp;&nbsp;• Trained on 6,000+ labeled images<br><br>
            <b>Accuracy:</b> ~70.4% | <b>Speed:</b> ~50ms<br>
            <b>Advantage:</b> Semantically aware segmentation
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top:10px;'>
        <b>&#128202; All Models Comparison:</b><br><br>
        <table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
            <thead>
                <tr style='background:rgba(76,201,240,0.2);'>
                    <th style='padding:10px;text-align:left;color:#fff;'>Model</th>
                    <th style='padding:10px;text-align:center;color:#fff;'>Section</th>
                    <th style='padding:10px;text-align:center;color:#fff;'>Accuracy</th>
                    <th style='padding:10px;text-align:center;color:#fff;'>Speed</th>
                    <th style='padding:10px;text-align:center;color:#fff;'>AI?</th>
                </tr>
            </thead>
            <tbody>
                <tr style='background:rgba(255,255,255,0.04);'>
                    <td style='padding:9px;color:#4cc9f0;'>U-Net (Custom Trained)</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>Upload, Webcam, Compare</td>
                    <td style='padding:9px;text-align:center;color:#4cc9f0;'>~70.4%</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>~50ms</td>
                    <td style='padding:9px;text-align:center;color:#00dc64;'>&#10003; Yes</td>
                </tr>
                <tr style='background:rgba(255,255,255,0.08);'>
                    <td style='padding:9px;color:#c084fc;'>DeepLabV3 (Pretrained)</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>Upload, Webcam</td>
                    <td style='padding:9px;text-align:center;color:#c084fc;'>~92% (est)</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>~120ms</td>
                    <td style='padding:9px;text-align:center;color:#00dc64;'>&#10003; Yes</td>
                </tr>
                <tr style='background:rgba(255,255,255,0.04);'>
                    <td style='padding:9px;color:#00dc64;'>Fast Mode (Otsu)</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>Webcam</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>~45%</td>
                    <td style='padding:9px;text-align:center;color:#00dc64;'>&lt;5ms</td>
                    <td style='padding:9px;text-align:center;color:#ff4444;'>&#10007; No</td>
                </tr>
                <tr style='background:rgba(255,255,255,0.08);'>
                    <td style='padding:9px;color:#ff9500;'>Canny Edge Detection</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>Webcam</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>N/A</td>
                    <td style='padding:9px;text-align:center;color:#00dc64;'>&lt;3ms</td>
                    <td style='padding:9px;text-align:center;color:#ff4444;'>&#10007; No</td>
                </tr>
                <tr style='background:rgba(255,255,255,0.04);'>
                    <td style='padding:9px;color:#ff9500;'>Simple Threshold</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>Compare</td>
                    <td style='padding:9px;text-align:center;color:#fff;'>~55%</td>
                    <td style='padding:9px;text-align:center;color:#00dc64;'>&lt;1ms</td>
                    <td style='padding:9px;text-align:center;color:#ff4444;'>&#10007; No</td>
                </tr>
            </tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── U-Net Architecture Deep Dive ──
    st.markdown("<div class='section-title'>&#9881; U-Net Architecture — Deep Dive</div>", unsafe_allow_html=True)

    arch1, arch2 = st.columns([1.4, 1], gap="large")
    with arch1:
        if os.path.exists(ARCH_IMG):
            st.image(ARCH_IMG, caption="U-Net Encoder-Decoder with Skip Connections", use_column_width=True)
        else:
            st.markdown("""
            <div class='info-box' style='text-align:center;padding:40px;'>
                <div style='font-size:3rem;'>&#127959;</div>
                <div style='font-size:1rem;color:rgba(255,255,255,0.6);margin-top:8px;'>Architecture diagram available after first training run.</div>
            </div>""", unsafe_allow_html=True)
    with arch2:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#4cc9f0;'>Encoder (Contracting Path)</b><br>
            Conv2D(64) &rarr; MaxPool &rarr; Conv2D(128) &rarr; MaxPool<br>
            Conv2D(256) &rarr; MaxPool &rarr; Conv2D(512) &rarr; MaxPool<br>
            <i style='color:rgba(255,255,255,0.5);'>Extracts features, doubles channels each step</i>
        </div>
        <div class='info-box' style='margin-top:8px;'>
            <b style='color:#ff9500;'>Bottleneck</b><br>
            Conv2D(1024) &rarr; Conv2D(1024)<br>
            <i style='color:rgba(255,255,255,0.5);'>Highest-level abstract representation</i>
        </div>
        <div class='info-box' style='margin-top:8px;'>
            <b style='color:#00dc64;'>Decoder (Expanding Path)</b><br>
            UpSample + Concat(skip) &rarr; Conv2D(512)<br>
            UpSample + Concat(skip) &rarr; Conv2D(256)<br>
            UpSample + Concat(skip) &rarr; Conv2D(128)<br>
            UpSample + Concat(skip) &rarr; Conv2D(64)<br>
            Conv2D(1, sigmoid) &rarr; Output Mask<br>
            <i style='color:rgba(255,255,255,0.5);'>Recovers spatial resolution with skip context</i>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training Details ──
    st.markdown("<div class='section-title'>&#128200; Training Configuration</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#4cc9f0;'>&#128190; Dataset</b><br><br>
            Oxford-IIIT Pet Dataset<br>
            37 pet breeds, ~7,000 images<br>
            Trimap pixel annotations<br>
            Input: 128&times;128 RGB
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#00dc64;'>&#9881; Training Setup</b><br><br>
            Optimizer: Adam (lr=1e-4)<br>
            Loss: BCE + Dice Loss<br>
            Epochs: 15&ndash;30 | Batch: 16<br>
            Framework: TensorFlow 2.x
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class='info-box'>
            <b style='color:#ff9500;'>&#127942; Results</b><br><br>
            Pixel Accuracy: 70.4%<br>
            Model Size: ~360 MB<br>
            Parameters: ~31M<br>
            Inference: ~50ms/image
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    report_path = os.path.join("docs", "report.html")
    if st.button("Generate Academic HTML Report", type="primary"):
        import subprocess
        subprocess.run(["python", "docs/report_generator.py"])
        st.success(f"Report Generated! Check {report_path}")
        if os.path.exists(report_path):
            st.info(f"Full project report available at: {report_path}")
