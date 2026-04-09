import streamlit as st 
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np

# =========================
# ⚙️ PARAMS
# =========================
DROPOUT = 0.624
HIDDEN_UNITS = 96
NUM_CLASSES = 6

device = torch.device("cpu")

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="🔆",
    layout="centered"
)

# =========================
# 🎨 CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Center everything */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background: rgba(255,255,255,0.03) !important;
    text-align: center;
}

[data-testid="stFileUploader"]:hover {
    border-color: #22c55e !important;
}

/* Prediction card */
.pred-box {
    padding: 25px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🏷️ HERO SECTION
# =========================
st.markdown("""
<div style='text-align:center; padding: 30px 0;'>

<h1 style="
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
☀️ Solar Panel Defect Classifier
</h1>

<p style="
    font-size:18px;
    color:#94a3b8;
    margin-top:10px;
">
AI-powered defect detection for solar panels ⚡
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# 🧠 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, HIDDEN_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_UNITS, NUM_CLASSES)
    )

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model

model = load_model()

# =========================
# 📤 UPLOAD SECTION (CENTERED CARD)
# =========================
st.markdown("""
<div style="
    display:flex;
    justify-content:center;
">
<div style="
    width: 500px;
    padding: 25px;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    text-align:center;
">
<h3 style="margin-bottom:15px;">📤 Upload Image</h3>
</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png']
)

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",             
    "Electrical-damage", 
    "Physical-damage",
    "Snow-Covered"
]

# =========================
# 🔍 PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocessing
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_class = CLASSES[pred_idx]

    # Prediction card
    st.markdown(f"""
    <div class="pred-box">
        <h2>Prediction: {predicted_class}</h2>
        <p style="font-size:18px;">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if predicted_class == "Clean":
        st.success("✅ Panel is healthy and clean")
    else:
        st.error("⚠️ Defect detected on panel")

    # Top 3
    st.write("### 📊 Top Predictions")
    top3 = torch.topk(probs, 3)

    for i in range(3):
        idx = top3.indices[0][i].item()
        prob = top3.values[0][i].item()
        st.progress(prob)
        st.write(f"{CLASSES[idx]} — {prob:.1%}")

# =========================
# 👤 SIDEBAR
# =========================
st.sidebar.title("⚡ About App")

st.sidebar.info("""
This AI model detects defects in solar panels using deep learning.

Built with:
- PyTorch
- EfficientNet
- Streamlit
""")

st.sidebar.markdown("### 👨‍💻 Connect")
st.sidebar.markdown("""
[GitHub](https://github.com/Shivanshvyas1729)  
[LinkedIn](https://www.linkedin.com/in/shivanshvyas)  
[Portfolio](https://shivanshvyas1729portfolio.netlify.app/)
""")

# =========================
# 👇 FOOTER
# =========================
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:gray;'>
Built with ❤️ by Shivansh Vyas
</div>
""", unsafe_allow_html=True)