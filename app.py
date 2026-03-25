import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cattle Disease Detector",
    page_icon="🐄",
    layout="centered"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .result-box {
        background: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 10px 0;
    }
    .healthy-box {
        background: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 10px 0;
    }
    .stFileUploader {
        background-color: #1e1e2e;
        border: 1px dashed #444;
        border-radius: 10px;
        padding: 10px;
    }
    .stProgress > div > div {
        background-color: #e74c3c;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    h1, h2, h3 { color: #ffffff !important; }
    p, label { color: #cccccc !important; }
    .stInfo { background-color: #1e1e2e; color: #aaaaaa; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.saved_model.load("best_cattle_disease_model")

CLASS_NAMES = [
    "Bovine Mastitis", "Dermatophilosis", "Healthy",
    "Pediculosis", "Ringworm", "Lumpy Skin", "Pinkeye"
]

DISEASE_ADVICE = {
    "Bovine Mastitis":  "⚠️ Udder infection detected. Consult a veterinarian immediately.",
    "Dermatophilosis":  "⚠️ Keep the animal dry. Consult vet for antibiotic treatment.",
    "Healthy":          "✅ The animal appears healthy. No action needed.",
    "Pediculosis":      "⚠️ Lice infestation detected. Apply approved insecticide.",
    "Ringworm":         "⚠️ Fungal infection detected. Apply antifungal cream.",
    "Lumpy Skin":       "🚨 Highly contagious! Isolate the animal immediately.",
    "Pinkeye":          "⚠️ Eye infection detected. Apply antibiotic eye drops.",
}

BAR_COLORS = {
    "Bovine Mastitis":  "#e74c3c",
    "Dermatophilosis":  "#9b59b6",
    "Healthy":          "#27ae60",
    "Pediculosis":      "#f39c12",
    "Ringworm":         "#1abc9c",
    "Lumpy Skin":       "#e67e22",
    "Pinkeye":          "#3498db",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🐄 Cattle Disease Detection System")
st.markdown("<p style='color:#888; margin-top:-10px;'>Upload a cattle photo to detect disease using AI</p>", unsafe_allow_html=True)
st.write("---")

model = load_model()
infer = model.signatures["serving_default"]

uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("🔬 Result")

        with st.spinner("Analyzing..."):
            img_resized = img.resize((224, 224)).convert("RGB")
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            output = infer(tf.constant(img_array))
            key = list(output.keys())[0]
            prediction = output[key].numpy()[0]

            idx = np.argmax(prediction)
            disease = CLASS_NAMES[idx]
            confidence = prediction[idx] * 100

        color = "#27ae60" if disease == "Healthy" else "#e74c3c"
        icon  = "✅" if disease == "Healthy" else "🔴"

        st.markdown(f"""
        <div style="background:#1e1e2e; padding:20px; border-radius:10px; border-left:5px solid {color}; margin:10px 0;">
            <h3 style="color:{color}; margin:0;">{icon} {disease}</h3>
            <p style="color:#aaa; margin:8px 0 0;">Confidence: <b style="color:white;">{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        advice = DISEASE_ADVICE.get(disease, "Please consult a veterinarian.")
        st.markdown(f"""
        <div style="background:#1a2a1a; padding:12px 16px; border-radius:8px; margin-top:10px; color:#aaffaa; font-size:0.9rem;">
            {advice}
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.write("---")
    st.subheader("📊 All Disease Probabilities")

    sorted_probs = sorted(
        zip(CLASS_NAMES, prediction),
        key=lambda x: x[1], reverse=True
    )

    for name, prob in sorted_probs:
        pct = prob * 100
        bar_color = BAR_COLORS.get(name, "#888")
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#ccc; font-size:0.85rem;">{name}</span>
                <span style="color:#fff; font-size:0.85rem; font-family:monospace;">{pct:.1f}%</span>
            </div>
            <div style="background:#2a2a3a; border-radius:999px; height:8px;">
                <div style="width:{pct}%; background:{bar_color}; height:8px; border-radius:999px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#444;">
        <div style="font-size:3rem;">📷</div>
        <p style="color:#555; margin-top:10px;">Upload a cattle image to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='color:#555; font-size:0.8rem; text-align:center;'>For screening only. Always consult a qualified veterinarian for proper diagnosis.</p>", unsafe_allow_html=True)