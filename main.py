import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f8faf7; }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        color: #1a2e1a;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .hero-sub {
        font-size: 1.1rem;
        color: #4a6741;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #4a9e4a;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .feature-card h4 {
        color: #1a2e1a;
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
        font-weight: 600;
    }

    .feature-card p {
        color: #6b7c6b;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .stat-box {
        background: linear-gradient(135deg, #2d5a2d, #4a9e4a);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }

    .stat-box .number {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        font-weight: 400;
    }

    .stat-box .label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 0.2rem;
    }

    .result-box {
        background: linear-gradient(135deg, #e8f5e8, #f0faf0);
        border: 2px solid #4a9e4a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-box .disease-name {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #1a2e1a;
        margin: 0.5rem 0;
    }

    .result-box .result-label {
        font-size: 0.85rem;
        color: #4a6741;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    .upload-area {
        background: white;
        border: 2px dashed #a8d4a8;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .about-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d5a2d, #4a9e4a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1a3d1a, #2d7a2d);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74,158,74,0.3);
    }

    section[data-testid="stSidebar"] {
        background: #1a2e1a;
    }

    section[data-testid="stSidebar"] * {
        color: #c8e6c8 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        color: #a8d4a8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    hr { border-color: #e0ece0; }
    </style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'Apple — Apple Scab', 'Apple — Black Rot', 'Apple — Cedar Apple Rust', 'Apple — Healthy',
    'Blueberry — Healthy',
    'Cherry — Powdery Mildew', 'Cherry — Healthy',
    'Corn — Cercospora / Gray Leaf Spot', 'Corn — Common Rust', 'Corn — Northern Leaf Blight', 'Corn — Healthy',
    'Grape — Black Rot', 'Grape — Esca (Black Measles)', 'Grape — Leaf Blight', 'Grape — Healthy',
    'Orange — Huanglongbing (Citrus Greening)',
    'Peach — Bacterial Spot', 'Peach — Healthy',
    'Pepper — Bacterial Spot', 'Pepper — Healthy',
    'Potato — Early Blight', 'Potato — Late Blight', 'Potato — Healthy',
    'Raspberry — Healthy',
    'Soybean — Healthy',
    'Squash — Powdery Mildew',
    'Strawberry — Leaf Scorch', 'Strawberry — Healthy',
    'Tomato — Bacterial Spot', 'Tomato — Early Blight', 'Tomato — Late Blight',
    'Tomato — Leaf Mold', 'Tomato — Septoria Leaf Spot',
    'Tomato — Spider Mites', 'Tomato — Target Spot',
    'Tomato — Yellow Leaf Curl Virus', 'Tomato — Mosaic Virus', 'Tomato — Healthy'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

def model_prediction(test_image):
    model = load_model()
    image = load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return result_index, confidence


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 PlantGuard")
    st.markdown("---")
    app_mode = st.selectbox(
        "Navigate",
        ["🏠  Home", "📖  About", "🔬  Disease Recognition"]
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#7aaa7a; line-height:1.6'>
    Powered by a CNN trained on<br>
    <strong style='color:#a8d4a8'>87,000+ leaf images</strong><br>
    across 38 disease classes.
    </div>
    """, unsafe_allow_html=True)


# ── Home ───────────────────────────────────────────────────────────────────────

if "Home" in app_mode:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="hero-title">Detect Plant Diseases<br>Instantly 🌿</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Upload a leaf photo — our AI identifies diseases across 14 crops in seconds.</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card"><h4>📤 Upload an Image</h4><p>Go to Disease Recognition and upload a clear photo of the affected leaf.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><h4>🧠 AI Analysis</h4><p>Our CNN model processes the image and identifies the disease pattern.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card"><h4>✅ Get Results</h4><p>Receive the disease name and confidence score in under 2 seconds.</p></div>', unsafe_allow_html=True)

    with col2:
        st.image("home_page.jpeg", use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-box"><div class="number">38</div><div class="label">Disease Classes</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-box"><div class="number">87K</div><div class="label">Training Images</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-box"><div class="number">14</div><div class="label">Crop Types</div></div>', unsafe_allow_html=True)


# ── About ──────────────────────────────────────────────────────────────────────

elif "About" in app_mode:
    st.markdown('<div class="hero-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Understanding the data and model behind PlantGuard.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3 style="color:#1a2e1a; font-family:'DM Serif Display',serif">📊 Dataset</h3>
            <p style="color:#4a6741; line-height:1.7">
            Built on the <strong>PlantVillage Dataset</strong> — recreated with offline augmentation
            for improved generalization. Contains ~87,000 RGB images of healthy and diseased
            crop leaves, categorized into <strong>38 classes</strong>.
            </p>
            <hr>
            <table style="width:100%; color:#4a6741; font-size:0.9rem">
                <tr><td>🌱 Training set</td><td><strong>70,295 images</strong></td></tr>
                <tr><td>✅ Validation set</td><td><strong>17,572 images</strong></td></tr>
                <tr><td>🧪 Test set</td><td><strong>33 images</strong></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3 style="color:#1a2e1a; font-family:'DM Serif Display',serif">🧠 Model</h3>
            <p style="color:#4a6741; line-height:1.7">
            A custom <strong>Convolutional Neural Network</strong> with 5 conv blocks,
            dropout regularization, and a 1500-unit dense layer.
            </p>
            <hr>
            <table style="width:100%; color:#4a6741; font-size:0.9rem">
                <tr><td>⚙️ Optimizer</td><td><strong>Adam (lr=0.0001)</strong></td></tr>
                <tr><td>📉 Loss</td><td><strong>Categorical Crossentropy</strong></td></tr>
                <tr><td>🔁 Epochs</td><td><strong>10</strong></td></tr>
                <tr><td>🖼️ Input size</td><td><strong>128 × 128 × 3</strong></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ── Disease Recognition ────────────────────────────────────────────────────────

elif "Recognition" in app_mode:
    st.markdown('<div class="hero-title">Disease Recognition 🔬</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a clear photo of a plant leaf to identify the disease.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        test_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("*Supported formats: JPG, JPEG, PNG*")
        st.markdown('</div>', unsafe_allow_html=True)

        if test_image:
            st.image(test_image, caption="Uploaded leaf image", use_column_width=True)

    with col2:
        if test_image:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍  Analyse Image"):
                with st.spinner("Analysing leaf..."):
                    result_index, confidence = model_prediction(test_image)
                    disease = CLASS_NAMES[result_index]
                    is_healthy = "Healthy" in disease

                    color = "#e8f5e8" if is_healthy else "#fff8e8"
                    border = "#4a9e4a" if is_healthy else "#e8a020"
                    icon = "✅" if is_healthy else "⚠️"

                    st.markdown(f"""
                    <div style="background:{color}; border:2px solid {border}; border-radius:16px; padding:1.5rem 2rem; text-align:center; margin-top:1rem">
                        <div style="font-size:2.5rem">{icon}</div>
                        <div style="font-size:0.8rem; color:#4a6741; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-top:0.5rem">Diagnosis</div>
                        <div style="font-family:'DM Serif Display',serif; font-size:1.6rem; color:#1a2e1a; margin:0.5rem 0">{disease}</div>
                        <div style="font-size:0.85rem; color:#6b7c6b">Confidence: <strong>{confidence:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:white; border-radius:16px; padding:3rem 2rem; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.06)">
                <div style="font-size:3rem">🍃</div>
                <div style="color:#4a6741; margin-top:1rem; font-size:1rem">Upload a leaf image on the left to get started</div>
            </div>
            """, unsafe_allow_html=True)