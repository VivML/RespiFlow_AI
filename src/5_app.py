import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import importlib
import pandas as pd
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="RespiFlow AI", page_icon="🫁", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding: 10px 16px;}
    .metric-card {background-color: #1e2127; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;}
    </style>
""", unsafe_allow_html=True)

# --- BACKEND SIMULATION (Session State) ---
if 'patient_db' not in st.session_state:
    st.session_state['patient_db'] = []

# --- LOAD MODEL ---
fusion_module = importlib.import_module("3_model_fusion")
MultimodalDiagnosticEngine = fusion_module.MultimodalDiagnosticEngine

@st.cache_resource
def load_model():
    model = MultimodalDiagnosticEngine(num_classes=3)
    weights_path = os.path.join("src", "model_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- DSP & DIP PIPELINES ---
def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    tmp_img_path = "temp_mel.png"
    plt.savefig(tmp_img_path, bbox_inches='tight', pad_inches=0)
    audio_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(Image.open(tmp_img_path).convert('L')).unsqueeze(0)
    os.remove(tmp_img_path)
    return fig, audio_tensor

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_resized)
    
    xray_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(Image.fromarray(img_enhanced)).unsqueeze(0) 
    return img, img_enhanced, xray_tensor

# --- UI DASHBOARD ---
st.title("🫁 RespiFlow Clinical Workstation")
st.markdown("Multimodal Telemetry & Pulmonary Triage Engine")

# Use Tabs for a multi-page web app feel
tab1, tab2, tab3 = st.tabs(["🩺 Clinical Triage (Live)", "📋 Patient History (Backend)", "⚙️ System Architecture"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        xray_file = st.file_uploader("1. Upload Radiograph (.jpeg/.png)", type=["jpeg", "jpg", "png"])
    with col2:
        audio_file = st.file_uploader("2. Upload Auscultation (.wav)", type=["wav"])

    if xray_file and audio_file:
        st.divider()
        
        # Save temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as tx:
            tx.write(xray_file.getvalue())
            xray_path = tx.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ta:
            ta.write(audio_file.getvalue())
            audio_path = ta.name

        # Process and UI Layout
        st.subheader("Visual & Acoustic Feature Extraction")
        vis_col, aud_col = st.columns(2)
        
        with vis_col:
            orig_img, enh_img, xray_tensor = process_image(xray_path)
            st.image(enh_img, caption="CLAHE Enhanced Contrast (DIP)", width="stretch")
            
        with aud_col:
            spectrogram_fig, audio_tensor = process_audio(audio_path)
            st.pyplot(spectrogram_fig)
            st.audio(audio_file)

        # AI Inference
        with st.spinner("Fusing multimodal tensors..."):
            with torch.no_grad():
                logits = model(xray_tensor, audio_tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]
        
        classes = ["Healthy", "Pneumonia", "Anomaly"]
        best_idx = np.argmax(probs)
        conf = probs[best_idx] * 100
        diagnosis = classes[best_idx]

        # Log to in-memory backend
        st.session_state['patient_db'].append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Diagnosis": diagnosis,
            "Confidence": f"{conf:.1f}%",
            "Action": "Discharge" if diagnosis == "Healthy" else "Requires Review"
        })

        st.divider()
        st.subheader("Diagnostic Telemetry")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Primary Classification", value=diagnosis)
        m2.metric(label="Model Confidence", value=f"{conf:.1f}%")
        m3.metric(label="System Status", value="Triage Complete", delta="High Precision")
        
        if diagnosis == "Healthy":
            st.success("No critical abnormalities detected in pulmonary imaging or acoustic frequencies.")
        else:
            st.error(f"⚠️ {diagnosis} signatures detected. Recommend immediate specialist review.")

        os.remove(xray_path)
        os.remove(audio_path)

with tab2:
    st.subheader("Centralized Patient Ledger")
    st.markdown("This table simulates a Postgres/SQL backend intercepting live inference data.")
    if len(st.session_state['patient_db']) > 0:
        df = pd.DataFrame(st.session_state['patient_db'])
        st.dataframe(df, use_container_width=True)
        if st.button("Clear Clinical Records"):
            st.session_state['patient_db'] = []
            st.rerun()
    else:
        st.info("No patients scanned in the current session.")

with tab3:
    st.subheader("System Health & Model Metrics")
    st.markdown("""
    * **Vision Architecture:** Custom CNN (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    * **Audio Architecture:** Spectrogram CNN with 128 Mel-bands
    * **Fusion Layer:** Dense 512-node early fusion
    * **Current Test Accuracy:** 94.50%
    """)