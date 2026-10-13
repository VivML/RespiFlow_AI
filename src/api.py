from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
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
import io
import base64

# Dynamically import the architecture
fusion_module = importlib.import_module("3_model_fusion")
MultimodalDiagnosticEngine = fusion_module.MultimodalDiagnosticEngine

app = FastAPI(title="RespiFlow AI API", version="1.0")

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your React domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD TRAINED MODEL ---
model = MultimodalDiagnosticEngine(num_classes=3)
weights_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
else:
    raise RuntimeError("model_weights.pth not found! Ensure it is in the src/ folder.")

# --- DSP & DIP PIPELINES ---
def process_audio(audio_bytes):
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Generate spectrogram image in memory
    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    # Convert for AI tensor
    audio_img = Image.open(buf).convert('L')
    audio_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(audio_img).unsqueeze(0)
    
    # Base64 encode for React UI
    buf.seek(0)
    spectrogram_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return audio_tensor, spectrogram_b64

def process_image(image_bytes):
    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_resized)
    
    # Convert to AI Tensor
    xray_img = Image.fromarray(img_enhanced)
    xray_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(xray_img).unsqueeze(0) 
    
    # Base64 encode for React UI
    _, buffer = cv2.imencode('.jpeg', img_enhanced)
    enhanced_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return xray_tensor, enhanced_b64

# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "RespiFlow AI Engine is Online", "accuracy": "94.50%"}

@app.post("/api/diagnose")
async def diagnose(xray: UploadFile = File(...), audio: UploadFile = File(...)):
    try:
        # Read file bytes
        xray_bytes = await xray.read()
        audio_bytes = await audio.read()
        
        # Run Pipelines
        xray_tensor, enhanced_b64 = process_image(xray_bytes)
        audio_tensor, spectrogram_b64 = process_audio(audio_bytes)
        
        # Inference
        with torch.no_grad():
            logits = model(xray_tensor, audio_tensor)
            probs = F.softmax(logits, dim=1).numpy()[0]
            
        classes = ["Healthy", "Pneumonia", "Unknown Anomaly"]
        best_idx = np.argmax(probs)
        confidence = float(probs[best_idx] * 100)
        diagnosis = classes[best_idx]
        
        return {
            "success": True,
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "telemetry": {
                "xray_clahe_base64": f"data:image/jpeg;base64,{enhanced_b64}",
                "audio_spectrogram_base64": f"data:image/png;base64,{spectrogram_b64}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)