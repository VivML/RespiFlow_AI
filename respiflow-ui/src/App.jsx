import React, { useState } from 'react';
import { UploadCloud, Activity, CheckCircle, AlertTriangle } from 'lucide-react';
import './App.css';

function App() {
  const [xrayFile, setXrayFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!xrayFile || !audioFile) {
      setError("Please upload both an X-Ray and an Audio file.");
      return;
    }
    
    setError("");
    setLoading(true);
    
    const formData = new FormData();
    formData.append("xray", xrayFile);
    formData.append("audio", audioFile);

    try {
      const response = await fetch("http://localhost:8000/api/diagnose", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) throw new Error("Backend inference failed");
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Failed to connect to RespiFlow API. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🫁 RespiFlow Clinical Workstation</h1>
        <p>Acoustic-Radiological Deep Learning Telemetry</p>
      </header>

      <div className="upload-grid">
        <div className="upload-card">
          <UploadCloud size={32} className="icon" />
          <h3>1. Upload Radiograph (.jpeg)</h3>
          <input type="file" accept="image/*" onChange={(e) => setXrayFile(e.target.files[0])} />
          {xrayFile && <p className="success-text">Loaded: {xrayFile.name}</p>}
        </div>
        
        <div className="upload-card">
          <Activity size={32} className="icon" />
          <h3>2. Upload Auscultation (.wav)</h3>
          <input type="file" accept="audio/*" onChange={(e) => setAudioFile(e.target.files[0])} />
          {audioFile && <p className="success-text">Loaded: {audioFile.name}</p>}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="action-container">
        <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
          {loading ? "Fusing Multimodal Tensors..." : "Initialize Triage Engine"}
        </button>
      </div>

      {result && (
        <div className="results-container">
          <div className={`diagnosis-banner ${result.diagnosis === 'Healthy' ? 'safe' : 'danger'}`}>
            {result.diagnosis === 'Healthy' ? <CheckCircle size={28} /> : <AlertTriangle size={28} />}
            <h2>{result.diagnosis} Detected</h2>
            <span className="confidence">Confidence: {result.confidence}%</span>
          </div>

          <div className="telemetry-grid">
            <div className="telemetry-card">
              <h4>DIP Pipeline: CLAHE Enhancement</h4>
              <img src={result.telemetry.xray_clahe_base64} alt="CLAHE X-Ray" />
            </div>
            <div className="telemetry-card">
              <h4>DSP Pipeline: Mel-Spectrogram</h4>
              <img src={result.telemetry.audio_spectrogram_base64} alt="Audio Spectrogram" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;