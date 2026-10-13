import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') # Required for headless batch processing
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import concurrent.futures
from tqdm import tqdm

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def process_single_audio(audio_path, output_dir):
    try:
        # Set exact output filename
        filename = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
        save_path = os.path.join(output_dir, filename)
        
        # Skip if already processed (lets you pause/resume the script)
        if os.path.exists(save_path):
            return True
            
        # DSP Pipeline
        y, sr = librosa.load(audio_path, sr=16000)
        b, a = butter_bandpass(100.0, 2000.0, sr, order=3)
        y_filtered = lfilter(b, a, y)
        
        mel_spect = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Save visual output
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw_audio")
    output_dir = os.path.join("data", "processed_spectrograms")
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    print(f"Starting batch DSP on {len(wav_files)} files utilizing full CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_audio, path, output_dir) for path in wav_files]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(wav_files), desc="Audio DSP"):
            pass
    print("Audio batch processing complete!")