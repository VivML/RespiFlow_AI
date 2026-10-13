import os
import shutil
import glob
import random
import csv

def build_massive_demo_set(num_samples_per_class=5):
    print("🔍 Parsing clinical labels and filtering datasets...")
    
    # 1. Parse Audio Diagnoses from the CSV
    csv_path = os.path.join("data", "patient_diagnosis.csv")
    audio_labels = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    audio_labels[row[0]] = row[1].strip().upper()
    else:
        print("⚠️ patient_diagnosis.csv missing from data/ folder!")
        return
                
    # 2. Categorize Raw Audio by Patient ID
    raw_audios = glob.glob(os.path.join("data", "raw_audio", "**", "*.wav"), recursive=True)
    audio_pools = {'HEALTHY': [], 'PNEUMONIA': [], 'COPD': []}
    
    for audio in raw_audios:
        filename = os.path.basename(audio)
        patient_id = filename.split('_')[0] # Extract the number (e.g., '101')
        if patient_id in audio_labels:
            diagnosis = audio_labels[patient_id]
            if diagnosis in audio_pools:
                audio_pools[diagnosis].append(audio)
                
    # 3. Categorize Raw X-Rays by Folder Name
    raw_xrays = glob.glob(os.path.join("data", "raw_xrays", "**", "*.jpeg"), recursive=True)
    xray_pools = {'HEALTHY': [], 'PNEUMONIA': [], 'COPD': []}
    
    for xray in raw_xrays:
        if "NORMAL" in xray.upper():
            xray_pools['HEALTHY'].append(xray)
            # Proxy: Use Normal X-Rays for COPD since dataset lacks COPD images
            xray_pools['COPD'].append(xray) 
        elif "PNEUMONIA" in xray.upper():
            xray_pools['PNEUMONIA'].append(xray)

    # 4. Generate the Demo Folder Structure
    base_dir = "demo_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    print(f"🏗️ Building {num_samples_per_class} distinct patient profiles per condition...")
    
    for condition in ['HEALTHY', 'PNEUMONIA', 'COPD']:
        for i in range(1, num_samples_per_class + 1):
            if not audio_pools[condition] or not xray_pools[condition]:
                continue
                
            # Create e.g., demo_data/Pneumonia/Patient_Pneumonia_1
            patient_name = f"Patient_{condition.capitalize()}_{i}"
            patient_dir = os.path.join(base_dir, condition.capitalize(), patient_name)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Select random paired data from the filtered pools
            selected_audio = random.choice(audio_pools[condition])
            selected_xray = random.choice(xray_pools[condition])
            
            # Copy and rename for a clean UI presentation
            shutil.copy(selected_audio, os.path.join(patient_dir, f"{patient_name}_Stethoscope.wav"))
            shutil.copy(selected_xray, os.path.join(patient_dir, f"{patient_name}_Chest_Xray.jpeg"))
            
    print(f"✅ Demo curation complete! {num_samples_per_class * 3} patients generated in demo_data/.")

if __name__ == "__main__":
    build_massive_demo_set(num_samples_per_class=5)