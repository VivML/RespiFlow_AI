import os
import glob
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm

def process_single_xray(image_path, raw_base_dir, output_base_dir):
    try:
        # Rebuild the train/test/val subfolder tree inside processed_xrays
        rel_path = os.path.relpath(image_path, raw_base_dir)
        filename = os.path.splitext(rel_path)[0] + ".png"
        save_path = os.path.join(output_base_dir, filename)
        
        # Skip if already processed
        if os.path.exists(save_path):
            return True
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # DIP Pipeline
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
            
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_resized)
        
        # Normalize to [0, 1] then save as 8-bit PNG
        img_normalized = img_enhanced.astype(np.float32) / 255.0
        cv2.imwrite(save_path, (img_normalized * 255).astype(np.uint8))
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw_xrays")
    output_dir = os.path.join("data", "processed_xrays")
    
    xray_files = glob.glob(os.path.join(raw_dir, "**", "*.jpeg"), recursive=True)
    
    print(f"Starting batch DIP on {len(xray_files)} X-rays utilizing full CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_xray, path, raw_dir, output_dir) for path in xray_files]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(xray_files), desc="Vision DIP"):
            pass
    print("X-Ray batch processing complete!")