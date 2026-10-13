import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import importlib

fusion_module = importlib.import_module("3_model_fusion")
MultimodalDiagnosticEngine = fusion_module.MultimodalDiagnosticEngine

class MedTechDataset(Dataset):
    def __init__(self, xray_dir, audio_dir, transform=None, max_samples=2000):
        self.transform = transform
        self.audio_files = glob.glob(os.path.join(audio_dir, "*.png"))
        
        all_xrays = glob.glob(os.path.join(xray_dir, "**", "*.png"), recursive=True)
        self.data = []
        
        # Load X-Rays
        for xray in all_xrays:
            if "NORMAL" in xray.upper():
                self.data.append((xray, 0)) # 0 = Healthy
            elif "PNEUMONIA" in xray.upper():
                self.data.append((xray, 1)) # 1 = Pneumonia
                
        random.shuffle(self.data)
        self.data = self.data[:max_samples] # Increased from 400 to 2000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xray_path, label = self.data[idx]
        
        # For a truly precise clinical model, you would map exact patient IDs.
        # For this enhanced prototype, we ensure the engine gets a paired tensor.
        audio_path = random.choice(self.audio_files) 
        
        xray_img = Image.open(xray_path).convert('L')
        audio_img = Image.open(audio_path).convert('L')
        
        if self.transform:
            xray_tensor = self.transform(xray_img)
            # We don't rotate the audio spectrogram, just resize and tensorize
            audio_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(audio_img)
            
        return xray_tensor, audio_tensor, label

def train_engine():
    print("🚀 Initializing Optimized MedTech Training Pipeline...")
    
    # 1. Hyperparameters (Tuned for higher accuracy)
    epochs = 10           # Increased epochs for better convergence
    batch_size = 32       # Increased batch size for more stable gradients
    learning_rate = 0.0005 # Lowered learning rate to prevent overshooting the minima
    
    # 2. Data Preparation with Aggressive Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to mirror X-ray
        transforms.RandomRotation(degrees=10),  # Slight tilt to simulate real patient posture
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() # No augmentation on test data!
    ])
    
    # We load the dataset and manually apply the correct transforms to train vs test
    full_dataset = MedTechDataset(
        xray_dir=os.path.join("data", "processed_xrays"),
        audio_dir=os.path.join("data", "processed_spectrograms"),
        max_samples=2000
    )
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Apply transforms dynamically using a wrapper
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            x, a, y = self.subset[idx]
            # MedTechDataset returns un-transformed images if transform=None
            # We will handle it locally for better control
            return x, a, y

    # Re-instantiate with transforms applied properly
    train_dataset = MedTechDataset(os.path.join("data", "processed_xrays"), os.path.join("data", "processed_spectrograms"), transform=train_transform, max_samples=2000)
    test_dataset = MedTechDataset(os.path.join("data", "processed_xrays"), os.path.join("data", "processed_spectrograms"), transform=test_transform, max_samples=400)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Training Hardware: {device.type.upper()}")
    print(f"📚 Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples.")
    
    # 3. Initialize Model with Weight Decay (L2 Regularization)
    model = MultimodalDiagnosticEngine(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    # Added weight_decay to penalize overly large weights (reduces overfitting)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    
    # 4. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xrays, audios, labels in loop:
            xrays, audios, labels = xrays.to(device), audios.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(xrays, audios)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
    # 5. Evaluate on Test Set
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for xrays, audios, labels in test_loader:
            xrays, audios, labels = xrays.to(device), audios.to(device), labels.to(device)
            outputs = model(xrays, audios)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"🎯 Final Optimized Test Accuracy: {test_acc:.2f}%")
    
    save_path = os.path.join("src", "model_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 High-precision weights saved to: {save_path}")

if __name__ == "__main__":
    train_engine()