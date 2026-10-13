import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalDiagnosticEngine(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalDiagnosticEngine, self).__init__()
        
        # -------------------------------------------
        # BRANCH A: VISION (For 224x224x1 X-Rays)
        # -------------------------------------------
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 28x28
            
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU()
        )
        # Output of Vision Branch: A 1D vector of 256 visual features

        # -------------------------------------------
        # BRANCH B: AUDIO (For 224x224x1 Mel-Spectrograms)
        # -------------------------------------------
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU()
        )
        # Output of Audio Branch: A 1D vector of 256 acoustic features

        # -------------------------------------------
        # LATE FUSION CLASSIFIER
        # -------------------------------------------
        # We concatenate the 256 vision features and 256 audio features (Total: 512)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevent overfitting
            nn.Linear(128, num_classes) # Final output: [Healthy, Pneumonia, COPD]
        )

    def forward(self, x_vision, x_audio):
        # 1. Pass data through independent branches
        vision_features = self.vision_cnn(x_vision)
        audio_features = self.audio_cnn(x_audio)
        
        # 2. Concatenate the feature vectors side-by-side
        combined_features = torch.cat((vision_features, audio_features), dim=1)
        
        # 3. Pass through final dense layers for diagnosis
        logits = self.fusion_classifier(combined_features)
        return logits

if __name__ == "__main__":
    print("Initializing Multimodal Diagnostic Engine...")
    
    # Instantiate the model predicting 3 classes: [Healthy, Pneumonia, COPD]
    model = MultimodalDiagnosticEngine(num_classes=3)
    
    # ---------------------------------------------------------
    # TEST BLOCK: Simulate the forward pass with dummy tensors
    # ---------------------------------------------------------
    batch_size = 1
    channels = 1
    height = 224
    width = 224
    
    # Create fake tensors representing one X-ray and one Spectrogram
    dummy_xray = torch.randn(batch_size, channels, height, width)
    dummy_audio = torch.randn(batch_size, channels, height, width)
    
    print("\nRunning simulated forward pass...")
    
    # Push the data through the engine
    model.eval() # Set to evaluation mode for testing
    with torch.no_grad():
        raw_output = model(dummy_xray, dummy_audio)
        probabilities = F.softmax(raw_output, dim=1)
    
    print(f"\nSuccess! The tensor shapes align perfectly.")
    print(f"Raw Logits Output Shape: {raw_output.shape}")
    print(f"Sample Output Probabilities: {probabilities.numpy()[0]}")
    print("Class mapping: [0: Healthy, 1: Pneumonia, 2: COPD]")