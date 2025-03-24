import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# --- Set device (Use GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- Paths ---
DATASET_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall-audio-detection-dataset"
LABELS_FILE = os.path.join(DATASET_PATH, "labeled_dataset.csv")

# --- Load labeled dataset ---
try:
    df_labels = pd.read_csv(LABELS_FILE)
except FileNotFoundError:
    print(f"[ERROR] Labels file not found: {LABELS_FILE}")
    exit()

file_to_label = dict(zip(df_labels["Filename"], df_labels["Label"]))

# --- Feature Extraction: MFCC Only ---
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)

        # Ignore silent audio
        max_amplitude = np.max(np.abs(y))
        if max_amplitude < 0.01:
            return None

        # Normalize the waveform
        y = librosa.util.normalize(y)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfcc = librosa.feature.delta(mfcc)  # First-order delta
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)  # Second-order delta

        # Combine features
        features = np.concatenate([np.mean(mfcc, axis=1),
                                   np.mean(delta_mfcc, axis=1),
                                   np.mean(delta2_mfcc, axis=1)])

        return features  # Shape: (60,) (20 MFCCs + 20 Delta + 20 Delta2)

    except Exception as e:
        print(f"[WARNING] Failed to extract features from {audio_file}: {e}")
        return None

# --- Custom Dataset Class ---
class FallDataset(Dataset):
    def __init__(self, dataset_path, file_to_label):
        self.file_paths = list(file_to_label.keys())
        self.labels = list(file_to_label.values())
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.file_paths[idx])
        features = extract_features(file_path)

        # Handle missing or corrupted files
        if features is None:
            print(f"[WARNING] Skipping corrupted file: {file_path}")
            return self.__getitem__((idx + 1) % len(self.file_paths))  # Skip to next sample safely

        label = float(self.labels[idx])  # Ensure labels are float32 for BCELoss
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# --- Load dataset ---
dataset = FallDataset(DATASET_PATH, file_to_label)

# --- Split dataset into training (80%) and validation (20%) ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"[INFO] Training samples: {train_size}, Validation samples: {val_size}")

class FallDetectionCNN(nn.Module):
    def __init__(self, input_size=960):  # 🔹 Update input size to 960 based on debug output
        super(FallDetectionCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(input_size, 32)  # 🔹 Match input size (960 from debug output)
        self.fc2 = nn.Linear(32, 1)

        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (Batch, 1, Features)
        x = self.relu(self.conv1(x))
        x = self.batch_norm1(self.pool(x))
        x = self.relu(self.conv2(x))
        x = self.batch_norm2(self.pool(x))

        x = x.view(x.size(0), -1)  # Flatten to match `fc1` input size
       
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x

# --- Initialize Model ---
fall_model = FallDetectionCNN().to(device)

# --- Use BCEWithLogitsLoss (Better than Sigmoid + BCELoss) ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(fall_model.parameters(), lr=0.001)

# --- Train Model ---
EPOCHS = 20
print("[INFO] Training started...")

for epoch in range(EPOCHS):
    fall_model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    # --- Training Loop ---
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        features, labels = features.to(device), labels.to(device).unsqueeze(1)  # Adjust for BCELoss

        optimizer.zero_grad()
        outputs = fall_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # --- Validation Loop ---
    fall_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
            features, labels = features.to(device), labels.to(device).unsqueeze(1)

            outputs = fall_model(features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - "
          f"Train Loss: {total_train_loss / len(train_loader):.4f} | "
          f"Val Loss: {total_val_loss / len(val_loader):.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

# --- Save the Trained Model ---
MODEL_SAVE_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall_detection_model.pth"
torch.save((fall_model), MODEL_SAVE_PATH) 
print(f"[INFO] New model saved at {MODEL_SAVE_PATH}")