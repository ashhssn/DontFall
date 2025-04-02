import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score

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
def extract_mfcc(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"[ERROR] Failed to extract features from {audio_file}: {e}")
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
        features = extract_mfcc(file_path)

        if features is None:
            print(f"[WARNING] Skipping corrupted file: {file_path}")
            return self.__getitem__((idx + 1) % len(self.file_paths))

        label = float(self.labels[idx])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# --- Load dataset ---
dataset = FallDataset(DATASET_PATH, file_to_label)

# --- Split dataset ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"[INFO] Training samples: {train_size}, Validation samples: {val_size}")

# --- Model ---
class FallDetectionModel(nn.Module):
    def __init__(self):
        super(FallDetectionModel, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

fall_model = FallDetectionModel().to(device)

# --- Loss and Optimizer ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(fall_model.parameters(), lr=0.001)

# --- Training ---
EPOCHS = 20
train_losses = []
val_losses = []
print("[INFO] Training started...")

for epoch in range(EPOCHS):
    fall_model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
        features, labels = features.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = fall_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    fall_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = fall_model(features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_accuracy:.4f}")

# --- Save the Model ---
MODEL_SAVE_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall_detection_model.pth"
torch.save(fall_model, MODEL_SAVE_PATH)
print(f"[INFO] New model saved at {MODEL_SAVE_PATH}")

# --- Compute F1 Score on Validation Set ---
fall_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)
        outputs = fall_model(features)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

f1 = f1_score(all_labels, all_preds)
print(f"[INFO] F1 Score on Validation Set: {f1:.4f}")

# --- Plot Loss Curve ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
