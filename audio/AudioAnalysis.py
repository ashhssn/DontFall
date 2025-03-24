import pyaudio
import wave
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
from vosk import Model, KaldiRecognizer
import json
import time

# --- Constants ---
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono
RATE = 16000  # Sampling rate
CHUNK = 1024  # Audio chunk size
MODEL_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/vosk-model-small-en-us-0.15"  # Vosk model for speech recognition
FALL_DETECTION_MODEL_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall_detection_model.pth"  # Pretrained fall detection model
RECORD_SECONDS = 6  # Duration for each audio capture
FALL_THRESHOLD = 0.9  # Increased threshold to reduce false positives

# --- Load PyTorch Model for Fall Detection ---
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

# --- Load the Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fall_model = FallDetectionCNN().to(device)

FALL_DETECTION_MODEL_PATH = "C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall_detection_model.pth"

try:
    fall_model.load_state_dict(torch.load(FALL_DETECTION_MODEL_PATH, map_location=device))
    fall_model.eval()
    print("[INFO] Model loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at: {FALL_DETECTION_MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

# --- Function to Record Audio ---
def record_audio(output_filename="recorded_audio.wav"):
    print(f"[INFO] Recording {RECORD_SECONDS} seconds of audio...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK))

    print("[INFO] Recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return output_filename

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
    
def test_audio(audio_file):
    print(f"[INFO] Testing audio: {audio_file}")

    features = extract_features(audio_file)

    if features is None:
        print("[ERROR] Could not extract features from the audio file.")
        return

    tensor_input = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = fall_model(tensor_input).item()

    print(f"[DEBUG] Fall Detection Model Output: {prediction}")

    if prediction > 0.8:
        print("[ALERT] Fall Detected!")
    else:
        print("[INFO] No Fall Detected.")

# --- Function to Detect Fall ---
def detect_fall(audio_file):
    mfcc_features = extract_features(audio_file)
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = fall_model(mfcc_tensor).item()

    print(f"[DEBUG] Fall Detection Model Output: {prediction}")

    if prediction > FALL_THRESHOLD:  # Increased threshold
        print("[ALERT] Fall Detected!")
        return True
    return False

"""
# --- Function to Perform Speech Recognition ---
def recognize_speech(audio_file):
    print("[INFO] Performing speech recognition...")
    
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)

    with wave.open(audio_file, "rb") as wf:
        while True:
            data = wf.readframes(CHUNK)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"[INFO] Recognized Text: {text}")

                    # Check for distress call
                    if any(word in text.lower() for word in ["help", "fall", "ouch", "emergency"]):
                        print("[ALERT] Patient asking for help detected!")
                        return text
    return None
"""
# --- Function to Capture Live Audio ---
def capture_live_audio():
    print("[INFO] Capturing live audio...")
    duration = 5  # Capture for 5 seconds
    audio_data = sd.rec(int(duration * RATE), samplerate=RATE, channels=1, dtype='int16')
    sd.wait()
    return audio_data

# --- Function to Process Live Audio for Fall Detection ---
def analyze_live_audio():
    audio_data = capture_live_audio()
    audio_data = np.squeeze(audio_data)  # Remove unnecessary dimensions
    mfcc_features = librosa.feature.mfcc(y=audio_data.astype(float), sr=RATE, n_mfcc=20)
    mfcc_features = np.mean(mfcc_features, axis=1)

    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        prediction = fall_model(mfcc_tensor).item()

    if prediction > 0.8:
        print("[ALERT] Fall Detected in Live Audio!")
        return True
    return False

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        print("\n[INFO] Capturing Audio...")
        recorded_file = record_audio()  # Capture audio

        # Fall Detection
        fall_detected = detect_fall(recorded_file)
        if fall_detected:
            print("[INFO] Sending fall alert to Raspberry Pi...")
            break  # Stop loop on fall detection

        """
        # Speech Recognition
        distress_text = recognize_speech(recorded_file)
        if distress_text:
            print("[INFO] Emergency detected, alerting caregivers!")
            break
        """
        time.sleep(1)  # Small delay before next capture
        
        #Capturing recorded audio to detect for fall/non-fall works but the real-time record is a bit inconsistent
        test_audio("C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall-audio-detection-dataset/01-022-07-014-01.wav")