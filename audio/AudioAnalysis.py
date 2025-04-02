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
import sqlite3
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.database import insert_data

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
class FallDetectionModel(torch.nn.Module):
    def __init__(self):
        super(FallDetectionModel, self).__init__()
        self.fc1 = torch.nn.Linear(20, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the trained fall detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fall_model = FallDetectionModel().to(device)
fall_model = torch.load(FALL_DETECTION_MODEL_PATH, map_location=device, weights_only=False)
fall_model.eval()

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

def extract_mfcc(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)
        
        # Extract MFCC (Only 20 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across time axis

        
        return mfcc_mean  # Only MFCC features (20 values)

    except Exception as e:
        print(f"[ERROR] Failed to extract features from {audio_file}: {e}")
        return None
    
def test_audio(audio_file):
    print(f"[INFO] Testing audio: {audio_file}")

    features = extract_mfcc(audio_file)

    if features is None:
        print("[ERROR] Could not extract features from the audio file.")
        return

    tensor_input = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = fall_model(tensor_input).item()

    print(f"Fall Detection Model Output: {prediction}")

    if prediction > 0.8:
        print("[ALERT]Status: ⚠️ Fall Detected!")
    else:
        print("[ALERT]Status: ✅ Fall Not Detected!")

# --- Function to Detect Fall ---
def detect_fall(audio_file):
    mfcc_features = extract_mfcc(audio_file)
    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = fall_model(mfcc_tensor).item()

    print(f"Fall Detection Model Output: {prediction:.4f}")

    if prediction > FALL_THRESHOLD:
        print("[ALERT]Status: ⚠️ Fall Detected!")
        return True, prediction
    else:
        print("[ALERT]Status: ✅ Fall Not Detected!")
        return False, prediction

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

def insert_microphone_data(content):
    conn = sqlite3.connect("database.db")  # 🔁 Replace with your actual DB path
    cursor = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO microphone (content, timestamp) VALUES (?, ?)", (content, timestamp))
    conn.commit()
    conn.close()

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        print("\n[INFO] Capturing Audio...")
        recorded_file = record_audio()  # Capture audio

        # Fall Detection + Confidence
        fall_detected, confidence = detect_fall(recorded_file)
        confidence = round(confidence * 100, 2)  # Convert to percentage
        
        if fall_detected:
            insert_data('microphone', f"Confidence: {confidence}%")
            print("[INFO] Sending fall alert to Raspberry Pi...")
            break
        else:
            insert_data('microphone', f"Confidence: {confidence}%")
        
        """
        # Speech Recognition
        distress_text = recognize_speech(recorded_file)
        if distress_text:
            print("[INFO] Emergency detected, alerting caregivers!")
            break
        """
        time.sleep(1)  # Small delay before next capture
        
        #Capturing recorded audio to detect for fall/non-fall works but the real-time record is a bit inconsistent
        #test_audio("C:/Users/bryan/OneDrive/Desktop/Project/DontFall/audio/fall-audio-detection-dataset/01-022-07-014-01.wav")

