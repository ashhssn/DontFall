import numpy as np

import cv2

import tflite_runtime.interpreter as tflite

import sounddevice as sd

import librosa

import onnxruntime as ort

import socket

import struct

import scipy.io.wavfile as wavfile

import time

import os

import threading



# Dashboard Server Configuration

HOST = '172.20.10.4'  # Dashboard IP

PORT = 5001           # Dashboard Port



# Global flags to control which detection method is active

audio_active = True

webcam_active = False



# Event to synchronize the switching between audio and webcam

switch_event = threading.Event()



# Load the TFLite model

interpreter = tflite.Interpreter(model_path="fall_detection_model.tflite")

interpreter.allocate_tensors()



# Get input and output tensors

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']



# Function to preprocess image for TFLite model

def preprocess_image(image):

    resized = cv2.resize(image, (224, 224))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    normalized = rgb / 255.0

    return np.expand_dims(normalized, axis=0).astype(np.float32)



# Function to send the captured frame to the dashboard

def send_data_to_server(frame):

    try:

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            client_socket.connect((HOST, PORT))

            print(f"Connected to server at {HOST}:{PORT}")



            # Convert frame to JPEG format

            ret, jpeg = cv2.imencode('.jpg', frame)

            if not ret:

                print("Failed to encode frame")

                return



            frame_bytes = jpeg.tobytes()



            # Send data length first

            data_len = struct.pack('!I', len(frame_bytes))

            client_socket.sendall(data_len)

            client_socket.sendall(frame_bytes)

            print("Sent image data to server")



    except Exception as e:

        print(f"Error sending data: {e}")



# Function to send text data to the dashboard

def send_text_to_server(text):

    try:

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            client_socket.connect((HOST, PORT))

            print(f"Connected to server at {HOST}:{PORT}")



            # Convert text to bytes

            text_bytes = text.encode('utf-8')



            # Send data length first

            data_len = struct.pack('!I', len(text_bytes))

            # client_socket.sendall(data_len)

            client_socket.sendall(text_bytes)

            print("Sent audio data to server")



    except Exception as e:

        print(f"Error sending audio data: {e}")



# Function to detect fall from webcam frames

def detect_fall_webcam():

    global webcam_active, audio_active

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open webcam")

        return



    CONFIDENCE_THRESHOLD = 0.7

    FRAME_INTERVAL = 10  # Process every 10th frame

    FALL_CONFIRMATION_FRAMES = 10 # Require 10 consecutive fall detections



    frame_count = 0

    fallen_count = 0

    fallen_state = False



    try:

        while True:

            # Wait until the webcam is activated

            switch_event.wait()



            ret, frame = cap.read()

            if not ret:

                print("Error: Failed to capture image")

                break



            frame_count += 1

            if frame_count % FRAME_INTERVAL != 0:

                continue



            # Preprocess the frame

            processed_frame = preprocess_image(frame)



            # Run inference

            interpreter.set_tensor(input_details[0]['index'], processed_frame)

            interpreter.invoke()



            # Get prediction

            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]



            # Decision logic - detect falls

            if prediction > CONFIDENCE_THRESHOLD:

                fallen_count += 1

                if fallen_count >= FALL_CONFIRMATION_FRAMES:

                    if not fallen_state:

                        print("FALL DETECTED! Sending alert...")



                        # Send frame to dashboard

                        send_data_to_server(frame)



                        fallen_state = True

                        # Deactivate webcam and activate audio

                        webcam_active = False

                        audio_active = True

                        switch_event.clear()  # Stop webcam processing

                        break  # Exit the loop to stop webcam



            else:

                fallen_count = 0

                fallen_state = False



            # Display prediction on frame

            status = f"Fallen: {prediction:.2f}" if prediction > CONFIDENCE_THRESHOLD else f"Not Fallen: {1-prediction:.2f}"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



            # Display the frame

            cv2.imshow('Fall Detection', frame)



            # Break on 'q' key press

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break



    finally:

        cap.release()

        cv2.destroyAllWindows()



# Class for microphone-based fall detection

class FallDetectionSystem:

    def __init__(self, onnx_model_path):

        if not os.path.exists(onnx_model_path):

            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

        

        self.SAMPLE_RATE = 16000

        self.DURATION = 5  # seconds

        self.CHANNELS = 1



        try:

            # Use CPU execution provider

            providers = ['CPUExecutionProvider']

            self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)

            print("[INFO] ONNX Model loaded successfully.")

        except Exception as e:

            print(f"[ERROR] Model loading failed: {e}")

            exit(1)



    def record_audio(self, filename='recorded_audio.wav'):

        try:

            print(f"[INFO] Recording {self.DURATION} seconds of audio...")

            recording = sd.rec(int(self.DURATION * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype='float32')

            sd.wait()  # Wait until recording is finished

            recording = (recording * 32767).astype(np.int16)

            wavfile.write(filename, self.SAMPLE_RATE, recording)

            print("[INFO] Audio recording saved.")

            return filename

        except Exception as e:

            print(f"[ERROR] Audio recording failed: {e}")

            return None



    def extract_features(self, audio_file):

        try:

            y, sr = librosa.load(audio_file, sr=self.SAMPLE_RATE)

            if np.max(np.abs(y)) < 0.01:

                print("[WARNING] Silent audio detected")

                return None



            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            delta_mfcc = librosa.feature.delta(mfcc)

            delta2_mfcc = librosa.feature.delta(mfcc, order=2)



            features = np.concatenate([np.mean(mfcc, axis=1), np.mean(delta_mfcc, axis=1), np.mean(delta2_mfcc, axis=1)]).astype(np.float32)

            return features

        except Exception as e:

            print(f"[ERROR] Feature extraction failed: {e}")

            return None



    def run_inference(self, features):

        try:

            input_data = features.reshape(1, -1)

            input_name = self.onnx_session.get_inputs()[0].name

            output_name = self.onnx_session.get_outputs()[0].name

            result = self.onnx_session.run([output_name], {input_name: input_data})[0][0][0]

            return result

        except Exception as e:

            print(f"[ERROR] Inference failed: {e}")

            return None



    def detect_fall(self):

        try:

            audio_file = self.record_audio()

            if audio_file is None:

                return False, 0.0



            features = self.extract_features(audio_file)

            if features is None:

                return False, 0.0



            fall_probability = self.run_inference(features)

            if fall_probability is None:

                return False, 0.0



            return fall_probability > 0.5, fall_probability

        except Exception as e:

            print(f"[ERROR] Fall detection process failed: {e}")

            return False, 0.0



def microphone_thread():

    global webcam_active, audio_active

    fall_detector = FallDetectionSystem("fall_detection_model.onnx")

    

    while True:

        if not audio_active:

            time.sleep(1)

            continue



        print("[INFO] Listening for fall via microphone...")

        fall_detected, confidence = fall_detector.detect_fall()



        print(f"[INFO] Microphone: {confidence:.2f}")



        if fall_detected:

            print(f"[ALERT] Fall detected via microphone with confidence: {confidence:.2f}, triggering webcam...")

            send_text_to_server(f"MICROPHONE: {confidence:.2f}")

            audio_active = False

            webcam_active = True

            switch_event.set()  # Activate webcam processing



if __name__ == "__main__":

    webcam_thread = threading.Thread(target=detect_fall_webcam)

    webcam_thread.daemon = True

    webcam_thread.start()



    mic_thread = threading.Thread(target=microphone_thread)

    mic_thread.daemon = True

    mic_thread.start()



    # Keep the main thread alive

    while True:

        time.sleep(1)

