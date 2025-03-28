import socket
import datetime
import threading
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import struct

# Server Configuration
HOST = '0.0.0.0'
PORT = 5001

# Dashboard Configuration
DASHBOARD_HOST = '172.20.10.4'  # Replace with your actual dashboard IP
DASHBOARD_PORT = 5001

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="fall_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Flag to control server listening while fall detection is running
is_fall_detection_running = False

# Function to preprocess image for TFLite model
def preprocess_image(image):
    resized = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb / 255.0
    return np.expand_dims(normalized, axis=0).astype(np.float32)

# Function to send image to dashboard
def send_data_to_server(frame):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((DASHBOARD_HOST, DASHBOARD_PORT))
            print(f"Connected to dashboard at {DASHBOARD_HOST}:{DASHBOARD_PORT}")

            # Convert frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                return
            
            frame_bytes = jpeg.tobytes()

            # Send data length first
            data_len = struct.pack('!I', len(frame_bytes))
            client_socket.sendall(data_len)
            client_socket.sendall(frame_bytes)
            print("Sent image to dashboard")

    except Exception as e:
        print(f"Error sending data: {e}")

# Function to run fall detection
def run_fall_detection():
    global is_fall_detection_running  # Access the flag

    print("Starting fall detection...")

    # Stop server from listening during fall detection
    is_fall_detection_running = True

    # Ensure webcam is properly released and reopened
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        is_fall_detection_running = False
        return

    CONFIDENCE_THRESHOLD = 0.7
    FRAME_INTERVAL = 10
    FALL_CONFIRMATION_FRAMES = 5

    frame_count = 0
    fallen_count = 0
    fallen_state = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            frame_count += 1
            if frame_count % FRAME_INTERVAL != 0:
                continue

            processed_frame = preprocess_image(frame)

            interpreter.set_tensor(input_details[0]['index'], processed_frame)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            if prediction > CONFIDENCE_THRESHOLD:
                fallen_count += 1
                if fallen_count >= FALL_CONFIRMATION_FRAMES:
                    if not fallen_state:
                        print("FALL DETECTED! Sending alert...")
                        send_data_to_server(frame)
                        fallen_state = True
            else:
                fallen_count = 0
                fallen_state = False

            # Display prediction on frame
            status = f"Fallen: {prediction:.2f}" if prediction > CONFIDENCE_THRESHOLD else f"Not Fallen: {1 - prediction:.2f}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Fall Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if fallen_state:  # If a fall is detected, stop webcam capture
                break

    finally:
        # Release the webcam properly and close windows
        cap.release()
        cv2.destroyAllWindows()

        # Fall detection is complete, resume server listening
        print("Fall detection stopped.")
        is_fall_detection_running = False

# Create and configure server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"Fall Detection Alert Server listening on port {PORT}")
print(f"Server IP: {socket.gethostbyname(socket.gethostname())}")
print("Waiting for alerts...")

while True:
    try:
        if is_fall_detection_running:
            # Server stops listening when fall detection is running
            continue

        client_socket, client_address = server_socket.accept()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\nConnection from {client_address} at {timestamp}")

        data = client_socket.recv(1024)
        if data:
            message = data.decode('utf-8')
            if "FALL_DETECTED" in message:
                print("\n*** ALERT RECEIVED! STARTING FALL DETECTION... ***")
                detection_thread = threading.Thread(target=run_fall_detection)
                detection_thread.start()

            elif message.startswith("DATA"):
                print(f"{message}")

            client_socket.send("Alert received".encode('utf-8'))
        
        client_socket.close()

    except Exception as e:
        print(f"Error: {e}")
