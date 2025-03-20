import numpy as np

import cv2

import tflite_runtime.interpreter as tflite

import time

import socket

import struct



# Dashboard Server Configuration

HOST = '172.20.10.4'  # Dashboard IP

PORT = 5001           # Dashboard Port



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



# Initialize webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam")

    exit()



# Constants

CONFIDENCE_THRESHOLD = 0.7

FRAME_INTERVAL = 10  # Process every 10th frame

FALL_CONFIRMATION_FRAMES = 5  # Require 5 consecutive fall detections



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

