import socket
import cv2
import time
import struct

def send_data_to_server(data_to_send):
    #HOST = '172.20.10.4'
    HOST = '127.0.0.1'
    PORT = 5001

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")

        if isinstance(data_to_send, bytes):
            # Pack the length of the data and send the header first.
            data_len = struct.pack('!I', len(data_to_send))
            client_socket.sendall(data_len)
            client_socket.sendall(data_to_send)
            print("Sent image data to server")
        else:
            client_socket.sendall(data_to_send.encode('utf-8'))
            print(f"Sent to server: {data_to_send}")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            break

    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret:
        frame_bytes = jpeg.tobytes()
        send_data_to_server(frame_bytes)
    else:
        print("Failed to encode frame")

    send_data_to_server("ACCELEROMETER: Hello from ACCELEROMETER!")
    send_data_to_server("MICROPHONE: 69")

    
    time.sleep(1)
