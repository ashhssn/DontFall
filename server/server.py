import socket
import struct
import numpy as np
import cv2

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def run_server():
    HOST = '0.0.0.0'
    PORT = 5001

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # First receive a 4-byte length header
                header = recvall(conn, 4)
                if not header:
                    break
                # Unpack the image size
                img_size = struct.unpack('!I', header)[0]

                # Now receive the full image data based on size
                data = recvall(conn, img_size)
                if not data:
                    break

                # Attempt to read as an image
                np_arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite("received_frame.jpg", img)
                    print("Image saved as received_frame.jpg")
                else:
                    # If decoding as image fails, try string decoding
                    try:
                        message = data.decode('utf-8')
                        print(f"Received message: {message}")
                    except UnicodeDecodeError:
                        print("Failed to decode data as an image or a string.")

if __name__ == "__main__":
    run_server()
