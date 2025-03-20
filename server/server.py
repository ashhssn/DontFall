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
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB max valid image size

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"Server listening on {HOST}:{PORT}...")

        # Main loop: continuously accept new connections.
        while True:
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            with conn:
                while True:
                    # Peek the first 4 bytes to decide the type of message.
                    header_peek = conn.recv(4, socket.MSG_PEEK)
                    if not header_peek:
                        # If no data is received, the client has disconnected.
                        print("Client disconnected")
                        break

                    # Attempt to interpret the peeked header as an unsigned int.
                    try:
                        potential_size = struct.unpack('!I', header_peek)[0]
                    except Exception:
                        potential_size = None

                    # If the potential_size is valid and within our max, assume an image.
                    if potential_size is not None and potential_size <= MAX_IMAGE_SIZE:
                        # Read the 4-byte header from the connection.
                        header = recvall(conn, 4)
                        if not header:
                            break
                        img_size = struct.unpack('!I', header)[0]

                        # Now read the image data based on the extracted size.
                        data = recvall(conn, img_size)
                        if not data:
                            break

                        # Decode the received data as an image.
                        np_arr = np.frombuffer(data, dtype=np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            cv2.imwrite("received_frame.jpg", img)
                            print("Image saved as received_frame.jpg")
                        else:
                            print("Failed to decode image from received data.")
                    else:
                        # Otherwise treat the incoming data as text.
                        text_data = conn.recv(4096)
                        if not text_data:
                            break
                        try:
                            message = text_data.decode('utf-8')
                            print(f"Received message: {message}")
                        except UnicodeDecodeError:
                            print("Failed to decode text data as UTF-8.")

if __name__ == "__main__":
    run_server()
