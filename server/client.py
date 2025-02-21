import socket

def send_data_to_server(data_to_send):

    HOST = '172.20.10.2'
    PORT = 5001  # Same port the server is listening on

    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        # Connect to the server
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")

        # Send data (string, bytes, etc.)
        client_socket.sendall(data_to_send.encode('utf-8'))
        print(f"Sent to server: {data_to_send}")

if __name__ == "__main__":
    # Example usage
    send_data_to_server("Hello from Raspberry Pi!")
