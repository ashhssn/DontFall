import socket

def run_server():
    # Define HOST and Port
    HOST = '0.0.0.0'
    PORT = 5001

    # Create a socket object using IPv4 (AF_INET) and TCP (SOCK_STREAM)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the given HOST and PORT
        server_socket.bind((HOST, PORT))
        # Start listening for incoming connections
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")

        
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")

            
            while True:
                data = conn.recv(1024)  # Buffer size
                if not data:
                    break

                # Process or handle the data here
                print(f"Received from client: {data.decode('utf-8')}")

if __name__ == "__main__":
    run_server()
