import socket
import datetime

# Server configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5001       # Port to listen on

# Create server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"Fall Detection Alert Server listening on port {PORT}")
print(f"Server IP: {socket.gethostbyname(socket.gethostname())}")
print("Waiting for alerts...")

while True:
    try:
        # Accept connection
        client_socket, client_address = server_socket.accept()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\nConnection from {client_address} at {timestamp}")
        
        # Receive data
        data = client_socket.recv(1024)
        if data:
            message = data.decode('utf-8')
            if "FALL_DETECTED" in message:
                print("\n***FALL DETECTED! ***")
                print(f"{timestamp} | {message}")
                print("*********************\n")
            elif message.startswith("DATA"):
                print(f"{message}")
            
            # Send acknowledgment
            client_socket.send("Alert received".encode('utf-8'))
        
        # Close connection
        client_socket.close()
        
    except Exception as e:
        print(f"Error: {e}")