import socket

# ESP32_IP = '192.168.0.204'    # MY HOUSE
# ESP32_IP = '192.168.0.135'    # GAMER HOUSE
ESP32_IP = '169.231.192.50'  # EDUROAM

PORT = 8001

def connect_to_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        s.connect((ESP32_IP, PORT))
        print("Connected Wrench to server.")
        
        try:
            while True:
                data = s.recv(1024)
                print(data.decode(), end='')
        except KeyboardInterrupt:
            print("Disconnected from server.")

if __name__ == '__main__':
    connect_to_server()