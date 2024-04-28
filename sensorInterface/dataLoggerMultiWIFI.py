import socket
import threading
import time

# List of ESP32 server IP addresses
# '192.168.0.1354' MY HOUSE

servers = [
    {'ip': '192.168.0.135', 'port': 80},  # Wrench
    ]

# This is the threaded function for handling server connections
def connect_to_server(ip, port):
    output_file = f"{ip.replace('.', '_')}_{port}_{time.strftime('%Y%m%d_%H%M%S')}.txt"  # Unique filename based on IP, port, and current time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            print(f"Begin connection sequence")
            s.connect((ip, port))
            print(f"Connected to server {ip}:{port}")
        
            with open(output_file, 'a') as f:  # Open the output file in append mode
                while True:
                    data = s.recv(1024)
                    if not data:
                        break  # Exit loop if server closes the connection
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Current time
                    formatted_data = f"[{timestamp}] Data from {ip}:{port}: {data.decode()}"
                    print(formatted_data, end='')  # Print to console
                    f.write(formatted_data + '\n')  # Write to file
        except KeyboardInterrupt:
            print(f"Manual interruption. Closing file and disconnecting from {ip}:{port}.")
            time.sleep(1)
        except Exception as e:
            print(f"Error during connection or file operation: {e}")
        finally:
            print(f"Disconnected from server {ip}:{port}")

if __name__ == '__main__':
    threads = []
    for server in servers:
        if server['ip']:  # Ensure the IP is not empty
            thread = threading.Thread(target=connect_to_server, args=(server['ip'], server['port']))
            thread.start()
            threads.append(thread)
    
    for thread in threads:
        thread.join()  # Wait for all threads to complete