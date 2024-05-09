import socket
import threading
import time

# Note the port numbers!!!
# # MY HOUSE AND FRANKS
# servers = [
#     {'ip': '192.168.0.135', 'port': 80},  # Double Flat
#     {'ip': '192.168.0.177', 'port': 81},  # Pedal Wrench
#     ]

# EDUROAM IPS
servers = [
    {'ip': '169.231.202.206', 'port': 8000},  # Double Flat
    {'ip': '169.231.193.109', 'port': 8001},  # Pedal Wrench
    ]

def connect_to_server(ip, port):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set a timeout after not connecting after 10 seconds
                s.settimeout(10)
                print(f"Attempting to connect to {ip}:{port}")
                s.connect((ip, port))
                print(f"\nConnected to server {ip}:{port}")

                complete_data = ""
                data_block = []
                try:
                    while True:
                        recv_data = s.recv(1024).decode()
                        if not recv_data:
                            if data_block:
                                # Remaining data that hsan't been processed
                                process_data_block(port, data_block)                  # Place holder for what datablock needs to be used for
                            break  # No more data, connection closed

                        complete_data += recv_data
                        while '\n' in complete_data:
                            line, complete_data = complete_data.split('\n', 1)
                            data_block.append(line.strip())
                            if len(data_block) == 3:
                                process_data_block(port, data_block)                  # Place holder for what datablock needs to be used for
                                data_block = []  # Reset for next block of data
                except Exception as e:
                    print(f"Error during connection or file operation: {e}")
                    break
            print(f"Disconnected from server {ip}:{port}")
        except socket.timeout:
                print(f"Connection to {ip}:{port} timed out. Retrying...")
                s.close()  # Ensure the socket is closed before retrying
                time.sleep(1)  # Wait a bit before retrying to avoid hammering the server too quickly
        except socket.error as err:
                print(f"Socket error: {err}")
                s.close()
                time.sleep(1)  # Wait a bit before retrying
        except Exception as e:
            print(f"Error during connection or file operation: {e}")
            s.close()
            break
        finally:
            s.close()

def process_data_block(port, data_block):
    # Placeholder function to process data blocks
    print(f"Processed data block from {port}:")
    for line in data_block:
        print(line)

def main():
    threads = []
    for tool in servers:
        thread = threading.Thread(target=connect_to_server, args=(tool['ip'], tool['port']))
        thread.start()
        threads.append(thread)

    print("ACTIVE THREADS: ", threads)

    for thread in threads:
        thread.join()  # Wait for all threads to complete

if __name__ == '__main__':
    main()