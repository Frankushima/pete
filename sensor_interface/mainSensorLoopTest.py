import socket
import threading
import time

# EDUROAM IPS
servers = [
    {'name': 'Double Flat', 'ip': '169.231.202.206', 'port': 8000},
    {'name': 'Pedal Wrench', 'ip': '169.231.198.85', 'port': 8001},
]

RECONNECT_DELAY = 1
CONNECT_TIMEOUT = 10
RECV_SIZE = 1024
LINES_PER_BLOCK = 3


def connect_to_server(name, ip, port):
    while True:
        try:
            print(f"[{name}] Attempting to connect to {ip}:{port}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(CONNECT_TIMEOUT)
                s.connect((ip, port))
                print(f"[{name}] Connected to server {ip}:{port}")

                complete_data = ""
                data_block = []

                while True:
                    recv_bytes = s.recv(RECV_SIZE)

                    if not recv_bytes:
                        print(f"[{name}] Connection closed by server")

                        # Process any leftover lines if desired
                        if data_block:
                            print(f"[{name}] Processing leftover partial block")
                            process_data_block(name, port, data_block)

                        break

                    recv_data = recv_bytes.decode("utf-8", errors="replace")
                    complete_data += recv_data

                    while '\n' in complete_data:
                        line, complete_data = complete_data.split('\n', 1)
                        cleaned_line = line.strip()

                        if cleaned_line:  # skip blank lines
                            data_block.append(cleaned_line)

                        if len(data_block) == LINES_PER_BLOCK:
                            process_data_block(name, port, data_block)
                            data_block = []

            print(f"[{name}] Disconnected from server {ip}:{port}")
            print(f"[{name}] Reconnecting in {RECONNECT_DELAY} second(s)...")
            time.sleep(RECONNECT_DELAY)

        except socket.timeout:
            print(f"[{name}] Connection to {ip}:{port} timed out. Retrying...")
            time.sleep(RECONNECT_DELAY)

        except socket.error as err:
            print(f"[{name}] Socket error: {err}")
            time.sleep(RECONNECT_DELAY)

        except Exception as e:
            print(f"[{name}] Unexpected error: {e}")
            time.sleep(RECONNECT_DELAY)


def process_data_block(name, port, data_block):
    print(f"\n[{name}] Processed data block from port {port}:")
    for i, line in enumerate(data_block, start=1):
        print(f"  Line {i}: {line}")
    print()


def main():
    threads = []

    for tool in servers:
        thread = threading.Thread(
            target=connect_to_server,
            args=(tool['name'], tool['ip'], tool['port']),
            daemon=True
        )
        thread.start()
        threads.append(thread)

    print("ACTIVE THREADS:", [thread.name for thread in threads])

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nMain program interrupted. Exiting...")


if __name__ == '__main__':
    main()