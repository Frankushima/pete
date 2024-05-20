import socket
import threading
import time
from pathlib import Path

# List of ESP32 server IP addresses
servers = [
    {"name": "Wrench", "ip": "192.168.0.135", "port": 80},
    # {"name": "Sensor2", "ip": "192.168.0.136", "port": 80},
]

stop_event = threading.Event()

# Settings
RECV_BUFFER_SIZE = 1024
SOCKET_TIMEOUT = 5              # seconds
RECONNECT_DELAY = 3             # seconds
NO_DATA_WARNING_TIME = 10       # seconds without data before warning
SEND_HEARTBEAT = False          # Pings ESP32 to check connection
HEARTBEAT_MESSAGE = "ping\n"
HEARTBEAT_INTERVAL = 5          # seconds


def safe_print(message: str):
    """Thread-safe print with timestamp."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def connect_to_server(name, ip, port):
    logs_dir = Path("sensor_logs")
    logs_dir.mkdir(exist_ok=True)

    output_file = logs_dir / f"{name}_{ip.replace('.', '_')}_{port}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    retry_count = 0

    while not stop_event.is_set():
        try:
            safe_print(f"{name}: Begin connection sequence to {ip}:{port}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(SOCKET_TIMEOUT)
                s.connect((ip, port))
                safe_print(f"{name}: Connected to server {ip}:{port}")

                retry_count = 0
                last_data_time = time.time()
                last_heartbeat_time = 0

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.flush()

                    while not stop_event.is_set():
                        # Optional heartbeat
                        if SEND_HEARTBEAT and (time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL):
                            try:
                                s.sendall(HEARTBEAT_MESSAGE.encode("utf-8"))
                                safe_print(f"{name}: Sent heartbeat")
                                last_heartbeat_time = time.time()
                            except Exception as e:
                                safe_print(f"{name}: Failed to send heartbeat: {e}")
                                break

                        try:
                            data = s.recv(RECV_BUFFER_SIZE)

                            if not data:
                                safe_print(f"{name}: Server closed connection")
                                break

                            last_data_time = time.time()

                            decoded_data = data.decode("utf-8", errors="replace").strip()
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                            formatted_data = f"[{timestamp}] {name} ({ip}:{port}): {decoded_data}"
                            print(formatted_data)
                            f.write(formatted_data + "\n")
                            f.flush()

                        except socket.timeout:
                            # No data received during timeout window
                            if time.time() - last_data_time >= NO_DATA_WARNING_TIME:
                                warning = f"{name}: No data received for {NO_DATA_WARNING_TIME}+ seconds"
                                safe_print(warning)
                                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: {warning}\n")
                                f.flush()
                                last_data_time = time.time()  # prevents warning spam

                        except Exception as e:
                            safe_print(f"{name}: Receive error: {e}")
                            break

        except KeyboardInterrupt:
            stop_event.set()
            safe_print(f"{name}: Manual interruption received")
            break

        except Exception as e:
            retry_count += 1
            safe_print(f"{name}: Connection error to {ip}:{port}: {e}")
            safe_print(f"{name}: Retry attempt #{retry_count} in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)

        finally:
            safe_print(f"{name}: Disconnected from server {ip}:{port}")

    safe_print(f"{name}: Thread exiting")


if __name__ == "__main__":
    threads = []

    try:
        for server in servers:
            if server.get("ip"):
                thread = threading.Thread(
                    target=connect_to_server,
                    args=(server["name"], server["ip"], server["port"]),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)

    except KeyboardInterrupt:
        safe_print("Main program interrupted. Stopping all threads...")
        stop_event.set()

    finally:
        for thread in threads:
            thread.join(timeout=2)

        safe_print("Program terminated cleanly.")