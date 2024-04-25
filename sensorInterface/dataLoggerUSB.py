import serial
import time

# Open the serial connection on the COM port your ESP32 is connected to
ser = serial.Serial('COM_PORT', 9600, timeout=1)

time.sleep(2)  # wait for the serial connection to initialize

def send_data(data):
    ser.write(data.encode())  # Send data as bytes

def read_data():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(f"Received: {line}")

while True:
    # Example of sending data to ESP32
    send_data("Hello, ESP32!")
    time.sleep(1)  # Wait a bit for ESP32 to respond

    # Read any incoming data
    read_data()