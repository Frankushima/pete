import matplotlib.pyplot as plt
import os

def extract_and_save_angles(input_filepath, output_filepath):
    # Open the input file and the output file
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        # Read through each line in the input file
        lines = infile.readlines()  # Read all lines into a list
        
        for line in lines:
            # Check if the line contains 'Angle X:'
            if 'Angle X:' in line:
                # Write this line to the output file
                outfile.write(line + '\n')  # Add a newline for better formatting in the output file

directory, filename = 'run_2024-05-05_23-28-11/192_168_0_177_81_2024-05-05_23-28-11.txt'.split('/')

input_filepath = os.path.join(directory, filename)

output_filepath = 'angles_output.txt'  # Path where you want to save the angle data
extract_and_save_angles(input_filepath, output_filepath)

# Initialize lists to store the angle data
angle_x = []
angle_y = []
angle_z = []

# Read the data from the file
with open('angles_output.txt', 'r') as file:
    for line in file:
        # Parse the line to extract angles
        parts = line.split()
        if len(parts) == 7:
            x = float(parts[2])
            y = float(parts[4])
            z = float(parts[6])
            # Append to lists
            angle_x.append(x)
            angle_y.append(y)
            angle_z.append(z)

# Calculate cumulative sums
cumulative_x = [sum(angle_x[:i+1]) for i in range(len(angle_x))]
cumulative_y = [sum(angle_y[:i+1]) for i in range(len(angle_y))]
cumulative_z = [sum(angle_z[:i+1]) for i in range(len(angle_z))]

max_x = max(cumulative_x)
max_y = max(cumulative_y)
max_z = max(cumulative_z)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(cumulative_x, label='Cumulative Angle X', marker='o')
plt.plot(cumulative_y, label='Cumulative Angle Y', marker='^')
plt.plot(cumulative_z, label='Cumulative Angle Z', marker='s')

max_x_index = cumulative_x.index(max_x)
max_y_index = cumulative_y.index(max_y)
max_z_index = cumulative_z.index(max_z)

plt.scatter(max_x_index, max_x, color='red', s=100, edgecolors='black', zorder=5)
plt.scatter(max_y_index, max_y, color='blue', s=100, edgecolors='black', zorder=5)
plt.scatter(max_z_index, max_z, color='green', s=100, edgecolors='black', zorder=5)

# Annotate the maximum points
plt.annotate(f'{max_x:.5f}', xy=(max_x_index, max_x), xytext=(max_x_index+5, max_x+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')
plt.annotate(f'{max_y:.5f}', xy=(max_y_index, max_y), xytext=(max_y_index+5, max_y+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')
plt.annotate(f'{max_z:.5f}', xy=(max_z_index, max_z), xytext=(max_z_index+5, max_z+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='black')

plt.title('Cumulative Angles over Time')
plt.xlabel('Index of Data Point')
plt.ylabel('Cumulative Angle (degrees)')
plt.legend()
plt.grid(True)
plt.show()