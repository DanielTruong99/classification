import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'raw_data/Aug30/TrueData_Updated.csv'
df_true = pd.read_csv(file_path)
file_path = 'raw_data/Aug30/Merged_FalseData.csv'
df_false = pd.read_csv(file_path)

# Add a new column to the dataframes
df_true['Estimated Collision Time'] = df_true['Distance'] / abs(df_true['Velocity'] + 1e-6); df_true.loc[df_true['Estimated Collision Time'] > 10000, 'Estimated Collision Time'] = 0.0
df_false['Estimated Collision Time'] = df_false['Distance'] / abs(df_false['Velocity'] + 1e-6); df_false.loc[df_false['Estimated Collision Time'] > 10000, 'Estimated Collision Time'] = 0.0

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# Plotting the histogram for the 'Velocity' column
axs[0, 0].hist(df_true['Velocity'], bins=30, alpha=0.7, label='True', color='blue', edgecolor='black')
axs[0, 0].hist(df_false['Velocity'], bins=30, alpha=0.7, label='False', color='orange', edgecolor='black')
axs[0, 0].set_title('Velocity Histogram')
axs[0, 0].set_xlabel('Velocity')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plotting the histogram for the 'Angle' column
axs[0, 1].hist(df_true['Angle'], bins=30, alpha=0.7, label='True', color='blue', edgecolor='black')
axs[0, 1].hist(df_false['Angle'], bins=30, alpha=0.7, label='False', color='orange', edgecolor='black')
axs[0, 1].set_title('Angle Histogram')
axs[0, 1].set_xlabel('Angle')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plotting the histogram for the 'Distance' column
axs[1, 0].hist(df_true['Distance'], bins=30, alpha=0.7, label='True', color='blue', edgecolor='black')
axs[1, 0].hist(df_false['Distance'], bins=30, alpha=0.7, label='False', color='orange', edgecolor='black')
axs[1, 0].set_title('Distance Histogram')
axs[1, 0].set_xlabel('Distance')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plotting the histogram for the 'Estimated Collision Time' column
axs[1, 1].hist(df_true['Estimated Collision Time'], bins=30, alpha=0.7, label='True', color='blue', edgecolor='black')
axs[1, 1].hist(df_false['Estimated Collision Time'], bins=30, alpha=0.7, label='False', color='orange', edgecolor='black')
axs[1, 1].set_title('Estimated Collision Time Histogram')
axs[1, 1].set_xlabel('Estimated Collision Time')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust the layout so the plots do not overlap
plt.tight_layout()

# Display the plot
plt.show()

# Save the modified data
df_true.to_csv('raw_data/Aug30/Modified_TrueData_Updated.csv', index=False)
df_false.to_csv('raw_data/Aug30/Modified_FalseData_Updated.csv', index=False)