import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
file_path = '/home/ryz2/DanielWorkspace/DL/classification/raw_data/Nov05/falseDetection3Wrist241105.xlsx'
data = pd.read_excel(file_path)

#! plot r1, r2, r3, r4 with q4, q5, q6 in the same plot
# # Extract the columns for r1, r2, r3, and r4
# r_columns = ['r1', 'r2', 'r3', 'r4']
# q_columns = ['q4', 'q5', 'q6']

# # Create subplots
# fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# # Plot r1, r2, r3, r4 with q4, q5, q6
# for i, r_col in enumerate(r_columns):
#     axs[i].plot(data[r_col], label=r_col)
#     for q_col in q_columns:
#         axs[i].plot(data[q_col], label=q_col)
#     axs[i].set_title(f'{r_col}, q4, q5, q6')
#     axs[i].set_xlabel('Sample Index')
#     axs[i].set_ylabel('Values')
#     axs[i].grid(True)  # Add grid to each subplot
#     axs[i].legend()

#! plot r1, r2, r3, r4 with q4, q5, q6 in different plots
# Create subplots
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

r_columns = ['r1', 'r2', 'r3', 'r4']
q_columns = ['q4', 'q5', 'q6']

# Plot r1, r2, r3, r4
for i, r_col in enumerate(r_columns):
    axs[i].plot(data[r_col], label=r_col)
    axs[i].set_title(f'{r_col}')
    axs[i].set_xlabel('Sample Index')
    axs[i].set_ylabel('Values')
    axs[i].grid(True)  # Add grid to each subplot
    axs[i].legend()

# Plot q4, q5, q6
for q_col in q_columns:
    axs[-1].plot(data[q_col], label=q_col)
    axs[-1].set_xlabel('Sample Index')
    axs[-1].set_ylabel('Values')
    axs[-1].grid(True)  # Add grid to each subplot
    axs[-1].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()