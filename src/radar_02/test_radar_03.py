import torch 
from learning import Learner
from src.radar_02.train_radar_cfg import RadarTrainCfg
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, roc_auc_score, average_precision_score
import os 
from learning.dataset import CSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time 
import pandas as pd
import math


model = torch.jit.load("colision_detector.pt")

# Read the CSV file
data = pd.read_csv('raw_data/Nov08/Cleaned_Data_without_Empty_Rows.csv')

# Replace 9999 with 1 in columns r1, r2, r3, r4
data[['r1', 'r2', 'r3', 'r4']] = data[['r1', 'r2', 'r3', 'r4']].replace(9999, 1)

# Add new columns based on the conditions
for index in range(4, 7):
    data[f's_q{index}'] = data[f'q{index}'].apply(lambda x: math.sin(x))
    data[f'c_q{index}'] = data[f'q{index}'].apply(lambda x: math.cos(x))

# Convert entire data to tensor
columns_to_save = ['r1', 'r2', 'r3', 'r4', 's_q4', 'c_q4', 's_q5', 'c_q5', 's_q6', 'c_q6']
inputs = torch.tensor(data[columns_to_save].values, dtype=torch.float32)

# Apply the model prediction
with torch.no_grad():
    logits = model(inputs)
    probs = torch.sigmoid(logits)
    predictions = (probs >= 0.7).float()

fig, axs = plt.subplots(4, 1, figsize=(10, 25))

for i in range(4):
    axs[i].plot(predictions[:2000, i], label=f'is_on_r{i+1}')
    axs[i].plot(data[f'r{i+1}'][:2000], label=f'r{i+1}')
    # axs[i].plot(data[f'q6'][:2000], label=f'q6')
    axs[i].legend()
    axs[i].grid(True)
    # axs[i].set_title(f'Predictions vs Actual for r{i+1} (First 1000 Data Points)')



plt.tight_layout()
plt.show()

