import os
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# Define the path to the folder containing the processed data files
folder_path = '/home/ryz2/DanielWorkspace/DL/classification/raw_data/Nov05/processed_data'

# Initialize an empty list to store the DataFrames
dataframes = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):  # Assuming the files are in CSV format
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path)
        dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
big_df = pd.concat(dataframes, ignore_index=True)
print("Merged DataFrame shape:", big_df.shape)

# Randomly shuffle the DataFrame
big_df = big_df.sample(frac=1).reset_index(drop=True)

# Define the proportions
train_size = 0.8
test_size = 0.1
validate_size = 0.1

# Select only the specified columns
columns_to_save = ['r1', 'r2', 'r3', 'r4', 'q4', 'q5', 'q6', 'is_on_1', 'is_on_2', 'is_on_3', 'is_on_4']
big_df = big_df[columns_to_save]
for index in range(4, 7):
    big_df[f's_q{index}'] = big_df[f'q{index}'].apply(lambda x: math.sin(x))
    big_df[f'c_q{index}'] = big_df[f'q{index}'].apply(lambda x: math.cos(x))


# Split the DataFrame into train, test, and validate sets
train_df, temp_df = train_test_split(big_df, test_size=(1 - train_size))
test_df, validate_df = train_test_split(temp_df, test_size=0.5)

# Optionally, save the train, test, and validate DataFrames to new CSV files
dataset_dir = '/home/ryz2/DanielWorkspace/DL/classification/raw_data/Nov05/dataset'
train_df.to_csv(os.path.join(dataset_dir, "train", "train.csv"), index=False)
test_df.to_csv(os.path.join(dataset_dir, "test", "test.csv"), index=False)
validate_df.to_csv(os.path.join(dataset_dir, "validate", "validate.csv"), index=False)

print("Train DataFrame shape:", train_df.shape)
print("Test DataFrame shape:", test_df.shape)
print("Validate DataFrame shape:", validate_df.shape)