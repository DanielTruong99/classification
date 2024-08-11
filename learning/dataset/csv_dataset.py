import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        file_name = os.listdir(csv_file)
        if len(file_name) > 1:
            raise ValueError('The directory should contain only one csv file')
        
        csv_file_path = os.path.join(csv_file, file_name[0])
        self.data_frame = pd.read_csv(csv_file_path)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_data = self.data_frame.iloc[idx, 1:-1]
        label = self.data_frame.iloc[idx, -1]
        
        sample = {'input': torch.tensor(input_data, dtype=torch.float32), 
                  'label': torch.tensor(label, dtype=torch.float32)}

        return sample