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


train_cfg = RadarTrainCfg(); train_cfg.device = 'cpu'
learner = Learner(train_cfg)
test_dataset = CSVDataset(os.path.join(train_cfg.dataset.dataset_path, 'test'), train_cfg.dataset.test)
learner.data_loaders['test'] = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset), 
    shuffle=False
)
learner.model.load_state_dict(torch.load('logs/colision_classifier_20241106_184820/model_20241106_184820_1736', map_location=learner.device))

learner.model.eval()

with torch.no_grad():
    learner.model.eval()
    for index, data in enumerate(learner.data_loaders['test']):
        inputs, labels = data['input'].to(learner.device), data['label'].to(learner.device)
        
        # Measure time taken for 1 sample computation
        start_time = time.time()
        logits = learner.model(inputs[0:1, :])
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.7).float()
        end_time = time.time()

        # Batch computation for measure accuracy
        logits = learner.model(inputs)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.7).float()

        print(f"Time taken for computation in {train_cfg.device}: {(end_time - start_time)/1e-3} milliseconds")


print(f"Accuracy based on metric Subset Accuracy: {accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())}")
print(f"Accuracy based on metric F1 Score: {f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='samples')}")
print(f"Accuracy based on metric Average Precision Score: {average_precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='samples')}")

#* Plot the confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import numpy as np

confusion_matrix = multilabel_confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for index, (matrix, label) in enumerate(zip(confusion_matrix, ['is_on_1', 'is_on_2', 'is_on_3', 'is_on_4'])):

    sns.heatmap(matrix, annot=True, fmt='d', ax=ax[index//2, index%2])
    ax[index//2, index%2].set_title(f'Confusion matrix for {label}')
    ax[index//2, index%2].set_xlabel('Predicted')
    ax[index//2, index%2].set_ylabel('True')

plt.show()



        