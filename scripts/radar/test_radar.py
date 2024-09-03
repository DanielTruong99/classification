import torch 
from learning import Learner
from scripts.radar.train_radar_cfg import RadarTrainCfg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
learner.model.load_state_dict(torch.load('logs/colision_classifier_20240903_004416/model_20240903_004416_10479', map_location=learner.device))

learner.model.eval()

with torch.no_grad():
    learner.model.eval()
    for index, data in enumerate(learner.data_loaders['test']):
        inputs, labels = data['input'].to(learner.device), data['label'].to(learner.device)
        
        start_time = time.time()
        learner.model(inputs[0:1, :])
        end_time = time.time()
        predicts = learner.model(inputs)

        print(f"Time taken for computation in {train_cfg.device}: {(end_time - start_time)/1e-3} milliseconds")

        predicts = (predicts >= 0.3).long()

        wrong_predictions_index = torch.where(predicts != labels)[0]
        wrong_inputs = inputs[wrong_predictions_index, :]
        wrong_true_labels = labels[wrong_predictions_index]
        wrong_predicted_labels = predicts[wrong_predictions_index]
        wrong_predictions = torch.cat([wrong_inputs, wrong_true_labels.unsqueeze(1), wrong_predicted_labels.unsqueeze(1)], dim=1)

        cm = confusion_matrix(labels.cpu().numpy(), predicts.cpu().numpy())

df = pd.DataFrame(wrong_predictions.cpu().numpy(), columns=['Distance', 'Estimated Collision Time', 'Velocity', 'Labels', 'Predictions'])
df.to_csv('wrong_predictions.csv', index=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

df_true = df[df['Labels'] == 1]; df_false = df[df['Labels'] == 0]
axs[0].hist(df_true['Estimated Collision Time'], bins=30, alpha=0.7, label='True but predict False', color='blue', edgecolor='black')
axs[0].hist(df_false['Estimated Collision Time'], bins=30, alpha=0.7, label='Falsebut predict True', color='red', edgecolor='black')
axs[0].set_title('Estimated Collision Time Histogram')
# axs[0].set_xlabel('Estimated Collision Time')
axs[0].legend()
axs[0].set_ylabel('Frequency')

axs[1].hist(df_false['Distance'], bins=30, alpha=0.7, label='True but predict False', color='blue', edgecolor='black')
axs[1].hist(df_false['Distance'], bins=30, alpha=0.7, label='Falsebut predict True', color='red', edgecolor='black')
axs[1].set_title('Distance Histogram')
# axs[1].set_xlabel('Distance')
axs[1].legend()
axs[1].set_ylabel('Frequency')

axs[2].hist(df_false['Velocity'], bins=30, alpha=0.7, label='True but predict False', color='blue', edgecolor='black')
axs[2].hist(df_false['Velocity'], bins=30, alpha=0.7, label='Falsebut predict True', color='red', edgecolor='black')
axs[2].set_title('Velocity Histogram')
# axs[2].set_xlabel('Velocity')
axs[2].legend()
axs[2].set_ylabel('Frequency')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
        