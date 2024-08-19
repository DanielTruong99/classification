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
learner.model.load_state_dict(torch.load('logs/colision_classifier_20240818_113814/model_20240818_113814_4478', map_location=learner.device))

with torch.no_grad():
    learner.model.eval()
    for index, data in enumerate(learner.data_loaders['test']):
        inputs, labels = data['input'].to(learner.device), data['label'].to(learner.device)
        
        start_time = time.time()
        learner.model(inputs[0:1, :])
        end_time = time.time()
        predicts = learner.model(inputs)

        print(f"Time taken for computation in {train_cfg.device}: {(end_time - start_time)/1e-3} milliseconds")

        predicts = (predicts >= 0.5).long()

        wrong_predictions_index = torch.where(predicts != labels)[0]
        wrong_inputs = inputs[wrong_predictions_index, :]
        wrong_true_labels = labels[wrong_predictions_index]
        wrong_predicted_labels = predicts[wrong_predictions_index]
        wrong_predictions = torch.cat([wrong_inputs, wrong_true_labels.unsqueeze(1), wrong_predicted_labels.unsqueeze(1)], dim=1)

        cm = confusion_matrix(labels.cpu().numpy(), predicts.cpu().numpy())

df = pd.DataFrame(wrong_predictions.cpu().numpy(), columns=['Distance', 'Angle', 'Velocity', 'Labels', 'Predictions'])
df.to_csv('wrong_predictions.csv', index=False)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
        