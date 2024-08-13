import torch 
from learning import Learner
from train_cfg import IrisTrainCfg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os 
from learning.dataset import CSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time 

train_cfg = IrisTrainCfg(); train_cfg.device = 'cpu'
learner = Learner(train_cfg)
test_dataset = CSVDataset(os.path.join(train_cfg.dataset.dataset_path, 'validate'))
learner.data_loaders['test'] = DataLoader(
    test_dataset, 
    batch_size=len(test_dataset), 
    shuffle=False
)
learner.model.load_state_dict(torch.load('logs/iris_classifier_20240812_012235/model_20240812_012235_713', map_location=learner.device))

with torch.no_grad():
    learner.model.eval()
    for index, data in enumerate(learner.data_loaders['test']):
        inputs, labels = data['input'].to(learner.device), data['label'].to(learner.device)
        
        start_time = time.time()
        learner.model(inputs[0:1, :])
        end_time = time.time()
        predicts = learner.model(inputs)

        print(f"Time taken for computation: {(end_time - start_time)/1e-3} milliseconds")

        predicts = (predicts >= 0.5).long()

        cm = confusion_matrix(labels.cpu().numpy(), predicts.cpu().numpy())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
        