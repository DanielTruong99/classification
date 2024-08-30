import torch
from ..dataset import CSVDataset
from torch.utils.data import Dataset, DataLoader
import os
from ..model import Classifier
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

class Learner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = self._get_model(cfg)
        self.optimizer = self._get_optimizer(cfg)
        self.criterion = self._get_criterion(cfg)
        self.writter = None
                

        self.data_loaders = {
            'train': DataLoader(
                CSVDataset(os.path.join(self.cfg.dataset.dataset_path, 'train'), self.cfg.dataset.train), 
                batch_size=self.cfg.dataset.train.batch_size, 
                # batch_size=len(CSVDataset(os.path.join(self.cfg.dataset.dataset_path, 'train'), self.cfg.dataset.train)),
                shuffle=self.cfg.dataset.train.is_shuffle
            ),
            'val': DataLoader(
                CSVDataset(os.path.join(self.cfg.dataset.dataset_path, 'validate'), self.cfg.dataset.val), 
                batch_size=self.cfg.dataset.val.batch_size, 
                # batch_size=len(CSVDataset(os.path.join(self.cfg.dataset.dataset_path, 'validate'), self.cfg.dataset.val)),
                shuffle=self.cfg.dataset.val.is_shuffle
            ),
            'test': DataLoader(
                CSVDataset(os.path.join(self.cfg.dataset.dataset_path, 'test'), self.cfg.dataset.test), 
                batch_size=self.cfg.dataset.test.batch_size, 
                shuffle=self.cfg.dataset.test.is_shuffle
            ),
        }
        
    def _get_model(self, cfg):
        return Classifier(cfg.critic).to(cfg.device)
        
    def _get_optimizer(self, cfg):
        optimizer_cfg = cfg.optimizer
        if optimizer_cfg.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=optimizer_cfg.learning_rate)
        else:
            raise NotImplementedError('Optimizer not implemented, Currently supported: Adam')
        
    def _get_criterion(self, cfg):
        optimizer_cfg = cfg.optimizer
        loss_fcn = None
        
        if optimizer_cfg.criterion == 'BCE':
            loss_fcn = torch.nn.BCELoss()
        elif optimizer_cfg.criterion == 'MSE':
            loss_fcn = torch.nn.MSELoss()
        else:
            raise NotImplementedError('Criterion not implemented, Currently supported: BCE, MSE') 
        
        return loss_fcn
        
    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = 'logs/{}_{}'.format(self.cfg.model_name, timestamp)
        self.writer = SummaryWriter(log_dir)
        epoch_number = 0
        best_vloss = 1_000_000. 

        for epoch in range(self.cfg.optimizer.epochs):
            print('Epoch {}/{}'.format(epoch_number + 1, self.cfg.optimizer.epochs))

            #* Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch()

            #! Set the model to evaluation mode, disabling dropout and using population
            #! statistics for batch normalization.
            self.model.eval()

            #* Disable gradient computation and reduce memory consumption.
            running_vloss = 0.0
            with torch.no_grad():
                for index, vdata in enumerate(self.data_loaders['val']):
                    vinputs, vlabels = vdata['input'], vdata['label']
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    voutputs = self.model(vinputs)
                    vloss = self.criterion(voutputs, vlabels)
                    running_vloss += vloss
            avg_vloss = running_vloss / (index + 1)

            #* Log the running loss averaged per batch
            #* for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            self.writer.flush()

            #* Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = '{}/model_{}_{}'.format(log_dir, timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
    
    def train_one_epoch(self):
        running_loss = 0.
        has_visualized_graph = True
        for index, data in enumerate(self.data_loaders['train']):
            inputs, labels = data['input'], data['label']
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            #* Zero your gradients for every batch!
            self.optimizer.zero_grad()

            #* Visualize the graph once
            if has_visualized_graph:
                has_visualized_graph = False
                self.writer.add_graph(self.model, inputs)

            #* Make predictions for this batch
            outputs = self.model(inputs)

            #* Compute the loss and its gradients
            loss = self.criterion(outputs, labels)
            loss.backward()

            #* Adjust learning weights
            self.optimizer.step()

            #* Gather data and report
            running_loss += loss.item()

        avg_vloss = running_loss / (index + 1)
        return avg_vloss

    def test(self, test_loader):
        raise NotImplementedError