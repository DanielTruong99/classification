import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import nn
from ..cfg import CriticCfg

class Classifier(torch.nn.Module):
    def __init__(self, cfg: CriticCfg):
        super(Classifier, self).__init__()
        
        architecture = cfg.architecture
        layers = []
        #* Add input layer
        layers.append(torch.nn.BatchNorm1d(cfg.input_dimension))
        layers.append(torch.nn.Linear(cfg.input_dimension, architecture[0]['hidden_dimension']))
        
        #* Add hidden layers
        for index, layer in enumerate(architecture):
            if index == len(architecture) - 1:
                break
            
            pre_process = getattr(torch.nn, layer['pre_process']) if 'pre_process' in layer or index == 0 else None
            if pre_process is not None:
                layers.append(pre_process(architecture[index - 1]['hidden_dimension']))

            dense_layer = torch.nn.Linear(layer['hidden_dimension'], architecture[index + 1]['hidden_dimension'])
            layers.append(dense_layer)
            
            post_process = self._get_activation(layer['activation']) if 'activation' in layer else None
            if post_process is not None:
                layers.append(post_process) 

        #* Add output layer
        layers.append(torch.nn.Linear(architecture[-1]['hidden_dimension'], cfg.output_dimension))

        #* Initialize the critic
        self.critic = torch.nn.Sequential(*layers)

    def _get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None
            
    def forward(self, x):
        logits = self.critic(x)
        probabilities = torch.sigmoid(logits)
        return probabilities.squeeze(1)