import torch 
from learning import Learner
from src.radar_02.train_radar_cfg import RadarTrainCfg

train_cfg = RadarTrainCfg(); train_cfg.device = 'cpu'
learner = Learner(train_cfg)
learner.model.load_state_dict(torch.load('logs/colision_classifier_20241106_184820/model_20241106_184820_1736', map_location=learner.device))

#! Compile the model to TorchScript
example_input = torch.rand(1000, 10)
traced_script_module = torch.jit.trace(learner.model, example_input)

#! Serialize the model to a file for use in C++
traced_script_module.save("colision_detector.pt")
