import torch 
from learning import Learner
from scripts.radar.train_radar_cfg import RadarTrainCfg

train_cfg = RadarTrainCfg(); train_cfg.device = 'cpu'
learner = Learner(train_cfg)
learner.model.load_state_dict(torch.load('logs/colision_classifier_20240820_233543/model_20240820_233543_1253', map_location=learner.device))

#! Compile the model to TorchScript
example_input = torch.rand(2, 3)
traced_script_module = torch.jit.trace(learner.model, example_input)

#! Serialize the model to a file for use in C++
traced_script_module.save("colision_detector.pt")
