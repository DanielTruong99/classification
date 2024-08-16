from learning import Learner
from train_cfg import IrisTrainCfg
from train_radar_cfg import RadarTrainCfg

# train_cfg = IrisTrainCfg()
train_cfg = RadarTrainCfg()
learner = Learner(train_cfg)

learner.train()