from learning import Learner
from scripts.iris.train_cfg import IrisTrainCfg
from scripts.radar.train_radar_cfg import RadarTrainCfg

# train_cfg = IrisTrainCfg()
train_cfg = RadarTrainCfg()
learner = Learner(train_cfg)

learner.train()