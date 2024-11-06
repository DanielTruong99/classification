from learning import Learner
from src.radar_02.train_radar_cfg import RadarTrainCfg

# train_cfg = IrisTrainCfg()
train_cfg = RadarTrainCfg()
learner = Learner(train_cfg)

learner.train()