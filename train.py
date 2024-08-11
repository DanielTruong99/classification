from learning import Learner
from train_cfg import IrisTrainCfg

train_cfg = IrisTrainCfg()
learner = Learner(train_cfg)

learner.train()