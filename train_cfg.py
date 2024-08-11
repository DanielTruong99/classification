from learning.cfg import TrainCfg

class IrisTrainCfg(TrainCfg):

    def __post_init__(self):
        self.critic.input_dimension = 4
        self.critic.output_dimension = 1
        self.critic.architecture = [
            {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
            {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
            {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        ]