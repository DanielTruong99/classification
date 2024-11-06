from dataclasses import dataclass
from typing import List, Optional, Dict
from learning.cfg import TrainCfg

class DataSetCfg:
    class BaseDatasetCfg:
        batch_size: int = 8192
        is_shuffle: bool = True
        num_workers: int = 4
        input_data_index: List[int] = [0, 1, 2, 3, 11, 12, 13, 14, 15, 16] # r1, r2, r3, r4, s_q4, c_q4, s_q5, c_q5, s_q6, c_q6
        label_data_index: List[int] = [7, 8, 9, 10] # is_on_1, is_on_2, is_on_3, is_on_4

    dataset_path: str = 'dataset/radar_02'
    train: BaseDatasetCfg = BaseDatasetCfg()
    val: BaseDatasetCfg = BaseDatasetCfg()
    test: BaseDatasetCfg = BaseDatasetCfg()

class CriticCfg:
    input_dimension: int = 10
    output_dimension: int = 4
    is_normalize_input: bool = True

    hidden_architecture: List[Dict] = [
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"},
        # {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
    ]

class OptimizerCfg:
    learning_rate: float = 3.0e-4
    epochs: int = 10000
    optimizer: str = "Adam"
    weight_decay: float = 1e-5
    criterion: str = "BCEWithLogitsLoss"

class RadarTrainCfg(TrainCfg):
    model_name: str = 'colision_classifier'
    device: str = 'cuda'
    critic: CriticCfg = CriticCfg()
    optimizer: OptimizerCfg = OptimizerCfg()
    dataset: DataSetCfg = DataSetCfg()
    
