from dataclasses import dataclass
from typing import List, Optional, Dict
from learning.cfg import TrainCfg

class DataSetCfg:
    class BaseDatasetCfg:
        batch_size: int = 128
        is_shuffle: bool = True
        num_workers: int = 4
        input_data_index: List[int] = [0, 1, 2]
        label_data_index: List[int] = [3]

    dataset_path: str = 'dataset/radar'
    train: BaseDatasetCfg = BaseDatasetCfg()
    val: BaseDatasetCfg = BaseDatasetCfg()
    test: BaseDatasetCfg = BaseDatasetCfg()

class CriticCfg:
    input_dimension: int = 3
    output_dimension: int = 1
    architecture: List[Dict] = [
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
        {"hidden_dimension": 256, "activation": "elu", "pre_process": "BatchNorm1d"}, 
    ]

class OptimizerCfg:
    learning_rate: float = 1e-3
    epochs: int = 1000
    optimizer: str = "Adam"

class RadarTrainCfg(TrainCfg):
    model_name: str = 'colision_classifier'
    device: str = 'cuda'
    critic: CriticCfg = CriticCfg()
    optimizer: OptimizerCfg = OptimizerCfg()
    dataset: DataSetCfg = DataSetCfg()
    