from dataclasses import dataclass
from typing import List, Optional, Dict

class DataSetCfg:
    class BaseDatasetCfg:
        batch_size: int = 32
        is_shuffle: bool = True
        num_workers: int = 4

    dataset_path: str = 'dataset/iris'
    train: BaseDatasetCfg = BaseDatasetCfg()
    val: BaseDatasetCfg = BaseDatasetCfg()
    test: BaseDatasetCfg = BaseDatasetCfg()

class CriticCfg:
    input_dimension: int = 4
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

class TrainCfg:
    model_name: str = 'iris_classifier'
    device: str = 'cuda'
    critic: CriticCfg = CriticCfg()
    optimizer: OptimizerCfg = OptimizerCfg()
    dataset: DataSetCfg = DataSetCfg()