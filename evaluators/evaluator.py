from abc import ABC
from checkpoint import CheckpointManager

import torch
from torch import nn


class Evaluator(ABC):
    def __init__(self, config) -> None:
        # Categorize configs
        self.config = config
        self.config_data = config["dataset"]
        self.config_train = config['train']
        self.config_eval = config['eval']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoints']

        # GPU configs
        self.device = "cuda" if config['gpu_ids'] else "cpu"

        # Define checkpoints manager
        self.checkpoint_manager = CheckpointManager(path=config['checkpoints']["path"])

        # Paths
        self.eval_path = self.config_data["eval_path"]
        self.normal_path = self.config_data["normal_path"]
        self.eval_results_path = self.config_eval["eval_results_path"]

        print(
            f"Loaded checkpoints with path {config['checkpoints']['path']} version {self.checkpoint_manager.version}"
        )

    def eval(self) -> None:
        raise NotImplementedError
