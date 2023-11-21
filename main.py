import argparse
import random

import torch
import yaml

from globals import *
from trainer import (
    BNNARHTCLTrainer,
    BNNARHTTrainer,
    EDLTrainer,
    DPNTrainer,
    DeepEnsemblesTrainer,
    DeepEnsemblesSimulationTrainer,
    MCDTrainer,
    WhatUncertaintiesTrainer,
    WhatUncertaintiesSimulationTrainer,
    ARHTMetricsTrainer,
    ARHTMetricsFreqTrainer,
    BNNARHTSimulationTrainer,
    MCDSimulatedTrainer,
)
from utils import ordered_yaml

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=612)

args = parser.parse_args()

opt_path = args.config
default_config_path = "BResNet_ARHT_ImageNet.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

####
# Set types (train/eval)
####
mode = "train"


def main():
    # Load configurations
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    if mode == "train":
        if config["train_type"] == "arht-uncertainty":
            trainer = BNNARHTTrainer(config)
        elif config["train_type"] == "arht-uncertainty-sim":
            trainer = BNNARHTSimulationTrainer(config)
        elif config["train_type"] == "arht-cl":
            trainer = BNNARHTCLTrainer(config)
        elif config["train_type"] == "arht-metrics":
            trainer = ARHTMetricsTrainer(config)
        elif config["train_type"] == "arht-metrics-freq":
            trainer = ARHTMetricsFreqTrainer(config)
        elif config["train_type"] == "edl":
            trainer = EDLTrainer(config)
        elif config["train_type"] == "dpn":
            trainer = DPNTrainer(config)
        elif config["train_type"] == "ensembles":
            trainer = DeepEnsemblesTrainer(config)
        elif config["train_type"] == "ensembles-sim":
            trainer = DeepEnsemblesSimulationTrainer(config)
        elif config["train_type"] == "mcd":
            trainer = MCDTrainer(config)
        elif config["train_type"] == "mcd-sim":
            trainer = MCDSimulatedTrainer(config)
        elif config["train_type"] == "what-uncertainties":
            trainer = WhatUncertaintiesTrainer(config)
        elif config["train_type"] == "what-uncertainties-sim":
            trainer = WhatUncertaintiesSimulationTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
        trainer.train()
    else:
        raise NotImplementedError("This mode is not implemented")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
