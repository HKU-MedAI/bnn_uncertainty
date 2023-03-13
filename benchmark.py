import argparse
import random

import torch
import yaml

from globals import *
from trainer import (
    BNNUncertaintyTrainer,
    BNNARHTTrainer,
    EDLTrainer,
    DPNTrainer,
    DeepEnsemblesTrainer,
    MCDTrainer,
    WhatUncertaintiesTrainer
)
from utils import ordered_yaml

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=612)

args = parser.parse_args()

opt_path = args.config
default_config_path = "LeNet_DPN_CIFAR.yml"

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

def parse_trainer(config):
    if mode == "train":
        if config["train_type"] == "bnn-uncertainty":
            trainer = BNNUncertaintyTrainer(config)
        elif config["train_type"] == "arht-uncertainty":
            trainer = BNNARHTTrainer(config)
        elif config["train_type"] == "edl":
            trainer = EDLTrainer(config)
        elif config["train_type"] == "dpn":
            trainer = DPNTrainer(config)
        elif config["train_type"] == "ensembles":
            trainer = DeepEnsemblesTrainer(config)
        elif config["train_type"] == "mcd":
            trainer = MCDTrainer(config)
        elif config["train_type"] == "what-uncertainties":
            trainer = WhatUncertaintiesTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
        return trainer
    else:
        raise NotImplementedError("This mode is not implemented")

def benchmark_lambda(config):

    for lamb in [0.0001, 0.001, 0.01, 0.1, 1]:
        config["train"]["init_lambda"] = lamb
        config["checkpoints"]["path"] = f"./checkpoints/BLeNet_ARHT_lamb{lamb}"

        trainer = parse_trainer(config)

        trainer.train()

def benchmark_sample_size(config):

    for n_2 in [5, 10, 20, 50, 100, 200, 500, 1000]:
        config["train"]["n_testing_samples"] = n_2
        config["checkpoints"]["path"] = f"./checkpoints/BLeNet_ARHT_n2_{n_2}"

        trainer = parse_trainer(config)

        trainer.train()

def benchmark_datasets(config):
    name = config["name"]
    archi = config["train"]["model_name"]
    in_datasets = ["MNIST", "CIFAR10"]
    out_datasets = ["FashionMNIST", "Omiglot", "SVHN"]

    for in_data in in_datasets:
        for out_data in out_datasets:
            config["dataset"]["in"] = in_data
            config["dataset"]["ood"] = out_data
            config["checkpoints"]["path"] = f"./checkpoints/{archi}_{name}_{in_data}_{out_data}"

            trainer = parse_trainer(config)

            trainer.train()

def benchmark_dimensions(config):
    pass

def benchmark_architectures(config):

    name = config["name"]
    in_data = "CIFAR10"
    out_data = "SVHN"

    # for archi in ["BAlexNet", "BLeNet", "BResNet"]:
    for archi in ["CNN", "AlexNet", "LeNet", "ResNet"]:
        config["train"]["model_name"] = archi
        config["checkpoints"]["path"] = f"./checkpoints/{archi}_{name}_{in_data}_{out_data}"

        trainer = parse_trainer(config)

        trainer.train()


def benchmark_methods(config):
    pass


def main():
    # Load configurations
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    configs = [
        "LeNet_EDL_MNIST.yml",
        "LeNet_DPN_MNIST.yml",
        "BLeNet_ARHT_CIFAR10.yml"
    ]

    # benchmark_lambda(config)
    # benchmark_sample_size(config)
    # benchmark_datasets(config)
    benchmark_architectures(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()