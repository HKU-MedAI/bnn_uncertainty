from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
import wandb

from checkpoint import CheckpointManager
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["dataset"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoints']
        self.config_logging = config["logging"]

        # Define checkpoints manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Load number of epochs
        self.n_epoch = self.config_train['num_epochs']
        self.starting_epoch = self.checkpoint_manager.version

        # Read batch size
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

    def train(self) -> None:
        raise NotImplementedError

    def initialize_logger(self, notes=""):
        name = "_".join(
            [
                self.config_train["model_name"],
                self.config["name"],
                self.config_data["in"],
                self.config_data["ood"],
            ]
        )
        tags = self.config["logging"]["tags"]
        tags += [
            self.config_data["in"],
            self.config_data["ood"]
        ]
        wandb.init(name=name,
                   project='BNN_Uncertainty',
                   notes=notes,
                   config=self.config,
                   tags=tags,
                   mode=self.config_logging["mode"]
                   )

    def visualize_scores(self, in_scores, out_scores, epoch):
        pth = self.checkpoint_manager.path / "visualizations"
        if not pth.exists():
            pth.mkdir()

        fig, ax = plt.subplots()
        ax.hist(in_scores, bins=100, color="g", range=(-1000, 5000), label="in", alpha=0.6)
        ax.hist(out_scores, bins=100, color="r", range=(-1000, 5000), label="ood", alpha=0.5)
        ax.legend()

        plt.savefig(pth / f"scors_ep{epoch}.png")
        wandb.log(fig)
        plt.close()

    @staticmethod
    def format_scores(scores):
        index = np.isposinf(scores)
        scores[np.isposinf(scores)] = 1e9
        maximum = np.amax(scores)
        scores[np.isposinf(scores)] = maximum + 1

        index = np.isneginf(scores)
        scores[np.isneginf(scores)] = -1e9
        minimum = np.amin(scores)
        scores[np.isneginf(scores)] = minimum - 1

        scores[np.isnan(scores)] = 0

        return scores

    def comp_aucs_ood(self, scores, labels_1, labels_2):
        auroc_1 = roc_auc_score(labels_1, scores)
        auroc_2 = roc_auc_score(labels_2, scores)
        auroc = max(auroc_1, auroc_2)

        precision, recall, thresholds = precision_recall_curve(labels_1, scores)
        aupr_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(labels_2, scores)
        aupr_2 = auc(recall, precision)

        aupr = max(aupr_1, aupr_2)

        return auroc, aupr, precision, recall

    def compute_entropy(self, alphas, alpha0):
        probs = alphas / alpha0
        entropy = -torch.sum(probs*torch.log(probs), dim=2)
        conf = torch.max(probs, dim=2).values

        return entropy.mean(0).numpy(), conf.mean(0).numpy()

    def compute_diff_entropy(self, alphas, alpha0):
        s = torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=2) - torch.lgamma(alpha0[:, :, 0])

        return s.mean(0).numpy()