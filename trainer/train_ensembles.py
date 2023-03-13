"""
Trainer of DPN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
import utils

from parse import (
    parse_loss,
    parse_optimizer,
    parse_frequentist_model
)
from losses import EdlLoss

import torchvision
import torch

from data import load_data
from utils import one_hot_embedding

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc

import torch.distributions as dist
from torch.distributions.dirichlet import Dirichlet


class DeepEnsemblesTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        image_size = self.config_data["image_size"]
        in_data_name = self.config_data["in"]
        ood_data_name = self.config_data["ood"]
        in_channel = self.config_train["in_channels"]

        train_in = load_data(in_data_name, True, image_size, in_channel)
        test_in = load_data(in_data_name, False, image_size, in_channel)
        train_out = load_data(ood_data_name, True, image_size, in_channel)
        test_out = load_data(ood_data_name, False, image_size, in_channel)


        # train_out.targets = torch.tensor(np.ones(len(train_out.targets)) * 10, dtype=torch.long)

        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_classes = self.config_train["out_channels"]
        self.n_models = self.config_train["n_models"]
        self.models = [parse_frequentist_model(self.config_train, image_size=image_size).to(self.device)
                      for _ in range(self.n_models)]
        self.optimzers = [parse_optimizer(self.config_optim, m.parameters()) for m in self.models]

        self.edl_loss = EdlLoss(self.device)

    @staticmethod
    def target_alpha(targets):
        target = targets.numpy()

        def gen_onehot(category, total_cat=10):
            label = np.ones(total_cat)
            label[category] = 20
            return label

        target_alphas = []
        for i in target:
            if i == 10:
                target_alphas.append(np.ones(10))
            else:
                target_alphas.append(gen_onehot(i))
        return torch.Tensor(target_alphas)

    def get_ood_label_score(self, test_in_score, test_out_score):
        score = np.concatenate([test_in_score, test_out_score])
        label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
        return label, score

    def train_one_step(self, data, label):

        label = utils.one_hot_embedding(label, self.n_classes).cuda()

        for i in range(self.n_models):
            # Training
            mu_train, sig_train = self.models[i](data)
            sig_train_pos = torch.log(1 + torch.exp(sig_train)) + 1e-6

            loss = torch.mean(
                0.5 * torch.log(sig_train_pos) + 0.5 * (torch.square(label - mu_train) / sig_train_pos)) + 1

            self.optimzers[i].zero_grad()
            loss.backward()
            self.optimzers[i].step()

        return loss.item()

    def valid_one_step(self, data, label):

        # Develop

        data = data.to(self.device)

        # s = [torch.exp(a) for a in scores]
        # s0 = [torch.sum(a, dim=0, keepdim=True) for a in s]
        # probs = [a / a0 for (a, a0) in zip(s, s0)]
        # ret = [-torch.sum(v * torch.log(v), dim=0) for v in probs]
        # entropy = torch.stack(ret).mean(0)
        # conf = torch.max(torch.stack(probs), dim=1).values

        with torch.no_grad():
            preds = [m(data) for m in self.models]
            preds_mu = torch.stack([p[0] for p in preds])
            preds_sig = torch.stack([p[1] for p in preds])
            preds_sig = torch.log(1 + torch.exp(preds_sig)) + 1e-6

        mu_final = preds_mu.mean(0)

        scores = torch.sqrt(preds_sig.mean(0) + (preds_mu.square().mean(0) - mu_final.square())).sum(1)

        return scores.detach().cpu().numpy()

    def validate(self, epoch):

        valid_loss_list = []
        in_score_list = []
        out_score_list = []

        for i, (data, label) in enumerate(self.test_in_loader):
            scores = self.valid_one_step(data, label)
            in_score_list.append(scores)

        for i, (data, label) in enumerate(self.test_out_loader):
            scores = self.valid_one_step(data, label)
            out_score_list.append(scores)

        in_scores = np.concatenate(in_score_list)
        out_scores = np.concatenate(out_score_list)

        labels_1 = torch.cat(
            [torch.zeros(in_scores.shape),
             torch.ones(out_scores.shape)]
        ).detach().cpu().numpy()

        labels_2 = torch.cat(
            [torch.ones(in_scores.shape),
             torch.zeros(out_scores.shape)]
        ).detach().cpu().numpy()

        scores = np.concatenate([in_scores, out_scores])

        index = np.isposinf(scores)
        scores[np.isposinf(scores)] = 1e9
        maximum = np.amax(scores)
        scores[np.isposinf(scores)] = maximum + 1

        index = np.isneginf(scores)
        scores[np.isneginf(scores)] = -1e9
        minimum = np.amin(scores)
        scores[np.isneginf(scores)] = minimum - 1

        scores[np.isnan(scores)] = 0

        def comp_aucs(scores, labels_1, labels_2):

            auroc_1 = roc_auc_score(labels_1, scores)
            auroc_2 = roc_auc_score(labels_2, scores)
            auroc = max(auroc_1, auroc_2)

            precision, recall, thresholds = precision_recall_curve(labels_1, scores)
            aupr_1 = auc(recall, precision)

            precision, recall, thresholds = precision_recall_curve(labels_2, scores)
            aupr_2 = auc(recall, precision)

            aupr = max(aupr_1, aupr_2)

            return auroc, aupr, precision, recall

        auroc, aupr, precision, recall = comp_aucs(scores, labels_1, labels_2)

        return auroc, aupr, precision, recall

    def train(self) -> None:
        print(f"Start training Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):
                data = data.to(self.device)

                res = self.train_one_step(data, label)

                training_loss_list.append(res)

                labels.append(label.detach().cpu().numpy())

            train_loss = np.mean(training_loss_list)

            valid_auc, valid_aupr, precision, recall = self.validate(epoch)

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f}'.format(
                    epoch, train_loss, valid_auc, valid_aupr))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Validation AUPR": valid_aupr,
                    "Validation AUC": valid_auc,
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.models[0].state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
