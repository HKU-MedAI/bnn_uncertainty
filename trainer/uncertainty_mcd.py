"""
Trainer of BNN
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
    parse_bayesian_model
)

from data import load_data

import torchvision.transforms as transforms
import torchvision
import torch

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc


class BNNUncertaintyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        train_in = load_data("MNIST", True, self.batch_size)
        test_in = load_data("MNIST", False, self.batch_size)
        train_out = load_data("FashionMNIST", True, self.batch_size)
        test_out = load_data("FashionMNIST", False, self.batch_size)

        train_out.targets = torch.tensor(np.ones(len(train_out.targets)) * 10, dtype=torch.long)

        train_all = train_in = load_data("MNIST", True, self.batch_size)
        train_all.data = torch.cat((train_in.data, train_out.data))
        train_all.targets = torch.cat((train_in.targets, train_out.targets))

        self.train_all_loader = DataLoader(train_all, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_samples = self.config_train["n_samples"]
        self.model = parse_bayesian_model(self.config_train, image_size=28)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def get_ood_label_score(self, test_in_score, test_out_score):
        score = np.concatenate([test_in_score, test_out_score])
        label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
        return label, score

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()
        data = data.permute(0, 2, 1, 3)

        pred = self.model(data)

        if pred.dim() > 2:
            pred = pred.mean(0)

        kl_loss = self.model.kl_loss()
        ce_loss = F.cross_entropy(pred, label, reduction='mean')

        loss = ce_loss + kl_loss.item() * self.beta
        # loss = ce_loss
        loss.backward()

        self.optimzer.step()

        if hasattr(self.model, "analytic_update"):
            self.model.analytic_update()

        acc = utils.acc(pred.data, label)

        return loss.item(), kl_loss.item(), ce_loss.item(), acc, pred

    def valid_one_step(self, data, label):

        # Develop

        data = data.to(self.device)

        # Monte Carlo samples from different dropout mask at test time
        with torch.no_grad():
            scores = [self.model(data) for _ in range(self.n_samples)]
            if scores[0].dim() > 2 :
                scores = [s.mean(0) for s in scores]
        s = [torch.exp(a) for a in scores]
        s0 = [torch.sum(a, dim=1, keepdim=True) for a in s]
        probs = [a / a0 for (a, a0) in zip(s, s0)]
        ret = [-torch.sum(v * torch.log(v), dim=1) for v in probs]
        entropy = torch.stack(ret).mean(0)

        return entropy

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

        in_scores = torch.cat(in_score_list)
        out_scores = torch.cat(out_score_list)

        labels = torch.cat(
            [torch.ones(in_scores.shape),
             torch.zeros(out_scores.shape)]
        ).detach().cpu().numpy()

        scores = torch.cat([in_scores, out_scores]).detach().cpu().numpy()

        index = np.isposinf(scores)
        scores[np.isposinf(scores)] = 1e9
        maximum = np.amax(scores)
        scores[np.isposinf(scores)] = maximum + 1

        index = np.isneginf(scores)
        scores[np.isneginf(scores)] = -1e9
        minimum = np.amin(scores)
        scores[np.isneginf(scores)] = minimum - 1

        scores[np.isnan(scores)] = 0

        auroc = roc_auc_score(labels, scores)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        aupr = auc(recall, precision)

        return auroc, aupr, precision, recall

    def train(self) -> None:
        print(f"Start training Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            acc_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.train_all_loader):
                data = data.to(self.device)
                data = data.permute(0, 3, 1, 2)
                label = label.to(self.device)

                res, kl, nll, acc, log_outputs = self.train_one_step(data, label)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)
                acc_list.append(acc)

                probs.append(log_outputs.softmax(1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            train_loss, train_acc, train_kl, train_nll = np.mean(training_loss_list), np.mean(acc_list), np.mean(
                kl_list), np.mean(nll_list)

            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average="weighted")

            valid_auc, valid_aupr, precision, recall = self.validate(epoch)

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f} \tTrain_kl_div: {:.4f} \tTrain_nll: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_auc, valid_aupr, train_kl, train_nll))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train NLL Loss": train_nll,
                    "Train KL Loss": train_kl,
                    "Train Accuracy": train_acc,
                    "Train F1": train_f1,
                    "Train AUC": train_aucroc,
                    "Validation AUPR": valid_aupr,
                    "Validation AUC": valid_auc,
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
