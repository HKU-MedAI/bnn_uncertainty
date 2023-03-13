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
    parse_frequentist_model
)
from losses import EdlLoss

import torchvision
import torch

from data import load_data
from utils import one_hot_embedding

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc


class EDLTrainer(Trainer):
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
        #
        # train_all = train_in
        # train_all.data = torch.cat((train_in.data, train_out.data))
        # train_all.targets = torch.cat((train_in.targets, train_out.targets))

        # self.train_all_loader = DataLoader(train_all, batch_size=self.batch_size, shuffle=True)
        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_classes = self.config_train["out_channels"]
        self.model = parse_frequentist_model(self.config_train, image_size=image_size).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.edl_loss = EdlLoss(self.device)

    def get_ood_label_score(self, test_in_score, test_out_score):
        score = np.concatenate([test_in_score, test_out_score])
        label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
        return label, score

    def train_one_step(self, data, label, epoch):
        self.optimzer.zero_grad()
        data = data.permute(0, 2, 1, 3)

        label_oh = one_hot_embedding(label, self.n_classes)

        pred = self.model(data)

        loss = self.edl_loss.edl_digamma_loss(
            pred, label_oh, epoch, self.n_classes, 10
        )

        loss.backward()

        self.optimzer.step()

        acc = utils.acc(pred.data, label)

        return loss.item(), acc, pred

    def valid_one_step(self, data, label):

        data = data.to(self.device)

        with torch.no_grad():
            scores = self.model(data)
            scores = F.relu(scores)
            scores += 1
            scores = scores.shape[1] / torch.sum(scores, dim=1, keepdim=True)

        scores = scores.squeeze()

        return scores

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

        labels_1 = torch.cat(
            [torch.ones(in_scores.shape),
             torch.zeros(out_scores.shape)]
        ).detach().cpu().numpy()
        labels_2 = torch.cat(
            [torch.zeros(in_scores.shape),
             torch.ones(out_scores.shape)]
        ).detach().cpu().numpy()

        scores = torch.cat([in_scores, out_scores]).detach().cpu().numpy()

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
            acc_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):
                data = data.to(self.device)
                data = data.permute(0, 3, 1, 2)
                label = label.to(self.device)

                res, acc, log_outputs = self.train_one_step(data, label, epoch)

                training_loss_list.append(res)
                acc_list.append(acc)

                probs.append(log_outputs.softmax(1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            train_loss, train_acc = np.mean(training_loss_list), np.mean(acc_list)

            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average="weighted")

            valid_auc, valid_aupr, precision, recall = self.validate(epoch)

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_auc, valid_aupr))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
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
