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
        try:
            test_in_name = self.config_data["test_in"]
        except KeyError:
            test_in_name = in_data_name

        in_channel = self.config_train["in_channels"]
        self.average = "binary" if self.config_train["out_channels"] == 2 else "weighted"

        train_in = load_data(in_data_name, True, image_size, in_channel)
        test_in = load_data(test_in_name, False, image_size, in_channel)
        test_out = load_data(ood_data_name, False, image_size, in_channel)

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

        auroc, aupr, precision, recall = self.comp_aucs_ood(scores, labels_1, labels_2)

        return auroc, aupr, precision, recall

    def test_classification(self):
        labels_list = []
        scores_list = []
        for i, (data, label) in enumerate(self.test_in_loader):
            data = data.to(self.device)
            with torch.no_grad():
                scores = self.model(data)
            labels_list.append(label)
            scores_list.append(scores)

        labels = torch.cat(labels_list).detach().cpu()
        scores = torch.cat(scores_list).detach().cpu().softmax(1)

        precision, recall, f1, classf_auroc = utils.metrics(scores, labels, self.average)
        clasff_accuracy = utils.acc(scores, labels)

        return {
            "classf_auroc": classf_auroc,
            "classf_acc": clasff_accuracy,
            "classf_f1": f1
        }

    def train(self) -> None:
        print(f"Start training Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            epoch_stats = {"Epoch": epoch + 1}
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

            # OOD Detection
            valid_auc, valid_aupr, precision, recall = self.validate(epoch)
            epoch_stats.update({
                "Train Loss": train_loss,
                "Validation AUPR": valid_aupr,
                "Validation AUC": valid_auc,
            })

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_auc, valid_aupr))

            # Classification
            classf_metrics = self.test_classification()
            epoch_stats.update(classf_metrics)

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
