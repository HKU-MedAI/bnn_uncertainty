"""
Trainer of MC Dropout
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


class MCDTrainer(Trainer):
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

        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_classes = self.config_train["out_channels"]
        self.n_test_samples = self.config_train["n_testing_samples"]
        self.model = parse_frequentist_model(self.config_train, image_size=image_size).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

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

    def train_one_step(self, data, label, epoch):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model.mc_dropout(data, 0.2)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = F.nll_loss(log_outputs, label)
        loss.backward()

        self.optimzer.step()

        return loss.item()

    def valid_one_step(self, data, label):

        # Develop

        data = data.to(self.device)

        # Monte Carlo samples from different dropout mask at test time
        with torch.no_grad():
            scores = self.model(data)

        # Compute embeddings
        with torch.no_grad():
            scores = [self.model.mc_dropout(data, 0.2) for _ in range(self.n_test_samples)]
            scores = torch.stack(scores)

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

        labels_1 = torch.cat(
            [torch.ones(in_scores.shape),
             torch.zeros(out_scores.shape)]
        ).detach().cpu().numpy()
        labels_2 = torch.cat(
            [torch.zeros(in_scores.shape),
             torch.ones(out_scores.shape)]
        ).detach().cpu().numpy()

        scores = torch.cat([in_scores, out_scores]).detach().cpu().numpy()

        scores = self.format_scores(scores)

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
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                res = self.train_one_step(data, label, epoch)

                training_loss_list.append(res)

                labels.append(label.detach().cpu().numpy())

            train_loss = np.mean(training_loss_list)

            # OOD Detection
            valid_auc, valid_aupr, precision, recall = self.validate(epoch)
            epoch_stats.update({
                "Train Loss": train_loss,
                "Validation AUPR": valid_aupr,
                "Validation AUC": valid_auc,
            })

            # Classification
            classf_metrics = self.test_classification()
            epoch_stats.update(classf_metrics)

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f}'.format(
                    epoch, train_loss, valid_auc, valid_aupr))

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
