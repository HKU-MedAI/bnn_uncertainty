"""
Trainer of DPN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from .trainer import Trainer

from parse import (
    parse_optimizer,
    parse_frequentist_model
)
from losses import EdlLoss

import torch

from data import load_data

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

import torch.distributions as dist
from torch.distributions.dirichlet import Dirichlet


class PostNetTrainer(Trainer):
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

        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_classes = self.config_train["out_channels"]
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

        # predict alpha
        target_a = self.target_alpha(label)
        target_a = target_a.to(self.device)
        output_alpha = torch.exp(self.model(data))
        dirichlet1 = Dirichlet(output_alpha)
        dirichlet2 = Dirichlet(target_a)

        loss = torch.sum(dist.kl.kl_divergence(dirichlet1, dirichlet2))

        loss.backward()

        self.optimzer.step()

        return loss.item()

    def valid_one_step(self, data, label):

        # Develop

        data = data.to(self.device)

        # Monte Carlo samples from different dropout mask at test time
        with torch.no_grad():
            scores = self.model(data)

        # s = [torch.exp(a) for a in scores]
        # s0 = [torch.sum(a, dim=0, keepdim=True) for a in s]
        # probs = [a / a0 for (a, a0) in zip(s, s0)]
        # ret = [-torch.sum(v * torch.log(v), dim=0) for v in probs]
        # entropy = torch.stack(ret).mean(0)
        # conf = torch.max(torch.stack(probs), dim=1).values

        alphas = torch.exp(self.model(data))
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        entropy = -torch.sum(probs*torch.log(probs), dim=1)
        conf = torch.max(probs, dim=1).values

        return entropy, conf

    def validate(self, epoch):

        valid_loss_list = []
        in_score_list_ent = []
        out_score_list_ent = []
        in_score_list_conf = []
        out_score_list_conf = []

        for i, (data, label) in enumerate(self.test_in_loader):
            in_scores_ent, in_scores_conf = self.valid_one_step(data, label)
            in_score_list_ent.append(in_scores_ent)
            in_score_list_conf.append(in_scores_conf)

        for i, (data, label) in enumerate(self.test_out_loader):
            out_scores_ent,  out_scores_conf = self.valid_one_step(data, label)
            out_score_list_ent.append(out_scores_ent)
            out_score_list_conf.append(out_scores_conf)

        in_scores_ent = torch.cat(in_score_list_ent)
        out_scores_ent = torch.cat(out_score_list_ent)
        in_scores_conf = torch.cat(in_score_list_conf)
        out_scores_conf = torch.cat(out_score_list_conf)

        labels_1 = torch.cat(
            [torch.ones(in_scores_ent.shape),
             torch.zeros(out_scores_ent.shape)]
        ).detach().cpu().numpy()
        labels_2 = torch.cat(
            [torch.zeros(in_scores_ent.shape),
             torch.ones(out_scores_ent.shape)]
        ).detach().cpu().numpy()

        ent_scores = torch.cat([in_scores_ent, out_scores_ent]).detach().cpu().numpy()
        conf_scores = torch.cat([in_scores_conf, out_scores_conf]).detach().cpu().numpy()

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

        ent_scores = format_scores(ent_scores)
        conf_scores = format_scores(conf_scores)

        def comp_aucs(scores, labels_1, labels_2):

            auroc_1 = roc_auc_score(labels_1, scores)
            auroc_2 = roc_auc_score(labels_2, scores)
            auroc = max(auroc_1, auroc_2)

            precision, recall, thresholds = precision_recall_curve(labels_1, scores)
            aupr_1 = auc(recall, precision)

            precision, recall, thresholds = precision_recall_curve(labels_2, scores)
            aupr_2 = auc(recall, precision)

            aupr = max(aupr_1, aupr_2)

            return auroc, aupr

        ent_auroc, ent_aupr = comp_aucs(ent_scores, labels_1, labels_2)
        conf_auroc, conf_aupr = comp_aucs(conf_scores, labels_1, labels_2)

        return ent_auroc, ent_aupr, conf_auroc, conf_aupr

    def train(self) -> None:
        print(f"Start training Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):
                data = data.to(self.device)

                res = self.train_one_step(data, label, epoch)

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
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
