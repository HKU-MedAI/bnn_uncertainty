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

from statistics_np import AdaptableRHT

from data import load_data

import torch

import wandb


class BNNARHTTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()
        self.compare_metrics = True

        in_data_name = self.config_data["in"]
        ood_data_name = self.config_data["ood"]
        try:
            test_in_name = self.config_data["test_in"]
        except KeyError:
            test_in_name = in_data_name

        image_size = self.config_data["image_size"]
        in_channel = self.config_train["in_channels"]

        self.average = "binary" if self.config_train["out_channels"] == 2 else "weighted"

        train_in = load_data(in_data_name, True, image_size, in_channel)
        test_in = load_data(test_in_name, False, image_size, in_channel)
        train_out = load_data(ood_data_name, True, image_size, in_channel)
        test_out = load_data(ood_data_name, False, image_size, in_channel)

        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        # self.train_all_loader = DataLoader(train_all, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_noraml_samples = self.config_train["n_normal_samples"]
        self.n_test_samples = self.config_train["n_testing_samples"]
        self.model = parse_bayesian_model(self.config_train, image_size=image_size)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

        self.task = self.config_train["task"]

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        pred = self.model(data)

        if pred.dim() > 2:
            pred = pred.mean(0)

        kl_loss = self.model.kl_loss()
        ce_loss = F.cross_entropy(pred, label, reduction='mean')

        loss = ce_loss + kl_loss.item() * self.beta
        # loss = ce_loss
        loss.backward()

        self.optimzer.step()

        acc = utils.acc(pred.data, label)

        return loss.item(), kl_loss.item(), ce_loss.item(), acc, pred

    def compute_normal_posterior(self, data):

        # Develop

        data = data.to(self.device)

        # Monte Carlo samples from different dropout mask at test time
        with torch.no_grad():
            scores = [self.model(data, get_emb=True) for _ in range(self.n_noraml_samples)]
            scores = torch.cat(scores)

        n_1 = scores.shape[0]
        s = scores.mean(0)
        diffs = scores - scores.mean(0)
        cov = torch.matmul(diffs.T, diffs)

        return s, cov, n_1

    def tune_lambda(self, tester, lambdas, priors):
        q_values = [tester.Q_function(lamb, priors) for lamb in lambdas]
        indx = np.argmax(q_values)
        return lambdas[indx]

    def get_embs(self, data):
        data = data.to(self.device)

        # Compute embeddings
        with torch.no_grad():
            scores = [self.model(data, get_emb=True) for _ in range(self.n_test_samples)]
            scores = torch.stack(scores)

        return scores

    def compute_p_values(self, mean_normal, cov_normal, n_1, embs):
        """
        Compute the ARHT test statistics and p value
        """

        # Bring data to numpy
        scores = embs.detach().cpu().numpy()
        mean_normal = mean_normal.detach().cpu().numpy() / n_1
        cov_normal = cov_normal.detach().cpu().numpy()

        # Compute mean and variance of testing embeddings
        test_mu = scores.mean(0)
        test_cov = np.einsum("ikj, ikl -> kjl", (scores - test_mu), (scores - test_mu))

        lamb = self.config_train["init_lambda"]
        # lambdas = [lamb, lamb * 5, lamb * 10, lamb * 50]
        lambdas = [lamb, lamb * 5, lamb * 10]
        n_2 = scores.shape[0]
        n = n_1 + n_2
        p = cov_normal.shape[0]

        cov = (cov_normal + test_cov) / (n_1 + n_2)

        tester = AdaptableRHT(lambdas, cov, n, p)
        lamb = tester.find_optimal_Q()

        t_stat = tester.adaptive_rht(lamb, mean_normal, test_mu, n_1, n_2, p)
        rht = tester.rht(lamb, mean_normal, test_mu, n_1, n_2)
        # pvalues = min(stats.norm.cdf(t_stat), 1 - stats.norm.cdf(t_stat))

        return t_stat
        # return t_stat, rht

    def validate(self, epoch):

        valid_loss_list = []
        in_score_list = []
        out_score_list = []

        # Compute in-distribution mean and covariance
        cov_normal = 0
        sum_normal = 0
        n_1 = 0
        for i, (data, label) in enumerate(self.train_in_loader):
            s, cov, n = self.compute_normal_posterior(data)
            cov_normal += cov
            sum_normal += s
            n_1 += n

        # Compute p values
        for i, (data, label) in enumerate(self.test_in_loader):
            embs = self.get_embs(data)
            scores = self.compute_p_values(sum_normal, cov_normal, n_1, embs)
            in_score_list.append(scores)

        for i, (data, label) in enumerate(self.test_out_loader):
            embs = self.get_embs(data)
            scores = self.compute_p_values(sum_normal, cov_normal, n_1, embs)
            out_score_list.append(scores)

        # Perform BH correction

        in_scores = np.concatenate(in_score_list)
        out_scores = np.concatenate(out_score_list)

        # Visualize the scores
        self.visualize_scores(in_scores, out_scores, epoch)

        labels_1 = torch.cat(
            [torch.zeros(in_scores.shape),
             torch.ones(out_scores.shape)]
        ).detach().cpu().numpy()

        labels_2 = torch.cat(
            [torch.ones(in_scores.shape),
             torch.zeros(out_scores.shape)]
        ).detach().cpu().numpy()

        scores = np.concatenate([in_scores, out_scores])
        scores = self.format_scores(scores)

        # Other scores

        auroc, aupr, precision, recall = self.comp_aucs_ood(scores, labels_1, labels_2)

        return auroc, aupr, precision, recall

    def test_classification(self):
        labels_list = []
        scores_list = []
        for i, (data, label) in enumerate(self.test_in_loader):
            data = data.to(self.device)
            with torch.no_grad():
                scores = [self.model(data) for _ in range(200)]
                scores = torch.stack(scores).mean(0)
            labels_list.append(label)
            scores_list.append(scores)

        labels = torch.cat(labels_list).detach().cpu()
        scores = torch.cat(scores_list).detach().cpu().softmax(1)

        # if self.average == "binary":
        #     fpr, tpr, thresholds = roc_curve(labels, scores)
        #     clssf_aucroc = auc(fpr, tpr)
        # else:
        #     classf_auroc = roc_auc_score(labels, scores, multi_class="ovo")
        precision, recall, f1, classf_auroc = utils.metrics(scores, labels, self.average)
        clasff_accuracy = utils.acc(scores, labels)

        return {
            "classf_auroc": classf_auroc,
            "classf_acc": clasff_accuracy,
            "classf_f1": f1
        }

    def train(self) -> None:
        print(f"Start training ARHT Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            epoch_stats = {"Epoch": epoch + 1}
            training_loss_list = []
            kl_list = []
            nll_list = []
            acc_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):
                data = data.to(self.device)
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
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average=self.average)

            if self.task == "ood":
                valid_auc, valid_aupr, precision, recall = self.validate(epoch)
                epoch_stats.update({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Train AUC": train_aucroc,
                    "Validation AUPR": valid_aupr,
                    "Validation AUC": valid_auc,
                })

                training_range.set_description(
                    'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation AUC: {:.4f} \tValidation AUPR: {:.4f} \tTrain_kl_div: {:.4f} \tTrain_nll: {:.4f}'.format(
                        epoch, train_loss, train_acc, valid_auc, valid_aupr, train_kl, train_nll))
            elif self.task == "classf":
                # Classification
                classf_metrics = self.test_classification()
                epoch_stats.update(classf_metrics)

                training_range.set_description(
                    'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation ACC: {:.4f} \tValidation AUC: {:.4f}'.format(
                        epoch, train_loss, train_acc, classf_metrics["classf_acc"], classf_metrics["classf_auroc"]))

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

            self.logging(epoch_stats)

    def get_ood_label_score(self, test_in_score, test_out_score):
        score = np.concatenate([test_in_score, test_out_score])
        label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
        return label, score

    def compute_entropy(self, alphas, alpha0):
        probs = alphas / alpha0
        entropy = -torch.sum(probs*torch.log(probs), dim=1)
        conf = torch.max(probs, dim=1).values

        return entropy, conf

    def compute_diff_entropy(self, alphas, alpha0):
        return torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1) - torch.lgamma(alpha0)

    def logging(self, epoch_stats):
        wandb.log(epoch_stats)

