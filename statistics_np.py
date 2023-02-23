from math import sqrt

import numpy as np
from numpy import linalg

import logging
logger = logging.getLogger(__name__)


"""
ref: https://arxiv.org/pdf/1609.08725.pdf
"""


class AdaptableRHT:
    def __init__(self, lambdas, cov, n, p):
        self.lambdas = lambdas
        self.lambda_dict = {lamb: i for i, lamb in enumerate(lambdas)}
        self.n = n
        self.p = p
        self.gamma = p / n

        self.cov = cov
        self.priors = [0, 1, 0]

        logger.debug("Computing regularized inverse covariance matrices")
        self.reg_inv_covs = [
            self.regularized_inverse(-lamb, cov)
            for lamb in self.lambdas
        ]
        logger.debug(f"Inverse shape {self.reg_inv_covs[0].shape}")

        logger.debug("Computing first derivatives")
        self.mz_list = [
            self.weighted_trace(self.reg_inv_covs[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]
        logger.debug(f"mz shape {self.mz_list[0].shape}")
        self.theta_1 = [
            self.get_theta_1(lamb, self.mz_list[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]
        self.theta_2 = [
            self.get_theta_2(lamb,
                             self.reg_inv_covs[self.lambda_dict[lamb]],
                             self.mz_list[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]
        logger.debug("Computing Q values")

    def regularized_inverse(self, z, cov):
        reg_cov = cov - z * np.eye(self.p)
        return linalg.inv(reg_cov)

    def weighted_trace(self, reg_inv_cov):
        m_z = np.trace(reg_inv_cov, axis1=1, axis2=2) / self.p
        return m_z

    def get_theta_1(self, lamb, m_z):
        a = 1 - lamb * m_z
        return a / (1 - self.gamma * a)

    def get_theta_2(self, lamb, reg_inv_cov, m_z):
        """

        :param lamb:
        :param reg_inv_cov: Shape (width*height, channel, channel)
        :param m_z: (width*height, channel, channel)
        :return:
        """
        a = 1 - lamb * m_z
        reg_inv_sq = reg_inv_cov @ reg_inv_cov
        logger.debug(f"Shape of square matrix {reg_inv_sq.shape}")
        m_prime_z = np.trace(reg_inv_sq, axis1=1, axis2=2) / self.p
        th = a / (1 - self.gamma * a) ** 3 - lamb * (m_z - lamb * m_prime_z) / (1 - self.gamma * a) ** 4

        th[th <= 0] = -th[th <= 0] + 10 ** -10

        return th

    def rht(self, lamb, mu_1, mu_2, n_1, n_2):
        reg_inv = self.reg_inv_covs[self.lambda_dict[lamb]]
        batch = mu_2.shape[0]

        diff = np.repeat(mu_1[None, ...], batch, axis=0) - mu_2
        temp = np.einsum('ij,ijj->ij', diff, reg_inv)
        temp = np.einsum('ij,ij->i', temp, temp)
        rht = n_1 * n_2 / (n_1 + n_2) * temp

        return rht

    def adaptive_rht(self, lamb, mu_1, mu_2, n_1, n_2, p):
        logger.debug(f"Shape of theta 1 {self.theta_1[self.lambda_dict[lamb]].shape}")
        t_stat = np.sqrt(p) * (self.rht(lamb, mu_1, mu_2, n_1, n_2)
                            / p - self.theta_1[self.lambda_dict[lamb]]) / np.sqrt(2 * self.theta_2[self.lambda_dict[lamb]])

        return t_stat

    def Q_function(self, lamb):
        """
        Q function for tuning lambda from data
        :param lamb:
        :param cov:
        :param n:
        :param p:
        :param priors: prior weights in list of 3
        :return:
        """
        phi = np.trace(self.cov, axis1=1, axis2=2) / self.p

        rho_1 = self.mz_list[self.lambda_dict[lamb]]
        rho_2 = self.theta_1[self.lambda_dict[lamb]]
        th_1 = rho_2
        rho_3 = (1 + self.gamma * th_1) * (phi - lamb * rho_1)

        Q = [pri * rho / np.sqrt(self.gamma * self.theta_2[self.lambda_dict[lamb]])
             for pri, rho in zip(self.priors, (rho_1, rho_2, rho_3))]
        Q = sum(Q).sum()

        return Q

    def find_optimal_Q(self):
        q_values = [self.Q_function(lamb) for lamb in self.lambdas]
        index = np.argmax(q_values)
        logger.debug(f"Q values are {q_values}")

        return self.lambdas[index]
