from torch import optim, nn
import torch.nn.functional as F

from models import (
    BBB3Conv3FC,
    BBBMultipleLinear,
    BBBLeNet,
    BBBAlexNet,
    BBBResNet,
    ResNet,
    CNN,
    MultipleLinear
)
from models.frequentists import LeNet, EfficientNetB4, AlexNet
from torchvision.models import resnet18, resnet50

from losses import ELBO


def parse_optimizer(config_optim, params):
    opt_method = config_optim["opt_method"].lower()
    alpha = config_optim["lr"]
    weight_decay = config_optim["weight_decay"]
    if opt_method == "adagrad":
        optimizer = optim.Adagrad(
            # model.parameters(),
            params,
            lr=alpha,
            lr_decay=weight_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "adadelta":
        optimizer = optim.Adadelta(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "adam":
        optimizer = optim.Adam(
            # model.parameters(),
            params,
            lr=alpha,
            # weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    return optimizer


def parse_loss(config_train):
    loss_name = config_train["loss"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    elif loss_name == "NLL":
        return nn.NLLLoss()
    elif loss_name == "ELBO":
        train_size = config_train["train_size"]
        return ELBO(train_size)
    elif loss_name == "cosine":
        return nn.CosineSimilarity(dim=-1)
    else:
        raise NotImplementedError("This Loss is not implemented")


def parse_bayesian_model(config_train, image_size=32):
    # Read input and output dimension
    in_dim = config_train["in_channels"]
    out_dim = config_train["out_channels"]

    model_name = config_train["model_name"]

    # Check deep ensembles
    try:
        is_de = config_train["is_de"]
    except KeyError:
        is_de = False

    try:
        p = config_train["emb_dim"]
    except KeyError:
        p = 84

    # Read priors for BNNs
    if model_name in ["BCNN", "BLeNet", "BAlexNet"]:
        priors = {
            'prior_mu': config_train["prior_mu"],
            'prior_sigma': config_train["prior_sigma"],
            'posterior_mu_initial': config_train["posterior_mu_initial"],
            'posterior_rho_initial': config_train["posterior_rho_initial"],
        }
    elif model_name in ["HorseshoeLeNet", "BHorseshoeAlexNet", "HorseshoeCNN", "HorseshoeMLP"]:
        priors = {
            "horseshoe_scale": config_train["horseshoe_scale"],
            "global_cauchy_scale": config_train["global_cauchy_scale"],
            "weight_cauchy_scale": config_train["weight_cauchy_scale"],
            "beta_rho_scale": config_train["beta_rho_scale"],
            "log_tau_mean": config_train["log_tau_mean"],
            "log_tau_rho_scale": config_train["log_tau_rho_scale"],
            "bias_rho_scale": config_train["bias_rho_scale"],
            "log_v_mean": config_train["log_v_mean"],
            "log_v_rho_scale": config_train["log_v_rho_scale"]
        }
    elif model_name in ["R2D2AlexNet", "R2D2LeNet", "R2D2CNN", "R2D2MLP"]:
        priors = {
            "r2d2_scale": config_train["r2d2_scale"],
            "prior_phi_prob": config_train["prior_phi_prob"],
            "prior_psi_shape": config_train["prior_psi_shape"],
            "beta_rho_scale": config_train["beta_rho_scale"],
            "bias_rho_scale": config_train["bias_rho_scale"],
            "weight_xi_shape": config_train["weight_xi_shape"],
            "weight_omega_shape": config_train["weight_omega_shape"],
        }
    else:
        priors = None

    if model_name == "BCNN":
        return BBB3Conv3FC(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            get_sig=is_de
        )
    elif model_name == "BMLP":
        n_blocks = config_train["n_blocks"]
        return BBBMultipleLinear(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "BLeNet":
        return BBBLeNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            image_size=image_size,
            de=is_de,
            p=p
        )
    elif model_name == "BAlexNet":
        model = BBBAlexNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            image_size=image_size,
            de=is_de
        )
        return model
    elif model_name == "BResNet":
        return BBBResNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            get_sig=is_de
        )
    else:
        raise NotImplementedError("This Model is not implemented")


def parse_frequentist_model(config_freq, image_size=32):
    # Read input and output dimension
    in_dim = config_freq["in_channels"]
    out_dim = config_freq["out_channels"]
    model_name = config_freq["model_name"]

    # Check deep ensembles
    try:
        is_de = config_freq["is_de"]
    except KeyError:
        is_de = False

    if model_name == "EfficientNet":
        return EfficientNetB4(
            inputs=in_dim,
            outputs=out_dim
        )
    elif model_name == "AlexNet":
        return AlexNet(
            inputs=in_dim,
            outputs=out_dim,
            get_sig=is_de
        )
    elif model_name == "LeNet":
        return LeNet(
            outputs=out_dim,
            inputs=in_dim,
            image_size=image_size,
            de=is_de
        )
    elif model_name == "ResNet":
        return ResNet(
            outputs=out_dim,
            inputs=in_dim,
            get_sig=is_de
        )
    elif model_name == "CNN":
        return CNN(
            outputs=out_dim,
            inputs=in_dim,
            get_sig=is_de
        )
    elif model_name == "MLP":
        n_blocks = config_freq["n_blocks"]
        return MultipleLinear(
            outputs=out_dim,
            inputs=in_dim,
            n_blocks=n_blocks
        )
    else:
        raise NotImplementedError("This Loss is not implemented")
