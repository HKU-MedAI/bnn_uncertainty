"""
Tesing the linear layer of R2D2 BNN
"""

from layers import R2D2LinearLayer
import torch

def main():
    input_ = torch.ones((32, 256))

    config_train = {
        "r2d2_scale": None,
        "prior_phi_prob": 0.6,
        "prior_psi_shape": 0.5,
        "beta_rho_scale": -5,
        "bias_rho_scale": -5,
        "weight_xi_shape": 0.3,
        "weight_omega_shape": 0.3,
    }

    parameters = {
        "r2d2_scale": config_train["r2d2_scale"],
        "prior_phi_prob": config_train["prior_phi_prob"],
        "prior_psi_shape": config_train["prior_psi_shape"],
        "beta_rho_scale": config_train["beta_rho_scale"],
        "bias_rho_scale": config_train["bias_rho_scale"],
        "weight_xi_shape": config_train["weight_xi_shape"],
        "weight_omega_shape": config_train["weight_omega_shape"],
    }
    layer = R2D2LinearLayer(256, 2, parameters)
    out = layer(input_)
    layer.analytic_update()
    print(out)

if __name__ == "__main__":
    main()