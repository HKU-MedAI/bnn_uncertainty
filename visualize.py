from matplotlib import pyplot as plt
from pathlib import Path

from parse import parse_bayesian_model

import numpy as np

# Load trained model

checkpoint_root = Path("./checkpoints/")
r2d2_checkpoints = [
    checkpoint_root / f"R2D2MLP_L{i}"
    for i in range(4)
]
horseshoe_checkpoints = [
    checkpoint_root / f"HorseshoeMLP_L{i}"
    for i in range(4)
]
bnn_checkpoints = [
    checkpoint_root / f"BMLP_L{i}"
    for i in range(4)
]

layers = [3]

for l in layers:
    r2d2_preds = np.load(str(r2d2_checkpoints[l] / "pred.npy"))
    horseshoe_preds = np.load(str(horseshoe_checkpoints[l] / "pred.npy"))
    bnn_preds = np.load(str(bnn_checkpoints[l] / "pred.npy"))

    r2d2_labels = np.load(str(r2d2_checkpoints[l] / "label.npy"))
    horseshoe_labels = np.load(str(horseshoe_checkpoints[l] / "label.npy"))
    bnn_labels = np.load(str(bnn_checkpoints[l] / "label.npy"))

    fig, ax = plt.subplots()
    ax.plot(x[indices, 0], label[indices])
    ax.fill_between(x[indices, 0], lower[indices], upper[indices], color='b', alpha=.1)

    plt.xlim([-5, 5])
    plt.ylim([-200, 200])



# cm = CheckpointManager(config['checkpoints']["path"])
# model = parse_bayesian_model(config["train"])
#
# sd = cm.load_model()
# model.load_state_dict(sd)
#
# l1 = model.dense_block.fc0
# l2 = model.dense_block.fc1

print("Model loaded")

# Plot prediction variance -> Prediction confidence interval of the regression task

