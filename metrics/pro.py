import os
# import mlflow
import numpy as np
import pandas as pd
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc


def compute_pro(category: str, masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    
    usage:
        pro_auc = compute_pro(class_name, np.squeeze(gt_mask, axis=1), scores)
    """


    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, f"set(masks.flatten()) must be {{0, 1}}, got {set(masks.flatten())}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    try:
        np.arange(min_th, max_th, delta)
    except ValueError:
        return 0

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    if not os.path.exists("pro_result"):
        os.mkdir("pro_result")
    df.to_csv(f"pro_result/{category}_pro_curve.csv", index=False)

    try:
        pro_auc = auc(df["fpr"], df["pro"])
        return pro_auc
    except ValueError:
        return 0

    # Logging pro_auc to mlflow server
    # mlflow.log_metric("pro_auc", value=pro_auc)

    # Logging pro_curve to mlflow server
    # TODO(inoue): step in log_metric only accept int, so fpr is multiplied by 100 and rounded.
    # for fpr, pro in zip(df["fpr"], df["pro"]):
        # mlflow.log_metric("pro_curve", value=round(pro * 100), step=round(fpr * 100))