from typing import Callable, Union

import torch

from ignite.metrics import EpochMetric
import pytest
import sklearn
import torch
from sklearn.metrics import average_precision_score


def mse_mae_fn(y_pred, y_true):
    mse = torch.mean((y_pred - y_true).float() ** 2)
    mae = torch.mean((y_pred - y_true).float().abs())
    return mse, mae

#### Inheriting EpochMetric ####
class Msemae(EpochMetric):

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Msemae, self).__init__(
            mse_mae_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )
