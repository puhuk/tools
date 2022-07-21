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


#### Overriding EpochMetric ####
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

    
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("PrecisionRecallCurve must have at least one example before it can be computed.")

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        ws = idist.get_world_size()
        if ws > 1 and not self._is_reduced:
            # All gather across all processes
            _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
            _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))
        self._is_reduced = True

        ###################################################################
        # precision, recall, thresholds = 0.0, 0.0, 0.0
        # if idist.get_rank() == 0:
        #     # Run compute_fn on zero rank only
        #     precision, recall, thresholds = self.compute_fn(_prediction_tensor, _target_tensor)

        # if ws > 1:
        #     # broadcast result to all processes
        #     precision = cast(float, idist.broadcast(precision, src=0))
        #     recall = cast(float, idist.broadcast(recall, src=0))
        #     thresholds = cast(float, idist.broadcast(thresholds, src=0))

        # return precision, recall, thresholds

        ###################################################################

        if idist.get_rank() == 0:
            # Run compute_fn on zero rank only
            mse, mae = self.compute_fn(_prediction_tensor, _target_tensor)
            mse = torch.tensor(mse)
            mae = torch.tensor(mae)
        else:
            mse, mae = None, None

        if ws > 1:
            # broadcast result to all processes
            mse = idist.broadcast(mse, src=0, safe_mode=True)
            mae = idist.broadcast(mae, src=0, safe_mode=True)

        return mse, mae
