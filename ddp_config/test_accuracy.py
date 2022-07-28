import os
import pytest
import torch

import ignite.distributed as idist
from sklearn.metrics import accuracy_score
import time
import numpy as np
import random
from accuracy import Accuracy
import torch.distributed as dist


def _test_distrib_accuracy(local_rank):
    rank = 0
    ws = 1

    torch.manual_seed(12)

    if torch.distributed.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()

    y_pred_all, y_true_all = (
        torch.randint(0, 10, size=(10 * ws * 2,)),
        torch.randint(0, 10, size=(10 * ws * 2,)),
    ) 

    y_pred, y_true = (
        y_pred_all[rank * 2 * 10 : (rank * 2 + 1) * 10],
        y_true_all[rank * 2 * 10 : (rank * 2 + 1) * 10],
    )  # seperate all y_pred and all y_true to y_pred and y_tre

    y_pred2, y_true2 = (
        y_pred_all[(rank * 2 + 1) * 10 : (rank * 2 + 2) * 10],
        y_true_all[(rank * 2 + 1) * 10 : (rank * 2 + 2) * 10],
    )  # seperate all y_pred and all y_true to y_pred and y_tre

    acc = Accuracy()  # initialize the Accuracy class
    acc.reset()  # reset acc
    acc.update(
        y_pred, y_true
    )  # count the number of correct items and whole items from each process
    acc.update(
        y_pred2, y_true2
    )  # count the number of correct items and whole items from each process
    res = (
        acc.compute()
    )  # compute the accuracy from sum of correct items and whole items of all processes

    acc1_num_correct, acc1_num_examples = acc._num_correct, acc._num_examples
    res2 = acc.compute()
    acc2_num_correct, acc2_num_examples = acc._num_correct, acc._num_examples

    assert accuracy_score(y_true_all, y_pred_all) == pytest.approx(
        res
    )  # check with reference value
    assert res == res2
    assert acc1_num_correct == acc2_num_correct
    assert acc1_num_examples == acc2_num_examples


if __name__ == "__main__":
    backend = "gloo"
    device = idist.device()
    with idist.Parallel(backend=None) as parallel:
        parallel.run(_test_distrib_accuracy)

    with idist.Parallel(
        backend=backend, nproc_per_node=4, init_method="file:///c:/tmp/sharedfile"
    ) as parallel:
        parallel.run(_test_distrib_accuracy)
