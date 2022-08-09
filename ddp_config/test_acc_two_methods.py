#python test_acc_two_methods.py

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

def _test_distrib_accuracy_with_slicing_batch(local_rank):
    rank = 0
    ws = 1

    n = 3
    batch_size = 10

    torch.manual_seed(12)

    if torch.distributed.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()

    batch_size = 10
    y_pred_all, y_true_all = (
        torch.randint(0, batch_size, size=(batch_size * ws * n,)),
        torch.randint(0, batch_size, size=(batch_size * ws * n,)),
    )

    acc = Accuracy()  # initialize the Accuracy class
    acc.reset()  # reset acc

    # count the number of correct items and whole items from each process
    for i in range(n):
        y_pred = y_pred_all[
            (i + rank * n) * batch_size : (i + rank * n + 1) * batch_size
        ]
        y_true = y_true_all[
            (i + rank * n) * batch_size : (i + rank * n + 1) * batch_size
        ]
        acc.update(y_pred, y_true)

    # compute the accuracy from sum of correct items and whole items of all processes
    res = acc.compute()

    acc1_num_correct, acc1_num_examples = acc._num_correct, acc._num_examples
    res2 = acc.compute()
    acc2_num_correct, acc2_num_examples = acc._num_correct, acc._num_examples

    acc.reset()  # reset acc
    assert acc._num_correct == 0
    assert acc._num_examples == 0

    # check with reference value
    assert accuracy_score(y_true_all, y_pred_all) == pytest.approx(res)

    assert res == res2
    assert acc1_num_correct == acc2_num_correct
    assert acc1_num_examples == acc2_num_examples


def _test_distrib_accuracy_with_different_seed_per_rank(local_rank):
    rank = 0
    ws = 1

    n = 3
    batch_size = 10

    if torch.distributed.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()

    batch_size = 10
    y_pred_all, y_true_all = [
        torch.Tensor(np.array(batch_size * n)) for i in range(ws)
    ], [torch.Tensor(np.array(batch_size * n)) for i in range(ws)]

    acc = Accuracy()  # initialize the Accuracy class
    acc.reset()  # reset acc

    # count the number of correct items and whole items from each process
    y_pred_rank, y_true_rank = [], []
    for i in range(n):
        torch.manual_seed(12 + rank + i)
        y_pred = torch.randint(0, batch_size, size=(batch_size,))
        y_true = torch.randint(0, batch_size, size=(batch_size,))  # boundary
        y_pred_rank.extend(y_pred.tolist())
        y_true_rank.extend(y_true.tolist())
        acc.update(y_pred, y_true)

    tmp_rank = torch.tensor(y_pred_rank, dtype=torch.float32)
    tmp_true = torch.tensor(y_true_rank, dtype=torch.float32)

    dist.all_gather(y_pred_all, tmp_rank)
    dist.all_gather(y_true_all, tmp_true)

    y_pred_all = [it.item() for item in y_pred_all for it in item]
    y_true_all = [it.item() for item in y_true_all for it in item]

    res = acc.compute()

    acc1_num_correct, acc1_num_examples = acc._num_correct, acc._num_examples
    res2 = acc.compute()
    acc2_num_correct, acc2_num_examples = acc._num_correct, acc._num_examples

    acc.reset()  # reset acc
    assert acc._num_correct == 0
    assert acc._num_examples == 0

    # check with reference value
    assert accuracy_score(y_true_all, y_pred_all) == pytest.approx(res)

    assert res == res2
    assert acc1_num_correct == acc2_num_correct
    assert acc1_num_examples == acc2_num_examples

if __name__ == "__main__":
    backend = "gloo"
    device = idist.device()
    # with idist.Parallel(backend=None) as parallel:
    #     parallel.run(_test_distrib_accuracy_1)

    # with idist.Parallel(
    #     backend=backend, nproc_per_node=4, init_method="file:///c:/tmp/sharedfile"
    # ) as parallel:
    #     parallel.run(_test_distrib_accuracy_1)

    with idist.Parallel(
        backend=backend, nproc_per_node=4
    ) as parallel:
        parallel.run(_test_distrib_accuracy_with_different_seed_per_rank)
