# python test_accuracy.py

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

def _test_distrib_accuracy(self):
    rank = dist.get_rank()
    torch.manual_seed(12)

    y_pred_all, y_true_all = (torch.randint(0, 10, size=(40,)), torch.randint(0, 10, size=(40,)))  # 4 x 10 items for all y_pred and all y_true
    y_pred, y_true = y_pred_all[rank*10:(rank+1)*10], y_true_all[rank*10:(rank+1)*10]  # seperate all y_pred and all y_true to y_pred and y_tre
    
    acc = Accuracy()    # initialize the Accuracy class
    acc.reset()         # reset acc
    acc.update(y_pred, y_true)  # count the number of correct items and whole items from each process
    res = acc.compute().cpu().detach().numpy() # compute the accuracy from sum of correct items and whole items of all processes

    assert accuracy_score(y_pred_all, y_true_all)== pytest.approx(res)  # check with reference value

if __name__ == '__main__':
    backend = "gloo"
    device = idist.device()
    with idist.Parallel(backend=backend, nproc_per_node=4) as parallel:
        parallel.run(_test_distrib_accuracy)
