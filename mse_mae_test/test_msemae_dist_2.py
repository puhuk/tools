# python test_msemae_dist_2.py
import os
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from msemae import Msemae
import time
import numpy as np
import random
from multiprocessing import Pool

def pprint(rank, msg):
    # We add sleep to avoid printing clutter
    time.sleep(0.5 * rank)
    print(rank, msg)

def _test_distrib_again2(local_rank, device):
    rank = idist.get_rank()
    device = idist.device()
    torch.manual_seed(12)

    # Generate length of random integer list with process(4) * length of list(10)
    y_pred, y = (torch.randint(0, 10, size=(40,)), torch.randint(0, 10, size=(40,)))
    # User partial list of y_pred and y as input of each process
    y_pred, y = y_pred[rank*10:(rank+1)*10], y[rank*10:(rank+1)*10]
    y_pred = y_pred.to(device)
    y = y.to(device)
    
    msemae = Msemae()
    torch.manual_seed(10 + rank)

    pprint(rank, f"Hello from process {rank} : {y_pred}, {y}")

    msemae.reset()
    msemae.update((y_pred, y))

    pprint(rank, f"before all_gather {rank} : {y_pred}, {y}")
    # gather y_pred, y
    y_pred = idist.all_gather(y_pred)
    y = idist.all_gather(y)

    pprint(rank, f"after all_gather {rank} : {y_pred}, {y}")

    np_y = y.cpu().numpy()
    np_y_pred = y_pred.cpu().numpy()

    res = msemae.compute()

    true_mse = mean_squared_error(np_y,  np_y_pred)
    true_mae = mean_absolute_error(np_y,  np_y_pred)

    print("RES, TRUE", res, true_mse, true_mae)

    assert true_mse== pytest.approx(res[0].cpu().numpy())
    assert true_mae == pytest.approx(res[1].cpu().numpy())

if __name__ == '__main__':
    backend = "gloo"
    device = idist.device()

    y_pred_list, y_list = (torch.randint(0, 10, size=(10*4,)), torch.randint(0, 10, size=(10*4,)))

    with idist.Parallel(backend=backend, nproc_per_node=4) as parallel:
        parallel.run(_test_distrib_again2, y_pred_list, y_list)

