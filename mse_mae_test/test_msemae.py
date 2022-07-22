# torchrun --nproc_per_node=4 test_msemae.py

import os
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from msemae import Msemae
import time


torch.manual_seed(12)

def pprint(rank, msg):
    # We add sleep to avoid printing clutter
    time.sleep(0.5 * rank)
    print(rank, msg)

def _test_non_distrib(device):
    torch.manual_seed(12)

    mse = Msemae()

    n = 10

    y_true = torch.randint(0, n, size=(5,)).to(device)
    y_preds = torch.randint(0, n, size=(5,)).to(device)

    print(y_true, y_preds)


    mse.reset()
    mse.update((y_preds, y_true))

    print(y_true, y_preds, mse.compute())

    np_y = y_true.numpy().ravel()
    np_y_pred = y_preds.numpy().ravel()

    print(mean_squared_error(np_y, np_y_pred))

    assert mean_squared_error(np_y, np_y_pred) == pytest.approx(mse.compute()[0])
    assert mean_absolute_error(np_y, np_y_pred) == pytest.approx(mse.compute()[1])
    
def _test_distrib_again_different_y(local_rank, device):
    rank = idist.get_rank()
    # torch.manual_seed(12)

    y_pred, y = (torch.randint(0, 10, size=(10,)), torch.randint(0, 10, size=(10,)))
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

def _test_distrib_again(local_rank, device):
    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(y_pred, y, metric_device):

        metric_device = torch.device(metric_device)
        msemae = Msemae(device=metric_device)

        torch.manual_seed(12)

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

    def get_test_cases():
        test_cases = [
            # Binary input data of shape (N,) or (N, 1)
            (torch.randint(0, 10, size=(10,)), torch.randint(0, 10, size=(10,))),
            (torch.randint(0, 2, size=(10, )), torch.randint(0, 2, size=(10,))),
        ]
        return test_cases

    for _ in range(2):
        test_cases = get_test_cases()
        for y_pred, y in test_cases:
            y_pred = y_pred.to(device)
            y = y.to(device)
            _test(y_pred, y, "cpu")
            if device.type != "xla":
                _test(y_pred, y, idist.device())

    # assert False

def _test_distrib(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 10

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)

        print(n_epochs, y_true.shape, y_preds.shape, y_true, y_preds, n_iters, s, rank, offset)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        msemae = Msemae(device=metric_device)
        msemae.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        print(res, y_preds, y_true)

        true_mse = mean_squared_error(y_true.numpy(),  y_preds.numpy())
        true_mae = mean_absolute_error(y_true.numpy(),  y_preds.numpy())
        print("true_res", res, pytest.approx(res[0]), true_mse)

        assert pytest.approx(res[0]) == true_mse
        assert pytest.approx(res[1]) == true_mae

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)

    assert False

def test_msemae_distrib_gloo_cpu_or_gpu():
    backend = "gloo"
    device = idist.device()
    # _test_distrib_again(device)
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(_test_distrib_again, device)


def test_msemae_gloo_cpu_or_gpu():

    device = idist.device()
    _test_non_distrib(device)

backend = "gloo"
device = idist.device()
with idist.Parallel(backend=backend) as parallel:
    parallel.run(_test_distrib_again, device)
