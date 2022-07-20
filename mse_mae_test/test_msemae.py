# pytest -n 3 test_msemae.py

import os
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from msemae import Msemae

torch.manual_seed(12)

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


def test_msemae_distrib_gloo_cpu_or_gpu():

    device = idist.device()
    _test_distrib(device)


def test_msemae_gloo_cpu_or_gpu():

    device = idist.device()
    _test_non_distrib(device)
