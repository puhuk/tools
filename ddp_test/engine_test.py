from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import torch
from ignite.metrics import Accuracy

torch.manual_seed(12)
n_iters = 3
batch_size = 4
n_classes = 2
y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,))
y_preds = torch.rand(n_iters * batch_size, n_classes)

print(y_true.shape, y_preds.shape, y_true, y_preds)

def update(engine, i):
    print("aa", acc._num_correct, acc._num_examples, y_preds[i * batch_size : (i + 1) * batch_size, :], y_true[i * batch_size : (i + 1) * batch_size])
    return (
        y_preds[i * batch_size : (i + 1) * batch_size, :],
        y_true[i * batch_size : (i + 1) * batch_size],
    )


engine = Engine(update)
acc = Accuracy()
acc.attach(engine, "acc")
print("num1", acc._num_correct, acc._num_examples)

data = list(range(n_iters))

print(y_true.shape, y_preds.shape)

n_epochs = 1
engine.run(data=data, max_epochs=n_epochs)
print("num2", acc._num_correct)
print(y_true.shape, y_preds.shape, acc._num_correct, acc._num_examples)

res = engine.state.metrics["acc"]
print("num3", acc._num_correct, acc._num_examples)
print(y_true.shape, y_preds.shape)
