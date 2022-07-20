# torchrun --nproc_per_node=4 ignite_dist_test.py

import ignite.distributed as idist
import torch
import time

backend = "gloo"  # or "horovod" if package is installed

config = {"key": "value"}



def pprint(rank, msg):
    # We add sleep to avoid printing clutter
    time.sleep(0.5 * rank)
    print(rank, msg)


def training(local_rank):
    y = None
    rank = idist.get_rank()
    ws = idist.get_world_size()
    if idist.get_rank() == 0:
        t1 = torch.rand(1, 2, device=idist.device())
        s1 = "abc"
        x = 12.3456
        y = torch.rand(1, 3, device=idist.device())
    else:
        t1 = torch.empty(1, 2, device=idist.device())
        s1 = ""
        x = 0.0

    # Broadcast tensor t1 from rank 0 to all processes
    pprint(rank, f"before data is {t1}, {s1}, {x}, {y}")
    t1 = idist.broadcast(t1, src=0)


    assert isinstance(t1, torch.Tensor)

    # Broadcast string s1 from rank 0 to all processes
    s1 = idist.broadcast(s1, src=0)
    # >>> s1 = "abc"

    # Broadcast float number x from rank 0 to all processes
    x = idist.broadcast(x, src=0)
    # >>> x = 12.3456

    # Broadcast any of those types from rank 0,
    # but other ranks do not define the placeholder
    y = idist.broadcast(y, src=0, safe_mode=False)
    y = idist.broadcast(y, src=0, safe_mode=True)
    pprint(rank, f"after data is {t1}, {s1}, {x}, {y}")
    assert isinstance(y, torch.Tensor)

with idist.Parallel(backend=backend) as parallel:
    parallel.run(training)
