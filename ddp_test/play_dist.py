# python -m torch.distributed.launch --nproc_per_node=4 --use_env prac1.py

import torch.distributed as dist
import torch
import time

def pprint(rank, msg):
    # We add sleep to avoid printing clutter
    time.sleep(0.5 * rank)
    print(rank, msg)

dist.init_process_group("gloo")
    
rank = dist.get_rank()
ws = dist.get_world_size()
if dist.get_rank() == 0:
    # Assumes world_size of 3.
    objects = ["foo", 'goo', 'aaa'] # any picklable object
else:
    objects = [None, None, None]
# Assumes backend is not NCCL

pprint(rank, f"before data_per_proc = {objects}")
device = torch.device("cpu")
dist.broadcast_object_list(objects, src=0, device=device)

pprint(rank, f"after data_per_proc = {objects}")
