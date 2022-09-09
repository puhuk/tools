Question 1: NativeModel.all_reduce -> where it goes in pytorch land ?

dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=gloo_group)
Assign group to group

Question 2: Horovod.all_reduce -> where it goes in horovod land ?

allreduce from horovod/torch/mpi_ops.py
def allreduce(tensor, average=None, name=None, compression=Compression.none, op=None, prescale_factor=1.0, postscale_factor=1.0, process_set=global_process_set):
Assign group to proecess_set

Question 3: XLA.all_reduce -> where it goes in XLA land ?

all_reduce from torch_xla.core.xla_model
def all_reduce(reduce_type,inputs, scale=1.0, groups=None, cctx=None, pin_layout=True):
Assign group to groups
