`export WORLD_SIZE=2`

`CUDA_VISIBLE_DEVICES="" pytest --dist=each --tx $WORLD_SIZE*popen//python=python -vvv tests/ignite/metrics/test_accuracy.py -m distributed`

`unset WORLD_SIZE`


`CUDA_VISIBLE_DEVICES="" pytest --dist=each --tx popen//python=python -vvv tests/ignite/distributed/utils/test_horovod.py -m distributed -k test_idist_all_reduce_hvd`
