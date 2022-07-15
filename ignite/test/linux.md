`export WORLD_SIZE=2`
`CUDA_VISIBLE_DEVICES="" pytest --dist=each --tx $WORLD_SIZE*popen//python=python -vvv tests/ignite/metrics/test_accuracy.py -m distributed`
