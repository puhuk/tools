SET WORLD_SIZE=2
SET CUDA_VISIBLE_DEVICES=""
pytest --dist=each --tx %WORLD_SIZE%*popen//python=python tests/ignite/metrics/test_accuracy.py -m distributed -vvv
