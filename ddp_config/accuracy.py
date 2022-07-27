import torch
import torch.distributed as dist


class Accuracy:
    def reset(self):  # reset values with zero
        self._num_correct = 0
        self._num_examples = 0

    def update(
        self, y_pred, y_true
    ):  # update each process's num_correct and num_examples
        self._num_correct += (y_true == y_pred).sum().item()
        self._num_examples += len(y_true)

    def compute(self):
        # collect each processes' num_correct and num_examples
        _num_correct_tmp = torch.tensor(self._num_correct)
        _num_examples_tmp = torch.tensor(self._num_examples)

        if torch.distributed.is_initialized():
            dist.all_reduce(
                _num_correct_tmp
            )  # all processes have same value of num_correct
            dist.all_reduce(
                _num_examples_tmp
            )  # all processes have same value of num_example

        return float(_num_correct_tmp) / float(
            _num_examples_tmp
        )  # return num_correct and num_examples separately
