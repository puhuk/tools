import torch
import torch.distributed as dist

class Accuracy():

    def __init__(self):
        super(Accuracy).__init__()

    def reset(self):    # reset values with zero
        self._num_correct = 0
        self._num_examples = 0

    def update(self, y_pred, y_true):   # update each process's num_correct and num_examples
        len_ = len(y_true)
        self._num_correct = (y_true == y_pred).sum()
        self._num_examples = torch.tensor(len_)

    def compute(self):
        # collect each processes' num_correct and num_examples 
        
        dist.all_reduce(torch.Tensor(self._num_correct))    # all processes have same value of num_correct
        dist.all_reduce(torch.Tensor(self._num_examples))   # all processes have same value of num_example

        return self._num_correct / self._num_examples       # return the accuracy, every process has same value
