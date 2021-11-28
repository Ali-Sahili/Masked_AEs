import torch
import random
import numpy as np


# Metric
class Metric_rank:
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def average(self):
        return self.sum / self.n


# random seed
def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# Calculate accuracy 
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1/batch_size).item()
            res.append(acc)  # unit: percentage (%)
            crr.append(correct_k)
        return res, crr



