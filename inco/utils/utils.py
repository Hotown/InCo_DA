import gc
import json
from logging import getLogger
import os
import random
import shutil
import string
from collections import Counter, OrderedDict
from subprocess import call
from turtle import back
import sys
import numpy as np
import torch
from dotmap import DotMap
import torch.nn as nn
sys.path.append('.')
import common.vision.models as models
# Specific

DOMAIN_V = {"source": "target", "target": "source"}


def reverse_domain(domain_name):
    return DOMAIN_V[domain_name]


def per(acc):
    return f"{acc * 100:.2f}%"


# Metric

def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


# Debug


def MB(x):
    return f"{x/1024/1024:.3} MB"


def GB(x):
    return f"{x/1024/1024/1024:.3} GB"


def print_occupied_mem(idx=0):
    print(GB(torch.cuda.memory_allocated(idx)))


def size_of_tensor(x):
    return MB(x.element_size() * x.nelement())


def randtext(length=10):
    return "".join([random.choice(string.ascii_letters) for i in range(length)])


# OS


def to_list(something):
    if something is not None and not isinstance(something, list):
        return [something]
    return something


def makedirs(dir_list):
    if not isinstance(dir_list, list):
        dir_list = [dir_list]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)


# Counter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name="default", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = getLogger()

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


# Json

def load_json(f_path):
    with open(f_path, "r") as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def is_div(freq, epoch, best=False):
    if freq is not None and best:
        return True
    return freq and epoch % freq == 0


def info_gpu_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


# dotmap
def exist_key(k):
    is_empty_dotmap = isinstance(k, DotMap) and len(k) == 0
    return isinstance(k, bool) or (not is_empty_dotmap and k is not None)


def set_default(cur_config, name, value=None, callback=None):
    if not exist_key(cur_config[name]):
        if value is not None:
            cur_config[name] = value
        elif callback is not None:
            assert exist_key(cur_config[callback])
            cur_config[name] = cur_config[callback]
        elif value is None and callback is None:
            cur_config[name] = value
        else:
            raise NotImplementedError
    return cur_config[name]

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower and not name.startswith('__')
        and callable(models.__dict__[name])
    )

def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        NotImplementedError
    return backbone