#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import os
from torch.autograd import Variable
import numpy as np

def accuracy(output, target):

    return (100.0 * np.sum(np.array(output) == np.array(target))
            / len(target))

def remove_pad(input, pad_id):
    """

    :param input: a 1-dim list
    :return:
    """
    for i in range(len(input)-1, -1, -1):
        if input[i] == pad_id:
            input.pop()
        else:
            break
    return input

def conlleval(eidx):
    score = os.popen('perl ./eval/conlleval.pl < ./result/result_epoch'+str(eidx)+'.txt')
    score = score.read().strip()
    return float(score.split('\n')[1].split('%')[0].split()[1])
