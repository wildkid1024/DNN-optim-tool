#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 9:04
# @Author  : wildkid1024
# @Site    : 
# @File    : MLP.py
# @Software: PyCharm

import torch.nn as nn
import torch

in_size = 28 * 28
layer_size = 256
layer_num = 3
out_size = 10


class MLP(nn.Module):
    def __init__(self, in_size=in_size, layer_size=layer_size, layer_num=layer_num, out_size=out_size):
        super(MLP, self).__init__()
        self.hidden_1 = nn.Linear(in_size, layer_size)
        for i in range(layer_num - 1):
            self.add_module("hidden_{0}".format(i + 2), nn.Linear(layer_size, layer_size))
        # self.hidden = nn.Linear(layer_size, layer_size)
        self.output = nn.Linear(layer_size, out_size)

    def forward(self, x):
        a = x
        for i in range(layer_num):
            fun_name = "hidden_" + str(i + 1)
            y = getattr(self, fun_name)(a)
            a = torch.sigmoid(y)
        y = self.output(a)
        out = torch.sigmoid(y)
        return out
