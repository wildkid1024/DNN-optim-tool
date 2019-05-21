#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 9:34
# @Author  : wildkid1024
# @Site    : 
# @File    : main.py
# @Software: PyCharm

from nets.MLP import *
from extra_params import *
from functions import *
from Base import update_weight, train, verify, avg_bvsb

import torch
import torch.optim as optim

import os
import pandas as pd


@print_run_time
def run(path):
    no_error_path = path + 'bvsb_no_error.json'
    inject_error_path = path + 'bvsb_inject_error.json'
    statistics_path = path + 'bvsb_result.json'

    model = MLP()
    model.load_state_dict(torch.load(model_path))
    res1 = verify(model)

    res2 = []
    for i in range(inject_times):
        model.load_state_dict(torch.load(model_path))
        res2.extend(verify(update_weight(model, path)))
    sta_data = avg_bvsb(res1, res2)

    # print(res1)
    save(res1, no_error_path)
    save(res2, inject_error_path)
    save(sta_data, statistics_path)


@print_run_time
def my_test_train():
    model = MLP()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("model:", model)

    pretrained_model = train(model, criterion, optimizer)

    print("num of params:", sum(param.numel() for param in pretrained_model.parameters()))

    torch.save(pretrained_model.state_dict(), model_path)


@print_run_time
def my_test():
    model1 = MLP()
    model1.load_state_dict(torch.load(model_path))
    res1 = verify(model1)

    model2 = MLP()
    model2.load_state_dict(torch.load(model_path))
    res2 = verify(model2)

    print(model1)
    print(model2)
    print("model1 and model2:", model1 == model2)
    print("res1 and res2:", res1 == res2)
    print(res1)
    print(res2)


if __name__ == '__main__':
    result_path = 'data-train/'
    my_test_train()
    run(result_path)
    # my_test()
