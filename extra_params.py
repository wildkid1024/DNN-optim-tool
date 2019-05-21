#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 9:25
# @Author  : wildkid1024
# @Site    : 
# @File    : extra_params.py
# @Software: PyCharm

in_size = 28 * 28
layer_size = 256
layer_num = 3
out_size = 10

batch_size = 32
learning_rate = 1e-2
num_epoches = 32

# 统计计算误差的图片个数
statistic_num = 50

dataset_root = './data-set/'
model_path = 'pretrained-model/mlp-mnist-pretrain2.pkl'

# 注入时的测试次数和数量
test_num = 256
inject_times = 10
