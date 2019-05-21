#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 9:23
# @Author  : wildkid1024
# @Site    : 
# @File    : functions.py
# @Software: PyCharm

import random
import numpy as np
import json
import time


def save(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
    return True


def load(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def get_all_voltage(path=''):
    """
    得到在该路径下的电压
    :param path:路径
    :return:voltage list
    """
    all_voltage = []
    with open(path + 'each_neuron_voltage.txt', 'r') as f:
        for line in f.readlines():
            v = float(line.split('\n')[0])
            all_voltage.append(v)
    return all_voltage


def error_equation(voltage):
    """
    错误率和电压之间的关系式，拟合函数
    :param voltage: the voltage matlab has allocation
    :return: the error num
    """
    error_rate = 5360.73087513152 - 38613.0129521685 * voltage + 117750.173386322 * (
            voltage ** 2) - 192432.524868796 * (voltage ** 3) + 183508.284537152 * (voltage ** 4) - \
                 104324.057385649 * (voltage ** 5) + 34279.920671022 * (voltage ** 6) - 5843.34008736431 * (
                         voltage ** 7) + 377.32294130795 * (voltage ** 8)
    return abs(error_rate)


def error_rate(all_voltage):
    """
    对应电压和单个神经元weight出错概率的函数,仅对mlp有效
    :param all_voltage: all the voltage
    :return:
    """
    all_error_rate = []
    for i, v in enumerate(all_voltage):
        if v >= 1.9:
            rate = 0
        elif v > 1.75:
            rate = random.randint(0, 1)
        else:
            rate = abs(error_equation(v))
            if i < 256:
                rate *= (784 / 256)
            # rate = rate / 2
            rate = int(rate)
            if rate == 0:
                rate = random.randint(0, 1)
        all_error_rate.append(rate)
    return all_error_rate


def bits_flip(v, flip_range=[0, 32]):
    pos = random.randint(flip_range[0], flip_range[1])
    v = v ^ (0x1 << pos)
    return v


def calculate_power(all_voltage):
    """
    得到该组电压下的能耗
    :param all_voltage:
    :return: the power
    """
    power = np.sum(np.array(all_voltage) ** 2)
    return power


def calculate_delay(all_voltage):
    """
    计算在该组电压下的延迟，仅对mlp有效
    :param all_voltage:
    :return: the delay
    """

    def cal(v_list):
        min_v = np.min(np.array(v_list))
        return min_v / (min_v - 0.4) ** 1.3

    delay = cal(all_voltage[:256]) + cal(all_voltage[256:512]) + cal(all_voltage[-256:])
    return delay


def print_run_time(func):
    """
    计时器函数
    :param func:
    :return:
    """

    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print('current Function [%s] run time is %.2f s' % (func.__name__, time.time() - local_time))

    return wrapper
