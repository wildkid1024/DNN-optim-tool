#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21 9:22
# @Author  : wildkid1024
# @Site    : 
# @File    : Base.py
# @Software: PyCharm

# 标准库
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.backends.cudnn as cudnn
# ---------------------

# 自定义库
from functions import *
from extra_params import dataset_root, batch_size, num_epoches
from extra_params import test_num, inject_times
from extra_params import out_size, layer_size, layer_num

# ---------------------

train_dataset = datasets.MNIST(
    root=dataset_root, train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(
    root=dataset_root, train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def update_weight(model, path):
    """
    修改model中的权值
    :param model: the model which has loaded data
    :param path: the voltage path
    :return model: 修改后的model
    """
    state_dict = model.state_dict()
    all_error_rate = error_rate(get_all_voltage(path))
    all_error_rate = np.array(all_error_rate).reshape((layer_num, layer_size))
    for id, rate in enumerate(all_error_rate):
        weight_name = 'hidden_' + str(id + 1) + '.weight'
        layer_weight_upper = list(state_dict[weight_name])
        num_each_cell = layer_weight_upper[0].size()[0]
        for i, value in enumerate(rate):
            value = min(value, num_each_cell)
            which_weight = random.sample(range(0, num_each_cell), value)  # 从第一个神经元开始，按照出错概率，随机生成数字（表示需要置反的权重），返回的是列表

            for value1 in which_weight:
                a = random.uniform(-0.95, -1.05)
                layer_weight_upper[i][value1] = layer_weight_upper[i][value1] / a

                # layer_weight_upper[i][value1] = bits_flip(layer_weight_upper[i][value1])

        dict_layer_weight_upper = {weight_name: layer_weight_upper}
        state_dict.update(dict_layer_weight_upper)
        return model


def train(model, criterion, optimizer):
    """
    训练网络模型
    :param model:
    :return: 训练好的模型
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # onehot转换函数
    def ConvertToOneHot(class_num, batch_size, label):
        label_onehot = torch.FloatTensor(batch_size, class_num)
        label_onehot.zero_()
        label_onehot.scatter_(1, label.data.unsqueeze(dim=1), 1.0)
        return Variable(label_onehot)

    for epoch in range(num_epoches):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            img = img.view(img.size(0), -1)
            img, label = img.to(device), label.to(device)
            label1 = label
            # 向前传播
            out = model(img)
            label = ConvertToOneHot(out_size, batch_size, label)
            loss = criterion(out, label)
            running_loss += loss.item() * label1.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label1).sum()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))

    return model


def verify(model):
    """
    测试数据模型检验
    :param model: 网络模型以及其参数
    :return res: 返回对应的列表
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    res = []
    for idx, data in enumerate(test_loader):
        img, label = data
        img, label = img.to(device), label.to(device)
        label2 = label.numpy()[0]
        img = img.view(img.size(0), -1)
        out = model(img)
        all_output = []
        for i in out.data:
            all_output.append(i.numpy())
        all_output = all_output[0]
        if max(all_output) == all_output[label2]:
            correct = True
        else:
            correct = False
        all_output = sorted(all_output, reverse=True)
        bvsb = all_output[0] - all_output[1]

        obj = {
            "label": int(label2),
            "correct": correct,
            "bvsb": float(bvsb)
        }

        res.append(obj)

        if idx >= test_num - 1:
            break
    return res


def avg_bvsb(normal_result, inject_result):
    """
    计算平均bvsb
    :param normal_result:
    :param inject_result:
    :return:
    """

    def compute(result):
        right_count = 0
        sum_bvsb = 0.0
        for res in result:
            # print(res)
            if res['correct'] == True:
                right_count = right_count + 1
                sum_bvsb += res['bvsb']
        if right_count == 0:
            return 0, 0
        return right_count, sum_bvsb / right_count

    no_inject_right_count, no_inject_avg_bvsb = compute(normal_result)
    inject_right_count, inject_avg_bvsb = compute(inject_result)

    sta_data = {'no_error_correct_num': no_inject_right_count,
                'no_error_fault_num': test_num - no_inject_right_count,
                'average_no_bvsb': no_inject_avg_bvsb,
                'inject_error_correct_num': inject_right_count,
                'inject_error_fault_num': test_num * inject_times - inject_right_count,
                'average_inject_bvsb': inject_avg_bvsb
                }
    return sta_data
