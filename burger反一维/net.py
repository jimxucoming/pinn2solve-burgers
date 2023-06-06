import torch
import torch.optim
from collections import OrderedDict
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        # Mlp全连接，seq_net为网络结构，输入为2维，输出为1维，中间有6层20个神经元的全连接，激活函数为tanh
        super().__init__() #调用nn.module的init方法
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
            # 格式化字符串：MLP_0 MLP_1   ，0-6都是MLP,共七层链接
        self.features = nn.ModuleDict(self.features) #整合网络
        self.active = activation #每层间的激活函数设置为tanh

        # 初始化bias为0
        for m in self.modules():
            if isinstance(m, nn.Linear):#如果是Linear全连接
                nn.init.constant_(m.bias, 0)#用0填充每层的bias值

    def forward(self, x):
        # x = x.view(-1, 2)
        #前向传播
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1:
                break
            i += 1  #共六层
            x = self.active(x)
        return x

