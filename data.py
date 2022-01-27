# 定义字典
zidian_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}

zidian_xr = [k for k, v in zidian_x.items()]

zidian_y = {k.upper(): v for k, v in zidian_x.items()}

zidian_yr = [k for k, v in zidian_y.items()]

import random

import numpy as np
import torch


def get_data():
    # 定义词集合
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
    ]

    # 定义每个词被选中的概率
    p = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ])
    p = p / p.sum()

    # 随机选n个词
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p)

    # 采样的结果就是x
    x = x.tolist()

    # y是对x的变换得到的
    # 字母大写,数字取10以内的互补数
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)

    y = [f(i) for i in x]
    y = y + [y[-1]]
    # 逆序
    y = y[::-1]

    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # 编码成数据
    x = [zidian_x[i] for i in x]
    y = [zidian_y[i] for i in y]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


# 两数相加测试,使用这份数据请把main.py中的训练次数改为10
# def get_data():
#     # 定义词集合
#     words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#     # 定义每个词被选中的概率
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     p = p / p.sum()
#
#     # 随机选n个词
#     n = random.randint(10, 20)
#     s1 = np.random.choice(words, size=n, replace=True, p=p)
#
#     # 采样的结果就是s1
#     s1 = s1.tolist()
#
#     # 同样的方法,再采出s2
#     n = random.randint(10, 20)
#     s2 = np.random.choice(words, size=n, replace=True, p=p)
#     s2 = s2.tolist()
#
#     # y等于s1和s2数值上的相加
#     y = int(''.join(s1)) + int(''.join(s2))
#     y = list(str(y))
#
#     # x等于s1和s2字符上的相加
#     x = s1 + ['a'] + s2
#
#     # 加上首尾符号
#     x = ['<SOS>'] + x + ['<EOS>']
#     y = ['<SOS>'] + y + ['<EOS>']
#
#     # 补pad到固定长度
#     x = x + ['<PAD>'] * 50
#     y = y + ['<PAD>'] * 51
#     x = x[:50]
#     y = y[:51]
#
#     # 编码成数据
#     x = [zidian_x[i] for i in x]
#     y = [zidian_y[i] for i in y]
#
#     # 转tensor
#     x = torch.LongTensor(x)
#     y = torch.LongTensor(y)
#
#     return x, y


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset(),
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)
