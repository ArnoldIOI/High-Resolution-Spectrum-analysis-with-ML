# -*- coding: utf-8 -*-

# from sklearn.datasets import load_iris
#
# data = load_iris()
# print(data)
import random
import math


def gendata():
    """
    产生一组有随机1～10个频率成分的信号
    :return: 一个数组，最后一个数据为频率成分个数，前一w个数据为产生的随机信号取样
    """
    A = random.uniform(-1, 1)
    n = random.randint(1, 10)
    f = []
    l = []
    for i in range(n):
        f.append(random.uniform(0, 100))
    f_c = random.choice(f)
    for j in range(10000):
        l.append(A*sum([math.sin(2*3.14*x+j*2*3.14*0.0001*f_c) for x in f]))
    l.append(n)
    return l

with open("./te_data.txt", "w") as f:
    for i in range(1):
        f.write(" ".join([str(x) for x in gendata()])+"\n")

with open("./train_data.txt", "w") as f:
    for i in range(1000):
        f.write(" ".join([str(x) for x in gendata()])+"\n")

with open("./test_data.txt", "w") as f:
    for i in range(200):
        f.write(" ".join([str(x) for x in gendata()])+"\n")
