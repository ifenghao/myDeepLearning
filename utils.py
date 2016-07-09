# coding:utf-8
__author__ = 'zfh'
import theano
from theano import shared
from theano import tensor as T
import numpy as np

# 转化为GPU格式的浮点数
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


# 随机初始化网络权重
def randomInit(shape, name):
    return shared(floatX(np.random.randn(*shape) * 0.01), name=name)


# 随机梯度下降得到权重更新
def sgd(cost, params, learningRate=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * learningRate])
    return updates
