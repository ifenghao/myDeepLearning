# coding:utf-8
__author__ = 'zfh'
import theano
from theano import shared
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

srng = RandomStreams()

# 转化为GPU格式的浮点数
def floatX(x):
    return x.astype(theano.config.floatX)


def weightInit(shape, name):
    return shared(floatX(np.random.randn(*shape) * 0.01), name=name, borrow=True)


def biasInit(shape, name):
    return shared(floatX(np.zeros(*shape)), name=name, borrow=True)


def reg(pramsIter):
    elementNum = 0
    regSum = shared(0., borrow=True)
    for p in pramsIter:
        regSum += T.sum(p ** 2)
        elementNum += np.prod(p.get_value(borrow=True, return_internal_type=True).shape)
    return regSum / elementNum


# 随机梯度下降得到权重更新
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


# 随机梯度下降增加动量项得到权重更新
def sgd_momentum(cost, params, momentum, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        pnew = shared(p.get_value() * 0., broadcastable=p.broadcastable)
        updates.append([p, p - pnew * lr])
        updates.append([pnew, momentum * pnew + (1. - momentum) * g])
    return updates


def dropout(X, pDrop=0.):
    if pDrop > 0:
        pRetain = 1 - pDrop
        X *= srng.binomial(X.shape, p=pRetain, dtype=theano.config.floatX)
        X /= pRetain
    return X


# rmsprop得到权重更新
def rmsprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)
        accNew = rho * acc + (1 - rho) * g ** 2
        gradScaling = T.sqrt(accNew + epsilon)
        g = g / gradScaling
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates
