# coding:utf-8
__author__ = 'zfh'
from theano import shared
from theano import tensor as T

'''
随机梯度下降及其各种变形，为了得到权重更新
'''


def sgd(cost, params, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


# 下降速度较快，但是学习率较大会发散
def sgdm(cost, params, lr=0.01, momentum=0.9, nesterov=False):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0., borrow=True)
        updates.append([v, momentum * v - lr * g])
        if nesterov:
            updates.append([p, p + momentum * v - lr * g])
        else:
            updates.append([p, p + v])
    return updates


# 较不容易发散，但是下降速度很慢
def sgdma(cost, params, lr=0.01, momentum=0.9):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0., borrow=True)
        updates.append([v, momentum * v + (1. - momentum) * g])
        updates.append([p, p - lr * v])
    return updates


def adagrad(cost, params, lr=0.01, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 加权累加器
        accNew = acc + T.square(g)
        g = g / T.sqrt(accNew + epsilon)
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


def adadelta(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 梯度加权累加器
        accDelta = shared(p.get_value() * 0., borrow=True)  # 更新加权累加器
        accNew = rho * acc + (1 - rho) * T.square(g)  # 梯度加权累加
        delta = g * T.sqrt(accDelta + epsilon) / T.sqrt(accNew + epsilon)  # 新的梯度累加器，旧的更新累加器
        accDeltaNew = rho * accDelta + (1 - rho) * T.square(delta)  # 更新加权累加
        updates.append((acc, accNew))
        updates.append((p, p - lr * delta))
        updates.append((accDelta, accDeltaNew))
    return updates


def rmsprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 加权累加器
        accNew = rho * acc + (1 - rho) * T.square(g)
        g = g / T.sqrt(accNew + epsilon)
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


def adam(cost, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mt = shared(p.get_value() * 0., borrow=True)
        vt = shared(p.get_value() * 0., borrow=True)
        mtNew = (beta1 * mt) + (1. - beta1) * g
        vtNew = (beta2 * vt) + (1. - beta2) * T.square(g)
        pNew = p - lr * mtNew / (T.sqrt(vtNew) + epsilon)
        updates.append((mt, mtNew))
        updates.append((vt, vtNew))
        updates.append((p, pNew))
    return updates
