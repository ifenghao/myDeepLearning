# coding:utf-8
__author__ = 'zfh'
import theano
import numpy as np

'''
图像预处理，只进行零均值化和归一化，在训练集上计算RGB三个通道每个位置的均值，分别在训练、验证、测试集上减去
不用归一化有时候会出现nan，即计算的数值太大
如果要使用标准差归一化，要注意有的位置上标准差为0，容易产生nan
'''
epsilon = 1e-6


def preprocess2d(trX, vateX):
    avg = np.mean(trX, axis=None, dtype=theano.config.floatX, keepdims=True)
    var = np.var(trX, axis=None, dtype=theano.config.floatX, keepdims=True)
    return (trX - avg) / np.sqrt(var + epsilon), (vateX - avg) / np.sqrt(var + epsilon)


def preprocess4d(trX, vateX):
    avg = np.mean(trX, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
    var = np.var(trX, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
    return (trX - avg) / np.sqrt(var + epsilon), (vateX - avg) / np.sqrt(var + epsilon)
