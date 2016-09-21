# coding:utf-8
__author__ = 'zfh'
from theano import shared
import numpy as np
from basicUtils import floatX

rng = np.random.RandomState(23455)

'''
参数初始化，在这里不能修改参数的broadcastable
否则会引起更新参数值TensorType{float32,4D}与原始值的类型cudaNdarray{float32,broadcastable}不匹配
'''


def weightInit(shape, name=None):
    return shared(floatX(rng.randn(*shape) * 0.1), name=name, borrow=True)


def biasInit(shape, name=None):
    return shared(floatX(np.zeros(shape)), name=name, borrow=True)


# 第二种参数初始化方法仅适用于零均值的输入情况，否则梯度下降很慢
def weightInitCNN2(shape, name=None):
    receptive_field_size = np.prod(shape[2:])
    fanIn = shape[1] * receptive_field_size
    fanOut = shape[0] * receptive_field_size
    bound = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(rng.uniform(-bound, bound, shape)), name=name, borrow=True)


def weightInitMLP2(shape, name=None):
    fanIn = shape[0]
    fanOut = shape[1]
    bound = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(rng.uniform(-bound, bound, shape)), name=name, borrow=True)


def weightInitMaxout2(shape, name=None):
    fanIn = shape[1]
    fanOut = shape[2]
    bound = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(rng.uniform(-bound, bound, shape)), name=name, borrow=True)


# 第三种参数初始化方法仅适用于零均值的输入，且使用ReLU神经元的情况
def weightInitCNN3(shape, name=None):
    receptive_field_size = np.prod(shape[2:])
    fanIn = shape[1] * receptive_field_size
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def weightInitMLP3(shape, name=None):
    fanIn = shape[0]
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def weightInitMaxout3(shape, name=None):
    fanIn = shape[1]
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def weightInitNIN1_3(shape, name=None):
    fanIn = shape[1] * np.prod(shape[3:])
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def weightInitNIN2_3(shape, name=None):
    fanIn = np.prod(shape[1:])
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def weightInitColfc(shape, name=None):
    fanIn = shape[-1]
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


'''
参数重置，方法和初始化配套
'''


def resetWeight(w):
    wShape = w.get_value(borrow=True).shape
    w.set_value(floatX(rng.randn(*wShape) * 0.1), borrow=True)


def resetBias(b):
    bShape = b.get_value(borrow=True).shape
    b.set_value(floatX(np.zeros(*bShape)), borrow=True)


# 2
def resetWeightCNN2(w):
    shape = w.get_value(borrow=True).shape
    fanIn = np.prod(shape[1:])
    fanOut = shape[0] * np.prod(shape[2:])
    bound = np.sqrt(6. / (fanIn + fanOut))
    w.set_value(floatX(rng.uniform(-bound, bound, shape)), borrow=True)


def resetWeightMLP2(w):
    shape = w.get_value(borrow=True).shape
    fanIn = shape[0]
    fanOut = shape[1]
    bound = np.sqrt(6. / (fanIn + fanOut))
    w.set_value(floatX(rng.uniform(-bound, bound, shape)), borrow=True)


def resetWeightMaxout2(w):
    shape = w.get_value(borrow=True).shape
    fanIn = shape[1]
    fanOut = shape[2]
    bound = np.sqrt(6. / (fanIn + fanOut))
    w.set_value(floatX(rng.uniform(-bound, bound, shape)), borrow=True)


# 3
def resetWeightCNN3(w):
    shape = w.get_value(borrow=True).shape
    fanIn = np.prod(shape[1:])
    bound = np.sqrt(2. / fanIn)
    w.set_value(floatX(rng.randn(*shape) * bound), borrow=True)


def resetWeightMLP3(w):
    shape = w.get_value(borrow=True).shape
    fanIn = shape[0]
    bound = np.sqrt(2. / fanIn)
    w.set_value(floatX(rng.randn(*shape) * bound), borrow=True)


def resetWeightMaxout3(w):
    shape = w.get_value(borrow=True).shape
    fanIn = shape[1]
    bound = np.sqrt(2. / fanIn)
    w.set_value(floatX(rng.randn(*shape) * bound), borrow=True)
