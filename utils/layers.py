# coding:utf-8
__author__ = 'zfh'
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal.pool import pool_2d

from utils import initial

'''
定义层类
'''


class Dense(object):
    def __init__(self, shape):
        self.shape = shape
        self.whidden = initial.weightInitMLP3(shape, name='whidden')
        self.bhidden = initial.biasInit(shape[1], name='bhidden')

    def getParams(self):
        return self.whidden, self.bhidden

    def resetParams(self):
        self.whidden = initial.weightInitMLP3(self.shape, name='whidden')
        self.bhidden = initial.biasInit(self.shape[1], name='bhidden')

    def setParams(self, params):
        self.whidden = params[0]
        self.bhidden = params[1]

    def getOutput(self, X):
        layer = T.dot(X, self.whidden) + self.bhidden.dimshuffle('x', 0)
        layer = relu(layer, alpha=0)
        return layer


class Convolution(object):
    def __init__(self, shape, border_mode='valid', subsample=(1, 1)):
        self.shape = shape
        self.mode = border_mode
        self.subsample = subsample
        self.wconv = initial.weightInitCNN3(shape, name='wconv')
        self.bconv = initial.biasInit(shape[0], name='bconv')

    def getParams(self):
        return self.wconv, self.bconv

    def resetParams(self):
        self.wconv = initial.weightInitCNN3(self.shape, name='wconv')
        self.bconv = initial.biasInit(self.shape[0], name='bconv')

    def setParams(self, params):
        self.wconv = params[0]
        self.bconv = params[1]

    def getOutput(self, X):
        layer = conv2d(X, self.wconv, border_mode=self.mode, subsample=self.subsample) + \
                self.bconv.dimshuffle('x', 0, 'x', 'x')
        layer = relu(layer, alpha=0)
        return layer


class Inception(object):
    def __init__(self, ninput, nconv1, nreduce3, nconv3, nreduce5, nconv5, npoolproj):
        self.ninput, self.nconv1, self.nreduce3, self.nconv3, self.nreduce5, self.nconv5, self.npoolproj = \
            ninput, nconv1, nreduce3, nconv3, nreduce5, nconv5, npoolproj
        self.wconv1 = initial.weightInitCNN3((nconv1, ninput, 1, 1), name='wconv1')
        self.bconv1 = initial.biasInit(nconv1, name='bconv1')
        self.wreduce3 = initial.weightInitCNN3((nreduce3, ninput, 1, 1), name='wreduce3')
        self.breduce3 = initial.biasInit(nreduce3, name='breduce3')
        self.wconv3 = initial.weightInitCNN3((nconv3, nreduce3, 3, 3), name='wconv3')
        self.bconv3 = initial.biasInit(nconv3, name='bconv3')
        self.wreduce5 = initial.weightInitCNN3((nreduce5, ninput, 1, 1), name='wreduce5')
        self.breduce5 = initial.biasInit(nreduce5, name='breduce5')
        self.wconv5 = initial.weightInitCNN3((nconv5, nreduce5, 5, 5), name='wconv5')
        self.bconv5 = initial.biasInit(nconv5, name='bconv5')
        self.wpoolproj = initial.weightInitCNN3((npoolproj, ninput, 1, 1), name='wpoolproj')
        self.bpoolproj = initial.biasInit(npoolproj, name='bpoolproj')

    def getParams(self):
        params = []
        params.extend((self.wconv1, self.bconv1))
        params.extend((self.wreduce3, self.breduce3))
        params.extend((self.wconv3, self.bconv3))
        params.extend((self.wreduce5, self.breduce5))
        params.extend((self.wconv5, self.bconv5))
        params.extend((self.wpoolproj, self.bpoolproj))
        return params

    def resetParams(self):
        self.wconv1 = initial.weightInitCNN3((self.nconv1, self.ninput, 1, 1), name='wconv1')
        self.bconv1 = initial.biasInit(self.nconv1, name='bconv1')
        self.wreduce3 = initial.weightInitCNN3((self.nreduce3, self.ninput, 1, 1), name='wreduce3')
        self.breduce3 = initial.biasInit(self.nreduce3, name='breduce3')
        self.wconv3 = initial.weightInitCNN3((self.nconv3, self.nreduce3, 3, 3), name='wconv3')
        self.bconv3 = initial.biasInit(self.nconv3, name='bconv3')
        self.wreduce5 = initial.weightInitCNN3((self.nreduce5, self.ninput, 1, 1), name='wreduce5')
        self.breduce5 = initial.biasInit(self.nreduce5, name='breduce5')
        self.wconv5 = initial.weightInitCNN3((self.nconv5, self.nreduce5, 5, 5), name='wconv5')
        self.bconv5 = initial.biasInit(self.nconv5, name='bconv5')
        self.wpoolproj = initial.weightInitCNN3((self.npoolproj, self.ninput, 1, 1), name='wpoolproj')
        self.bpoolproj = initial.biasInit(self.npoolproj, name='bpoolproj')

    def setParams(self, params):
        self.wconv1, self.bconv1 = params[0]
        self.wreduce3, self.breduce3 = params[1]
        self.wconv3, self.bconv3 = params[2]
        self.wreduce5, self.breduce5 = params[3]
        self.wconv5, self.bconv5 = params[4]
        self.wpoolproj, self.bpoolproj = params[5]

    def getOutput(self, X):
        conv1 = conv2d(X, self.wconv1, border_mode='valid') + self.bconv1.dimshuffle('x', 0, 'x', 'x')
        reduce3 = conv2d(X, self.wreduce3, border_mode='valid') + self.breduce3.dimshuffle('x', 0, 'x', 'x')
        conv3 = conv2d(reduce3, self.wconv3, border_mode='half') + self.bconv3.dimshuffle('x', 0, 'x', 'x')
        reduce5 = conv2d(X, self.wreduce5, border_mode='valid') + self.breduce5.dimshuffle('x', 0, 'x', 'x')
        conv5 = conv2d(reduce5, self.wconv5, border_mode='half') + self.bconv5.dimshuffle('x', 0, 'x', 'x')
        pool3 = pool_2d(X, (3, 3), st=(1, 1), padding=(1, 1), ignore_border=True, mode='max')
        poolproj = conv2d(pool3, self.wpoolproj, border_mode='valid') + self.bpoolproj.dimshuffle('x', 0, 'x', 'x')
        return T.concatenate([conv1, conv3, conv5, poolproj], axis=1)
