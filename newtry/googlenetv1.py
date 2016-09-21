# coding:utf-8
__author__ = 'zfh'
'''
Inception模块
'''
import time
from compiler.ast import flatten
from copy import copy
import h5py
from scipy.misc import imread, imresize, imshow

import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import KFold
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d

from load import cifar
from utils import basicUtils, gradient, initial, preprocess
from utils import layers


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 模型构建，返回给定样本判定为某类别的概率
class Model(object):
    def __init__(self):
        self.conv1 = layers.Convolution((64, 3, 7, 7), border_mode='half', subsample=(2, 2))
        self.conv2reduce = layers.Convolution((64, 64, 1, 1), border_mode='valid', subsample=(1, 1))
        self.conv2 = layers.Convolution((192, 64, 3, 3), border_mode='half', subsample=(1, 1))
        self.inception3a = layers.Inception(ninput=192, nconv1=64, nreduce3=96, nconv3=128,
                                            nreduce5=16, nconv5=32, npoolproj=32)
        self.inception3b = layers.Inception(ninput=256, nconv1=128, nreduce3=128, nconv3=192,
                                            nreduce5=32, nconv5=96, npoolproj=64)
        self.inception4a = layers.Inception(ninput=480, nconv1=192, nreduce3=96, nconv3=208,
                                            nreduce5=16, nconv5=48, npoolproj=64)
        self.inception4b = layers.Inception(ninput=512, nconv1=160, nreduce3=112, nconv3=224,
                                            nreduce5=24, nconv5=64, npoolproj=64)
        self.inception4c = layers.Inception(ninput=512, nconv1=128, nreduce3=128, nconv3=256,
                                            nreduce5=24, nconv5=64, npoolproj=64)
        self.inception4d = layers.Inception(ninput=512, nconv1=112, nreduce3=144, nconv3=288,
                                            nreduce5=32, nconv5=64, npoolproj=64)
        self.inception4e = layers.Inception(ninput=528, nconv1=256, nreduce3=160, nconv3=320,
                                            nreduce5=32, nconv5=128, npoolproj=128)
        self.inception5a = layers.Inception(ninput=832, nconv1=256, nreduce3=160, nconv3=320,
                                            nreduce5=32, nconv5=128, npoolproj=128)
        self.inception5b = layers.Inception(ninput=832, nconv1=384, nreduce3=192, nconv3=384,
                                            nreduce5=48, nconv5=128, npoolproj=128)
        self.prob3mlp = layers.Dense((1024, 1000))
        self.mainLayers = [self.conv1, self.conv2reduce, self.conv2, self.inception3a, self.inception3b,
                           self.inception4a, self.inception4b, self.inception4c, self.inception4d,
                           self.inception4e, self.inception5a, self.inception5b, self.prob3mlp]
        self.prob1conv = layers.Convolution((128, 512, 1, 1), border_mode='valid', subsample=(1, 1))
        self.prob1mlp0 = layers.Dense((128 * 4 * 4, 1024))
        self.prob1mlp1 = layers.Dense((1024, 1000))
        self.aux1Layers = [self.prob1conv, self.prob1mlp0, self.prob1mlp1]
        self.prob2conv = layers.Convolution((128, 528, 1, 1), border_mode='valid', subsample=(1, 1))
        self.prob2mlp0 = layers.Dense((128 * 4 * 4, 1024))
        self.prob2mlp1 = layers.Dense((1024, 1000))
        self.aux2Layers = [self.prob2conv, self.prob2mlp0, self.prob2mlp1]

    def getParams(self):
        mainParams = []
        for mainLayer in self.mainLayers:
            mainParams.append(mainLayer.getParams())
        auxParams1 = []
        for aux1Layer in self.aux1Layers:
            auxParams1.append(aux1Layer.getParams())
        auxParams2 = []
        for aux2Layer in self.aux2Layers:
            auxParams2.append(aux2Layer.getParams())
        return mainParams, auxParams1, auxParams2

    def resetParams(self):
        allLayers = []
        allLayers.extend(self.mainLayers)
        allLayers.extend(self.aux1Layers)
        allLayers.extend(self.aux2Layers)
        for layer in allLayers:
            layer.resetParams()

    def setParams(self, params):
        self.conv1.setParams(params['conv1'])
        self.conv2reduce.setParams(params['conv2reduce'])
        self.conv2.setParams(params['conv2'])
        self.inception3a.setParams(params['inception3a'])
        self.inception3b.setParams(params['inception3b'])
        self.inception4a.setParams(params['inception4a'])
        self.prob1conv.setParams(params['prob1conv'])
        self.prob1mlp0.setParams(params['prob1mlp0'])
        self.prob1mlp1.setParams(params['prob1mlp1'])
        self.inception4b.setParams(params['inception4b'])
        self.inception4c.setParams(params['inception4c'])
        self.inception4d.setParams(params['inception4d'])
        self.prob2conv.setParams(params['prob2conv'])
        self.prob2mlp0.setParams(params['prob2mlp0'])
        self.prob2mlp1.setParams(params['prob2mlp1'])
        self.inception4e.setParams(params['inception4e'])
        self.inception5a.setParams(params['inception5a'])
        self.inception5b.setParams(params['inception5b'])
        self.prob3mlp.setParams(params['prob3mlp'])

    def _loadGroup(self, hf, groupName):
        data = hf.get(groupName)
        w, b = data.values()
        w = np.array(w, dtype=theano.config.floatX)
        b = np.array(b, dtype=theano.config.floatX)
        return w, b

    def loadParams(self, fileName):
        hf = h5py.File(fileName, 'r')
        params = {}
        params['conv1'] = self._loadGroup(hf, 'conv1/7x7_s2/conv1')
        params['conv2reduce'] = self._loadGroup(hf, 'conv2/3x3_reduce/conv2')
        params['conv2'] = self._loadGroup(hf, 'conv2/3x3/conv2')
        params['inception3a'] = []
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/1x1/inception_3a'))
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/3x3_reduce/inception_3a'))
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/3x3/inception_3a'))
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/5x5_reduce/inception_3a'))
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/5x5/inception_3a'))
        params['inception3a'].append(self._loadGroup(hf, 'inception_3a/pool_proj/inception_3a'))
        params['inception3b'] = []
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/1x1/inception_3b'))
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/3x3_reduce/inception_3b'))
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/3x3/inception_3b'))
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/5x5_reduce/inception_3b'))
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/5x5/inception_3b'))
        params['inception3b'].append(self._loadGroup(hf, 'inception_3b/pool_proj/inception_3b'))
        params['inception4a'] = []
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/1x1/inception_4a'))
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/3x3_reduce/inception_4a'))
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/3x3/inception_4a'))
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/5x5_reduce/inception_4a'))
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/5x5/inception_4a'))
        params['inception4a'].append(self._loadGroup(hf, 'inception_4a/pool_proj/inception_4a'))
        params['prob1conv'] = self._loadGroup(hf, 'loss1/conv/loss1')
        params['prob1mlp0'] = self._loadGroup(hf, 'loss1/fc/loss1')
        params['prob1mlp1'] = self._loadGroup(hf, 'loss1/classifier/loss1')
        params['inception4b'] = []
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/1x1/inception_4b'))
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/3x3_reduce/inception_4b'))
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/3x3/inception_4b'))
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/5x5_reduce/inception_4b'))
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/5x5/inception_4b'))
        params['inception4b'].append(self._loadGroup(hf, 'inception_4b/pool_proj/inception_4b'))
        params['inception4c'] = []
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/1x1/inception_4c'))
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/3x3_reduce/inception_4c'))
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/3x3/inception_4c'))
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/5x5_reduce/inception_4c'))
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/5x5/inception_4c'))
        params['inception4c'].append(self._loadGroup(hf, 'inception_4c/pool_proj/inception_4c'))
        params['inception4d'] = []
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/1x1/inception_4d'))
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/3x3_reduce/inception_4d'))
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/3x3/inception_4d'))
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/5x5_reduce/inception_4d'))
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/5x5/inception_4d'))
        params['inception4d'].append(self._loadGroup(hf, 'inception_4d/pool_proj/inception_4d'))
        params['prob2conv'] = self._loadGroup(hf, 'loss2/conv/loss2')
        params['prob2mlp0'] = self._loadGroup(hf, 'loss2/fc/loss2')
        params['prob2mlp1'] = self._loadGroup(hf, 'loss2/classifier/loss2')
        params['inception4e'] = []
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/1x1/inception_4e'))
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/3x3_reduce/inception_4e'))
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/3x3/inception_4e'))
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/5x5_reduce/inception_4e'))
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/5x5/inception_4e'))
        params['inception4e'].append(self._loadGroup(hf, 'inception_4e/pool_proj/inception_4e'))
        params['inception5a'] = []
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/1x1/inception_5a'))
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/3x3_reduce/inception_5a'))
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/3x3/inception_5a'))
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/5x5_reduce/inception_5a'))
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/5x5/inception_5a'))
        params['inception5a'].append(self._loadGroup(hf, 'inception_5a/pool_proj/inception_5a'))
        params['inception5b'] = []
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/1x1/inception_5b'))
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/3x3_reduce/inception_5b'))
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/3x3/inception_5b'))
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/5x5_reduce/inception_5b'))
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/5x5/inception_5b'))
        params['inception5b'].append(self._loadGroup(hf, 'inception_5b/pool_proj/inception_5b'))
        params['prob3mlp'] = self._loadGroup(hf, 'loss3/classifier/loss3')
        return params

    def getOutput(self, X, drop):
        layer = self.conv1.getOutput(X)
        layer = basicUtils.pad2d(layer, (0, 1, 0, 1))
        layer = pool_2d(layer, (3, 3), st=(2, 2), ignore_border=False, mode='max')
        layer = basicUtils.LRN(layer)
        layer = self.conv2reduce.getOutput(layer)
        layer = self.conv2.getOutput(layer)
        layer = basicUtils.LRN(layer)
        layer = basicUtils.pad2d(layer, (0, 1, 0, 1))
        layer = pool_2d(layer, (3, 3), st=(2, 2), ignore_border=False, mode='max')
        layer = self.inception3a.getOutput(layer)
        layer = self.inception3b.getOutput(layer)
        layer = basicUtils.pad2d(layer, (0, 1, 0, 1))
        layer = pool_2d(layer, (3, 3), st=(2, 2), ignore_border=False, mode='max')
        layer = self.inception4a.getOutput(layer)
        probAux1 = self._auxiliaryClassifier1(layer, drop)
        layer = self.inception4b.getOutput(layer)
        layer = self.inception4c.getOutput(layer)
        layer = self.inception4d.getOutput(layer)
        probAux2 = self._auxiliaryClassifier2(layer, drop)
        layer = self.inception4e.getOutput(layer)
        layer = basicUtils.pad2d(layer, (0, 1, 0, 1))
        layer = pool_2d(layer, (3, 3), st=(2, 2), ignore_border=False, mode='max')
        layer = self.inception5a.getOutput(layer)
        layer = self.inception5b.getOutput(layer)
        probMain = self._mainClassifier(layer, drop)
        return probMain, probAux1, probAux2

    def _auxiliaryClassifier1(self, X, drop):
        prob = pool_2d(X, (5, 5), st=(3, 3), ignore_border=False, mode='average_exc_pad')
        prob = self.prob1conv.getOutput(prob)
        prob = T.flatten(prob, outdim=2)
        prob = self.prob1mlp0.getOutput(prob)
        if drop: prob = basicUtils.dropout(prob, 0.7)
        prob = self.prob1mlp1.getOutput(prob)
        prob = softmax(prob)
        return prob

    def _auxiliaryClassifier2(self, X, drop):
        prob = pool_2d(X, (5, 5), st=(3, 3), ignore_border=False, mode='average_exc_pad')
        prob = self.prob2conv.getOutput(prob)
        prob = T.flatten(prob, outdim=2)
        prob = self.prob2mlp0.getOutput(prob)
        if drop: prob = basicUtils.dropout(prob, 0.7)
        prob = self.prob2mlp1.getOutput(prob)
        prob = softmax(prob)
        return prob

    def _mainClassifier(self, X, drop):
        prob = pool_2d(X, (7, 7), st=(1, 1), ignore_border=False, mode='average_exc_pad')
        prob = T.flatten(prob, outdim=2)
        if drop: prob = basicUtils.dropout(prob, 0.4)
        prob = self.prob3mlp.getOutput(prob)
        prob = softmax(prob)
        return prob


# 卷积网络，输入一组超参数，返回该网络的训练、验证、预测函数
class GoogLeNet(object):
    def __init__(self, lr, C):
        # 超参数
        self.lr = lr
        self.C = C
        self.model = Model()
        self.mainParams, self.auxParams1, self.auxParams2 = self.model.getParams()
        self.params = self.mainParams + self.auxParams1 + self.auxParams2

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.tensor4('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProbMain, YDropProbAux1, YDropProbAux2 = self.model.getOutput(self.X, True)
        self.trNeqsMain = basicUtils.neqs(YDropProbMain, self.Y)
        self.trNeqsAux1 = basicUtils.neqs(YDropProbAux1, self.Y)
        self.trNeqsAux2 = basicUtils.neqs(YDropProbAux2, self.Y)
        trCrossEntropyMain = categorical_crossentropy(YDropProbMain, self.Y)
        trCrossEntropyAux1 = categorical_crossentropy(YDropProbAux1, self.Y)
        trCrossEntropyAux2 = categorical_crossentropy(YDropProbAux2, self.Y)
        trCostMain = T.mean(trCrossEntropyMain) + self.C * basicUtils.regularizer(flatten(self.mainParams))
        trCostAux1 = T.mean(trCrossEntropyAux1) + self.C * basicUtils.regularizer(flatten(self.auxParams1))
        trCostAux2 = T.mean(trCrossEntropyAux2) + self.C * basicUtils.regularizer(flatten(self.auxParams2))
        self.trCost = trCostMain + 0.3 * trCostAux1 + 0.3 * trCostAux2

        # 测试验证集代价函数
        YFullProbMain, YFullProbAux1, YFullProbAux2 = self.model.getOutput(self.X, False)
        self.vateNeqsMain = basicUtils.neqs(YFullProbMain, self.Y)
        self.vateNeqsAux1 = basicUtils.neqs(YFullProbAux1, self.Y)
        self.vateNeqsAux2 = basicUtils.neqs(YFullProbAux2, self.Y)
        vateCrossEntropyMain = categorical_crossentropy(YFullProbMain, self.Y)
        vateCrossEntropyAux1 = categorical_crossentropy(YFullProbAux1, self.Y)
        vateCrossEntropyAux2 = categorical_crossentropy(YFullProbAux2, self.Y)
        vateCostMain = T.mean(vateCrossEntropyMain) + self.C * basicUtils.regularizer(flatten(self.mainParams))
        vateCostAux1 = T.mean(vateCrossEntropyAux1) + self.C * basicUtils.regularizer(flatten(self.auxParams1))
        vateCostAux2 = T.mean(vateCrossEntropyAux2) + self.C * basicUtils.regularizer(flatten(self.auxParams2))
        self.vateCost = vateCostMain + 0.3 * vateCostAux1 + 0.3 * vateCostAux2
        self.YPred = YFullProbMain

    def setParams(self, fileName):
        params = self.model.loadParams(fileName)
        self.model.setParams(params)

    # 重置优化参数，以重新训练模型
    def resetParams(self):
        self.model.resetParams()

    def predictcn(self, X):
        predict = basicUtils.makeFunc([self.X], [self.YPred], None)
        return predict(X)

    # 训练卷积网络，最终返回在测试集上的误差
    def traincn(self, trX, teX, trY, teY, batchSize=128, maxIter=100, verbose=True,
                start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
        # 训练函数，输入训练集，输出训练损失和误差
        updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
        train = basicUtils.makeFunc([self.X, self.Y],
                                    [self.trCost, self.trNeqsMain, self.trNeqsAux1, self.trNeqsAux2],
                                    updates)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        valtest = basicUtils.makeFunc([self.X, self.Y],
                                      [self.vateCost, self.vateNeqsMain, self.vateNeqsAux1, self.vateNeqsAux2],
                                      None)
        trX, teX = preprocess.preprocess4d(trX, teX)
        earlyStop = basicUtils.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in range(maxIter):  # every epoch
            startTime = time.time()
            trCost, _, _, _ = basicUtils.miniBatchMO(trX, trY, train, batchSize, verbose)
            teCost, _, _, _ = basicUtils.miniBatchMO(teX, teY, valtest, batchSize, verbose)
            if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
            if earlyStop.send((trCost, teCost)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                train = basicUtils.makeFunc([self.X, self.Y],
                                            [self.trCost, self.trNeqsMain, self.trNeqsAux1, self.trNeqsAux2],
                                            updates)
                totalStopCount += 1
                if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                if verbose: print 'learning rate decreases to ', lr

    # 交叉验证得到一组平均验证误差，使用早停止
    def cv(self, X, Y, folds=5, batchSize=128, maxIter=100, verbose=True,
           start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        # 训练集分为n折交叉验证集
        kf = KFold(X.shape[0], n_folds=folds, random_state=42)
        vaErrorList = []
        for trIndex, vaIndex in kf:
            lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
            # 训练函数，输入训练集，输出训练损失和误差
            updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
            train = basicUtils.makeFunc([self.X, self.Y],
                                        [self.trCost, self.trNeqsMain, self.trNeqsAux1, self.trNeqsAux2],
                                        updates)
            # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
            valtest = basicUtils.makeFunc([self.X, self.Y],
                                          [self.vateCost, self.vateNeqsMain, self.vateNeqsAux1, self.vateNeqsAux2],
                                          None)
            # 分割训练集
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            trX, vaX = preprocess.preprocess4d(trX, vaX)
            earlyStop = basicUtils.earlyStopGen(start, period, threshold, earlyStopTol)
            earlyStop.next()  # 初始化生成器
            totalStopCount = 0
            vaErrorOpt = 1.
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost, trError, _, _ = basicUtils.miniBatchMO(trX, trY, train, batchSize, verbose)
                vaCost, vaError, _, _ = basicUtils.miniBatchMO(vaX, vaY, valtest, batchSize, verbose)
                if vaError < vaErrorOpt: vaErrorOpt = vaError
                if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
                if earlyStop.send((trCost, vaCost)):
                    lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                    updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                    train = basicUtils.makeFunc([self.X, self.Y],
                                                [self.trCost, self.trNeqsMain, self.trNeqsAux1, self.trNeqsAux2],
                                                updates)
                    totalStopCount += 1
                    if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                    if verbose: print 'learning rate decreases to ', lr
            vaErrorList.append(copy(vaErrorOpt))
            if verbose: print '*' * 10, 'one validation done, best vaError', vaErrorList, '*' * 10
            self.resetParams()
            break  # 只进行一轮验证
        return np.mean(vaErrorList)


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    # trX, teX, trY, teY = cifar(onehot=True)
    # f1, nin1, f2, nin2, f3, nin3, expand, h1 = 16, 5, 32, 3, 64, 2, 2, 64
    # params = basicUtils.randomSearch(nIter=10)
    # cvErrorList = []
    # for param, num in zip(params, range(len(params))):
    #     lr, C = param
    #     print '*' * 40, num, 'parameters', param, '*' * 40
    #     convNet = CConvNet(3, f1, nin1, f2, nin2, f3, nin3, expand, h1, 10, lr, C, 0.2, 0.5)
    #     cvError = convNet.cv(trX, trY)
    #     cvErrorList.append(copy(cvError))
    # optIndex = np.argmin(cvErrorList, axis=0)
    # lr, C = params[optIndex]
    # print 'retraining', params[optIndex]
    # convNet = CConvNet(3, f1, nin1, f2, nin2, f3, nin3, expand, h1, 10, lr, C, 0.2, 0.5)
    # convNet.traincn(trX, teX, trY, teY)
    img = imread("/home/zhufenghao/persian-cat.jpg", mode='RGB')
    img = imresize(img, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    avg = np.mean(img, axis=(1, 2), keepdims=True)
    img -= avg
    img = np.expand_dims(img, axis=0)
    googlenet = GoogLeNet(0.1, 0.0002)
    googlenet.setParams('/home/zhufenghao/googlenet_weights.h5')
    out = googlenet.predictcn(img)
    print np.argmax(out)


if __name__ == '__main__':
    main()
