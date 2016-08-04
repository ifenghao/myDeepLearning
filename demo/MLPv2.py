# coding:utf-8
__author__ = 'zfh'
'''
使用和CNNv4相同的格式
'''
from compiler.ast import flatten
import time
from copy import copy

import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy, relu
from sklearn.cross_validation import KFold
import numpy as np

from load import mnist
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def layerMLPParams(shape):
    w = utils.weightInitMLP3(shape, 'w')
    b = utils.biasInit(shape[1], 'b')
    return [w, b]


# 模型构建，返回给定样本判定为某类别的概率
# dimshuffle在偏置插入维度使之与相加矩阵相同（1，本层特征图个数，1，1），插入维度的broadcastable=True
# 每次调用dropout的模式都不同，即在每轮训练中网络结构都不同
# 本层的每个特征图和上层的所有特征图连接，可以不用去选择一些组合来部分连接
def model(X, params, pDropHidden1, pDropHidden2):
    lnum = 0
    layer = T.dot(X, params[lnum][0]) + params[lnum][1]
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropHidden1)
    lnum += 1
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1]
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropHidden2)
    lnum += 1
    return softmax(T.dot(layer, params[lnum][0]) + params[lnum][1])  # 如果使用nnet中的softmax训练产生NAN


class CMLP(object):
    def __init__(self, fin, h1, h2, outputs,
                 lr, C, pDropHidden1=0.2, pDropHidden2=0.5):
        # 超参数
        self.lr = lr
        self.C = C
        self.pDropHidden1 = pDropHidden1
        self.pDropHidden2 = pDropHidden2
        # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        self.params = []
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        self.params.append(layerMLPParams((fin, h1)))
        self.params.append(layerMLPParams((h1, h2)))
        self.params.append(layerMLPParams((h2, outputs)))

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.matrix('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProb = model(self.X, self.params, pDropHidden1, pDropHidden2)
        self.trNeqs = utils.neqs(YDropProb, self.Y)
        trCrossEntropy = categorical_crossentropy(YDropProb, self.Y)
        self.trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(self.params))

        # 测试验证集代价函数
        YFullProb = model(self.X, self.params, 0., 0.)
        self.vateNeqs = utils.neqs(YFullProb, self.Y)
        self.YPred = T.argmax(YFullProb, axis=1)
        vateCrossEntropy = categorical_crossentropy(YFullProb, self.Y)
        self.vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(self.params))

    # 重置优化参数，以重新训练模型
    def resetPrams(self):
        for p in self.params:
            utils.resetWeightMLP3(p[0])
            utils.resetBias(p[1])

    # 训练卷积网络，最终返回在测试集上的误差
    def trainmlp(self, trX, teX, trY, teY, batchSize=128, maxIter=100, verbose=True,
                 start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
        # 训练函数，输入训练集，输出训练损失和误差
        updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
        train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        valtest = utils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
        trX, teX = utils.preprocess(trX, teX)
        earlyStop = utils.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in range(maxIter):  # every epoch
            startTime = time.time()
            trCost, trError = utils.miniBatch(trX, trY, train, batchSize, verbose)
            teCost, teError = utils.miniBatch(teX, teY, valtest, batchSize, verbose)
            if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
            if earlyStop.send((trCost, teCost)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
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
            updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
            train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
            # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
            valtest = utils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
            # 分割训练集
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            trX, vaX = utils.preprocess(trX, vaX)
            earlyStop = utils.earlyStopGen(start, period, threshold, earlyStopTol)
            earlyStop.next()  # 初始化生成器
            totalStopCount = 0
            vaErrorOpt = 1.
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost, trError = utils.miniBatch(trX, trY, train, batchSize, verbose)
                vaCost, vaError = utils.miniBatch(vaX, vaY, valtest, batchSize, verbose)
                if vaError < vaErrorOpt: vaErrorOpt = vaError
                if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
                if earlyStop.send((trCost, vaCost)):
                    lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                    updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                    train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
                    totalStopCount += 1
                    if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                    if verbose: print 'learning rate decreases to ', lr
            vaErrorList.append(copy(vaErrorOpt))
            if verbose: print '*' * 10, 'one validation done, best vaError', vaErrorList, '*' * 10
            self.resetPrams()
            break  # 只进行一轮验证
        return np.mean(vaErrorList)


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    trX, teX, trY, teY = mnist(onehot=True)
    h1, h2 = 625, 625
    params = utils.randomSearch(nIter=10)
    cvErrorList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 40, num, 'parameters', param, '*' * 40
        mlp = CMLP(28 * 28, h1, h2, 10, lr, C, 0.2, 0.5)
        cvError = mlp.cv(trX, trY)
        cvErrorList.append(copy(cvError))
    optIndex = np.argmin(cvErrorList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    mlp = CMLP(28 * 28, h1, h2, 10, lr, C, 0.2, 0.5)
    mlp.trainmlp(trX, teX, trY, teY)


if __name__ == '__main__':
    main()