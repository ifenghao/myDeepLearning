# coding:utf-8
__author__ = 'zfh'
'''
Maxout Network的Conv版本
使用和CNNv4相同的格式
输出层采用relu全连接层
'''
import time
from compiler.ast import flatten
from copy import copy

import numpy as np
import theano.tensor as T
from sklearn.cross_validation import KFold
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d

from load import cifar
from utils import basicUtils, gradient, initial, preprocess


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# maxout激活函数，只需要把本层特征图维度拆分出分段维度，在分段维度上求最大值
# 输入（样本数，输入特征图，行1，列1）* maxout层（本层特征图*分段数，上层特征图，3，3）=（样本数，本层特征图*分段数，行2，列2）
# 输出（样本数，本层特征图，行2，列2）
def maxout(X, featMap, piece):
    Xr = T.reshape(X, (-1, featMap, piece, X.shape[-2], X.shape[-1]))  # （样本数，本层特征图，分段数，行2，列2）
    return T.max(Xr, axis=2)  # （样本数，本层特征图，行2，列2）


# 将maxout等效为一般的CNN使用同样的参数维度
def layerCNNParams(shape):
    w = initial.weightInitCNN3(shape, 'w')
    b = initial.biasInit(shape[0], 'b')
    return [w, b]


def layerMLPParams(shape):
    w = initial.weightInitMLP3(shape, 'w')
    b = initial.biasInit(shape[1], 'b')
    return [w, b]


def model(X, params, featMaps, pieces, pDropConv, pDropHidden):
    lnum = 0  # conv: (32, 32) pool: (16, 16)
    layer = conv2d(X, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = maxout(layer, featMaps[lnum], pieces[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1  # conv: (16, 16) pool: (8, 8)
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = maxout(layer, featMaps[lnum], pieces[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1  # conv: (8, 8) pool: (4, 4)
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = maxout(layer, featMaps[lnum], pieces[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1
    layer = T.flatten(layer, outdim=2)
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0)
    layer = relu(layer, alpha=0)
    layer = basicUtils.dropout(layer, pDropHidden)
    lnum += 1
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0)
    layer = relu(layer, alpha=0)
    layer = basicUtils.dropout(layer, pDropHidden)
    lnum += 1
    return softmax(T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0))  # 如果使用nnet中的softmax训练产生NAN


class CMaxoutconv(object):
    def __init__(self, fin, f1, piece1, f2, piece2, f3, piece3, h1, h2, outputs,
                 lr, C, pDropConv=0.2, pDropHidden=0.5):
        # 超参数
        self.lr = lr
        self.C = C
        self.pDropConv = pDropConv
        self.pDropHidden = pDropHidden
        # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        self.params = []
        self.paramsCNN = []
        self.paramsMLP = []
        featMaps = []
        pieces = []
        # 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
        self.paramsCNN.append(layerCNNParams((f1 * piece1, fin, 3, 3)))  # conv: (32, 32) pool: (16, 16)
        featMaps.append(f1)
        pieces.append(piece1)
        self.paramsCNN.append(layerCNNParams((f2 * piece2, f1, 3, 3)))  # conv: (16, 16) pool: (8, 8)
        featMaps.append(f2)
        pieces.append(piece2)
        self.paramsCNN.append(layerCNNParams((f3 * piece3, f2, 3, 3)))  # conv: (8, 8) pool: (4, 4)
        featMaps.append(f3)
        pieces.append(piece3)
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        self.paramsMLP.append(layerMLPParams((f3 * 4 * 4, h1)))
        self.paramsMLP.append(layerMLPParams((h1, h2)))
        self.paramsMLP.append(layerMLPParams((h2, outputs)))
        self.params = self.paramsCNN + self.paramsMLP

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.tensor4('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProb = model(self.X, self.params, featMaps, pieces, pDropConv, pDropHidden)
        self.trNeqs = basicUtils.neqs(YDropProb, self.Y)
        trCrossEntropy = categorical_crossentropy(YDropProb, self.Y)
        self.trCost = T.mean(trCrossEntropy) + C * basicUtils.regularizer(flatten(self.params))

        # 测试验证集代价函数
        YFullProb = model(self.X, self.params, featMaps, pieces, 0., 0.)
        self.vateNeqs = basicUtils.neqs(YFullProb, self.Y)
        self.YPred = T.argmax(YFullProb, axis=1)
        vateCrossEntropy = categorical_crossentropy(YFullProb, self.Y)
        self.vateCost = T.mean(vateCrossEntropy) + C * basicUtils.regularizer(flatten(self.params))

    # 重置优化参数，以重新训练模型
    def resetParams(self):
        for p in self.paramsCNN:
            initial.resetWeightMaxout3(p[0])
            initial.resetBias(p[1])
        for p in self.paramsMLP:
            initial.resetWeightMLP3(p[0])
            initial.resetBias(p[1])

    # 训练卷积网络，最终返回在测试集上的误差
    def trainmaxout(self, trX, teX, trY, teY, batchSize=128, maxIter=100, verbose=True,
                    start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
        # 训练函数，输入训练集，输出训练损失和误差
        updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
        train = basicUtils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        valtest = basicUtils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
        trX, teX = preprocess.preprocess4d(trX, teX)
        earlyStop = basicUtils.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in range(maxIter):  # every epoch
            startTime = time.time()
            trCost, trError = basicUtils.miniBatch(trX, trY, train, batchSize, verbose)
            teCost, teError = basicUtils.miniBatch(teX, teY, valtest, batchSize, verbose)
            if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
            if earlyStop.send((trCost, teCost)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                train = basicUtils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
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
            train = basicUtils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
            # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
            valtest = basicUtils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
            # 分割训练集
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            trX, vaX = preprocess.preprocess4d(trX, vaX)
            earlyStop = basicUtils.earlyStopGen(start, period, threshold, earlyStopTol)
            earlyStop.next()  # 初始化生成器
            totalStopCount = 0
            vaErrorOpt = 1.
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost, trError = basicUtils.miniBatch(trX, trY, train, batchSize, verbose)
                vaCost, vaError = basicUtils.miniBatch(vaX, vaY, valtest, batchSize, verbose)
                if vaError < vaErrorOpt: vaErrorOpt = vaError
                if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
                if earlyStop.send((trCost, vaCost)):
                    lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                    updates = gradient.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                    train = basicUtils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
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
    trX, teX, trY, teY = cifar(onehot=True)
    f1, piece1, f2, piece2, f3, piece3, h1, h2 = 32, 3, 64, 3, 128, 3, 512, 256
    params = basicUtils.randomSearch(nIter=10)
    cvErrorList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 40, num, 'parameters', param, '*' * 40
        maxout = CMaxoutconv(3, f1, piece1, f2, piece2, f3, piece3, h1, h2, 10, lr, C, 0.2, 0.5)
        cvError = maxout.cv(trX, trY)
        cvErrorList.append(copy(cvError))
    optIndex = np.argmin(cvErrorList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    maxout = CMaxoutconv(3, f1, piece1, f2, piece2, f3, piece3, h1, h2, 10, lr, C, 0.2, 0.5)
    maxout.trainmaxout(trX, teX, trY, teY)


if __name__ == '__main__':
    main()
