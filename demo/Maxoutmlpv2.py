# coding:utf-8
__author__ = 'zfh'
'''
Maxout Network的MLP版本
使用和CNNv4相同的格式
将maxout等效为一般的MLP不增加参数维度，采用了跨通道池化的方法，与maxoutconv保持通用
'''
import time
from compiler.ast import flatten
from copy import copy

import numpy as np
import theano.tensor as T
from sklearn.cross_validation import KFold
from theano.tensor.nnet import categorical_crossentropy

from load import mnist
from utils import basicUtils, gradient, initial, preprocess


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# maxout激活函数，reshape的顺序是按行的，所以需要先转置
# 输入（样本数，输入特征数）* maxout层（输入数，输出数*分段数）=（样本数，输出数*分段数）
# 输出（样本数，输出数）
def maxout(X, hidden, piece):
    Xt = T.transpose(X)  # （输出数*分段数，样本数）
    Xr = T.reshape(Xt, (hidden, piece, -1))  # （输出数，分段数，样本数）
    Xmax = T.max(Xr, axis=1)  # （输出数，样本数）
    return T.transpose(Xmax)  # （样本数，输出数）


# 将maxout等效为一般的MLP使用同样的参数维度
def layerMLPParams(shape):
    w = initial.weightInitMLP3(shape, 'w')
    b = initial.biasInit(shape[1], 'b')
    return [w, b]


def model(X, params, hiddens, pieces, pDropHidden1, pDropHidden2):
    lnum = 0
    layer = T.dot(X, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0)
    layer = maxout(layer, hiddens[lnum], pieces[lnum])
    layer = basicUtils.dropout(layer, pDropHidden1)
    lnum += 1
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0)
    layer = maxout(layer, hiddens[lnum], pieces[lnum])
    layer = basicUtils.dropout(layer, pDropHidden2)
    lnum += 1
    return softmax(T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0))  # 如果使用nnet中的softmax训练产生NAN


class CMaxoutmlp(object):
    def __init__(self, fin, h1, piece1, h2, piece2, outputs,
                 lr, C, pDropHidden1=0.2, pDropHidden2=0.5):
        # 超参数
        self.lr = lr
        self.C = C
        self.pDropHidden1 = pDropHidden1
        self.pDropHidden2 = pDropHidden2
        # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        self.params = []
        hiddens = []
        pieces = []
        # maxout层，指定piece表示分段线性函数的段数，即使用隐隐层的个数，维度与一般MLP相同，使用跨通道最大池化
        self.params.append(layerMLPParams((fin, h1 * piece1)))
        hiddens.append(h1)
        pieces.append(piece1)
        self.params.append(layerMLPParams((h1, h2 * piece2)))
        hiddens.append(h2)
        pieces.append(piece2)
        self.params.append(layerMLPParams((h2, outputs)))

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.matrix('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProb = model(self.X, self.params, hiddens, pieces, pDropHidden1, pDropHidden2)
        self.trNeqs = basicUtils.neqs(YDropProb, self.Y)
        trCrossEntropy = categorical_crossentropy(YDropProb, self.Y)
        self.trCost = T.mean(trCrossEntropy) + C * basicUtils.regularizer(flatten(self.params))

        # 测试验证集代价函数
        YFullProb = model(self.X, self.params, hiddens, pieces, 0., 0.)
        self.vateNeqs = basicUtils.neqs(YFullProb, self.Y)
        self.YPred = T.argmax(YFullProb, axis=1)
        vateCrossEntropy = categorical_crossentropy(YFullProb, self.Y)
        self.vateCost = T.mean(vateCrossEntropy) + C * basicUtils.regularizer(flatten(self.params))

    # 重置优化参数，以重新训练模型
    def resetParams(self):
        for p in self.params:
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
    trX, teX, trY, teY = mnist(onehot=True)
    h1, hpiece1, h2, hpiece2 = 625, 5, 625, 5
    params = basicUtils.randomSearch(nIter=10)
    cvErrorList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 40, num, 'parameters', param, '*' * 40
        maxout = CMaxoutmlp(28 * 28, h1, hpiece1, h2, hpiece2, 10, lr, C, 0.2, 0.5)
        cvError = maxout.cv(trX, trY)
        cvErrorList.append(copy(cvError))
    optIndex = np.argmin(cvErrorList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    maxout = CMaxoutmlp(28 * 28, h1, hpiece1, h2, hpiece2, 10, lr, C, 0.2, 0.5)
    maxout.trainmaxout(trX, teX, trY, teY)


if __name__ == '__main__':
    main()
