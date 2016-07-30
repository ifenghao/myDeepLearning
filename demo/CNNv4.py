# coding:utf-8
__author__ = 'zfh'
'''
训练技巧：
1、使用线性修正单元（relu）作为激活函数
2、每一层加入dropout
3、参数更新方式采用rmsprop
4、使用mini-batch分批训练
5、使用borrow=True属性
6、基于v1的结构增加了验证集上的超参数选取
'''
from compiler.ast import flatten
import time
import theano.tensor as T
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d
from sklearn.cross_validation import KFold
from copy import copy
import numpy as np
from load import cifar
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def layerParams(shape):
    w = utils.weightInit3(shape, 'w')
    b = utils.biasInit(shape, 'b')
    return [w, b]


# 模型构建，返回给定样本判定为某类别的概率
# dimshuffle在偏置插入维度使之与相加矩阵相同（1，本层特征图个数，1，1），插入维度的broadcastable=True
# 每次调用dropout的模式都不同，即在每轮训练中网络结构都不同
# 本层的每个特征图和上层的所有特征图连接，可以不用去选择一些组合来部分连接
def model(X, params, pDropConv, pDropHidden):
    # conv: (32, 32) = (32, 32)
    # pool: (32/2, 32/2) = (16, 16)
    lnum = 0
    layer = conv2d(X, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    # conv: (16, 16) = (16, 16)
    lnum += 1
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropConv)
    # conv: (32, 32) = (32, 32)
    # pool: (32-1, 32-1) = (31, 31)
    lnum += 1
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    # conv: (32, 32) = (32, 32)
    lnum += 1
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropConv)
    # conv: (32, 32) = (32, 32)
    # pool: (32-1, 32-1) = (31, 31)
    lnum += 1
    layer = conv2d(layer, params[lnum][0], border_mode='half') + \
            params[lnum][1].dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    # 128
    lnum += 1
    layer = T.flatten(layer, outdim=2)
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1]
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropHidden)
    lnum += 1
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1]
    layer = relu(layer, alpha=0)
    layer = utils.dropout(layer, pDropHidden)
    lnum += 1
    return softmax(T.dot(layer, params[lnum][0]) + params[lnum][1])  # 如果使用nnet中的softmax训练产生NAN


# 卷积网络，输入一组超参数，返回该网络的训练、验证、预测函数
class CConvNet(object):
    def __init__(self, fin, f1, f2, f3, f4, f5, h1, h2, outputs,
                 lr, C, pDropConv=0., pDropHidden=0.5):
        # 超参数
        self.lr = lr
        self.C = C
        self.pDropConv = pDropConv
        self.pDropHidden = pDropHidden
        # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        self.params = []
        # 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
        self.params.append(layerParams((f1, fin, 3, 3)))
        self.params.append(layerParams((f2, f1, 3, 3)))
        self.params.append(layerParams((f3, f2, 3, 3)))
        self.params.append(layerParams((f4, f3, 3, 3)))
        self.params.append(layerParams((f5, f4, 3, 3)))
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        self.params.append(layerParams((f5 * 4 * 4, h1)))
        self.params.append(layerParams((h1, h2)))
        self.params.append(layerParams((h2, outputs)))

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.tensor4('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProb = model(self.X, self.params, pDropConv, pDropHidden)
        self.trNeqs = utils.neqs(YDropProb, self.Y)
        trCrossEntropy = categorical_crossentropy(YDropProb, self.Y)
        self.trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(self.params))

        # 测试验证集代价函数
        YFullProb = model(self.X, self.params, 0., 0.)
        self.vateNeqs = utils.neqs(YFullProb, self.Y)
        self.YPred = T.argmax(YFullProb, axis=1)
        vateCrossEntropy = categorical_crossentropy(YFullProb, self.Y)
        self.vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(self.params))

    # 训练卷积网络，最终返回在测试集上的误差
    def traincn(self, trX, teX, trY, teY, batchSize=128, maxIter=100, verbose=True,
                start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=3):
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
            trCost = utils.miniBatch(trX, trY, train, batchSize, verbose)
            teCost = utils.miniBatch(teX, teY, valtest, batchSize, verbose)
            if verbose: print ' time: %10.5f' % (time.time() - startTime)
            if earlyStop.send((trCost, teCost)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
                totalStopCount += 1
                if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                if verbose: print 'learning rate decreases to ', lr

    # 交叉验证得到一组平均验证误差，使用早停止
    def cv(self, X, Y, folds=3, batchSize=128, maxIter=100, verbose=True,
           start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=3):
        # 训练集分为n折交叉验证集
        kf = KFold(X.shape[0], n_folds=folds, random_state=42)
        vaCostList = []
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
            vaCostOpt = np.inf
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost = utils.miniBatch(trX, trY, train, batchSize, verbose)
                vaCost = utils.miniBatch(vaX, vaY, valtest, batchSize, verbose)
                if vaCost < vaCostOpt: vaCostOpt = vaCost
                if verbose: print ' time: %10.5f' % (time.time() - startTime)
                if earlyStop.send((trCost, vaCost)):
                    lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                    updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                    train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
                    totalStopCount += 1
                    if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                    if verbose: print 'learning rate decreases to ', lr
            vaCostList.append(copy(vaCostOpt))
            if verbose: print '*' * 10, 'one validation done, best vaCost', vaCostList, '*' * 10
            self.resetPrams()
        return np.mean(vaCostList)

    # 重置优化参数，以重新训练模型
    def resetPrams(self):
        for p in self.params:
            utils.resetWeight3(p[0])
            utils.resetBias(p[1])


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    trX, teX, trY, teY = cifar(onehot=True)
    f1, f2, f3, f4, f5, h1, h2 = 32, 64, 64, 128, 128, 1024, 1024
    params = utils.randomSearch(nIter=10)
    cvCostList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 40, num + 1, 'parameters', param, '*' * 40
        convNet = CConvNet(3, f1, f2, f3, f4, f5, h1, h2, 10, lr, C, 0, 0.5)
        cvCost = convNet.cv(trX, trY)
        cvCostList.append(copy(cvCost))
    optIndex = np.argmin(cvCostList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    convNet = CConvNet(3, f1, f2, f3, f4, f5, h1, h2, 10, lr, C, 0, 0.5)
    convNet.traincn(trX, teX, trY, teY)


if __name__ == '__main__':
    main()
