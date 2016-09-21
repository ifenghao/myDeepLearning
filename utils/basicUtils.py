# coding:utf-8
__author__ = 'zfh'
import cPickle
import os
from copy import copy
from random import uniform

import numpy as np
import theano
from theano import shared, function, In, Out
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

'''
theano相关的操作
'''


# 转化为GPU格式的浮点数
def floatX(x):
    return x.astype(theano.config.floatX)


# 将数据集直接复制到GPU内存，减少每次调用mini-batch时从CPU复制到GPU的时间消耗
def datasetShared(X, Y, Xname, Yname):
    XShared = shared(floatX(X), name=Xname, borrow=True)
    YShared = shared(floatX(Y), name=Yname, borrow=True)
    return XShared, YShared


def makeFunc(inList, outList, updates):
    inputs = []
    for i in inList:
        inputs.append(In(i, borrow=True, allow_downcast=True))
    outputs = []
    for o in outList:
        outputs.append(Out(o, borrow=True))
    return function(
        inputs=inputs,
        outputs=outputs,  # 减少返回参数节省时间
        updates=updates,
        allow_input_downcast=True
    )


def convOutputShape(inputShape, filterShape, border_mode, stride):
    mapRow, mapCol = inputShape
    filterRow, filterCol = filterShape
    rowStride, colStride = stride
    if border_mode == 'half':
        mapRow += 2 * (filterRow // 2)
        mapCol += 2 * (filterCol // 2)
    elif border_mode == 'full':
        mapRow += 2 * (filterRow - 1)
        mapCol += 2 * (filterCol - 1)
    outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
    return outRow, outCol


def poolOutputShape(inputShape, poolSize, ignore_border=False, stride=None):
    if stride is None:
        stride = poolSize
    mapRow, mapCol = inputShape
    poolRow, poolCol = poolSize
    rowStride, colStride = stride
    outRow, outCol = (mapRow - poolRow) // rowStride + 1, (mapCol - poolCol) // colStride + 1
    if not ignore_border:
        if (mapRow - poolRow) % rowStride:
            outRow += 1
        if (mapCol - poolCol) % colStride:
            outCol += 1
    return outRow, outCol


# pad的顺序依次：上下左右
def pad2d(X, padding=(0, 0, 0, 0)):
    inputShape = X.shape
    outputShape = (inputShape[0],
                   inputShape[1],
                   inputShape[2] + padding[0] + padding[1],
                   inputShape[3] + padding[2] + padding[3])
    output = T.zeros(outputShape)
    indices = (slice(None),
               slice(None),
               slice(padding[0], inputShape[2] + padding[0]),  # 上下
               slice(padding[2], inputShape[3] + padding[2]))  # 左右
    return T.set_subtensor(output[indices], X)


def genIndex(XShape, filterShape, border_mode='valid', stride=(1, 1)):
    nSample, nMap, mapRow, mapCol = XShape
    _, _, filterRow, filterCol = filterShape
    rowStride, colStride = stride
    outRow, outCol = convOutputShape((mapRow, mapCol), (filterRow, filterCol), border_mode, stride)
    block1 = np.arange(filterCol, dtype='int')
    block2 = []
    for i in xrange(filterRow):
        block2.append(block1 + i * mapCol)
    block2 = np.hstack(block2)
    block3 = []
    for i in xrange(outCol):
        block3.append(block2 + i * colStride)
    block3 = np.hstack(block3)
    block4 = []
    for i in xrange(outRow):
        block4.append(block3 + i * mapCol * rowStride)
    block4 = np.hstack(block4)
    out = []
    for i in xrange(nSample * nMap):
        out.append(block4 + i * mapRow * mapCol)
    return np.hstack(out).astype('int')


def convdot2d(X, f, index, border_mode='valid', stride=(1, 1), filter_flip=True):
    nSample, nMap, mapRow, mapCol = X.shape
    _, _, filterRow, filterCol = f.shape
    outRow, outCol = convOutputShape((mapRow, mapCol), (filterRow, filterCol), border_mode, stride)
    if border_mode == 'half':
        X = pad2d(X, (filterRow // 2, filterRow // 2, filterCol // 2, filterCol // 2))
    elif border_mode == 'full':
        X = pad2d(X, (filterRow - 1, filterRow - 1, filterCol - 1, filterCol - 1))
    X = T.flatten(X, outdim=1)
    X = X[index].reshape((nSample, nMap, outRow, outCol, filterRow, filterCol))
    if filter_flip:
        f = f[:, :, ::-1, ::-1]
    out = T.tensordot(X, f, axes=[[1, X.ndim - 2, X.ndim - 1], [1, f.ndim - 2, f.ndim - 1]])
    return out.dimshuffle(0, 3, 1, 2)


'''
网络结构中需要计算的参量
'''


# 使用GPU时误差的计算，输入都必须是TensorType
def neqs(YProb, Y):
    if YProb.ndim != Y.ndim:
        raise TypeError('dim mismatch', (YProb.type, Y.type))
    YProb = T.argmax(YProb, axis=1)
    Y = T.argmax(Y, axis=1)
    return T.sum(T.neq(YProb, Y))  # 返回不相等元素个数


# 正则项
def regularizer(pramsIter):
    elementNum = 0
    regSum = shared(0., borrow=True)
    for p in pramsIter:
        regSum += T.sum(p ** 2)
        elementNum += np.prod(p.get_value(borrow=True).shape)
    return regSum / elementNum


# binomial生成与X维度一样的随机分布的0-1矩阵，其中1的比例为p，且每次调用生成的矩阵都不同
def dropout(X, pDrop=0.):
    if pDrop > 0:
        pRetain = 1 - pDrop
        X *= srng.binomial(X.shape, p=pRetain, dtype=theano.config.floatX)
        X /= pRetain
    return X


# 局部相应归一化
def LRN(X, alpha=0.0001, k=1, beta=0.75, n=5):
    b, ch, r, c = X.shape
    nHalf = n // 2
    Xsqr = T.sqr(X)
    Xextend = T.zeros((b, ch + 2 * nHalf, r, c), dtype=theano.config.floatX)
    Xsqr = T.set_subtensor(Xextend[:, nHalf:ch + nHalf, :, :], Xsqr)
    alphaNorm = alpha / n
    scale = k
    for i in xrange(n):
        scale += alphaNorm * Xsqr[:, i:i + ch, :, :]
    scale = scale ** beta
    X = X / scale
    return X


'''
训练过程中的方法
'''


def miniBatch(X, Y, func, batchSize=128, verbose=True):
    '''
    分批训练训练集
    :param X: 训练数据X
    :param Y: 训练数据Y
    :param func: theano编译的训练函数
    :param batchSize: 分批的大小，控制着参数的更新次数
    :param verbose: 是否输出训练信息
    :return: 每批训练损失的平均
    '''
    size = X.shape[0]
    costList = []
    neqAcc = 0
    startRange = range(0, size - batchSize, batchSize)
    endRange = range(batchSize, size, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        cost, neq = func(X[start:end], Y[start:end])
        costList.append(copy(cost))
        neqAcc += neq
    costMean, error = np.mean(costList), float(neqAcc) / size
    if verbose: print 'costMean: %8.5f   error: %.5f' % (costMean, error),
    return costMean, error


def miniBatchMO(X, Y, func, batchSize=128, verbose=True):
    '''
    多输出，用于googLeNet
    :param X: 训练数据X
    :param Y: 训练数据Y
    :param func: theano编译的训练函数
    :param batchSize: 分批的大小，控制着参数的更新次数
    :param verbose: 是否输出训练信息
    :return: 每批训练损失的平均
    '''
    size = X.shape[0]
    costList = []
    neqMainAcc = 0
    neqAux1Acc = 0
    neqAux2Acc = 0
    startRange = range(0, size - batchSize, batchSize)
    endRange = range(batchSize, size, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        cost, neqMain, neqAux1, neqAux2 = func(X[start:end], Y[start:end])
        costList.append(copy(cost))
        neqMainAcc += neqMain
        neqAux1Acc += neqAux1
        neqAux2Acc += neqAux2
    costMean = np.mean(costList)
    errorMain = float(neqMainAcc) / size
    errorAux1 = float(neqAux1Acc) / size
    errorAux2 = float(neqAux2Acc) / size
    if verbose: print 'costMean: %8.5f   errors: %.5f  %.5f  %.5f' % (costMean, errorMain, errorAux1, errorAux2)
    return costMean, errorMain, errorAux1, errorAux2


def earlyStopGen(start=5, period=3, threshold=10, tol=2):
    '''
    早停止生成器，生成器可以保存之前传入的参数，从而在不断send入参数的过程中判断是否早停止训练
    :param start: 开始检测早停止的epoch，即至少完成多少epoch后才可以早停止
    :param period: 监视早停止标志的周期，每period个epoch计算一次stopMetric
    :param threshold: stopMetric的阈值，超过此阈值则早停止标志计数一次
    :param tol: 早停止标志计数的容忍限度，当计数超过此限度则立即执行早停止
    :return: 是否执行早停止
    '''
    trCostPeriod = []
    vaCostPeriod = []
    vaCostOpt = np.inf
    epoch = 0
    stopSign = False
    stopCount = 0
    while True:
        newCosts = (yield stopSign)  # 返回是否早停止
        epoch += 1
        if stopSign:  # 返回一个早停止标志后，重新检测
            stopSign = False
            stopCount = 0
        if epoch > start and newCosts is not None:  # send进来的元组在newCosts中
            trCost, vaCost = newCosts
            trCostPeriod.append(trCost)
            vaCostPeriod.append(vaCost)
            if vaCost < vaCostOpt:
                vaCostOpt = vaCost
            if len(trCostPeriod) >= period:
                P = np.mean(trCostPeriod) / np.min(trCostPeriod) - 1
                GL = np.mean(vaCostPeriod) / vaCostOpt - 1
                stopMetric = GL / P  # 停止的度量策略
                if stopMetric >= threshold:
                    stopCount += 1
                    if stopCount >= tol:
                        stopSign = True
                trCostPeriod = []  # 清空列表以继续判断下个周期
                vaCostPeriod = []


def randomSearch(nIter):
    '''
    随机生成超参数组合搜索最优结果
    :param nIter: 迭代次数，即超参数组合个数
    :return: 超参数组合的二维列表
    '''
    lr = [uniform(1, 20) * 1e-3 for _ in range(nIter)]
    C = [uniform(1, 500) * 1e-1 for _ in range(nIter)]
    # pDropConv = [uniform(0., 0.3) for _ in range(nIter)]
    # pDropHidden = [uniform(0., 0.5) for _ in range(nIter)]
    return zip(lr, C)


# 保存网络参数
def dumpModel(convNet):
    modelPickleFile = os.path.join(os.getcwd(), 'convNet' + '.pkl')
    with open(modelPickleFile, 'w') as file:
        cPickle.dump(convNet.params, file)
