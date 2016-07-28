# coding:utf-8
__author__ = 'zfh'
import theano
from theano import shared, function, In, Out
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal.pool import pool_2d
from random import uniform, randint
import numpy as np
import pylab
import time
from copy import copy

srng = RandomStreams()
rng = np.random.RandomState(23455)

# 转化为GPU格式的浮点数
def floatX(x):
    return x.astype(theano.config.floatX)


# 将数据集直接复制到GPU内存，减少每次调用mini-batch时从CPU复制到GPU的时间消耗
def datasetShared(X, Y, Xname, Yname):
    XShared = shared(floatX(X), name=Xname, borrow=True)
    YShared = shared(floatX(Y), name=Yname, borrow=True)
    return XShared, YShared


# 图像预处理，只进行零均值化和归一化，在训练集上计算RGB三个通道每个位置的均值，分别在训练、验证、测试集上减去
def preprocess(trX, vateX):
    avg = np.mean(trX, axis=0, dtype=theano.config.floatX, keepdims=True)
    std = np.std(trX, axis=0, dtype=theano.config.floatX, keepdims=True)
    return (trX - avg) / std, (vateX - avg) / std


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


# 参数初始化，在这里不能修改参数的broadcastable
# 否则会引起更新参数值TensorType{float32,4D}与原始值的类型cudaNdarray{float32,broadcastable}不匹配
def weightInit(shape, name):
    return shared(floatX(rng.randn(*shape) * 0.1), name=name, borrow=True)


# 第二种参数初始化方法仅适用于零均值的输入情况，否则梯度下降很慢
def weightInit2(shape, name):
    if len(shape) > 2:
        fanIn = np.prod(shape[1:])
        fanOut = shape[0] * np.prod(shape[2:])
    else:
        fanIn = shape[0]
        fanOut = shape[1]
    bound = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(rng.uniform(-bound, bound, shape)), name=name, borrow=True)


# 第三种参数初始化方法仅适用于零均值的输入，且使用ReLU神经元的情况
def weightInit3(shape, name):
    if len(shape) > 2:
        fanIn = np.prod(shape[1:])
    else:
        fanIn = shape[0]
    bound = np.sqrt(2. / fanIn)
    return shared(floatX(rng.randn(*shape) * bound), name=name, borrow=True)


def biasInit(shape, name):
    if len(shape) > 2:
        bais = shared(floatX(np.zeros(shape[0])), name=name, borrow=True)
    else:
        bais = shared(floatX(np.zeros(shape[1])), name=name, borrow=True)
    return bais


def resetWeight(w):
    wShape = w.get_value(borrow=True).shape
    w.set_value(floatX(rng.randn(*wShape) * 0.1), borrow=True)


def resetWeight2(w):
    shape = w.get_value(borrow=True).shape
    if len(shape) > 2:
        fanIn = np.prod(shape[1:])
        fanOut = shape[0] * np.prod(shape[2:])
    else:
        fanIn = shape[0]
        fanOut = shape[1]
    bound = np.sqrt(6. / (fanIn + fanOut))
    w.set_value(floatX(rng.uniform(-bound, bound, shape)), borrow=True)


def resetWeight3(w):
    shape = w.get_value(borrow=True).shape
    if len(shape) > 2:
        fanIn = np.prod(shape[1:])
    else:
        fanIn = shape[0]
    bound = np.sqrt(2. / fanIn)
    w.set_value(floatX(rng.randn(*shape) * bound), borrow=True)


def resetBias(b):
    bShape = b.get_value(borrow=True).shape
    b.set_value(floatX(np.zeros(*bShape)), borrow=True)


# 使用GPU时误差的计算，输入都必须是TensorType
def neqs(YProb, Y):
    if YProb.ndim != Y.ndim:
        raise TypeError('dim mismatch', (YProb.type, Y.type))
    YProb = T.argmax(YProb, axis=1)
    Y = T.argmax(Y, axis=1)
    return T.sum(T.neq(YProb, Y))  # 返回不相等元素个数


# 正则项
def reg(pramsIter):
    elementNum = 0
    regSum = shared(0., borrow=True)
    for p in pramsIter:
        regSum += T.sum(p ** 2)
        elementNum += np.prod(p.get_value(borrow=True).shape)
    return regSum / elementNum


# 随机梯度下降得到权重更新
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


# 增加动量项
def sgd_momentum(cost, params, momentum, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        pnew = shared(p.get_value() * 0., broadcastable=p.broadcastable)
        updates.append([p, p - pnew * lr])
        updates.append([pnew, momentum * pnew + (1. - momentum) * g])
    return updates


# binomial生成与X维度一样的随机分布的0-1矩阵，其中1的比例为p，且每次调用生成的矩阵都不同
def dropout(X, pDrop=0.):
    if pDrop > 0:
        pRetain = 1 - pDrop
        X *= srng.binomial(X.shape, p=pRetain, dtype=theano.config.floatX)
        X /= pRetain
    return X


# rmsprop得到权重更新
def rmsprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 加权累加器
        accNew = rho * acc + (1 - rho) * g ** 2
        gradScaling = T.sqrt(accNew + epsilon)
        g = g / gradScaling
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


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
    return costMean


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
    lr = [uniform(5, 50) * 1e-4 for _ in range(nIter)]
    C = [uniform(1, 100) * 1e-1 for _ in range(nIter)]
    # pDropConv = [uniform(0., 0.3) for _ in range(nIter)]
    # pDropHidden = [uniform(0., 0.5) for _ in range(nIter)]
    return zip(lr, C)


# 网络可视化
def modelFlow(X, params):
    lconv1 = relu(conv2d(X, params[0][0], border_mode='full') +
                  params[0][1].dimshuffle('x', 0, 'x', 'x'))
    lds1 = pool_2d(lconv1, (2, 2))

    lconv2 = relu(conv2d(lds1, params[1][0]) +
                  params[1][1].dimshuffle('x', 0, 'x', 'x'))
    lds2 = pool_2d(lconv2, (2, 2))

    lconv3 = relu(conv2d(lds2, params[2][0]) +
                  params[2][1].dimshuffle('x', 0, 'x', 'x'))
    lds3 = pool_2d(lconv3, (2, 2))
    return X, lconv1, lds1, lconv2, lds2, lconv3, lds3


def addPad(map2D, padWidth):
    row, col = map2D.shape
    hPad = np.zeros((row, padWidth))
    map2D = np.hstack((hPad, map2D, hPad))
    vPad = np.zeros((padWidth, col + 2 * padWidth))
    map2D = np.vstack((vPad, map2D, vPad))
    return map2D


def squareStack(map3D):
    mapNum = map3D.shape[0]
    row, col = map3D.shape[1:]
    side = int(np.ceil(np.sqrt(mapNum)))
    lack = side ** 2 - mapNum
    map3D = np.vstack((map3D, np.zeros((lack, row, col))))
    map2Ds = [addPad(map3D[i], 2) for i in range(side ** 2)]
    return np.vstack([np.hstack(map2Ds[i:i + side])
                      for i in range(0, side ** 2, side)])


def listFeatureMap(inputX, params):
    X = T.tensor4('X')
    featureMaps = modelFlow(X, params)
    makeMap = function([X], featureMaps, allow_input_downcast=True)
    return makeMap(inputX)


def showFeatureMap(featureMaps):
    batchSize = featureMaps[0].shape[0]
    for i in range(batchSize):
        mapList = []
        mapList.append(featureMaps[0][i][0])
        for j in range(1, len(featureMaps)):
            mapList.append(squareStack(featureMaps[j][i]))
        pylab.figure(i)
        subRow = 2
        subCol = len(mapList) // subRow + 1
        pylab.subplot(subRow, subCol, 1)  # 显示原始图片
        pylab.imshow(mapList[0])
        pylab.gray()
        plotPlace = 2
        for k in range(1, len(mapList), 2):
            pylab.subplot(subRow, subCol, plotPlace)  # 显示卷积特征图
            pylab.imshow(mapList[k])
            pylab.gray()
            pylab.subplot(subRow, subCol, plotPlace + subCol)
            pylab.imshow(mapList[k + 1])
            pylab.gray()
            plotPlace += 1
    pylab.show()
