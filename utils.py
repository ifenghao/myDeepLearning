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
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import cPickle, os

srng = RandomStreams()
rng = np.random.RandomState(23455)

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
图像预处理，只进行零均值化和归一化，在训练集上计算RGB三个通道每个位置的均值，分别在训练、验证、测试集上减去
不用归一化有时候会出现nan，即计算的数值太大
如果要使用标准差归一化，要注意有的位置上标准差为0，容易产生nan
'''


def preprocess(trX, vateX):
    avg = np.mean(trX, axis=0, dtype=theano.config.floatX, keepdims=True)
    return (trX - avg) / 128, (vateX - avg) / 128


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
    fanIn = np.prod(shape[1:])
    fanOut = shape[0] * np.prod(shape[2:])
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
    fanIn = np.prod(shape[1:])
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
def reg(pramsIter):
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


'''
随机梯度下降及其各种变形，为了得到权重更新
'''


def sgd(cost, params, lr=0.01):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


# 下降速度较快，但是学习率较大会发散
def sgdm(cost, params, lr=0.01, momentum=0.9, nesterov=False):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0., borrow=True)
        updates.append([v, momentum * v - lr * g])
        if nesterov:
            updates.append([p, p + momentum * v - lr * g])
        else:
            updates.append([p, p + v])
    return updates


# 较不容易发散，但是下降速度很慢
def sgdma(cost, params, lr=0.01, momentum=0.9):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        v = shared(p.get_value() * 0., borrow=True)
        updates.append([v, momentum * v + (1. - momentum) * g])
        updates.append([p, p - lr * v])
    return updates


def adagrad(cost, params, lr=0.01, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 加权累加器
        accNew = acc + T.square(g)
        g = g / T.sqrt(accNew + epsilon)
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


def adadelta(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 梯度加权累加器
        accDelta = shared(p.get_value() * 0., borrow=True)  # 更新加权累加器
        accNew = rho * acc + (1 - rho) * T.square(g)  # 梯度加权累加
        delta = g * T.sqrt(accDelta + epsilon) / T.sqrt(accNew + epsilon)  # 新的梯度累加器，旧的更新累加器
        accDeltaNew = rho * accDelta + (1 - rho) * T.square(delta)  # 更新加权累加
        updates.append((acc, accNew))
        updates.append((p, p - lr * delta))
        updates.append((accDelta, accDeltaNew))
    return updates


def rmsprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = shared(p.get_value() * 0., borrow=True)  # 加权累加器
        accNew = rho * acc + (1 - rho) * T.square(g)
        g = g / T.sqrt(accNew + epsilon)
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


def adam(cost, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mt = shared(p.get_value() * 0., borrow=True)
        vt = shared(p.get_value() * 0., borrow=True)
        mtNew = (beta1 * mt) + (1. - beta1) * g
        vtNew = (beta2 * vt) + (1. - beta2) * T.square(g)
        pNew = p - lr * mtNew / (T.sqrt(vtNew) + epsilon)
        updates.append((mt, mtNew))
        updates.append((vt, vtNew))
        updates.append((p, pNew))
    return updates


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


'''
辅助方法
'''


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


# 保存网络参数
def dumpModel(convNet):
    modelPickleFile = os.path.join(os.getcwd(), 'convNet' + '.pkl')
    with open(modelPickleFile, 'w') as file:
        cPickle.dump(convNet.params, file)


# 参数选择可视化
def surface(x, y, z):
    ax = pylab.axes(projection='3d')
    xx, yy = np.meshgrid(x, y)
    ax.plot_surface(xx, yy, z)
    pylab.show()


def scatter(x, y, z):
    ax = pylab.axes(projection='3d')
    ax.scatter(x, y, z, c=z)
    pylab.show()
