# coding:utf-8
__author__ = 'zfh'
import theano
from theano import shared, function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
import pylab

srng = RandomStreams()

# 转化为GPU格式的浮点数
def floatX(x):
    return x.astype(theano.config.floatX)


# 参数初始化，在这里不能修改参数的broadcastable，否则会引起更新参数值TensorType{float32,4D}与原始值的类型cudaNdarray{float32,broadcastable}不匹配
def weightInit(shape, name):
    return shared(floatX(np.random.randn(*shape) * 0.01), name=name, borrow=True)


def weightInitConv(shape, poolSize, name):
    fanIn = np.prod(shape[1:])
    fanOut = shape[0] * np.prod(shape[2:]) / poolSize
    var = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(np.random.randn(*shape) * var), name=name, borrow=True)


def weightInitMLP(shape, name):
    fanIn = shape[0]
    fanOut = shape[1]
    var = np.sqrt(6. / (fanIn + fanOut))
    return shared(floatX(np.random.randn(*shape) * var), name=name, borrow=True)


def biasInit(shape, name):
    return shared(floatX(np.zeros(*shape)), name=name, borrow=True)


# 正则项
def reg(pramsIter):
    elementNum = 0
    regSum = shared(0., borrow=True)
    for p in pramsIter:
        regSum += T.sum(p ** 2)
        elementNum += np.prod(p.get_value(borrow=True, return_internal_type=True).shape)
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
        acc = shared(p.get_value() * 0., borrow=True)
        accNew = rho * acc + (1 - rho) * g ** 2
        gradScaling = T.sqrt(accNew + epsilon)
        g = g / gradScaling
        updates.append((acc, accNew))
        updates.append((p, p - lr * g))
    return updates


# 网络可视化
def modelFlow(X, prams):
    lconv1 = T.nnet.relu(conv2d(X, prams[0][0], border_mode='full') +
                         prams[0][1].dimshuffle('x', 0, 'x', 'x'))
    lds1 = max_pool_2d(lconv1, (2, 2))

    lconv2 = T.nnet.relu(conv2d(lds1, prams[1][0]) +
                         prams[1][1].dimshuffle('x', 0, 'x', 'x'))
    lds2 = max_pool_2d(lconv2, (2, 2))

    lconv3 = T.nnet.relu(conv2d(lds2, prams[2][0]) +
                         prams[2][1].dimshuffle('x', 0, 'x', 'x'))
    lds3 = max_pool_2d(lconv3, (2, 2))
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


def listFeatureMap(inputX, prams):
    X = T.tensor4('X')
    featureMaps = modelFlow(X, prams)
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
