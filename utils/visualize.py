# coding:utf-8
__author__ = 'zfh'
import numpy as np
import pylab
from theano import function
from theano import tensor as T
from theano.tensor.nnet import conv2d, relu
from theano.tensor.signal.pool import pool_2d


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
