# coding:utf-8
__author__ = 'zfh'
'''
32滤波器产生32个特征图，32个按列全连接产生下一层的2个特征图
顶层使用全局平均池化层
'''
from compiler.ast import flatten
import time
from copy import copy
import theano

import theano.tensor as T
from theano import scan, function
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu

from theano.tensor.signal.pool import pool_2d
import pylab
import numpy as np
from load import cifar
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# def nin(X, param, shape):
#     w1, w2 = param
#     for i in xrange(64):
#         for j in xrange(32):
#             (n,1,r,c)**(1,1,3,3)=(n,1,r,c)
#             relu
#         concatenate(32*(n,1,r,c), axis=1)
#         (n,32,r,c)**(2,32,1,1)=(n,2,r,c)
#         relu
#     return concatenate(64*(n,2,r,c), axis=1)

# scan的一次元操作
def metaOp1(i, j, X, w1, b1):
    # (n,1,r,c)**(1,1,3,3)=(n,1,r,c)
    hiddens = conv2d(X[:, j, :, :, :], w1[i, j, :, :, :, :], border_mode='half') + b1[i, j, :, :, :, :]
    hiddens = T.nnet.relu(hiddens, alpha=0)
    return hiddens


def metaOp2(i, X, w2, b2):
    # (n,32,r,c)**(2,32,1,1)=(n,2,r,c)
    hiddens = conv2d(X[i, :, :, :, :], w2[i, :, :, :, :], border_mode='valid') + b2[i, :, :, :, :]
    hiddens = T.nnet.relu(hiddens, alpha=0)
    return hiddens


def nin(X, param):
    w1, w2, b1, b2 = param
    X = X.dimshuffle(0, 1, 'x', 2, 3)  # (n,32,1,r,c)
    w1 = w1.dimshuffle(0, 1, 'x', 'x', 2, 3)  # (64,32,1,1,3,3)
    w2 = w2.dimshuffle(0, 1, 2, 'x', 'x')  # (64,2,32,1,1)
    b1 = b1.dimshuffle(0, 1, 'x', 'x', 'x', 'x')  # (64,32,1,1,1,1)
    b2 = b2.dimshuffle(0, 'x', 1, 'x', 'x')  # (64,1,2,1,1)
    indexi = T.arange(w1.shape[0], dtype='int32')  # (0:64)
    indexi = T.repeat(indexi, w1.shape[1], axis=0)
    indexj = T.arange(w1.shape[1], dtype='int32')  # (0:64)
    indexj = T.tile(indexj, w1.shape[0])
    results, updates = scan(fn=metaOp1,
                            sequences=[indexi, indexj],
                            outputs_info=None,
                            non_sequences=[X, w1, b1],
                            strict=True)  # (64*32,n,1,r,c)
    metaShape1 = results.shape[-4], results.shape[-2], results.shape[-1]
    reshaped1 = results.reshape((w1.shape[0], w1.shape[1]) + metaShape1)  # (64,32,n,r,c)
    permuted1 = T.transpose(reshaped1, axes=(0, 2, 1, 3, 4))  # (64,n,32,r,c)
    indexi = T.arange(w1.shape[0], dtype='int32')  # (0:64)
    results, updates = scan(fn=metaOp2,
                            sequences=[indexi],
                            outputs_info=None,
                            non_sequences=[permuted1, w2, b2],
                            strict=True)  # (64,n,2,r,c)
    permuted2 = T.transpose(results, axes=(1, 0, 2, 3, 4))  # (n,64,2,r,c)
    metaShape2 = permuted2.shape[-2], permuted2.shape[-1]
    reshaped2 = permuted2.reshape((permuted2.shape[0], -1) + metaShape2)  # (n,128,r,c)
    return reshaped2


def gap(X):
    layer = T.mean(X, axis=(2, 3))
    return layer


def conv1t1(X, param):
    wconv, bconv = param
    layer = conv2d(X, wconv, border_mode='valid') + bconv.dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    return layer


# 模型构建，返回给定样本判定为某类别的概率
def model(X, params, pDropConv, pDropHidden):
    lnum = 0  # conv: (32, 32) pool: (16, 16)
    layer = nin(X, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    lnum += 1  # conv: (16, 16) pool: (8, 8)
    layer = nin(layer, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    lnum += 1  # conv: (8, 8) pool: (4, 4)
    layer = nin(layer, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = utils.dropout(layer, pDropConv)
    lnum += 1
    layer = conv1t1(layer, params[lnum])
    layer = utils.dropout(layer, pDropHidden)
    lnum += 1
    layer = conv1t1(layer, params[lnum])
    layer = gap(layer)
    return softmax(layer)  # 如果使用nnet中的softmax训练产生NAN


# 常量
iterSteps = 100
lr = 0.001
C = 0.001
fin = 3
f1 = 16
f2 = 32
f3 = 64
h1 = 64
outputs = 10
expand = 2
batchSize = 128

# 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
trX, teX, trY, teY = cifar(onehot=True)
trSize = trX.shape[0]  # 训练集样本数
teSize = teX.shape[0]  # 测试集样本数

params = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
# 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
shape = (f1, fin, 3, 3)
w11 = utils.weightInitCNN3(shape, 'w1')
w12 = utils.weightInitColfc((shape[0], expand, shape[1]), 'w2')
b11 = utils.biasInit(shape[:2], 'b1')
b12 = utils.biasInit((shape[0], expand), 'b2')
params.append([w11, w12, b11, b12])
shape = (f2, f1 * expand, 3, 3)
w21 = utils.weightInitCNN3(shape, 'w1')
w22 = utils.weightInitColfc((shape[0], expand, shape[1]), 'w2')
b21 = utils.biasInit(shape[:2], 'b1')
b22 = utils.biasInit((shape[0], expand), 'b2')
params.append([w21, w22, b21, b22])
shape = (f3, f2 * expand, 3, 3)
w31 = utils.weightInitCNN3(shape, 'w1')
w32 = utils.weightInitColfc((shape[0], expand, shape[1]), 'w2')
b31 = utils.biasInit(shape[:2], 'b1')
b32 = utils.biasInit((shape[0], expand), 'b2')
params.append([w31, w32, b31, b32])
# 全局平均池化
wgap1 = utils.weightInitCNN3((h1, f3 * expand, 1, 1), 'wgap')
bgap1 = utils.biasInit((h1,), 'bgap')
params.append([wgap1, bgap1])
wgap2 = utils.weightInitCNN3((outputs, h1, 1, 1), 'wgap')
bgap2 = utils.biasInit((outputs,), 'bgap')
params.append([wgap2, bgap2])

# 定义 Theano 符号变量，并构建 Theano 表达式
X = T.tensor4('X')
Y = T.matrix('Y')
# 训练集代价函数
YDropProb = model(X, params, 0.2, 0.5)
trNeqs = utils.neqs(YDropProb, Y)
trCrossEntropy = categorical_crossentropy(YDropProb, Y)
trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(params))

# 测试验证集代价函数
YFullProb = model(X, params, 0., 0.)
vateNeqs = utils.neqs(YFullProb, Y)
YPred = T.argmax(YFullProb, axis=1)
vateCrossEntropy = categorical_crossentropy(YFullProb, Y)
vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(params))
updates = utils.sgdm(trCost, flatten(params), lr, nesterov=True)
train = function(
    inputs=[X, Y],
    outputs=[trCost, trNeqs],  # 减少返回参数节省时间
    updates=updates,
    allow_input_downcast=True
)

# 训练迭代，一次迭代分为多batch训练
start = time.time()
for i in range(iterSteps):
    epochStart = time.time()
    count = 0
    for start, end in zip(range(0, trSize, batchSize), range(batchSize, trSize, batchSize)):
        batchStart = time.time()
        trCost, trError = train(trX[start:end], trY[start:end])
        print count, 'trCost: %8.5f   trError: %d   batch time: %.5f' % (trCost, trError, time.time() - batchStart)
        count += 1
    print 'epoch time:', time.time() - epochStart
print 'total time:', time.time() - start
