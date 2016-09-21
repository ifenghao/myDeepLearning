# coding:utf-8
__author__ = 'zfh'
'''
在NIN的顶层使用全连接层
在scan的元操作中包含所有relu激活，而在scan外部仅做组合
如果在sacn外部使用对所有元素relu激活，会出现结果的nan
'''
import time
from compiler.ast import flatten

import theano.tensor as T
from theano import scan, function
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d

from load import cifar
from utils import basicUtils, gradient, initial, preprocess


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 直接使用python的for循环的表示，导致编译失败（递归栈溢出）
# def nin(X, param, shape):
#     w1, w2 = param
#     map0 = []
#     for i in xrange(shape[0]):
#         map1 = []
#         for j in xrange(shape[1]):
#             Xj = X[:, j, :, :].dimshuffle(0, 'x', 1, 2)
#             w1ij = w1[i, j, :, :, :].dimshuffle(0, 'x', 1, 2)
#             w2ij = w2[i, j, :].dimshuffle('x', 0, 'x', 'x')
#             tmp = conv2d(Xj, w1ij, border_mode='half')
#             tmp = relu(tmp, alpha=0)
#             map1.append(conv2d(tmp, w2ij, border_mode='valid'))
#         map0.append(relu(T.sum(map1, axis=0), alpha=0))
#     return T.concatenate(map0, axis=1)

# scan的一次元操作
def metaOp(i, j, X, w1, w2, b1, b2):
    # (n,1,r,c)**(16,1,3,3)=(n,16,r,c)
    hiddens = conv2d(X[:, j, :, :, :], w1[i, j, :, :, :, :], border_mode='half') + b1[i, j, :, :, :, :]
    hiddens = relu(hiddens, alpha=0)
    # 在元操作中就需要包含relu激活
    # return conv2d(hiddens, w2[i, j, :, :, :, :], border_mode='valid') + b2[i, j, :, :, :, :]
    # (n,16,r,c)**(1,16,1,1)=(n,1,r,c)
    outputs = conv2d(hiddens, w2[i, j, :, :, :, :], border_mode='valid') + b2[i, j, :, :, :, :]
    return T.nnet.relu(outputs)


def nin(X, param):
    w1, w2, b1, b2 = param
    X = X.dimshuffle(0, 1, 'x', 2, 3)  # (n,32,1,r,c)
    w1 = w1.dimshuffle(0, 1, 2, 'x', 3, 4)  # (64,32,16,1,3,3)
    w2 = w2.dimshuffle(0, 1, 'x', 2, 'x', 'x')  # (64,32,1,16,1,1)
    b1 = b1.dimshuffle(0, 1, 'x', 2, 'x', 'x')  # (64,32,1,16,1,1)
    b2 = b2.dimshuffle(0, 1, 'x', 2, 'x', 'x')  # (64,32,1,1,1,1)
    indexi = T.arange(w1.shape[0], dtype='int32')  # (0:64)
    indexi = T.repeat(indexi, w1.shape[1], axis=0)
    indexj = T.arange(w1.shape[1], dtype='int32')  # (0:32)
    indexj = T.tile(indexj, w1.shape[0])
    results, updates = scan(fn=metaOp,
                            sequences=[indexi, indexj],
                            outputs_info=None,
                            non_sequences=[X, w1, w2, b1, b2],
                            strict=True)  # (64*32,n,1,r,c)
    metaShape = results.shape[-4], results.shape[-2], results.shape[-1]
    reshaped = results.reshape((w1.shape[0], w1.shape[1]) + metaShape)  # (64,32,n,r,c)
    sumed = T.sum(reshaped, axis=1)  # (64,n,r,c)
    permuted = T.transpose(sumed, axes=(1, 0, 2, 3))  # (n,64,r,c)
    # 在scan外部不能对全体元素relu激活
    # return relu(permuted, alpha=0)
    return permuted


# 模型构建，返回给定样本判定为某类别的概率
def model(X, params, pDropConv, pDropHidden):
    lnum = 0  # conv: (32, 32) pool: (16, 16)
    layer = nin(X, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1  # conv: (16, 16) pool: (8, 8)
    layer = nin(layer, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1  # conv: (8, 8) pool: (4, 4)
    layer = nin(layer, params[lnum])
    layer = pool_2d(layer, (2, 2), st=(2, 2), ignore_border=False, mode='max')
    layer = basicUtils.dropout(layer, pDropConv)
    lnum += 1
    layer = T.flatten(layer, outdim=2)
    layer = T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0)
    layer = relu(layer, alpha=0)
    layer = basicUtils.dropout(layer, pDropHidden)
    lnum += 1
    return softmax(T.dot(layer, params[lnum][0]) + params[lnum][1].dimshuffle('x', 0))  # 如果使用nnet中的softmax训练产生NAN


# 常量
iterSteps = 100
lr = 0.001
C = 0.001
fin = 3
f1 = 32
nin1 = 4
f2 = 64
nin2 = 4
f3 = 128
nin3 = 4
hiddens = 256
outputs = 10
batchSize = 128

# 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
trX, teX, trY, teY = cifar(onehot=True)
trSize = trX.shape[0]  # 训练集样本数
teSize = teX.shape[0]  # 测试集样本数

params = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
# 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
shape = (f1, fin, nin1, 3, 3)
w11 = initial.weightInitNIN1_3(shape, 'w1')
w12 = initial.weightInitNIN2_3(shape[:3], 'w2')
b11 = initial.biasInit(shape[:3], 'b1')
b12 = initial.biasInit(shape[:2] + (1,), 'b2')
params.append([w11, w12, b11, b12])
shape = (f2, f1, nin2, 3, 3)
w21 = initial.weightInitNIN1_3(shape, 'w1')
w22 = initial.weightInitNIN2_3(shape[:3], 'w2')
b21 = initial.biasInit(shape[:3], 'b1')
b22 = initial.biasInit(shape[:2] + (1,), 'b2')
params.append([w21, w22, b21, b22])
shape = (f3, f2, nin3, 3, 3)
w31 = initial.weightInitNIN1_3(shape, 'w1')
w32 = initial.weightInitNIN2_3(shape[:3], 'w2')
b31 = initial.biasInit(shape[:3], 'b1')
b32 = initial.biasInit(shape[:2] + (1,), 'b2')
params.append([w31, w32, b31, b32])
# 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
wfull = initial.weightInitMLP3((f3 * 4 * 4, hiddens), 'wfull')
bfull = initial.biasInit((hiddens,), 'bfull')
params.append([wfull, bfull])
wout = initial.weightInitMLP3((hiddens, outputs), 'wout')
bout = initial.biasInit((outputs,), 'bout')
params.append([wout, bout])

# 定义 Theano 符号变量，并构建 Theano 表达式
X = T.tensor4('X')
Y = T.matrix('Y')
# 训练集代价函数
YDropProb = model(X, params, 0.2, 0.5)
trNeqs = basicUtils.neqs(YDropProb, Y)
trCrossEntropy = categorical_crossentropy(YDropProb, Y)
trCost = T.mean(trCrossEntropy) + C * basicUtils.regularizer(flatten(params))

# 测试验证集代价函数
YFullProb = model(X, params, 0., 0.)
vateNeqs = basicUtils.neqs(YFullProb, Y)
YPred = T.argmax(YFullProb, axis=1)
vateCrossEntropy = categorical_crossentropy(YFullProb, Y)
vateCost = T.mean(vateCrossEntropy) + C * basicUtils.regularizer(flatten(params))
updates = gradient.sgdm(trCost, flatten(params), lr, nesterov=True)
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
