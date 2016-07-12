# coding:utf-8
__author__ = 'zfh'

from compiler.ast import flatten
import time

import numpy as np
import theano.tensor as T
from theano import function
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import matplotlib.pyplot as plt

from demo.load_mnist import mnist
import utils



# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 模型构建，返回给定样本判定为某类别的概率,偏置在某些维度上展开与相加矩阵相同
def model(X, prams, pDropConv, pDropHidden):
    lconv1 = T.nnet.relu(conv2d(X, prams[0][0], border_mode='full') +
                         prams[0][1].dimshuffle('x', 0, 'x', 'x'))
    lds1 = max_pool_2d(lconv1, (2, 2))
    lds1 = utils.dropout(lds1, pDropConv)

    lconv2 = T.nnet.relu(conv2d(lds1, prams[1][0]) +
                         prams[1][1].dimshuffle('x', 0, 'x', 'x'))
    lds2 = max_pool_2d(lconv2, (2, 2))
    lds2 = utils.dropout(lds2, pDropConv)

    lconv3 = T.nnet.relu(conv2d(lds2, prams[2][0]) +
                         prams[2][1].dimshuffle('x', 0, 'x', 'x'))
    lds3 = max_pool_2d(lconv3, (2, 2))
    lds3 = utils.dropout(lds3, pDropConv)

    lflat = T.flatten(lds3, outdim=2)
    lfull = T.nnet.relu(T.dot(lflat, prams[3][0]) + prams[3][1])
    lfull = utils.dropout(lfull, pDropHidden)
    return T.nnet.softmax(T.dot(lfull, prams[4][0]) + prams[4][1])

# 常量
m = 60000  # 样本数
n = 784  # 特征维度
iterSteps = 5000
learningRate = 0.001
C = 0.001
hiddens = 625
outputs = 10
batchSize = 200

# 数据集
trX, teX, trY, teY = mnist(onehot=True)
# 数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

# Theano 符号变量
X = T.tensor4('X')
Y = T.matrix('Y')
prams = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
# 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
wconv1 = utils.weightInit((32, 1, 5, 5), 'wconv1')
bconv1 = utils.biasInit((32,), 'bconv1')
prams.append([wconv1, bconv1])
wconv2 = utils.weightInit((64, 32, 3, 3), 'wconv2')
bconv2 = utils.biasInit((64,), 'bconv2')
prams.append([wconv2, bconv2])
wconv3 = utils.weightInit((128, 64, 3, 3), 'wconv3')
bconv3 = utils.biasInit((128,), 'bconv3')
prams.append([wconv3, bconv3])
# 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
wfull = utils.weightInit((128 * 3 * 3, hiddens), 'wfull')
bfull = utils.biasInit((hiddens,), 'bfull')
prams.append([wfull, bfull])
wout = utils.weightInit((hiddens, outputs), 'wout')
bout = utils.biasInit((outputs,), 'bout')
prams.append([wout, bout])

# 构建 Theano 表达式
yDropProb = model(X, prams, 0.2, 0.5)
yFullProb = model(X, prams, 0., 0.)
yPred = T.argmax(yFullProb, axis=1)
crossEntropy = T.nnet.categorical_crossentropy(yDropProb, Y)
cost = T.mean(crossEntropy) + C * utils.reg(flatten(prams))
updates = utils.rmsprop(cost, flatten(prams), lr=learningRate)

# 编译函数
train = function(
    inputs=[X, Y],
    outputs=[yPred, cost],
    updates=updates,
    allow_input_downcast=True
)
predict = function(
    inputs=[X],
    outputs=yPred,
    allow_input_downcast=True
)

# 训练迭代，一次迭代分为多batch训练
accuracyTrace = []
start = time.time()
for i in range(iterSteps):
    epochStart = time.time()
    for start, end in zip(range(0, m, batchSize), range(batchSize, m, batchSize)):
        pred, err = train(trX[start:end], trY[start:end])
    accuracy = np.mean(predict(teX) == np.argmax(teY, axis=1))
    accuracyTrace.append(accuracy)
    print 'accuracy:', accuracy, 'time delay:', time.time() - epochStart
print 'total time:', time.time() - start

plt.plot(accuracyTrace, 'b-')
plt.show()
