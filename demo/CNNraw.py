# coding:utf-8
__author__ = 'zfh'

import time

import numpy as np
import theano.tensor as T
from theano import function
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import matplotlib.pyplot as plt

from load import mnist
import utils


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 模型构建，返回给定样本判定为某类别的概率
def model(X, wconv1, bconv1, wconv2, bconv2, wconv3, bconv3, wfull, bfull, wout, bout):
    lconv1 = T.nnet.sigmoid(conv2d(X, wconv1, border_mode='full') + bconv1.dimshuffle('x', 0, 'x', 'x'))
    lds1 = max_pool_2d(lconv1, (2, 2))
    lconv2 = T.nnet.sigmoid(conv2d(lds1, wconv2) + bconv2.dimshuffle('x', 0, 'x', 'x'))
    lds2 = max_pool_2d(lconv2, (2, 2))
    lconv3 = T.nnet.sigmoid(conv2d(lds2, wconv3) + bconv3.dimshuffle('x', 0, 'x', 'x'))
    lds3 = max_pool_2d(lconv3, (2, 2))
    lflat = T.flatten(lds3, outdim=2)
    lfull = T.nnet.sigmoid(T.dot(lflat, wfull) + bfull)
    return softmax(T.dot(lfull, wout) + bout)

# 常量
m = 60000  # 样本数
n = 784  # 特征维度
iterSteps = 5000
learningRate = 0.01
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
# 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（1，本层特征图个数，1，1）
wconv1 = utils.weightInit((32, 1, 3, 3), 'wconv1')
bconv1 = utils.biasInit((32,), 'bconv1')
wconv2 = utils.weightInit((64, 32, 3, 3), 'wconv2')
bconv2 = utils.biasInit((64,), 'bconv2')
wconv3 = utils.weightInit((128, 64, 3, 3), 'wconv3')
bconv3 = utils.biasInit((128,), 'bconv3')
# 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
wfull = utils.weightInit((128 * 3 * 3, hiddens), 'wfull')
bfull = utils.biasInit((hiddens,), 'bfull')
wout = utils.weightInit((hiddens, outputs), 'wout')
bout = utils.biasInit((outputs,), 'bout')

# 构建 Theano 表达式
yProb = model(X, wconv1, bconv1, wconv2, bconv2, wconv3, bconv3, wfull, bfull, wout, bout)
yPred = T.argmax(yProb, axis=1)
crossEntropy = T.nnet.categorical_crossentropy(yProb, Y)
cost = T.mean(crossEntropy) + C * (
    T.mean(wconv1 ** 2) + T.mean(bconv1 ** 2) +
    T.mean(wconv2 ** 2) + T.mean(bconv2 ** 2) +
    T.mean(wconv3 ** 2) + T.mean(bconv3 ** 2) +
    T.mean(wfull ** 2) + T.mean(bfull ** 2) +
    T.mean(wout ** 2) + T.mean(bout ** 2)
)
gradPrams = [wconv1, bconv1, wconv2, bconv2, wconv3, bconv3, wfull, bfull, wout, bout]  # 所有需要优化的参数放入列表中
updates = utils.sgd_momentum(cost, gradPrams, learningRate)

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
errList = []
start = time.time()
for i in range(iterSteps):
    for start, end in zip(range(0, m, batchSize), range(batchSize, m, batchSize)):
        pred, err = train(trX[start:end], trY[start:end])
        errList.append(err)
    print 'accuracy: ', np.mean(predict(teX) == np.argmax(teY, axis=1))
print 'time delay ', time.time() - start

plt.plot(errList, 'b-')
plt.show()
