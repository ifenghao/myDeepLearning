# coding:utf-8
__author__ = 'zfh'

from compiler.ast import flatten
import time

import numpy as np
import theano.tensor as T
from theano import function
import matplotlib.pyplot as plt

from demo.load_mnist import mnist
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 模型构建，返回给定样本判定为某类别的概率
def model(X, prams, pDropInput, pDropHidden):
    X = utils.dropout(X, pDropInput)
    h = T.nnet.relu(T.dot(X, prams[0][0]) + prams[0][1])
    h = utils.dropout(h, pDropHidden)
    return softmax(T.dot(h, prams[1][0]) + prams[1][1])

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

# Theano 符号变量
X = T.matrix('X')
Y = T.matrix('Y')
prams = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
w1 = utils.weightInit((n, hiddens), 'w1')
b1 = utils.biasInit((hiddens,), 'b1')
prams.append([w1, b1])
w2 = utils.weightInit((hiddens, outputs), 'w2')
b2 = utils.biasInit((outputs,), 'b2')
prams.append([w2, b2])

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
    print 'accuracy: ', accuracy, 'time delay ', time.time() - epochStart
print 'total time ', time.time() - start

plt.plot(accuracyTrace, 'b-')
plt.show()
