# coding:utf-8
__author__ = 'zfh'

import numpy as np
from numpy import random as rng
import theano.tensor as T
from theano import function, shared
import utils
import matplotlib.pyplot as plt
import time

# 模型构建，返回给定样本判定为某类别的概率
def model(X, w1, b1, w2, b2):
    h = T.nnet.sigmoid(T.dot(X, w1) + b1)
    return T.nnet.sigmoid(T.dot(h, w2) + b2)

# 常量
m = 400  # 样本数
n = 784  # 特征维度
D = (rng.randn(m, n), rng.randint(size=m, low=0, high=2))  # 生成数据集
iterSteps = 500
learningRate = 0.1
C = 0.01
hiddens = 100

# Theano 符号变量
X = T.matrix('X')
y = T.vector('y')
w1 = utils.randomInit((n, hiddens), 'w1')
b1 = shared(utils.floatX(np.zeros((hiddens,))), name='b1')
w2 = utils.randomInit((hiddens,), 'w2')
b2 = shared(0., name='b2')

# 构建 Theano 表达式
yProb = model(X, w1, b1, w2, b2)
yPred = yProb > 0.5
crossEntropy = -y * T.log(yProb) - (1 - y) * T.log(1 - yProb)
cost = T.mean(crossEntropy) + C * (T.sum(w1 ** 2) + T.sum(w2 ** 2))
gradPrams = [w1, b1, w2, b2]  # 所有需要优化的参数放入列表中
updates = utils.sgd(cost, gradPrams, learningRate)

# 编译训练函数
train = function(
    inputs=[X, y],
    outputs=[yPred, cost],
    updates=updates,
    allow_input_downcast=True
)
predict = function(
    inputs=[X],
    outputs=yPred,
    allow_input_downcast=True
)

# 训练迭代
errList = []
start=time.time()
for i in range(iterSteps):
    pred, err = train(D[0], D[1])
    errList.append(err)
print 'time delay ', time.time()-start
print 'target values for D: ', D[1]
print 'predictions on D: ', predict(D[0])
print 'accuracy: ', np.mean(predict(D[0]) == D[1])
plt.plot(errList, 'b-')
plt.show()
