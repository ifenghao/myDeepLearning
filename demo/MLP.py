# coding:utf-8
__author__ = 'zfh'

import time

import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from numpy import random as rng
from theano import function, shared

from utils import basicUtils, gradient, initial, preprocess


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
w1 = shared(basicUtils.floatX(np.random.randn(n, hiddens) * 0.01), name='w1', borrow=True)
b1 = shared(basicUtils.floatX(np.ones((hiddens,))), name='b1', borrow=True)
w2 = shared(basicUtils.floatX(np.random.randn(hiddens) * 0.01), name='w2', borrow=True)
b2 = shared(1., name='b2', borrow=True)

# 构建 Theano 表达式
yProb = model(X, w1, b1, w2, b2)
yPred = yProb > 0.5
crossEntropy = T.nnet.categorical_crossentropy(yProb, y)
cost = T.mean(crossEntropy) + C * (T.mean(w1 ** 2) + T.mean(b1 ** 2) + T.mean(w2 ** 2) + T.mean(b2 ** 2))
gradPrams = [w1, b1, w2, b2]  # 所有需要优化的参数放入列表中
updates = gradient.sgd(cost, gradPrams, learningRate)

# 编译函数
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
start = time.time()
for i in range(iterSteps):
    pred, err = train(D[0], D[1])
    errList.append(err)
print 'time delay ', time.time() - start
print 'target values for D: ', D[1]
print 'predictions on D: ', predict(D[0])
print 'accuracy: ', np.mean(predict(D[0]) == D[1])
plt.plot(errList, 'b-')
plt.show()
