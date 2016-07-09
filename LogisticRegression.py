# coding:utf-8
__author__ = 'zfh'

import numpy as np
from numpy import random as rng
import theano.tensor as T
from theano import function, shared
import utils
import matplotlib.pyplot as plt

# 模型构建，返回给定样本判定为某类别的概率
def model(X, w, b):
    # return 1 / (1 + T.exp(-T.dot(X, w)-b))
    return T.nnet.sigmoid(T.dot(X, w) + b)

# 常量
m = 400  # 样本数
n = 784  # 特征维度
D = (rng.randn(m, n), rng.randint(size=m, low=0, high=2))  # 生成数据集
iterSteps = 1000
learningRate = 0.1
C = 0.01

# Theano 符号变量
X = T.matrix('X')
y = T.vector('y')
w = utils.randomInit((n,), 'w')
b = shared(0., name='b')

# 构建 Theano 表达式
yProb = model(X, w, b)
yPred = yProb > 0.5
crossEntropy = -y * T.log(yProb) - (1 - y) * T.log(1 - yProb)
cost = T.mean(crossEntropy) + C * T.sum(w ** 2)
prams = [w, b]  # 所有需要优化的参数放入列表中
updates = utils.sgd(cost, prams, learningRate)

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
for i in range(iterSteps):
    pred, err = train(D[0], D[1])
    errList.append(err)
print 'Final model:'
print w.get_value(), b.get_value()
print 'target values for D: ', D[1]
print 'predictions on D: ', predict(D[0])
print 'accuracy: ', np.mean(predict(D[0]) == D[1])
plt.plot(errList, 'b-')
plt.show()
