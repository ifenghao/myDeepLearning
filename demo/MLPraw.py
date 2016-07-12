# coding:utf-8
__author__ = 'zfh'

import time

import numpy as np
import theano.tensor as T
from theano import function, shared
import matplotlib.pyplot as plt

from demo.load_mnist import mnist
import utils


# 模型构建，返回给定样本判定为某类别的概率
def model(X, w1, b1, w2, b2):
    h = T.nnet.sigmoid(T.dot(X, w1) + b1)
    return T.nnet.softmax(T.dot(h, w2) + b2)

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

# Theano 符号变量
X = T.matrix('X')
Y = T.matrix('Y')
w1 = shared(utils.floatX(np.random.randn(n, hiddens) * 0.01), name='w1', borrow=True)
b1 = shared(utils.floatX(np.zeros(hiddens)), name='b1', borrow=True)
w2 = shared(utils.floatX(np.random.randn(hiddens, outputs) * 0.01), name='w2', borrow=True)
b2 = shared(utils.floatX(np.zeros(outputs)), name='b2', borrow=True)

# 构建 Theano 表达式
yProb = model(X, w1, b1, w2, b2)
yPred = T.argmax(yProb, axis=1)
crossEntropy = T.nnet.categorical_crossentropy(yProb, Y)
cost = T.mean(crossEntropy) + C * utils.reg((w1, b1, w2, b2))
gradPrams = [w1, b1, w2, b2]  # 所有需要优化的参数放入列表中
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
