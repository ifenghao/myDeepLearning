# coding:utf-8
__author__ = 'zfh'
'''
network in network
在NIN的顶层使用全连接层会导致输出结果包含nan
替换为全局平均池化层就可以正常计算
'''
from compiler.ast import flatten
import time
import theano.tensor as T
from theano import scan
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d
from sklearn.cross_validation import KFold
from copy import copy
import numpy as np
from load import cifar
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# scan的一次元操作
def metaOp(i, j, X, w1, w2, b1, b2):
    hiddens = conv2d(X[:, j, :, :, :], w1[i, j, :, :, :, :], border_mode='half') + b1[i, j, :, :, :, :]
    hiddens = T.nnet.relu(hiddens, alpha=0)
    outputs = conv2d(hiddens, w2[i, j, :, :, :, :], border_mode='valid') + b2[i, j, :, :, :, :]
    return relu(outputs)


def nin(X, param):
    w1, w2, b1, b2 = param
    X = X.dimshuffle(0, 1, 'x', 2, 3)
    w1 = w1.dimshuffle(0, 1, 2, 'x', 3, 4)
    w2 = w2.dimshuffle(0, 1, 'x', 2, 'x', 'x')
    b1 = b1.dimshuffle(0, 1, 'x', 2, 'x', 'x')
    b2 = b2.dimshuffle(0, 1, 'x', 2, 'x', 'x')
    indexi = T.arange(w1.shape[0], dtype='int32')
    indexi = T.repeat(indexi, w1.shape[1], axis=0)
    indexj = T.arange(w1.shape[1], dtype='int32')
    indexj = T.tile(indexj, w1.shape[0])
    results, updates = scan(fn=metaOp,
                            sequences=[indexi, indexj],
                            outputs_info=None,
                            non_sequences=[X, w1, w2, b1, b2],
                            strict=True)
    metaShape = results.shape[-4], results.shape[-2], results.shape[-1]
    reshaped = results.reshape((w1.shape[0], w1.shape[1]) + metaShape)
    sumed = T.sum(reshaped, axis=1)
    permuted = T.transpose(sumed, axes=(1, 0, 2, 3))
    return permuted


def gap(X):
    layer = T.mean(X, axis=(2, 3))
    return layer


def conv1t1(X, param):
    wconv, bconv = param
    layer = conv2d(X, wconv, border_mode='valid') + bconv.dimshuffle('x', 0, 'x', 'x')
    layer = relu(layer, alpha=0)
    return layer


def fc(X, param):
    w, b = param
    layer = T.dot(X, w) + b.dimshuffle('x', 0)
    layer = relu(layer, alpha=0)
    return layer


def addNINLayer(inputShape, filterShape, params, border_mode, stride=(1, 1)):
    params.append(layerNINParams(filterShape))
    outputShape = utils.convOutputShape(inputShape[-2:], filterShape[-2:], border_mode, stride)
    return outputShape


def addPoolLayer(inputShape, poolSize, ignore_border=False, stride=None):
    outputShape = utils.poolOutputShape(inputShape, poolSize, ignore_border, stride)
    return outputShape


def layerNINParams(shape):
    w1 = utils.weightInitNIN1_3(shape, 'w1')
    w2 = utils.weightInitNIN2_3(shape[:3], 'w2')
    b1 = utils.biasInit(shape[:3], 'b1')
    b2 = utils.biasInit(shape[:2] + (1,), 'b2')
    return [w1, w2, b1, b2]


# 全局平均池化使用和卷积层一样的参数
def layerConvParams(shape):
    w = utils.weightInitCNN3(shape, 'w')
    b = utils.biasInit(shape[0], 'b')
    return [w, b]


def layerMLPParams(shape):
    w = utils.weightInitMLP3(shape, 'w')
    b = utils.biasInit(shape[1], 'b')
    return [w, b]


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
    # 全连接层
    # layer = T.flatten(layer, outdim=2)
    # lnum += 1
    # layer = fc(layer, params[lnum])
    # layer = utils.dropout(layer, pDropHidden)
    # lnum += 1
    # layer = fc(layer, params[lnum])
    # 全局平均池化
    lnum += 1
    layer = conv1t1(layer, params[lnum])
    layer = utils.dropout(layer, pDropHidden)
    lnum += 1
    layer = conv1t1(layer, params[lnum])
    layer = gap(layer)
    return softmax(layer)  # 如果使用nnet中的softmax训练产生NAN


# 卷积网络，输入一组超参数，返回该网络的训练、验证、预测函数
class CConvNet(object):
    def __init__(self, fin, f1, nin1, f2, nin2, f3, nin3, h1, outputs,
                 lr, C, pDropConv=0.2, pDropHidden=0.5):
        # 超参数
        self.lr = lr
        self.C = C
        self.pDropConv = pDropConv
        self.pDropHidden = pDropHidden
        # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        self.params = []
        self.paramsNIN = []
        self.paramsFCorConv = []
        # 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
        inputShape = (32, 32)
        layerShape = addNINLayer(inputShape, (f1, fin, nin1, 3, 3), self.paramsNIN, 'half')
        layerShape = addPoolLayer(layerShape, (2, 2))
        layerShape = addNINLayer(layerShape, (f2, f1, nin2, 3, 3), self.paramsNIN, 'half')
        layerShape = addPoolLayer(layerShape, (2, 2))
        layerShape = addNINLayer(layerShape, (f3, f2, nin3, 3, 3), self.paramsNIN, 'half')
        layerShape = addPoolLayer(layerShape, (2, 2))
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        # self.paramsFCorGAP.append(layerMLPParams((f3 * np.prod(layerShape), h1)))
        # self.paramsFCorGAP.append(layerMLPParams((h1, outputs)))
        # 全局平均池化层
        self.paramsFCorConv.append(layerConvParams((h1, f3, 1, 1)))
        self.paramsFCorConv.append(layerConvParams((outputs, h1, 1, 1)))
        self.params = self.paramsNIN + self.paramsFCorConv

        # 定义 Theano 符号变量，并构建 Theano 表达式
        self.X = T.tensor4('X')
        self.Y = T.matrix('Y')
        # 训练集代价函数
        YDropProb = model(self.X, self.params, pDropConv, pDropHidden)
        self.trNeqs = utils.neqs(YDropProb, self.Y)
        trCrossEntropy = categorical_crossentropy(YDropProb, self.Y)
        self.trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(self.params))

        # 测试验证集代价函数
        YFullProb = model(self.X, self.params, 0., 0.)
        self.vateNeqs = utils.neqs(YFullProb, self.Y)
        self.YPred = T.argmax(YFullProb, axis=1)
        vateCrossEntropy = categorical_crossentropy(YFullProb, self.Y)
        self.vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(self.params))

    # 重置优化参数，以重新训练模型
    def resetPrams(self):
        for p in self.paramsNIN:
            utils.resetWeightCNN3(p[0])
            utils.resetBias(p[1])
        for p in self.paramsFCorConv:
            utils.resetWeightMLP3(p[0])
            utils.resetBias(p[1])

    # 训练卷积网络，最终返回在测试集上的误差
    def traincn(self, trX, teX, trY, teY, batchSize=128, maxIter=100, verbose=True,
                start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
        # 训练函数，输入训练集，输出训练损失和误差
        updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
        train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        valtest = utils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
        trX, teX = utils.preprocess(trX, teX)
        earlyStop = utils.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in range(maxIter):  # every epoch
            startTime = time.time()
            trCost, trError = utils.miniBatch(trX, trY, train, batchSize, verbose)
            teCost, teError = utils.miniBatch(teX, teY, valtest, batchSize, verbose)
            if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
            if earlyStop.send((trCost, teCost)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
                totalStopCount += 1
                if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                if verbose: print 'learning rate decreases to ', lr

    # 交叉验证得到一组平均验证误差，使用早停止
    def cv(self, X, Y, folds=5, batchSize=128, maxIter=100, verbose=True,
           start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        # 训练集分为n折交叉验证集
        kf = KFold(X.shape[0], n_folds=folds, random_state=42)
        vaErrorList = []
        for trIndex, vaIndex in kf:
            lr = self.lr  # 当验证损失不再下降而早停止后，降低学习率继续迭代
            # 训练函数，输入训练集，输出训练损失和误差
            updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
            train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
            # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
            valtest = utils.makeFunc([self.X, self.Y], [self.vateCost, self.vateNeqs], None)
            # 分割训练集
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            trX, vaX = utils.preprocess(trX, vaX)
            earlyStop = utils.earlyStopGen(start, period, threshold, earlyStopTol)
            earlyStop.next()  # 初始化生成器
            totalStopCount = 0
            vaErrorOpt = 1.
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost, trError = utils.miniBatch(trX, trY, train, batchSize, verbose)
                vaCost, vaError = utils.miniBatch(vaX, vaY, valtest, batchSize, verbose)
                if vaError < vaErrorOpt: vaErrorOpt = vaError
                if verbose: print ' time: %10.5f' % (time.time() - startTime), 'epoch ', epoch
                if earlyStop.send((trCost, vaCost)):
                    lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                    updates = utils.sgdm(self.trCost, flatten(self.params), lr, nesterov=True)
                    train = utils.makeFunc([self.X, self.Y], [self.trCost, self.trNeqs], updates)
                    totalStopCount += 1
                    if totalStopCount > totalStopTol: break  # 如果学习率降低仍然发生早停止，则退出迭代
                    if verbose: print 'learning rate decreases to ', lr
            vaErrorList.append(copy(vaErrorOpt))
            if verbose: print '*' * 10, 'one validation done, best vaError', vaErrorList, '*' * 10
            self.resetPrams()
            break  # 只进行一轮验证
        return np.mean(vaErrorList)


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    trX, teX, trY, teY = cifar(onehot=True)
    f1, nin1, f2, nin2, f3, nin3, h1 = 32, 5, 64, 3, 128, 2, 64
    params = utils.randomSearch(nIter=10)
    cvErrorList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 40, num, 'parameters', param, '*' * 40
        convNet = CConvNet(3, f1, nin1, f2, nin2, f3, nin3, h1, 10, lr, C, 0.2, 0.5)
        cvError = convNet.cv(trX, trY)
        cvErrorList.append(copy(cvError))
    optIndex = np.argmin(cvErrorList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    convNet = CConvNet(3, f1, nin1, f2, nin2, f3, nin3, h1, 10, lr, C, 0.2, 0.5)
    convNet.traincn(trX, teX, trY, teY)


if __name__ == '__main__':
    main()
