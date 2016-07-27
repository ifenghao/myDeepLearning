# coding:utf-8
import numpy as np
import os, getpass
import cPickle

datasets_dir = '/home/' + getpass.getuser() + '/dataset'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


# 数据格式为2D矩阵（样本数，图像行数*图像列数）
def mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX / 255.
    teX = teX / 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


# 数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
def cifar(onehot=True):
    data_dir = os.path.join(datasets_dir, 'cifar-10-batches-py/')
    allFiles = os.listdir(data_dir)
    trFiles = [f for f in allFiles if f.startswith('data_batch')]
    trX = []
    trY = []
    for file in trFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        trX.append(batchData)
        trY.extend(batchLabel)
        fd.close()
    trX = np.vstack(trX)
    teFiles = [f for f in allFiles if f.find('test_batch') != -1]
    teX = []
    teY = []
    for file in teFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        teX.append(batchData)
        teY.extend(batchLabel)
        fd.close()
    teX = np.vstack(teX)
    trX -= 128.
    teX -= 128.
    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)
    return trX, teX, trY, teY
