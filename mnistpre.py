__author__ = 'zfh'

import os
import struct
import numpy as np
from PIL import Image


def readHead(bytes):
    length = len(bytes) / 4
    s = struct.unpack('>' + 'L' * length, bytes)  # int(a.encode('hex'), 16)
    return s


def bytes2image(bytes, row, col):
    tup = struct.unpack('>' + 'B' * (row * col), bytes)
    mat = np.matrix(tup).reshape((row, col))
    image = Image.fromarray(mat.astype(np.uint8))
    return image


def bytes2label(bytes):
    tup = struct.unpack('B', bytes)
    return tup[0]


fImage = open(os.path.join(os.getcwd(), 'dataset', 'MNIST', 'train-images.idx3-ubyte'), 'rb')
fLabel = open(os.path.join(os.getcwd(), 'dataset', 'MNIST', 'train-labels.idx1-ubyte'), 'rb')
hImage = fImage.read(16)
s = readHead(hImage)
num, row, col = s[1:]
hLabel = fLabel.read(8)
labelCount = np.zeros(10, dtype=np.int)
for _ in range(num):
    image = bytes2image(fImage.read(row * col), row, col)
    label = bytes2label(fLabel.read(1))
    image.save(os.path.join(os.getcwd(), 'dataset', 'MNIST', 'picture',
                            str(label) + '_' + str(labelCount[label - 1]) + '.png'))
    labelCount[label - 1] += 1
print labelCount, labelCount.sum()
fImage.close()
