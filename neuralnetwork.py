#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def visualImg(img):
    plt.figure(1)
    plt.imshow(img)
    plt.show()

def signmoid(x):
    return 1. / (1. + np.exp(-x))

def dev_signmoid(x):
    return signmoid(x) * (1. - signmoid(x))

def getNorm(vec):
    return sum(map(np.sum, map(lambda x: x*x, vec)))

def rangeRandom(size, radius, center = 0.):
    return (np.random.random(size) - 0.5) * (radius / 0.5) + center

class NeuralNetwork:
    def __init__(self, layer, nlabel, weightDecayRate):
        self.layer = layer
        self.nlabel = nlabel
        self.weightDecayRate = weightDecayRate 
        self.weight = [rangeRandom((layer[i+1], layer[i]), 0.01) for i in xrange(len(layer) - 1)]
        self.weight.append(rangeRandom((nlabel, layer[-1]), 0.01))
        self.intercept = [rangeRandom(layer[i+1], 0.01) for i in xrange(len(layer) - 1)]

        self.z = [np.zeros(layer[i]) for i in xrange(len(layer))]
        self.z.append(np.zeros(nlabel))
        self.a = [np.zeros(layer[i]) for i in xrange(len(layer))]
        self.a.append(np.zeros(nlabel))

    def forwardPropagation(self, x):
        self.a[0] = x
        for i in xrange(len(self.layer)):
            if i >= 1:
                self.a[i] = signmoid(self.z[i])
            self.z[i + 1] = self.weight[i].dot(self.a[i])
            if i + 1 < len(self.layer):
                self.z[i + 1] += self.intercept[i]
        self.a[-1] = np.exp(self.z[-1])
        self.a[-1] /= np.sum(self.a[-1])
        return self.a[-1]

    def getCost(self, data, label):
        res = 0.
        for i in xrange(len(data)):
            self.forwardPropagation(data[i])
            res += -np.log(self.a[-1][ label[i] ])
        return res

    def test(self, data, label):
        errCount = 0
        for idx, (edata, elabel) in enumerate(zip(data, label)):
            ans = getAnswer(self.forwardPropagation(edata))
            if ans != elabel:
                print idx, ans, elabel
                errCount += 1.
        print 'Error rate : ', errCount / len(data) 
 
    def BPtraining(self, data, label, learningRate, repeat = 100):
        dataNum = len(data)

        for rep in xrange(repeat):
            delta = [np.zeros(self.layer[i]) for i in xrange(len(self.layer))]
            delta.append(np.zeros(self.nlabel))
            dWeight = [np.zeros(self.weight[i].shape) for i in xrange(len(self.layer))]
            dIntercept = [np.zeros(self.intercept[i].shape) for i in xrange(len(self.layer) - 1)]

            for i in xrange(dataNum):
                self.forwardPropagation(data[i])
                delta[-1] = np.array([np.exp(self.z[-1][row]) for row in xrange(self.nlabel)])
                delta[-1] /= np.sum(delta[-1])
                delta[-1][ label[i] ] -= 1.
                for l in reversed(xrange(len(self.layer))):
                    delta[l] = np.multiply(delta[l + 1].dot(self.weight[l]), dev_signmoid(self.z[l]))
                    dWeight[l] += delta[l+1].reshape(len(delta[l+1]), 1).dot(self.a[l].reshape(1, len(self.a[l])))
                    if l < len(self.layer) - 1:
                        dIntercept[l] += delta[l + 1]

            for l in reversed(xrange(len(self.layer))):
                self.weight[l] -= learningRate * (dWeight[l] / dataNum + self.weightDecayRate * self.weight[l])
                if l < len(self.layer) - 1:
                    self.intercept[l] -= learningRate * (dIntercept[l] / dataNum)

            print 'rep = ', rep, 'div = ', getNorm(dWeight) + getNorm(dIntercept), self.getCost(data, label)


def getAnswer(distribute):
    curVal, curAns = 0., 0
    for idx, val in enumerate(distribute):
        if val > curVal:
            curAns, curVal = idx, val
    return curAns

def test():
    ftrainData = open('./MNIST/train-images-idx3-ubyte', 'rb')
    ftrainData.read(4)
    dataNum = np.fromfile(ftrainData, dtype = '>i4', count = 1)[0]
    print dataNum
    trainData = []
    rows = np.fromfile(ftrainData, dtype = '>i4', count = 1)[0]
    cols = np.fromfile(ftrainData, dtype = '>i4', count = 1)[0]
    print rows, cols
    for i in xrange(dataNum):
        trainData.append(np.fromfile(ftrainData, dtype = np.uint8, count = rows * cols))
        #visualImg(trainData[-1])
    ftrainData.close()

    ftrainLabel = open('./MNIST/train-labels-idx1-ubyte', 'rb')
    ftrainLabel.read(8)
    trainLabel = np.fromfile(ftrainLabel, dtype = '>i1', count = dataNum)
    ftrainLabel.close()

    ftestData = open('./MNIST/t10k-images-idx3-ubyte', 'rb')
    ftestData.read(4)
    testNum = np.fromfile(ftestData, dtype = '>i4', count = 1)[0]
    print testNum
    testData = []
    ftestData.read(8)
    for i in xrange(testNum):
        testData.append(np.fromfile(ftestData, dtype = np.uint8, count = rows * cols))
    ftestData.close()

    ftestLabel = open('./MNIST/t10k-labels-idx1-ubyte', 'rb')
    ftestLabel.read(8)
    testLabel = np.fromfile(ftestLabel, dtype = '>i1', count = testNum)
    ftestLabel.close()

#    for i in xrange(20):
#        visualImg(testData[i])
#        print testLabel[i]

    trainData = map(lambda x : x / 255., trainData)
    testData = map(lambda x : x / 255., testData)

    nn = NeuralNetwork([rows * cols, 10], 10, 0.0)
    nn.BPtraining(trainData[:], trainLabel[:], 2.5, 100)

    print 'training data:'
    nn.test(trainData, trainLabel)
    print 'test data:'
    nn.test(testData, testLabel)

if __name__ == '__main__':
    test()
