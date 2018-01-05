#!/usr/bin/python

from __future__ import print_function
import sys
import numpy as np
import PyDeepCL

if len(sys.argv) != 2:
    print(
        'usage: python ' + sys.argv[0] +
        ' [mnist data directory (containing the .mat files)]')
    sys.exit(-1)

mnistFilePath = sys.argv[1] + '/t10k-images-idx3-ubyte'

cl = PyDeepCL.DeepCL()
net = PyDeepCL.NeuralNet(cl)
net = PyDeepCL.NeuralNet(cl)
net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(28))
net.addLayer(
    PyDeepCL.FullyConnectedMaker().numPlanes(150).imageSize(1).biased())
net.addLayer(PyDeepCL.ActivationMaker().scaledTanh())
net.addLayer(
    PyDeepCL.FullyConnectedMaker().numPlanes(10).imageSize(1).biased())
net.addLayer(PyDeepCL.SoftMaxMaker())

(N, planes, size) = PyDeepCL.GenericLoader.getDimensions(mnistFilePath)
print((N, planes, size))

N = 1280
batchSize = 128
numEpochs = 30

images = np.zeros((N, planes, size, size), dtype=np.float32)
labels = np.zeros((N,), dtype=np.int32)
PyDeepCL.GenericLoader.load(mnistFilePath, images, labels, 0, N)

sgd = PyDeepCL.SGD(cl, 0.002, 0)
sgd.setMomentum(0.0001)
print('trainer', sgd)
netLearner = PyDeepCL.NetLearner(
    sgd, net,
    N, images, labels,
    N, images, labels,
    128)

trainer = PyDeepCL.Adagrad(cl, 0.002, 0)
print('trainer', trainer)
netLearner = PyDeepCL.NetLearner(
    trainer, net,
    N, images, labels,
    N, images, labels,
    128)

trainer = PyDeepCL.Nesterov(cl, 0.002, 0)
print('trainer', trainer)
netLearner = PyDeepCL.NetLearner(
    trainer, net,
    N, images, labels,
    N, images, labels,
    128)

trainer = PyDeepCL.Rmsprop(cl, 0.002, 0)
print('trainer', trainer)
netLearner = PyDeepCL.NetLearner(
    trainer, net,
    N, images, labels,
    N, images, labels,
    128)

trainer = PyDeepCL.Annealer(cl, 0.002, 0)
print('trainer', trainer)
netLearner = PyDeepCL.NetLearner(
    trainer, net,
    N, images, labels,
    N, images, labels,
    128)

