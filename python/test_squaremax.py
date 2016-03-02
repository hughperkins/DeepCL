#!/usr/bin/python

from __future__ import print_function
import array
import PyDeepCL
import sys
print('imports done')

cl = PyDeepCL.DeepCL()
net = PyDeepCL.NeuralNet(cl)
sgd = PyDeepCL.SGD(cl, 0.002, 0)
sgd.setMomentum(0.0001)

net = PyDeepCL.NeuralNet(cl, 1, 28)
net.addLayer(PyDeepCL.NormalizationLayerMaker().translate(-0.5).scale(1 / 255.0))
net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
net.addLayer(PyDeepCL.PoolingMaker().poolingSize(2))
net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(1).imageSize(1))
net.addLayer(PyDeepCL.SquareLossMaker())

print(net.asString())
N = 1280
batchSize = 128
planes = 1
size = 28
numEpochs = 3

images = array.array('f', [0] * (N*planes*size*size))
targets = array.array('f', [0] * N)

net.setBatchSize(batchSize)
for epoch in range(numEpochs):
    print('epoch', epoch)
    numRight = 0
    context = PyDeepCL.TrainingContext(epoch, 0)
    for batch in range(N // batchSize):
        sgd.train(
            net,
            context,
            images[batch * batchSize * planes * size * size:],
            targets[batch * batchSize:])

