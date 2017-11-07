# Copyright Hugh Perkins 2016
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.
"""
Simple example of xor
"""
from __future__ import print_function
import PyDeepCL
import random
import numpy as np

def go():
    print('xor')

    data = [
        {'data': [-1, -1], 'label': 0},
        {'data': [1, -1], 'label': 1},
        {'data': [-1, 1], 'label': 1},
        {'data': [1, 1], 'label': 0}
    ]

    N = len(data)
    batchSize = N
    planes = 2
    size = 1
    learningRate = 0.1
    numEpochs = 3000

    # seed just makes it repeatable.  Normally you wouldnt want this (except for unit tests, or for
    # comparing with other networks etc, for correctness checking)
    PyDeepCL.RandomSingleton.seed(1234)
    cl = PyDeepCL.DeepCL(gpuindex=1)
    net = PyDeepCL.NeuralNet(cl, planes, size)
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(2).filterSize(1).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().sigmoid())
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(2).filterSize(1).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().sigmoid())
    net.addLayer(PyDeepCL.SoftMaxMaker())
    print(net.asString())

    images = np.zeros((N, planes, size, size), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int32)
    for n in range(N):
        for plane in range(planes):
            images[n,plane,0,0] = data[n]['data'][plane]
        labels[n] = data[n]['label']

    sgd = PyDeepCL.SGD(cl, learningRate, 0.0)
    netLearner = PyDeepCL.NetLearner(
        sgd, net,
        N, images, labels,
        N, images, labels,
        batchSize)
    netLearner.setSchedule(numEpochs)
    netLearner.run()

if __name__ == '__main__':
    go()

