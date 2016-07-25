# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function

# import sys
# import array
import PyDeepCL
import random
import numpy as np

def go():
    print('xor')

    random.seed(1)
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl, 2, 1)
    # net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(2).imageSize(1))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(2).filterSize(1).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().sigmoid())
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(2).filterSize(1).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().sigmoid())
    # net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(2).imageSize(1).biased().relu())
    # net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(2).imageSize(1).biased().relu())
    # net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(10).imageSize(1).biased().linear() )
    #net.addLayer( PyDeepCL.SquareLossMaker() )
    net.addLayer(PyDeepCL.SoftMaxMaker())
    print(net.asString())

    data = [
        {'data': [-1, -1], 'label': 0},
        {'data': [1, -1], 'label': 1},
        {'data': [-1, 1], 'label': 1},
        {'data': [1, 1], 'label': 0}
    ]

    N = len(data)
    planes = 2
    size = 1
    images = np.zeros((N, planes, size, size), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int32)
    for n in range(N):
        images[n,0,0,0] = data[n]['data'][0]
        images[n,1,0,0] = data[n]['data'][1]
        labels[n] = data[n]['label']

    sgd = PyDeepCL.SGD(cl, 0.1, 0.0)
    netLearner = PyDeepCL.NetLearner(
        sgd, net,
        N, images.reshape((images.size,)), labels,
        N, images.reshape((images.size,)), labels,
        N)
    netLearner.setSchedule(2000)
    netLearner.run()

if __name__ == '__main__':
    go()

