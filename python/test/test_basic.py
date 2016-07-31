# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function

import sys
import numpy as np
import PyDeepCL

def test_buildnet():
    print('test_buildnet')
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(28))
    net.addLayer(PyDeepCL.NormalizationLayerMaker().translate(-0.5).scale(1/255.0))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.PoolingMaker().poolingSize(2))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.PoolingMaker().poolingSize(3))
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(150).imageSize(1).biased())
    net.addLayer(PyDeepCL.ActivationMaker().tanh())
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(10).imageSize(1).biased())
    #net.addLayer(PyDeepCL.SquareLossMaker())
    net.addLayer(PyDeepCL.SoftMaxMaker())
    print(net.asString())
    assert 12 == net.getNumLayers()

    assert 1 == net.getLayer(0).getOutputPlanes()
    assert 1 == net.getLayer(1).getOutputPlanes()
    assert 8 == net.getLayer(2).getOutputPlanes()
    assert 8 == net.getLayer(3).getOutputPlanes()
    assert 150 == net.getLayer(9).getOutputPlanes()
    assert 10 == net.getLayer(10).getOutputPlanes()
    assert 10 == net.getLayer(11).getOutputPlanes()

    exceptionCalled = False
    try:
        net.getLayer(14).getOutputPlanes()
    except:
        exceptionCalled = True
    assert exceptionCalled

    assert 28 == net.getLayer(0).getOutputSize()
    assert 28 == net.getLayer(1).getOutputSize()
    assert 28 == net.getLayer(2).getOutputSize()
    assert 14 == net.getLayer(4).getOutputSize()
    assert 1 == net.getLayer(8).getOutputSize()

#    assert not net.getLayer(0).getBiased()
#    assert net.getLayer(2).getBiased()
#    assert net.getLayer(5).getBiased()
#    assert net.getLayer(6).getBiased()

def test_getResults():
    # we probalby need to do some learning first, otherwise how can we 
    # know what we have is correct?

    # maybe try input layer first?
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2))
    net.addLayer(PyDeepCL.NormalizationLayerMaker().translate(-2.0).scale(1/4.0))
    print(net.asString())
    assert 2 == net.getNumLayers()
    net.setBatchSize(2)
    inputValues = np.array([ 1,2,3,4, 5,6,7,8], dtype=np.float32)
    net.forward(inputValues)

    results = net.getLayer(0).getOutput()
    print('results', results)
    assert (np.array([1,2,3,4,5,6,7,8], dtype=np.float32).reshape(2, 1, 2, 2) == results).all()

    results = net.getLayer(1).getOutput()
    print('results', results)
    expected_ls = [(x - 2.0)/4.0 for x in [1,2,3,4,5,6,7,8]]
    print('expected_ls', expected_ls)
    assert (np.array(expected_ls, dtype=np.float32).reshape(2,1,2,2) == results).all()

    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2))
    net.addLayer(PyDeepCL.PoolingMaker().poolingSize(2))
    print(net.asString())
    assert 2 == net.getNumLayers()
    net.setBatchSize(2)
    net.forward(inputValues)
    results = net.getLayer(1).getOutput()
    print('results', results)
    assert (np.array([4,8], dtype=np.float32).reshape(2,1,1,1) == results).all()

    # check net.getResults() ,should be the same
    results = net.getOutput()
    print('results', results)
    assert (np.array([4,8], dtype=np.float32).reshape(2,1,1,1) == results).all()

def test_getsetweights():
    # set some weights, try getting them, should give same values, and
    # not crash etc :-)
    # then try forward propagating, and it should prop according to these
    # weigths, eg try setting all weights to 0, or 1, or 2, to make
    # it easy ish, and turn off bias
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(1).filterSize(1).biased(False))
    net.setBatchSize(1)
    print(net.asString())
    assert 2 == net.getNumLayers()
    print('net.getLayer(1).getBiased()', net.getLayer(1).getBiased())
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert weights.size == 1 # since not biased
    assert weights[0] != 0 # since it's random, not much more we can say :-)

    # set weights, and check we can get them again
    net.getLayer(1).setWeights(np.array([2.5], dtype=np.float32))
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 1 # since not biased
    assert weights[0] == 2.5

    # try with biased, check there are two weights now, or 5, since we make filter bigger
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(1).filterSize(2).biased())
    net.setBatchSize(1)
    print(net.asString())
    assert 2 == net.getNumLayers()
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 5 # since not biased
    for weight in weights:
        assert weight != 0 # since it's random, not much more we can say :-)

    # set weights, and check we can get them again
    net.getLayer(1).setWeights(np.array([2.5,1,3,2,7], dtype=np.float32))
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 5 # since not biased
    assert weights.tolist() == [2.5,1,3,2,7]

def test_convmaker():
    convmaker = PyDeepCL.ConvolutionalMaker()
    convmaker.padZeros()
    convmaker.padZeros(True)
    convmaker.padZeros(False)

def test_forcebackprop():
    # check can be instantiated ok
    # not sure how we check if it forces backprop...

    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(3).imageSize(28))
    net.addLayer(PyDeepCL.ForceBackpropMaker())
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
    net.setBatchSize(32)
    print(net.asString())
    assert 3 == net.getNumLayers()
    print(net.getLayer(1))
    print(net.getLayer(1).asString())
    print(net.getLayer(1).getClassName())
    assert "ForceBackpropMaker", net.getLayer(1).getClassName() 

def test_getoutputcubesize():
    batchSize = 32
    planes = 3
    size = 28
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(PyDeepCL.RandomTranslationsMaker().translateSize(3))
    net.setBatchSize(batchSize)
    images = np.zeros((batchSize, planes, size, size), dtype=np.float32)
    net.forward(images)
    print('net.getOutputCubeSize()', net.getLayer(1).getOutputCubeSize())
    assert net.getLayer(1).getOutputCubeSize() == batchSize * planes * size * size

def test_getweights():
    batchSize = 32
    planes = 3
    size = 28
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(PyDeepCL.RandomTranslationsMaker().translateSize(3))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(4).filterSize(3).padZeros().biased())
    net.setBatchSize(batchSize)
    images = np.zeros((batchSize, planes, size, size), dtype=np.float32)
    net.forward(images)
    print('net.getLayer(1).getWeights()', net.getLayer(1).getWeights())
    print('net.getLayer(2).getWeights().shape', net.getLayer(2).getWeights().shape)

def test_setweights():
    batchSize = 32
    planes = 3
    size = 28
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(PyDeepCL.RandomTranslationsMaker().translateSize(3))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(4).filterSize(3).padZeros().biased())
    for i in range(net.getNumLayers()):
        layer = net.getLayer(i)
        weights = layer.getWeights()
        if weights is None:
            continue
        weightsSize = weights.size
        print('i', i, 'weightsSize', weightsSize)
        weights2 = np.zeros((weightsSize,), dtype=np.float32)
        weights2[:] = np.random.uniform(-0.5, 0.5, weights2.shape)
        layer.setWeights(weights2)
        weights3 = layer.getWeights()
        for j in range(weights2.size):
            assert weights2[j] == weights3[j]

