# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function

import sys
import array
import PyDeepCL

def test_buildnet():
    print('test_buildnet')
    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(28) )
    net.addLayer( PyDeepCL.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
    net.addLayer( PyDeepCL.PoolingMaker().poolingSize(2) )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
    net.addLayer( PyDeepCL.PoolingMaker().poolingSize(3) )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(150).imageSize(1).biased().tanh() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(10).imageSize(1).biased().linear() )
    #net.addLayer( PyDeepCL.SquareLossMaker() )
    net.addLayer( PyDeepCL.SoftMaxMaker() )
    print( net.asString() )
    assert 9 == net.getNumLayers()

    assert 1 == net.getLayer(0).getOutputPlanes()
    assert 1 == net.getLayer(1).getOutputPlanes()
    assert 8 == net.getLayer(2).getOutputPlanes()
    assert 8 == net.getLayer(3).getOutputPlanes()
    assert 150 == net.getLayer(6).getOutputPlanes()
    assert 10 == net.getLayer(7).getOutputPlanes()
    assert 10 == net.getLayer(8).getOutputPlanes()

    exceptionCalled = False
    try:
        net.getLayer(10).getOutputPlanes()
    except:
        exceptionCalled = True
    assert exceptionCalled

    assert 28 == net.getLayer(0).getOutputImageSize()
    assert 28 == net.getLayer(1).getOutputImageSize()
    assert 28 == net.getLayer(2).getOutputImageSize()
    assert 14 == net.getLayer(3).getOutputImageSize()
    assert 1 == net.getLayer(8).getOutputImageSize()

#    assert not net.getLayer(0).getBiased()
#    assert net.getLayer(2).getBiased()
#    assert net.getLayer(5).getBiased()
#    assert net.getLayer(6).getBiased()

def test_getResults():
    # we probalby need to do some learning first, otherwise how can we 
    # know what we have is correct?

    # maybe try input layer first?
    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2) )
    net.addLayer( PyDeepCL.NormalizationLayerMaker().translate(-2.0).scale(1/4.0) )
    print( net.asString() )
    assert 2 == net.getNumLayers()
    net.setBatchSize(2)
    inputValues = [ 1,2,3,4, 5,6,7,8]
    net.propagateList( inputValues )

    results = net.getLayer(0).getResults()
    print('results', results )
    assert [1,2,3,4,5,6,7,8] == results.tolist()

    results = net.getLayer(1).getResults()
    print('results', results )
    assert map( lambda x: ( x - 2.0)/4.0, [1,2,3,4,5,6,7,8] ) == results.tolist()

    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2) )
    net.addLayer( PyDeepCL.PoolingMaker().poolingSize(2) )
    print( net.asString() )
    assert 2 == net.getNumLayers()
    net.setBatchSize(2)
    net.propagateList( inputValues )
    results = net.getLayer(1).getResults()
    print('results', results )
    assert [4,8] == results.tolist()

    # check net.getResults() ,should be the same
    results = net.getResults()
    print('results', results )
    assert [4,8] == results.tolist()

def test_getsetweights():
    # set some weights, try getting them, should give same values, and
    # not crash etc :-)
    # then try forward propagating, and it should prop according to these
    # weigths, eg try setting all weights to 0, or 1, or 2, to make
    # it easy ish, and turn off bias
    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2) )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(1).filterSize(1).linear() )
    net.setBatchSize(1)
    print( net.asString() )
    assert 2 == net.getNumLayers()
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 1 # since not biased
    assert weights[0] != 0 # since it's random, not much more we can say :-)

    # set weights, and check we can get them again
    net.getLayer(1).setWeightsList([2.5])
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 1 # since not biased
    assert weights[0] == 2.5

    # try with biased, check there are two weights now, or 5, since we make filter bigger
    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(2) )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(1).filterSize(2).biased().linear() )
    net.setBatchSize(1)
    print( net.asString() )
    assert 2 == net.getNumLayers()
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 5 # since not biased
    for weight in weights:
        assert weight != 0 # since it's random, not much more we can say :-)

    # set weights, and check we can get them again
    net.getLayer(1).setWeightsList([2.5,1,3,2,7])
    weights = net.getLayer(1).getWeights()
    print('weights',weights)
    assert len(weights) == 5 # since not biased
    assert weights.tolist() == [2.5,1,3,2,7]

def test_forcebackprop():
    # check can be instantiated ok
    # not sure how we check if it forces backprop...

    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(3).imageSize(28) )
    net.addLayer( PyDeepCL.ForceBackpropMaker() )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().linear() )
    net.setBatchSize(32)
    print( net.asString() )
    assert 3 == net.getNumLayers()
    print( net.getLayer(1) )
    print( net.getLayer(1).asString() )
    print( net.getLayer(1).getClassName() )
    assert "ForceBackpropMaker", net.getLayer(1).getClassName() 

