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

