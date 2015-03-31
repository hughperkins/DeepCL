#!/usr/bin/python

from __future__ import print_function

#from array import array
import sys
import array
import PyClConvolve

if len(sys.argv) != 2:
    print('usage: python ' + sys.argv[0] + ' [mnist data directory (containing the .mat files)]')
    sys.exit(-1)

mnistFilePath = sys.argv[1] + '/t10k-dat.mat' # '../ClConvolve/data/mnist/t10k-dat.mat'

net = PyClConvolve.NeuralNet()
net.addLayer( PyClConvolve.InputLayerMaker().numPlanes(1).imageSize(28) )
net.addLayer( PyClConvolve.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
net.addLayer( PyClConvolve.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
net.addLayer( PyClConvolve.PoolingMaker().poolingSize(2) )
net.addLayer( PyClConvolve.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
net.addLayer( PyClConvolve.PoolingMaker().poolingSize(3) )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(150).imageSize(1).biased().tanh() )
net.addLayer( PyClConvolve.FullyConnectedMaker().numPlanes(10).imageSize(1).biased().linear() )
#net.addLayer( PyClConvolve.SquareLossMaker() )
net.addLayer( PyClConvolve.SoftMaxMaker() )
print( net.asString() )

#mnistFilePath = '../ClConvolve/data/mnist/t10k-dat.mat'
(N,planes,size) = PyClConvolve.GenericLoader.getDimensions(mnistFilePath)
print( (N,planes,size) )

N = 1280
batchSize = 128
numEpochs = 30

images = array.array( 'f', [0] * (N*planes*size*size) )
labels = array.array('i',[0] * N )
PyClConvolve.GenericLoader.load(mnistFilePath, images, labels, 0, N )

net.setBatchSize(batchSize)
for epoch in range(numEpochs): 
    numRight = 0
    for batch in range( N // batchSize ):
        net.propagate( images[batch * batchSize:] )
        net.backPropFromLabels( 0.002, labels[batch * batchSize:] )
        numRight += net.calcNumRight( labels[batch * batchSize:] )
        # print( 'numright ' + str( net.calcNumRight( labels ) ) )
#    print( 'loss ' + str( loss ) )
    print( 'num right: ' + str(numRight) )

