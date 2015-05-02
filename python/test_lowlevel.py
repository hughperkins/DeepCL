#!/usr/bin/python

from __future__ import print_function

#from array import array
import sys
import array
import PyDeepCL

if len(sys.argv) != 2:
    print('usage: python ' + sys.argv[0] + ' [mnist data directory (containing the .mat files)]')
    sys.exit(-1)

mnistFilePath = sys.argv[1] + '/t10k-images-idx3-ubyte' 

cl = PyDeepCL.EasyCL()
net = PyDeepCL.NeuralNet(cl)
sgd = PyDeepCL.SGD( cl, 0.002, 0 )
sgd.setMomentum( 0.0001 )
net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(28) )
net.addLayer( PyDeepCL.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased() )
net.addLayer( PyDeepCL.ActivationMaker().relu() )
net.addLayer( PyDeepCL.PoolingMaker().poolingSize(2) )
net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased() )
net.addLayer( PyDeepCL.ActivationMaker().relu() )
net.addLayer( PyDeepCL.PoolingMaker().poolingSize(3) )
net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(150).imageSize(1).biased() )
net.addLayer( PyDeepCL.ActivationMaker().tanh() )
net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(10).imageSize(1).biased() )
#net.addLayer( PyDeepCL.SquareLossMaker() )
net.addLayer( PyDeepCL.SoftMaxMaker() )
print( net.asString() )

(N,planes,size) = PyDeepCL.GenericLoader.getDimensions(mnistFilePath)
print( (N,planes,size) )

N = 1280
batchSize = 128
numEpochs = 30

images = array.array( 'f', [0] * (N*planes*size*size) )
labels = array.array('i',[0] * N )
PyDeepCL.GenericLoader.load(mnistFilePath, images, labels, 0, N )

net.setBatchSize(batchSize)
for epoch in range(numEpochs): 
    numRight = 0
    context = PyDeepCL.TrainingContext(epoch, 0)
    for batch in range( N // batchSize ):
        sgd.trainFromLabels( net, context, images[batch * batchSize * planes * size * size:], labels[batch * batchSize:] )
#        net.forward( images[batch * batchSize * planes * size * size:] )
#        net.backwardFromLabels( labels[batch * batchSize:] )
        net.forward( images[batch * batchSize * planes * size * size:] )
        numRight += net.calcNumRight( labels[batch * batchSize:] )
        # print( 'numright ' + str( net.calcNumRight( labels ) ) )
#    print( 'loss ' + str( loss ) )
    print( 'num right: ' + str(numRight) )

