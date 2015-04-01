#!/usr/bin/python

from __future__ import print_function

#from array import array
import array
import PyClConvolve
import sys
print('imports done')

if len(sys.argv) != 2:
    print('usage: python ' + sys.argv[0] + ' [mnist data directory (containing the .mat files)]')
    sys.exit(-1)

mnistFilePath = sys.argv[1] + '/t10k-dat.mat' # '../ClConvolve/data/mnist/t10k-dat.mat'

net = PyClConvolve.NeuralNet(1,28)
print('created net')
print( net.asString() )
print('printed net')
net.addLayer( PyClConvolve.NormalizationLayerMaker().translate(-0.5).scale(1/255.0) )
print('added layer ')
PyClConvolve.NetdefToNet.createNetFromNetdef( net, "rt2-8c5-mp2-16c5-mp3-150n-10n" ) 
print( net.asString() )
 
(N,planes,size) = PyClConvolve.GenericLoader.getDimensions(mnistFilePath)
print( (N,planes,size) )

N = 1280
images = array.array( 'f', [0] * (N*planes*size*size) )
labels = array.array('i',[0] * N )
PyClConvolve.GenericLoader.load(mnistFilePath, images, labels, 0, N )

netLearner = PyClConvolve.NetLearner( net )
netLearner.setTrainingData( N, images, labels )
netLearner.setTestingData( N, images, labels )
netLearner.setSchedule( 12 )
netLearner.setBatchSize( 128 )
netLearner.learn( 0.002 )
 

