# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from cython cimport view
from cpython cimport array as c_array
from array import array
import threading
from libcpp.string cimport string
cimport cClConvolve

def checkException():
    cdef int threwException = 0
    cdef string message = ""
    cClConvolve.checkException( &threwException, &message)
    # print('threwException: ' + str(threwException) + ' ' + message ) 
    if threwException:
        raise RuntimeError(message)

def interruptableCall( function, args ):
    mythread = threading.Thread( target=function, args = args )
    mythread.daemon = True
    mythread.start()
    while mythread.isAlive():
        mythread.join(0.1)
        #print('join timed out')

def toCppString( pyString ):
    if isinstance( pyString, unicode ):
        return pyString.encode('utf8')
    return pyString

cdef class NeuralNet:
    cdef cClConvolve.NeuralNet *thisptr

    def __cinit__(self, planes = None, size = None):
        print( '__cinit__(planes,size)')
        if planes == None and size == None:
             self.thisptr = new cClConvolve.NeuralNet()
        else:
            self.thisptr = new cClConvolve.NeuralNet(planes, size)

    def asString(self):
        return self.thisptr.asString()

#    def myprint(self):
#        self.thisptr.print()

    def setBatchSize( self, int batchSize ):
        self.thisptr.setBatchSize( batchSize ) 
    #def propagate( self, const unsigned char[:] images):
    #    self.thisptr.propagate( &images[0] )
    def propagate( self, const float[:] images):
        self.thisptr.propagate( &images[0] )
    def backPropFromLabels( self, float learningRate, int[:] labels):
        return self.thisptr.backPropFromLabels( learningRate, &labels[0] ) 
    def backProp( self, float learningRate, float[:] expectedResults):
        return self.thisptr.backProp( learningRate, &expectedResults[0] )
    def calcNumRight( self, int[:] labels ):
        return self.thisptr.calcNumRight( &labels[0] )
    def addLayer( self, LayerMaker2 layerMaker ):
        self.thisptr.addLayer( layerMaker.baseptr )

cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        return cClConvolve.NetdefToNet.createNetFromNetdef( neuralnet.thisptr, toCppString( netdef ) )

cdef class NetLearner: 
    cdef cClConvolve.CyNetLearner[float] *thisptr
    def __cinit__( self, NeuralNet neuralnet ):
        self.thisptr = new cClConvolve.CyNetLearner[float]( neuralnet.thisptr )
    def setTrainingData( self, Ntrain, float[:] trainData, int[:] trainLabels ):
        self.thisptr.setTrainingData( Ntrain, &trainData[0], &trainLabels[0] )
    def setTestingData( self, Ntest, float[:] testData, int[:] testLabels ):
        self.thisptr.setTestingData( Ntest, &testData[0], &testLabels[0] )
    def setSchedule( self, numEpochs ):
        self.thisptr.setSchedule( numEpochs )
    def setDumpTimings( self, bint dumpTimings ):
        self.thisptr.setDumpTimings( dumpTimings )
    def setBatchSize( self, batchSize ):
        self.thisptr.setBatchSize( batchSize )
    def _learn( self, float learningRate ):
        with nogil:
            self.thisptr.learn( learningRate )
    def learn( self, float learningRate ):
        interruptableCall( self._learn, [ learningRate ] ) 
#        with nogil:
#            thisptr._learn( learningRate )
        checkException()

cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilePath ):
        cdef int N
        cdef int planes
        cdef int size
        cClConvolve.GenericLoader.getDimensions( toCppString( trainFilePath ), &N, &planes, &size )
        # print( N )
        return (N,planes,size)
    @staticmethod 
    def loaduc( trainFilepath, unsigned char[:] images, int[:] labels, startN, numExamples ):
        #(N, planes, size) = getDimensions(trainFilepath)
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        cClConvolve.GenericLoader.load( toCppString( trainFilepath ), &images[0], &labels[0], startN , numExamples )
        #return (images, labels)
    @staticmethod 
    def load( trainFilepath, float[:] images, int[:] labels, startN, numExamples ):
        (N, planes, size) = GenericLoader.getDimensions(toCppString(trainFilepath))
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        #cdef unsigned char ucImages[numExamples * planes * size * size]
        print( (N, planes, size ) )
        cdef c_array.array ucImages = array('B', [0] * (numExamples * planes * size * size) )
        cdef unsigned char[:] ucImagesMv = ucImages
        cClConvolve.GenericLoader.load( toCppString(trainFilepath), &ucImagesMv[0], &labels[0], startN , numExamples )
        #return (images, labels)
        cdef int i
        cdef int total
        total = numExamples * planes * size * size
        print(total)
        for i in range(total):
            images[i] = ucImagesMv[i]

cdef class LayerMaker2:
    cdef cClConvolve.LayerMaker2 *baseptr

cdef class NormalizationLayerMaker(LayerMaker2):
    cdef cClConvolve.NormalizationLayerMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.NormalizationLayerMaker()
        self.baseptr = self.thisptr
    def translate( self, float _translate ):
        self.thisptr.translate( _translate )
        return self
    def scale( self, float _scale ):
        self.thisptr.scale( _scale )
        return self
    @staticmethod
    def instance():
        return NormalizationLayerMaker()

cdef class FullyConnectedMaker(LayerMaker2):
    cdef cClConvolve.FullyConnectedMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.FullyConnectedMaker()
        self.baseptr = self.thisptr
    def numPlanes( self, int _numPlanes ):
        self.thisptr.numPlanes( _numPlanes )
        return self
    def imageSize( self, int _imageSize ):
        self.thisptr.imageSize( _imageSize )
        return self
    def biased(self):
        self.thisptr.biased()
        return self
    def biased(self, int _biased):
        self.thisptr.biased( _biased )
        return self
    def linear(self):
        self.thisptr.linear()
        return self
    def tanh(self):
        self.thisptr.tanh()
        return self
    def sigmoid(self):
        self.thisptr.sigmoid()
        return self
    def relu(self):
        self.thisptr.relu()
        return self
    @staticmethod
    def instance():
        return FullyConnectedMaker()

cdef class ConvolutionalMaker(LayerMaker2):
    cdef cClConvolve.ConvolutionalMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.ConvolutionalMaker()
        self.baseptr = self.thisptr
    def numFilters( self, int _numFilters ):
        self.thisptr.numFilters( _numFilters )
        return self
    def filterSize( self, int _filterSize ):
        self.thisptr.filterSize( _filterSize )
        return self
    def padZeros(self):
        self.thisptr.padZeros()
        return self
    def padZeros(self, bint _padZeros):
        self.thisptr.padZeros( _padZeros )
        return self
    def biased(self):
        self.thisptr.biased()
        return self
    def biased(self, bint _biased):
        self.thisptr.biased( _biased )
        return self
    def linear(self):
        self.thisptr.linear()
        return self
    def tanh(self):
        self.thisptr.tanh()
        return self
    def sigmoid(self):
        self.thisptr.sigmoid()
        return self
    def relu(self):
        self.thisptr.relu()
        return self
    @staticmethod
    def instance():
        return ConvolutionalMaker()

cdef class PoolingMaker(LayerMaker2):
    cdef cClConvolve.PoolingMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.PoolingMaker()
        self.baseptr = self.thisptr
    def poolingSize( self, int _poolingSize ):
        self.thisptr.poolingSize( _poolingSize )
        return self
    @staticmethod
    def instance():
        return PoolingMaker()

cdef class SquareLossMaker(LayerMaker2):
    cdef cClConvolve.SquareLossMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.SquareLossMaker()
        self.baseptr = self.thisptr
    @staticmethod
    def instance():
        return SquareLossMaker()

cdef class SoftMaxMaker(LayerMaker2):
    cdef cClConvolve.SoftMaxMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.SoftMaxMaker()
        self.baseptr = self.thisptr
    @staticmethod
    def instance():
        return SoftMaxMaker()

cdef class InputLayerMaker(LayerMaker2):
    cdef cClConvolve.InputLayerMaker[float] *thisptr
    def __cinit__( self ):
        self.thisptr = new cClConvolve.InputLayerMaker[float]()
        self.baseptr = self.thisptr
    def numPlanes( self, int _numPlanes ):
        self.thisptr.numPlanes( _numPlanes )
        return self
    def imageSize( self, int _imageSize ):
        self.thisptr.imageSize( _imageSize )
        return self
    @staticmethod
    def instance():
        return InputLayerMaker()


