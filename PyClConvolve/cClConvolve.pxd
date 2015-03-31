# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "CyWrappers.h":
    cdef void checkException( int *wasRaised, string *message )

cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        #pass
        NeuralNet() except +
        #void print()
        NeuralNet( int numPlanes, int size ) except +
        string asString() except +
        void setBatchSize( int batchSize ) except +
        void propagate( const float *images) except +
        void backPropFromLabels( float learningRate, const int *labels) except +
        void backProp( float learningRate, const float *expectedResults) except +
        int calcNumRight( const int *labels ) except +
        void addLayer( LayerMaker2 *maker ) except +

cdef extern from "NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdef( NeuralNet *net, string netdef ) except +

cdef extern from "CyNetLearner.h":
    cdef cppclass CyNetLearner[T]:
        CyNetLearner( NeuralNet *net ) except +
        void setTrainingData( int Ntrain, T *trainData, int *trainLabels ) except +
        void setTestingData( int Ntest, T *testData, int *testLabels ) except +
        void setSchedule( int numEpochs ) except +
        void setDumpTimings( bool dumpTimings ) except +
        void setBatchSize( int batchSize ) except +
        void learn( float learningRate ) nogil
        #void setSchedule( int numEpochs, int startEpoch )
        # VIRTUAL void addPostEpochAction( PostEpochAction *action );
        #void learn( float learningRate, float annealLearningRate )

cdef extern from "GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) except +
        @staticmethod
        void load( string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples ) except +

cdef extern from "LayerMaker.h":
    cdef cppclass LayerMaker2:
        pass

cdef extern from "NormalizationLayerMaker.h":
    cdef cppclass NormalizationLayerMaker(LayerMaker2):
        NormalizationLayerMaker *translate( float translate ) except +
        NormalizationLayerMaker *scale( float scale ) except +
        @staticmethod
        NormalizationLayerMaker *instance() except +

cdef extern from "FullyConnectedMaker.h":
    cdef cppclass FullyConnectedMaker(LayerMaker2):
        FullyConnectedMaker *numPlanes( int numPlanes ) except +
        FullyConnectedMaker *imageSize( int imageSize ) except +
        FullyConnectedMaker *biased() except +
        FullyConnectedMaker *biased(bint _biased) except +
        FullyConnectedMaker *linear() except +
        FullyConnectedMaker *tanh() except +
        FullyConnectedMaker *sigmoid() except +
        FullyConnectedMaker *relu() except +
        @staticmethod
        FullyConnectedMaker *instance() except +

cdef extern from "ConvolutionalMaker.h":
    cdef cppclass ConvolutionalMaker(LayerMaker2):
        ConvolutionalMaker *numFilters( int numFilters ) except +
        ConvolutionalMaker *filterSize( int imageSize ) except +
        ConvolutionalMaker *padZeros() except +
        ConvolutionalMaker *padZeros(bint _padZeros) except +
        ConvolutionalMaker *biased() except +
        ConvolutionalMaker *biased(bint _biased) except +
        ConvolutionalMaker *linear() except +
        ConvolutionalMaker *tanh() except +
        ConvolutionalMaker *sigmoid() except +
        ConvolutionalMaker *relu() except +
        @staticmethod
        ConvolutionalMaker *instance() except +

cdef extern from "PoolingMaker.h":
    cdef cppclass PoolingMaker(LayerMaker2):
        PoolingMaker *poolingSize( int _poolingsize ) except +
        PoolingMaker *padZeros( int _padZeros ) except +
        @staticmethod
        PoolingMaker *instance() except +

cdef extern from "LayerMaker.h":
    cdef cppclass SquareLossMaker(LayerMaker2):
        @staticmethod
        SquareLossMaker *instance() except +

cdef extern from "LayerMaker.h":
    cdef cppclass SoftMaxMaker(LayerMaker2):
        @staticmethod
        SoftMaxMaker *instance()

cdef extern from "InputLayerMaker.h":
    cdef cppclass InputLayerMaker[T](LayerMaker2):
        InputLayerMaker *numPlanes( int _numPlanes ) except +
        InputLayerMaker *imageSize( int _imageSize ) except +
        @staticmethod
        InputLayerMaker *instance() except +

cdef extern from "QLearner.h":
    cdef cppclass QLearner:
        QLearner( CyScenario *scenario, NeuralNet *net ) except +
        void run() except +

cdef extern from "CyScenario.h":
    cdef cppclass CyScenario:
        CyScenario(void *pyObject)
        #void printScenario()
        #void printQRepresentation(NeuralNet *net)
        #int getPerceptionSize()
        #int getPerceptionPlanes()
        #void getPerception( float *perception )
        #void reset()
        #int getNumActions()
        #float act( int index )
        #bool hasFinished()


