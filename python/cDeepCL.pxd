# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from libcpp.string cimport string
from libcpp cimport bool

include "cLayerMaker.pxd"
include "cNeuralNet.pxd"
include "cGenericLoader.pxd"

cdef extern from "CyWrappers.h":
    cdef void checkException( int *wasRaised, string *message )

cdef extern from "Layer.h":
    cdef cppclass Layer:
        void propagate()
        void backProp( float learningRate )
        bool needsBackProp()
        bool getBiased()
        int getOutputCubeSize()
        int getOutputPlanes()
        int getOutputImageSize()
        float * getResults()
        int getResultsSize()
        int getPersistSize()
        void persistToArray(float *array)
        void unpersistFromArray(const float *array)
        string asString()
        string getClassName()

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


