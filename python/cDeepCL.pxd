# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from libcpp.string cimport string
from libcpp cimport bool

include "cEasyCL.pxd"
include "cLayerMaker.pxd"
include "cNeuralNet.pxd"
include "cSGD.pxd"
include "cGenericLoader.pxd"
include "cNetDefToNet.pxd"
include "cNetLearner.pxd"
include "cLayer.pxd"

cdef extern from "CyWrappers.h":
    cdef void checkException( int *wasRaised, string *message )

cdef extern from "QLearner.h":
    cdef cppclass QLearner:
        QLearner( SGD *sgd, CyScenario *scenario, NeuralNet *net ) except +
        void run() except +
        void setLambda( float thislambda )
        void setMaxSamples( int maxSamples )
        void setEpsilon( float epsilon )
        # void setLearningRate( float learningRate )

cdef extern from "CyScenario.h":
    #[[[cog
    # import ScenarioDefs
    # import cog_cython
    # cog_cython.pxd_write_proxy_class( 'CyScenario', ScenarioDefs.defs )
    #]]]
    # generated using cog (as far as the [[end]] bit:
    ctypedef int(*CyScenario_getPerceptionSizeDef)( void *pyObject)
    ctypedef int(*CyScenario_getPerceptionPlanesDef)( void *pyObject)
    ctypedef void(*CyScenario_getPerceptionDef)(float * perception, void *pyObject)
    ctypedef void(*CyScenario_resetDef)( void *pyObject)
    ctypedef int(*CyScenario_getNumActionsDef)( void *pyObject)
    ctypedef float(*CyScenario_actDef)(int index, void *pyObject)
    ctypedef bool(*CyScenario_hasFinishedDef)( void *pyObject)
    cdef cppclass CyScenario:
        CyScenario(void *pyObject)

        void setGetPerceptionSize ( CyScenario_getPerceptionSizeDef cGetPerceptionSize )
        void setGetPerceptionPlanes ( CyScenario_getPerceptionPlanesDef cGetPerceptionPlanes )
        void setGetPerception ( CyScenario_getPerceptionDef cGetPerception )
        void setReset ( CyScenario_resetDef cReset )
        void setGetNumActions ( CyScenario_getNumActionsDef cGetNumActions )
        void setAct ( CyScenario_actDef cAct )
        void setHasFinished ( CyScenario_hasFinishedDef cHasFinished )
    #[[[end]]]


