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
include "cNetDefToNet.pxd"
include "cNetLearner.pxd"
include "cLayer.pxd"

cdef extern from "CyWrappers.h":
    cdef void checkException( int *wasRaised, string *message )

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


