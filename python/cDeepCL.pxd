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

#[[[cog
# import ScenarioDefs
# defs = ScenarioDefs.defs
# upperFirst = ScenarioDefs.upperFirst
#]]]
#[[[end]]]

#[[[cog
# cog.outl("# generated using cog:")
# import ScenarioDefs
# defs = ScenarioDefs.defs
# upperFirst = ScenarioDefs.upperFirst
#
# for thisdef in defs:
#     ( name, returnType, parameters ) = thisdef
#     cog.out('ctypedef ' + returnType + '(*CyScenario_' + name + 'Def)(')
#     for parameter in parameters:
#         (ptype,pname) = parameter
#         cog.out( ptype + ' ' + pname + ',')
#     cog.outl( ' void *pyObject)')
#]]]
# generated using cog:
ctypedef void(*CyScenario_printDef)( void *pyObject)
ctypedef void(*CyScenario_printQRepresentationDef)(NeuralNet * net, void *pyObject)
ctypedef int(*CyScenario_getPerceptionSizeDef)( void *pyObject)
ctypedef int(*CyScenario_getPerceptionPlanesDef)( void *pyObject)
ctypedef void(*CyScenario_getPerceptionDef)(float * perception, void *pyObject)
ctypedef void(*CyScenario_resetDef)( void *pyObject)
ctypedef int(*CyScenario_getNumActionsDef)( void *pyObject)
ctypedef float(*CyScenario_actDef)(int index, void *pyObject)
ctypedef bool(*CyScenario_hasFinishedDef)( void *pyObject)
#[[[end]]]

cdef extern from "CyScenario.h":
    cdef cppclass CyScenario:
        CyScenario(void *pyObject)

        #[[[cog
        # cog.outl("# generated using cog:")
        # for thisdef in defs:
        #     ( name, returnType, parameters ) = thisdef
        #     cog.outl( 'void set' + upperFirst( name ) + ' ( CyScenario_' + name + 'Def c' + upperFirst( name ) + ' )')
        # ]]]
        # generated using cog:
        void setPrint ( CyScenario_printDef cPrint )
        void setPrintQRepresentation ( CyScenario_printQRepresentationDef cPrintQRepresentation )
        void setGetPerceptionSize ( CyScenario_getPerceptionSizeDef cGetPerceptionSize )
        void setGetPerceptionPlanes ( CyScenario_getPerceptionPlanesDef cGetPerceptionPlanes )
        void setGetPerception ( CyScenario_getPerceptionDef cGetPerception )
        void setReset ( CyScenario_resetDef cReset )
        void setGetNumActions ( CyScenario_getNumActionsDef cGetNumActions )
        void setAct ( CyScenario_actDef cAct )
        void setHasFinished ( CyScenario_hasFinishedDef cHasFinished )
        # [[[end]]]

    ctypedef int(*CyMyInterface_getNumberDef)(void *pyObject)
    ctypedef void(*CyMyInterface_getFloatsDef)(float *floats, void *pyObject)

