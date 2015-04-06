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
from libcpp cimport bool

cimport cDeepCL

include "NeuralNet.pyx"
include "Layer.pyx"
include "LayerMaker.pyx"
include "GenericLoader.pyx"
include "NetLearner.pyx"
include "NetDefToNet.pyx"

def checkException():
    cdef int threwException = 0
    cdef string message = ""
    cDeepCL.checkException( &threwException, &message)
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

#[[[cog
# cog.outl("# generated using cog:")
# import ScenarioDefs
# defs = ScenarioDefs.defs
# upperFirst = ScenarioDefs.upperFirst
#
# for thisdef in defs:
#     ( name, returnType, parameters ) = thisdef
#     if not name in ['printQRepresentation', 'getPerception', 'print']:
#         cog.out('cdef ' + returnType + ' Scenario_' + name + '( ')
#         for (ptype,pname) in parameters:
#             cog.out( ptype + ' ' + pname + ', ' )
#             isFirst = False
#         cog.outl( ' void *pyObject ):')
#         cog.out( '    ')
#         if returnType != 'void':
#             cog.out( 'return ')
#         cog.out( '(<object>pyObject).' + name + '(')
#         isFirst = True
#         for (ptype,pname) in parameters:
#             if not isFirst:
#                 cog.out(', ')
#             cog.out( pname )
#             isFirst = False
#         cog.outl( ')' )
#         cog.outl( '' )
#]]]
# generated using cog:
cdef int Scenario_getPerceptionSize(  void *pyObject ):
    return (<object>pyObject).getPerceptionSize()

cdef int Scenario_getPerceptionPlanes(  void *pyObject ):
    return (<object>pyObject).getPerceptionPlanes()

cdef void Scenario_reset(  void *pyObject ):
    (<object>pyObject).reset()

cdef int Scenario_getNumActions(  void *pyObject ):
    return (<object>pyObject).getNumActions()

cdef float Scenario_act( int index,  void *pyObject ):
    return (<object>pyObject).act(index)

cdef bool Scenario_hasFinished(  void *pyObject ):
    return (<object>pyObject).hasFinished()

#[[[end]]]

cdef void Scenario_print(  void *pyObject ):
    (<object>pyObject).output()

#cdef void MyInterface_getFloats( float *floats, void *pyObject ):
##    cdef float[:]floatsMv = floats
#    pyFloats = (<object>pyObject).getFloats()
#    for i in range(len(pyFloats)):
#        floats[i] = pyFloats[i]

cdef class Scenario:
    cdef cDeepCL.CyScenario *thisptr
    def __cinit__(self):
        self.thisptr = new cDeepCL.CyScenario(<void *>self )
        #[[[cog
        # cog.outl("# generated using cog:")
        # for thisdef in defs:
        #     ( name, returnType, parameters ) = thisdef
        #     if name in ['printQRepresentation', 'getPerception']:
        #         continue
        #     cog.outl('self.thisptr.set' + upperFirst( name ) + '( Scenario_' + name + ' )' )
        #]]]
        # generated using cog:
        self.thisptr.setPrint( Scenario_print )
        self.thisptr.setGetPerceptionSize( Scenario_getPerceptionSize )
        self.thisptr.setGetPerceptionPlanes( Scenario_getPerceptionPlanes )
        self.thisptr.setReset( Scenario_reset )
        self.thisptr.setGetNumActions( Scenario_getNumActions )
        self.thisptr.setAct( Scenario_act )
        self.thisptr.setHasFinished( Scenario_hasFinished )
        #[[[end]]]
#        self.thisptr.setGetNumberCallback( MyInterface_getNumber )
#        self.thisptr.setGetFloatsCallback( MyInterface_getFloats )
    def __dealloc__(self):
        del self.thisptr
#    def getNumber(self):
#        return 0 # placeholder
#    def getFloats(self):
#        return [] # placeholder

