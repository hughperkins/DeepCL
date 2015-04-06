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
#         cog.outl('    print("cdef Scenario_' + name + '")')
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
    print("cdef Scenario_getPerceptionSize")
    return (<object>pyObject).getPerceptionSize()

cdef int Scenario_getPerceptionPlanes(  void *pyObject ):
    print("cdef Scenario_getPerceptionPlanes")
    return (<object>pyObject).getPerceptionPlanes()

cdef void Scenario_reset(  void *pyObject ):
    print("cdef Scenario_reset")
    (<object>pyObject).reset()

cdef int Scenario_getNumActions(  void *pyObject ):
    print("cdef Scenario_getNumActions")
    return (<object>pyObject).getNumActions()

cdef float Scenario_act( int index,  void *pyObject ):
    print("cdef Scenario_act")
    return (<object>pyObject).act(index)

cdef bool Scenario_hasFinished(  void *pyObject ):
    print("cdef Scenario_hasFinished")
    return (<object>pyObject).hasFinished()

#[[[end]]]

cdef void Scenario_print(  void *pyObject ):
    (<object>pyObject).show()

cdef void Scenario_getPerception( float *perception, void *pyObject ):
    pyPerception = (<object>pyObject).getPerception()
    for i in range(len(pyPerception)):
        perception[i] = pyPerception[i]

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
        #     if name in ['printQRepresentation']:
        #         continue
        #     cog.outl('self.thisptr.set' + upperFirst( name ) + '( Scenario_' + name + ' )' )
        #]]]
        # generated using cog:
        self.thisptr.setPrint( Scenario_print )
        self.thisptr.setGetPerceptionSize( Scenario_getPerceptionSize )
        self.thisptr.setGetPerceptionPlanes( Scenario_getPerceptionPlanes )
        self.thisptr.setGetPerception( Scenario_getPerception )
        self.thisptr.setReset( Scenario_reset )
        self.thisptr.setGetNumActions( Scenario_getNumActions )
        self.thisptr.setAct( Scenario_act )
        self.thisptr.setHasFinished( Scenario_hasFinished )
        #[[[end]]]
#        self.thisptr.setGetNumberCallback( MyInterface_getNumber )
#        self.thisptr.setGetFloatsCallback( MyInterface_getFloats )
    def __dealloc__(self):
        del self.thisptr

    #[[[cog
    # cog.outl("# generated using cog:")
    # for thisdef in defs:
    #     ( name, returnType, parameters ) = thisdef
    #     if name in ['print']:
    #         continue
    #     cog.out( 'def ' + name + '(self')
    #     isFirst = True
    #     for (ptype,pname) in parameters:
    #         #if not isFirst:
    #             #cog.out(', ')
    #         cog.out( ', ' + pname )
    #         isFirst = False
    #     cog.outl( '):')
    #     cog.outl('    print("' + name + '()")')
    #     cog.out( '    ')
    #     #if returnType != 'void':
    #     #cog.out('return ')
    #     cog.outl('#return None # placeholder')
    #     cog.outl('    raise Exception("Method needs to be overridden: Scenario.' + name + '()")')
    #     
    #]]]
    # generated using cog:
    def printQRepresentation(self, net):
        print("printQRepresentation()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.printQRepresentation()")
    def getPerceptionSize(self):
        print("getPerceptionSize()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.getPerceptionSize()")
    def getPerceptionPlanes(self):
        print("getPerceptionPlanes()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.getPerceptionPlanes()")
    def getPerception(self, perception):
        print("getPerception()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.getPerception()")
    def reset(self):
        print("reset()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.reset()")
    def getNumActions(self):
        print("getNumActions()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.getNumActions()")
    def act(self, index):
        print("act()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.act()")
    def hasFinished(self):
        print("hasFinished()")
        #return None # placeholder
        raise Exception("Method needs to be overridden: Scenario.hasFinished()")
    #[[[end]]]
    def show(self):
        print('show')
        raise Exception("Method needs to be overridden: Scenario.show()")

#    def getNumber(self):
#        return 0 # placeholder
#    def getFloats(self):
#        return [] # placeholder

cdef class QLearner:
    cdef cDeepCL.QLearner *thisptr
    def __cinit__(self,Scenario scenario,NeuralNet net):
        self.thisptr = new cDeepCL.QLearner(scenario.thisptr, net.thisptr)
    def __dealloc__(self):
        del self.thisptr
    def run(self):
        self.thisptr.run()

