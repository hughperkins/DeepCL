# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
from cython cimport view
from cpython cimport array as c_array
# from array import array
import threading
from libcpp cimport bool
import platform

cimport CppRuntimeBoundary

cimport cDeepCL

major_version = platform.python_version_tuple()[0]
if major_version == '2':
  intArrayType = b'i'
  floatArrayType = b'f'
else:
  intArrayType = 'i'
  floatArrayType = 'f'

include "DeepCL.pyx"
include "RandomSingleton.pyx"
#include "DeepCL.pyx"
include "Trainer.pyx"
include "SGD.pyx"
include "Annealer.pyx"
include "Nesterov.pyx"
include "Adagrad.pyx"
include "Rmsprop.pyx"
include "Adadelta.pyx"
include "NeuralNet.pyx"
include "Layer.pyx"
include "LayerMaker.pyx"
include "GenericLoader.pyx"
include "NetLearner.pyx"
include "NetDefToNet.pyx"
include "QLearning.pyx"

#def checkException():
#    cdef int threwException = 0
#    cdef string message = ""
#    cDeepCL.checkException( &threwException, &message)
#    # print('threwException: ' + str(threwException) + ' ' + message ) 
#    if threwException:
#        raise RuntimeError(message)

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


