# Copyright Hugh Perkins 2015
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from libcpp cimport bool

cdef extern from "DeepCL.h":
    cdef cppclass DeepCL:
        @staticmethod
        DeepCL *createForFirstGpuOtherwiseCpu() except+
        @staticmethod
        DeepCL *createForIndexedGpu( int gpu ) except+

        void deleteMe()

        void setProfiling(bool profiling)
        void dumpProfiling();

        int getComputeUnits()
        int getLocalMemorySize()
        int getLocalMemorySizeKB()
        int getMaxWorkgroupSize()
        int getMaxAllocSizeMB()

include "cRandomSingleton.pxd"
include "cLayerMaker.pxd"
include "cNeuralNet.pxd"
include "cTrainer.pxd"
include "cSGD.pxd"
include "cAnnealer.pxd"
include "cNesterov.pxd"
include "cAdagrad.pxd"
include "cRmsprop.pxd"
include "cAdadelta.pxd"
include "cNetLearner.pxd"
include "cGenericLoader.pxd"
include "cNetDefToNet.pxd"
include "cLayer.pxd"
include "cQLearning.pxd"

# cdef extern from "CyWrappers.h":
#     cdef void checkException( int *wasRaised, string *message )

