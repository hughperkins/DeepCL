cdef class DeepCL:
    cdef cDeepCL.DeepCL *thisptr

    def __cinit__(self, gpuindex=None ):
#        print( '__cinit__(planes,size)')
        if gpuindex is None:
             self.thisptr = cDeepCL.DeepCL.createForFirstGpuOtherwiseCpu()
        else:
            self.thisptr = cDeepCL.DeepCL.createForIndexedGpu(gpuindex)

    def __dealloc__(self):
        self.thisptr.deleteMe()

    def setProfiling(self, profiling):
        self.thisptr.setProfiling(profiling)

    def dumpProfiling(self):
        self.thisptr.dumpProfiling()

    def getComputeUnits(self):
        return self.thisptr.getComputeUnits()

    def getLocalMemorySize(self):
        return self.thisptr.getLocalMemorySize()

    def getLocalMemorySizeKB(self):
        return self.thisptr.getLocalMemorySizeKB()

    def getMaxWorkgroupSize(self):
        return self.thisptr.getMaxWorkgroupSize()

    def getMaxAllocSizeMB(self):
        return self.thisptr.getMaxAllocSizeMB()

