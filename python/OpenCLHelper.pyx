cdef class OpenCLHelper:
    cdef cDeepCL.OpenCLHelper *thisptr

    def __cinit__(self, gpuindex=None ):
#        print( '__cinit__(planes,size)')
        if gpuindex is None:
             self.thisptr = cDeepCL.OpenCLHelper.createForFirstGpuOtherwiseCpu()
        else:
            self.thisptr = cDeepCL.OpenCLHelper.createForIndexedGpu(gpuindex)

    def __dealloc(self):
        del self.thisptr 

