cdef class EasyCL:
    cdef cDeepCL.EasyCL *thisptr

    def __cinit__(self, gpuindex=None ):
#        print( '__cinit__(planes,size)')
        if gpuindex is None:
             self.thisptr = cDeepCL.EasyCL.createForFirstGpuOtherwiseCpu()
        else:
            self.thisptr = cDeepCL.EasyCL.createForIndexedGpu(gpuindex)

    def __dealloc(self):
        del self.thisptr 

