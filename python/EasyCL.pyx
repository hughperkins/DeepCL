#cdef class DeepCL:
#    cdef cDeepCL.DeepCL *thisptr

#    def __cinit__(self, gpuindex=None ):
##        print( '__cinit__(planes,size)')
#        if gpuindex is None:
#             self.thisptr = cDeepCL.DeepCL.createForFirstGpuOtherwiseCpu()
#        else:
#            self.thisptr = cDeepCL.DeepCL.createForIndexedGpu(gpuindex)

#    def __dealloc__(self):
#        del self.thisptr 

