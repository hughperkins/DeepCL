cdef class Layer:
    cdef cDeepCL.Layer *thisptr

    def __cinit__(self):
        pass
    cdef set_thisptr(self, cDeepCL.Layer *thisptr):
        self.thisptr = thisptr
    def forward(self):
        self.thisptr.forward()
    def backward(self):
        self.thisptr.backward()
    def needsBackProp(self):
        return self.thisptr.needsBackProp()
#    def getBiased( self ):
#        return self.thisptr.getBiased()
    def getOutputCubeSize(self):
        return self.thisptr.getOutputCubeSize()
    def getOutputPlanes(self):
        return self.thisptr.getOutputPlanes()
    def getOutputImageSize(self):
        return self.thisptr.getOutputImageSize()
    def getOutput(self):
        # the underlying c++ method returns a pointer
        # to a block of memory that we dont own
        # we should probably copy it I suppose
        cdef float *output = self.thisptr.getOutput()
        cdef int outputSize = self.thisptr.getOutputSize()
        cdef c_array.array outputArray = array('f', [0] * outputSize )
        for i in range(outputSize):
            outputArray[i] = output[i]
#        cdef float[:] outputMv = output
#        cdef float[:] outputArrayMv = outputArray
#        outputArrayMv[:] = outputMv
#        outputArrayMv = self.thisptr.getOutput()
        return outputArray
    def getWeights(self):
        cdef int weightsSize = self.thisptr.getPersistSize()
        cdef c_array.array weightsArray = array('f', [0] * weightsSize )
        cdef float[:] weightsArray_view = weightsArray
        self.thisptr.persistToArray( &weightsArray_view[0] )
        return weightsArray
    def setWeights(self, float[:] weights):
        cdef int weightsSize = self.thisptr.getPersistSize()
        assert weightsSize == len(weights)
#        cdef c_array.array weightsArray = array('f', [0] * weightsSize )
        self.thisptr.unpersistFromArray( &weights[0] )

#        int getPersistSize()
#        void persistToArray(float *array)
#        void unpersistFromArray(const float *array)
    def setWeightsList(self, weightsList):
        cdef c_array.array weightsArray = array('f')
        weightsArray.fromlist( weightsList )
        self.setWeights( weightsArray )
    def asString(self):
        return self.thisptr.asString()
    def getClassName(self):
        return self.thisptr.getClassName()


