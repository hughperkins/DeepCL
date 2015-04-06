cdef class Layer:
    cdef cDeepCL.Layer *thisptr

    def __cinit__(self):
        pass
    cdef set_thisptr( self, cDeepCL.Layer *thisptr):
        self.thisptr = thisptr
    def propagate(self):
        self.thisptr.propagate()
    def backProp(self, float learningRate):
        self.thisptr.backProp( learningRate )
    def needsBackProp( self ):
        return self.thisptr.needsBackProp()
#    def getBiased( self ):
#        return self.thisptr.getBiased()
    def getOutputCubeSize( self ):
        return self.thisptr.getOutputCubeSize()
    def getOutputPlanes( self ):
        return self.thisptr.getOutputPlanes()
    def getOutputImageSize( self ):
        return self.thisptr.getOutputImageSize()
    def getResults(self):
        # the underlying c++ method returns a pointer
        # to a block of memory that we dont own
        # we should probably copy it I suppose
        cdef float *results = self.thisptr.getResults()
        cdef int resultsSize = self.thisptr.getResultsSize()
        cdef c_array.array resultsArray = array('f', [0] * resultsSize )
        for i in range(resultsSize):
            resultsArray[i] = results[i]
#        cdef float[:] resultsMv = results
#        cdef float[:] resultsArrayMv = resultsArray
#        resultsArrayMv[:] = resultsMv
#        resultsArrayMv = self.thisptr.getResults()
        return resultsArray
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


