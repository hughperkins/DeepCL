cdef class SGD: 
    cdef cDeepCL.SGD *thisptr
    def __cinit__( self, OpenCLHelper cl ):
        self.thisptr = new cDeepCL.SGD(cl.thisptr)
    def __dealloc(self):
        del self.thisptr
    def setLearningRate(self, float learningRate):
        self.thisptr.setLearningRate(learningRate)
    def setMomentum(self, float momentum):
        self.thisptr.setMomentum(momentum)

