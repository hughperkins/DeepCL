cdef class SGD: 
    cdef cDeepCL.SGD *thisptr
    def __cinit__( self, EasyCL cl, learningRate, momentum=0.0 ):
        self.thisptr = new cDeepCL.SGD(cl.thisptr)
        self.thisptr.setLearningRate(learningRate)
        self.thisptr.setMomentum(momentum)
    def __dealloc(self):
        del self.thisptr
    def setLearningRate(self, float learningRate):
        self.thisptr.setLearningRate(learningRate)
    def setMomentum(self, float momentum):
        self.thisptr.setMomentum(momentum)

