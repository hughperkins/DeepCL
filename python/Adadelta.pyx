cdef class Adadelta(Trainer): 
    cdef cDeepCL.Adadelta *thisptr
    def __cinit__( self, DeepCL cl, rho=0.9 ):
        self.thisptr = new cDeepCL.Adadelta(cl.thisptr, rho)
        self.baseptr = self.thisptr
    def __dealloc__(self):
        del self.thisptr
    def train(self, NeuralNet net, TrainingContext context,
        inputdata, float[:] expectedOutput ):
        cdef float[:] inputdata_ = inputdata.reshape(-1)
        cdef cDeepCL.BatchResult result = self.thisptr.train(
            net.thisptr, context.thisptr, &inputdata_[0], &expectedOutput[0])
        return result.getLoss()
    def trainFromLabels(self, NeuralNet net, TrainingContext context,
        inputdata, int[:] labels):
        cdef float[:] inputdata_ = inputdata.reshape(-1)
        cdef cDeepCL.BatchResult result = self.thisptr.trainFromLabels(
            net.thisptr, context.thisptr, &inputdata_[0], &labels[0])
        return ( result.getLoss(), result.getNumRight() )

