cdef class Adadelta: 
    cdef cDeepCL.Adadelta *thisptr
    def __cinit__( self, EasyCL cl, rho=0.9 ):
        self.thisptr = new cDeepCL.Adadelta(cl.thisptr, rho)
    def __dealloc(self):
        del self.thisptr
    def train(self, NeuralNet net, TrainingContext context,
        float[:] inputdata, float[:] expectedOutput ):
        cdef cDeepCL.BatchResult result = self.thisptr.train(
            net.thisptr, context.thisptr, &inputdata[0], &expectedOutput[0])
        return result.getLoss()
    def trainFromLabels(self, NeuralNet net, TrainingContext context,
        float[:] inputdata, int[:] labels):
        cdef cDeepCL.BatchResult result = self.thisptr.trainFromLabels(
            net.thisptr, context.thisptr, &inputdata[0], &labels[0])
        return ( result.getLoss(), result.getNumRight() )

