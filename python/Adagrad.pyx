cdef class Adagrad: 
    cdef cDeepCL.Adagrad *thisptr
    def __cinit__( self, EasyCL cl, learningRate, momentum=0.0 ):
        self.thisptr = new cDeepCL.Adagrad(cl.thisptr)
        self.thisptr.setLearningRate(learningRate)
    def __dealloc(self):
        del self.thisptr
    def setLearningRate(self, float learningRate):
        self.thisptr.setLearningRate(learningRate)
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

