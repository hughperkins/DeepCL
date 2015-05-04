cdef class Nesterov: 
    cdef cDeepCL.Nesterov *thisptr
    def __cinit__( self, EasyCL cl, learningRate, momentum=0.0 ):
        self.thisptr = new cDeepCL.Nesterov(cl.thisptr)
        self.thisptr.setLearningRate(learningRate)
        self.thisptr.setMomentum(momentum)
    def __dealloc(self):
        del self.thisptr
    def setLearningRate(self, float learningRate):
        self.thisptr.setLearningRate(learningRate)
    def setMomentum(self, float momentum):
        self.thisptr.setMomentum(momentum)
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

