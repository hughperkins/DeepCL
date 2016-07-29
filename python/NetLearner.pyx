cdef class NetLearner: 
    cdef cDeepCL.CyNetLearner *thisptr
    def __cinit__(self, Trainer trainer, NeuralNet neuralnet,
            Ntrain, trainData, int[:] trainLabels,
            Ntest, testData, int[:] testLabels,
            batchSize):
        cdef float[:] trainData_ = trainData.reshape(-1)
        cdef float[:] testData_ = testData.reshape(-1)
        self.thisptr = new cDeepCL.CyNetLearner(
            trainer.baseptr, neuralnet.thisptr,
            Ntrain, &trainData_[0], &trainLabels[0],
            Ntest, &testData_[0], &testLabels[0],
            batchSize)
    def __dealloc__(self):
        del self.thisptr
#    def setTrainingData(self, Ntrain, float[:] trainData, int[:] trainLabels):
#        self.thisptr.setTrainingData(Ntrain, &trainData[0], &trainLabels[0])
#    def setTestingData(self, Ntest, float[:] testData, int[:] testLabels):
#        self.thisptr.setTestingData(Ntest, &testData[0], &testLabels[0])
    def setSchedule(self, numEpochs):
        self.thisptr.setSchedule(numEpochs)
    def setDumpTimings(self, bint dumpTimings):
        self.thisptr.setDumpTimings(dumpTimings)
#    def setBatchSize(self, batchSize):
#        self.thisptr.setBatchSize(batchSize)
    def _run(self):
        with nogil:
           self.thisptr.run()
    def run(self):
        interruptableCall(self._run, []) 
##        with nogil:
##            thisptr._learn(learningRate)
        # checkException()


