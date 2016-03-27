cdef extern from "CyWrappers.h":
    cdef cppclass CyNetLearner:
        CyNetLearner(Trainer *trainer, NeuralNet *net,
            int Ntrain, float *trainData, int *trainLabels,
            int Ntest, float *testData, int *testLabels,
            int batchSize) except +
        void run() nogil
        void setSchedule(int numEpochs) except +
        void setDumpTimings(bool dumpTimings) except +
        # void setBatchSize(int batchSize) except +
        #void setSchedule(int numEpochs, int startEpoch)

