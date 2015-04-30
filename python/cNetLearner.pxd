cdef extern from "CyWrappers.h":
    cdef cppclass CyNetLearner:
        CyNetLearner( SGD *sgd, NeuralNet *net,
            int Ntrain, float *trainData, int *trainLabels,
            int Ntest, float *testData, int *testLabels,
            int batchSize ) except +
        # void setTrainingData( int Ntrain, float *trainData, int *trainLabels ) except +
        # void setTestingData( int Ntest, float *testData, int *testLabels ) except +
        void setSchedule( int numEpochs ) except +
        void setDumpTimings( bool dumpTimings ) except +
        # void setBatchSize( int batchSize ) except +
        # void learn( float learningRate ) nogil
        #void setSchedule( int numEpochs, int startEpoch )
        # VIRTUAL void addPostEpochAction( PostEpochAction *action );
        #void learn( float learningRate, float annealLearningRate )


