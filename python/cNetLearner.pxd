cdef extern from "CyNetLearner.h":
    cdef cppclass CyNetLearner[T]:
        CyNetLearner( NeuralNet *net ) except +
        void setTrainingData( int Ntrain, T *trainData, int *trainLabels ) except +
        void setTestingData( int Ntest, T *testData, int *testLabels ) except +
        void setSchedule( int numEpochs ) except +
        void setDumpTimings( bool dumpTimings ) except +
        void setBatchSize( int batchSize ) except +
        void learn( float learningRate ) nogil
        #void setSchedule( int numEpochs, int startEpoch )
        # VIRTUAL void addPostEpochAction( PostEpochAction *action );
        #void learn( float learningRate, float annealLearningRate )


