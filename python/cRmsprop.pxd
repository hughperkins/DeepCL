cdef extern from "trainers/Rmsprop.h":
    cdef cppclass Rmsprop:
        Rmsprop( EasyCL *cl ) except +
        void setLearningRate( float learningRate )
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )

