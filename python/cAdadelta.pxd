cdef extern from "trainers/Adadelta.h":
    cdef cppclass Adadelta:
        Adadelta( EasyCL *cl, float rho ) except +
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )

