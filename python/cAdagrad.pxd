cdef extern from "trainers/Adagrad.h":
    cdef cppclass Adagrad(Trainer):
        Adagrad( DeepCL *cl ) except +
        void setLearningRate( float learningRate )
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )

