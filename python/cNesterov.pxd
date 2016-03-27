cdef extern from "trainers/Nesterov.h":
    cdef cppclass Nesterov(Trainer):
        Nesterov( DeepCL *cl ) except +
        void setLearningRate( float learningRate )
        void setMomentum( float momentum )
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )

