cdef extern from "trainers/Annealer.h":
    cdef cppclass Annealer(Trainer):
        Annealer( DeepCL *cl ) except +
        void setLearningRate( float learningRate )
        void setAnneal( float anneal )
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )

