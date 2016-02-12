cdef extern from "trainers/Trainer.h":
    cdef cppclass BatchResult:
        float getLoss()
        int getNumRight()

cdef extern from "trainers/TrainingContext.h":
    cdef cppclass TrainingContext:
        TrainingContext( int epoch, int batch )

cdef extern from "trainers/SGD.h":
    cdef cppclass SGD(Trainer):
        SGD( DeepCL *cl ) except +
        void setLearningRate( float learningRate )
        void setMomentum( float momentum )
        void setWeightDecay( float weightDecay )
        BatchResult train( NeuralNet *net, TrainingContext *context,
            const float *input, const float *expectedOutput )
        BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
            const float *input, const int *labels )
