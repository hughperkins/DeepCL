cdef extern from "net/NeuralNet.h":
    cdef cppclass NeuralNet:
        @staticmethod
        NeuralNet *instance(DeepCL *cl) except +
        @staticmethod
        NeuralNet *instance3(DeepCL *cl, int numPlanes, int size) except +
        const char *asNewCharStar() except +
        void setBatchSize( int batchSize ) except +
        void forward( const float *images) except +
        void backwardFromLabels( const int *labels) except +
        void backward( const float *expectedOutput) except +
        int calcNumRight( const int *labels ) except +
        void addLayer( LayerMaker2 *maker ) except +
        Layer *getLayer( int index )
        int getNumLayers()
        const float *getOutput()
        int getOutputNumElements()
        void setTraining( bool training )
        void deleteMe()
