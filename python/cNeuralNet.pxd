cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        #pass
        NeuralNet( EasyCL *cl ) except +
        #void print()
        NeuralNet( EasyCL *cl, int numPlanes, int size ) except +
        string asString() except +
        void setBatchSize( int batchSize ) except +
        void forward( const float *images) except +
        void backwardFromLabels( const int *labels) except +
        void backward( const float *expectedOutput) except +
        int calcNumRight( const int *labels ) except +
        void addLayer( LayerMaker2 *maker ) except +
        Layer *getLayer( int index )
        int getNumLayers()
        const float *getOutput()
        int getOutputSize()
        void setTraining( bool training )


