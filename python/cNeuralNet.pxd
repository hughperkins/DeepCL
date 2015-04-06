cdef extern from "NeuralNet.h":
    cdef cppclass NeuralNet:
        #pass
        NeuralNet() except +
        #void print()
        NeuralNet( int numPlanes, int size ) except +
        string asString() except +
        void setBatchSize( int batchSize ) except +
        void propagate( const float *images) except +
        void backPropFromLabels( float learningRate, const int *labels) except +
        void backProp( float learningRate, const float *expectedResults) except +
        int calcNumRight( const int *labels ) except +
        void addLayer( LayerMaker2 *maker ) except +
        Layer *getLayer( int index )
        int getNumLayers()
        const float *getResults()
        int getResultsSize()
        void setTraining( bool training )


