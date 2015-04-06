cdef extern from "Layer.h":
    cdef cppclass Layer:
        void propagate()
        void backProp( float learningRate )
        bool needsBackProp()
        bool getBiased()
        int getOutputCubeSize()
        int getOutputPlanes()
        int getOutputImageSize()
        float * getResults()
        int getResultsSize()
        int getPersistSize()
        void persistToArray(float *array)
        void unpersistFromArray(const float *array)
        string asString()
        string getClassName()


