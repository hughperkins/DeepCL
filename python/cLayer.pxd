cdef extern from "layer/Layer.h":
    cdef cppclass Layer:
        void forward()
        void backward()
        bool needsBackProp()
        bool getBiased()
        int getOutputCubeSize()
        int getOutputPlanes()
        int getOutputSize()
        float * getOutput()
        int getOutputNumElements()
        int getPersistSize()
        void persistToArray(float *array)
        void unpersistFromArray(const float *array)
        string asString()
        string getClassName()


