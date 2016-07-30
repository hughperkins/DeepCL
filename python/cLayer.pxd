cdef extern from "layer/Layer.h":
    cdef cppclass Layer:
        void forward()
        void backward()
        bool needsBackProp()
        bool getBiased()
        int getOutputCubeSize() except+
        int getOutputPlanes()
        int getOutputSize()
        float * getOutput()
        int getOutputNumElements()
        int getPersistSize()
        void persistToArray(float *array)
        void unpersistFromArray(const float *array)
        const char *asNewCharStar()
        const char *getClassNameAsCharStar()

cdef extern from "loss/SoftMaxLayer.h":
    cdef cppclass SoftMaxLayer(Layer):
        int getBatchSize()
        void getLabels(int *labels)

