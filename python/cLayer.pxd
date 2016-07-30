cdef extern from "layer/Layer.h":
    cdef cppclass Layer:
        void forward()
        void backward()
        bool needsBackProp()
        bool biased() except+
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

cdef extern from "patches/RandomTranslations.h":
    cdef cppclass RandomTranslations(Layer):
        int getTranslationSize()

cdef extern from "conv/ConvolutionalLayer.h":
    cdef cppclass ConvolutionalLayer(Layer):
        int getFilterSize()
        bool getPadZeros()

cdef extern from "pooling/PoolingLayer.h":
    cdef cppclass PoolingLayer(Layer):
        int getPoolingSize()
        bool getPadZeros()

cdef extern from "activate/ActivationLayer.h":
    cdef cppclass ActivationLayer(Layer):
        const char *getActivationAsCharStar()

