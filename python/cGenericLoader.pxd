cdef extern from "loaders/GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( const char * trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) except +
        @staticmethod
        void load( const char * trainFilepath, float *images, int *labels, int startN, int numExamples ) except +
