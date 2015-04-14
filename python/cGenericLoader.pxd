cdef extern from "GenericLoader.h":
    cdef cppclass GenericLoader:
        @staticmethod
        void getDimensions( string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) except +
        @staticmethod
        void load( string trainFilepath, float *images, int *labels, int startN, int numExamples ) except +


