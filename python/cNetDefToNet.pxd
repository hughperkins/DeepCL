cdef extern from "netdef/NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdefCharStar( NeuralNet *net, const char * netdef ) except +


