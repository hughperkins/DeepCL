cdef extern from "NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdef( NeuralNet *net, string netdef ) except +


