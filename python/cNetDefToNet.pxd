cdef extern from "netdef/NetdefToNet.h":
    cdef cppclass NetdefToNet:
        @staticmethod
        bool createNetFromNetdef( NeuralNet *net, string netdef ) except +


