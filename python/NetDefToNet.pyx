cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        cdef const char *netdef_charstar = netdef
        return cDeepCL.NetdefToNet.createNetFromNetdefCharStar(neuralnet.thisptr, netdef_charstar)
