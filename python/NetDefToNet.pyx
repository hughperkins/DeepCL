cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        cdef const char *netdef_charstar
        encodedString = toCppString(netdef)
        netdef_charstar = encodedString
        net = cDeepCL.NetdefToNet.createNetFromNetdefCharStar(neuralnet.thisptr, netdef_charstar)
        return net
