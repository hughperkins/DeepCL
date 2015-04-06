cdef class NetdefToNet:
    @staticmethod
    def createNetFromNetdef( NeuralNet neuralnet, netdef ):
        return cDeepCL.NetdefToNet.createNetFromNetdef( neuralnet.thisptr, toCppString( netdef ) )


