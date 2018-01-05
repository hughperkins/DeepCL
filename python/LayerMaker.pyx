cdef class LayerMaker2:
    cdef cDeepCL.LayerMaker2 *baseptr

cdef class NormalizationLayerMaker(LayerMaker2):
    cdef cDeepCL.NormalizationLayerMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.NormalizationLayerMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    def translate( self, float _translate ):
        self.thisptr.translate( _translate )
        return self
    def scale( self, float _scale ):
        self.thisptr.scale( _scale )
        return self
    @staticmethod
    def instance():
        return NormalizationLayerMaker()

cdef class FullyConnectedMaker(LayerMaker2):
    cdef cDeepCL.FullyConnectedMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.FullyConnectedMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    def numPlanes( self, int _numPlanes ):
        self.thisptr.numPlanes( _numPlanes )
        return self
    def imageSize( self, int _imageSize ):
        self.thisptr.imageSize( _imageSize )
        return self
    def biased(self):
        self.thisptr.biased()
        return self
    def biased(self, int _biased):
        self.thisptr.biased( _biased )
        return self
    @staticmethod
    def instance():
        return FullyConnectedMaker()

cdef class ConvolutionalMaker(LayerMaker2):
    cdef cDeepCL.ConvolutionalMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.ConvolutionalMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
        #del self.thisptr
    def numFilters( self, int _numFilters ):
        self.thisptr.numFilters( _numFilters )
        return self
    def filterSize( self, int _filterSize ):
        self.thisptr.filterSize( _filterSize )
        return self
    #def padZeros(self):
    #    self.thisptr.padZeros()
    #    return self
    def padZeros(self, bint _padZeros = True):
        self.thisptr.padZeros(_padZeros)
        return self
    #def biased(self):
    #    self.thisptr.biased()
    #    return self
    def biased(self, bint _biased=True):
        self.thisptr.biased( _biased )
        return self
    @staticmethod
    def instance():
        return ConvolutionalMaker()

cdef class PoolingMaker(LayerMaker2):
    cdef cDeepCL.PoolingMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.PoolingMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    def poolingSize( self, int _poolingSize ):
        self.thisptr.poolingSize( _poolingSize )
        return self
    @staticmethod
    def instance():
        return PoolingMaker()

cdef class DropoutMaker(LayerMaker2):
    cdef cDeepCL.DropoutMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.DropoutMaker()
        self.baseptr = self.thisptr
    def dropRatio(self, float _dropRatio):
        self.thisptr.dropRatio(_dropRatio)
        return self
    @staticmethod
    def instance():
        return ActivationMaker()

cdef class ActivationMaker(LayerMaker2):
    cdef cDeepCL.ActivationMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.ActivationMaker()
        self.baseptr = self.thisptr
    def relu(self):
        self.thisptr.relu()
        return self
    def elu(self):
        self.thisptr.elu()
        return self
    def sigmoid(self):
        self.thisptr.sigmoid()
        return self
    def scaledTanh(self):
        self.thisptr.scaledTanh()
        return self
    def tanh(self):
        self.thisptr.tanh()
        return self
    def linear(self):
        self.thisptr.linear()
        return self
    @staticmethod
    def instance():
        return ActivationMaker()

cdef class ForceBackpropMaker(LayerMaker2):
    cdef cDeepCL.ForceBackpropLayerMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.ForceBackpropLayerMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    @staticmethod
    def instance():
        return ForceBackpropMaker()

cdef class SquareLossMaker(LayerMaker2):
    cdef cDeepCL.SquareLossMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.SquareLossMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    @staticmethod
    def instance():
        return SquareLossMaker()


cdef class RandomTranslationsMaker(LayerMaker2):
    cdef cDeepCL.RandomTranslationsMaker *thisptr

    def __cinit__(self):
        self.thisptr = new cDeepCL.RandomTranslationsMaker()
        self.baseptr = self.thisptr

    @staticmethod
    def instance():
        return RandomTranslationsMaker()

    def translateSize(self, int _translateSize):
        self.thisptr.translateSize(_translateSize)
        return self


cdef class SoftMaxMaker(LayerMaker2):
    cdef cDeepCL.SoftMaxMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.SoftMaxMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    @staticmethod
    def instance():
        return SoftMaxMaker()

cdef class InputLayerMaker(LayerMaker2):
    cdef cDeepCL.InputLayerMaker *thisptr
    def __cinit__( self ):
        self.thisptr = new cDeepCL.InputLayerMaker()
        self.baseptr = self.thisptr
#    def __dealloc__(self):
#        del self.thisptr
    def numPlanes( self, int _numPlanes ):
        self.thisptr.numPlanes( _numPlanes )
        return self
    def imageSize( self, int _imageSize ):
        self.thisptr.imageSize( _imageSize )
        return self
    @staticmethod
    def instance():
        return InputLayerMaker()


