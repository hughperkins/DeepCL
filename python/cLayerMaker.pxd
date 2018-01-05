
cdef extern from "layer/LayerMaker.h":
    cdef cppclass LayerMaker2:
        pass

cdef extern from "input/InputLayerMaker.h":
    cdef cppclass InputLayerMaker(LayerMaker2):
        InputLayerMaker *numPlanes( int _numPlanes ) except +
        InputLayerMaker *imageSize( int _imageSize ) except +
        @staticmethod
        InputLayerMaker *instance() except +

cdef extern from "dropout/DropoutMaker.h":
    cdef cppclass DropoutMaker(LayerMaker2):
        DropoutMaker *dropRatio( float _dropRatio ) except +
        @staticmethod
        DropoutMaker *instance() except +

cdef extern from "activate/ActivationMaker.h":
    cdef cppclass ActivationMaker(LayerMaker2):
        ActivationMaker *relu() except +
        ActivationMaker *elu() except +
        ActivationMaker *tanh() except +
        ActivationMaker *linear() except +
        ActivationMaker *sigmoid() except +
        ActivationMaker *scaledTanh() except +
        @staticmethod
        ActivationMaker *instance() except +

cdef extern from "normalize/NormalizationLayerMaker.h":
    cdef cppclass NormalizationLayerMaker(LayerMaker2):
        NormalizationLayerMaker *translate( float translate ) except +
        NormalizationLayerMaker *scale( float scale ) except +
        @staticmethod
        NormalizationLayerMaker *instance() except +

cdef extern from "fc/FullyConnectedMaker.h":
    cdef cppclass FullyConnectedMaker(LayerMaker2):
        FullyConnectedMaker *numPlanes( int numPlanes ) except +
        FullyConnectedMaker *imageSize( int imageSize ) except +
        FullyConnectedMaker *biased() except +
        FullyConnectedMaker *biased(bint _biased) except +
        @staticmethod
        FullyConnectedMaker *instance() except +

cdef extern from "conv/ConvolutionalMaker.h":
    cdef cppclass ConvolutionalMaker(LayerMaker2):
        ConvolutionalMaker *numFilters( int numFilters ) except +
        ConvolutionalMaker *filterSize( int imageSize ) except +
        ConvolutionalMaker *padZeros() except +
        ConvolutionalMaker *padZeros(bint _padZeros) except +
        ConvolutionalMaker *biased() except +
        ConvolutionalMaker *biased(bint _biased) except +
        @staticmethod
        ConvolutionalMaker *instance() except +

cdef extern from "pooling/PoolingMaker.h":
    cdef cppclass PoolingMaker(LayerMaker2):
        PoolingMaker *poolingSize( int _poolingsize ) except +
        PoolingMaker *padZeros( int _padZeros ) except +
        @staticmethod
        PoolingMaker *instance() except +

cdef extern from "forcebackprop/ForceBackpropLayerMaker.h":
    cdef cppclass ForceBackpropLayerMaker(LayerMaker2):
        @staticmethod
        ForceBackpropLayerMaker *instance() except +

cdef extern from "layer/LayerMaker.h":
    cdef cppclass SquareLossMaker(LayerMaker2):
        @staticmethod
        SquareLossMaker *instance() except +

cdef extern from "layer/LayerMaker.h":
    cdef cppclass SoftMaxMaker(LayerMaker2):
        @staticmethod
        SoftMaxMaker *instance()

cdef extern from "patches/RandomTranslationsMaker.h":
    cdef cppclass RandomTranslationsMaker(LayerMaker2):
        @staticmethod
        RandomTranslationsMaker *instance()
        RandomTranslationsMaker *translateSize(int translateSize) except+

