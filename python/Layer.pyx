import numpy as np


cdef class Layer:
    cdef cDeepCL.Layer *thisptr

    def __cinit__(self):
        pass
    cdef set_thisptr(self, cDeepCL.Layer *thisptr):
        self.thisptr = thisptr
    def getNetdefString(self):
        className = self.getClassName()
        # print('className', className)
        if className == 'ConvolutionalLayer':
            numFilters = self.getOutputPlanes()
            filterSize = self.getFilterSize()
            padZerosString = 'z' if self.getPadZeros() else ''
            return "%sc%s%s" % (numFilters, filterSize, padZerosString)
        elif className == 'ActivationLayer':
            activation = self.getActivation()
            return activation
        elif className == 'RandomTranslations':
            translationSize = self.getTranslationSize()
            return 'rt%s' % translationSize
        elif className == 'PoolingLayer':
            poolingSize = self.getPoolingSize()
            padZerosString = 'z' if self.getPadZeros() else ''
            return 'mp%s%s' % (poolingSize, padZerosString)
        elif className == 'SoftMaxLayer':
            return ''
        elif className == 'NormalizationLayer':
            return ''
        elif className == 'InputLayer':
            return ''
        elif className == 'FullyConnectedLayer':
            numNeurons = self.getOutputPlanes()
            return '%sn' % numNeurons
        else:
            raise Exception('getNetdefString not implemented for %s' % className)
    def forward(self):
        self.thisptr.forward()
    def backward(self):
        self.thisptr.backward()
    def needsBackProp(self):
        return self.thisptr.needsBackProp()
    def getBiased( self ):
        return self.thisptr.biased()
    def getOutputCubeSize(self):
        return self.thisptr.getOutputCubeSize()
    def getOutputPlanes(self):
        return self.thisptr.getOutputPlanes()
    def getOutputSize(self):
        return self.thisptr.getOutputSize()
    def getOutput(self):
        # the underlying c++ method returns a pointer
        # to a block of memory that we dont own
        # we should probably copy it I suppose
        cdef float *output = self.thisptr.getOutput()
        cdef int outputNumElements = self.thisptr.getOutputNumElements()
        planes = self.getOutputPlanes()
        size = self.getOutputSize()
        batchSize = outputNumElements // planes // size // size
        outputArray = np.zeros((batchSize, planes, size, size), dtype=np.float32)
        # cdef c_array.array outputArray = array(floatArrayType, [0] * outputNumElements )
        outreshape = outputArray.reshape(-1)
        # outreshape = output
        for i in range(outputNumElements):
            outreshape[i] = output[i]
        return outputArray
    def getWeights(self):
        cdef int weightsSize = self.thisptr.getPersistSize()
        if weightsSize == 0:
            return None

        weightsArray = np.zeros((weightsSize,), dtype=np.float32)
        # cdef c_array.array weightsArray = weightsArray # array(floatArrayType, [0] * weightsSize )
        cdef float[:] weightsArray_view = weightsArray
        self.thisptr.persistToArray( &weightsArray_view[0] )
        return weightsArray
    def setWeights(self, float[:] weights):
        cdef int weightsSize = self.thisptr.getPersistSize()
        assert weightsSize == len(weights)
#        cdef c_array.array weightsArray = array('f', [0] * weightsSize )
        self.thisptr.unpersistFromArray( &weights[0] )

#        int getPersistSize()
#        void persistToArray(float *array)
#        void unpersistFromArray(const float *array)
    #def setWeightsList(self, weightsList):
        #cdef c_array.array weightsArray = array(floatArrayType)
        #weightsArray.fromlist( weightsList )
        #self.setWeights( weightsArray )
    def asString(self):
        cdef const char *res_charstar = self.thisptr.asNewCharStar()
        cdef str res = str(res_charstar.decode('UTF-8'))
        CppRuntimeBoundary.deepcl_deleteCharStar(res_charstar)
        return res
    def getClassName(self):
        cdef const char *res_charstar = self.thisptr.getClassNameAsCharStar()
        cdef str res = str(res_charstar.decode('UTF-8'))
        CppRuntimeBoundary.deepcl_deleteCharStar(res_charstar)
        return res

cdef class RandomTranslations(Layer):
    def __cinit__(self):
        pass

    def getTranslationSize(self):
        cdef cDeepCL.RandomTranslations *cRandomTranslations = <cDeepCL.RandomTranslations *>(self.thisptr)
        cdef int translationSize = cRandomTranslations.getTranslationSize()
        return translationSize

cdef class PoolingLayer(Layer):
    def __cinit__(self):
        pass

    def getPoolingSize(self):
        cdef cDeepCL.PoolingLayer *cPoolingLayer = <cDeepCL.PoolingLayer *>(self.thisptr)
        cdef int poolingSize = cPoolingLayer.getPoolingSize()
        return poolingSize
    def getPadZeros(self):
        cdef cDeepCL.PoolingLayer *cPoolingLayer = <cDeepCL.PoolingLayer *>(self.thisptr)
        cdef bool padZeros = cPoolingLayer.getPadZeros()
        return padZeros

cdef class ConvolutionalLayer(Layer):
    def __cinit__(self):
        pass
    def getFilterSize(self):
        cdef cDeepCL.ConvolutionalLayer *cConvolutionalLayer = <cDeepCL.ConvolutionalLayer *>(self.thisptr)
        cdef int filterSize = cConvolutionalLayer.getFilterSize()
        return filterSize
    def getPadZeros(self):
        cdef cDeepCL.ConvolutionalLayer *cConvolutionalLayer = <cDeepCL.ConvolutionalLayer *>(self.thisptr)
        cdef bool padZeros = cConvolutionalLayer.getPadZeros()
        return padZeros

cdef class ActivationLayer(Layer):
    def __cinit__(self):
        pass
    def getActivation(self):
        cdef cDeepCL.ActivationLayer *cActivationLayer = <cDeepCL.ActivationLayer *>(self.thisptr)
        cdef const char *res_charstar = cActivationLayer.getActivationAsCharStar()
        cdef str res = str(res_charstar.decode('UTF-8'))
        CppRuntimeBoundary.deepcl_deleteCharStar(res_charstar)
        return res

cdef class SoftMax(Layer):
    def __cinit__(self):
        pass

    def getBatchSize(self):
        cdef cDeepCL.SoftMaxLayer *cSoftMax = <cDeepCL.SoftMaxLayer *>(self.thisptr)
        cdef int batchSize = cSoftMax.getBatchSize()
        return batchSize        

    def getLabels(self):
        cdef cDeepCL.SoftMaxLayer *cSoftMax = <cDeepCL.SoftMaxLayer *>(self.thisptr)
        cdef int batchSize = cSoftMax.getBatchSize()
        labelsArray = np.zeros((batchSize), dtype=np.int32)
        # cdef c_array.array labelsArray = array(intArrayType, [0] * batchSize)
        cdef int[:] labelsArray_view = labelsArray
        cSoftMax.getLabels(&labelsArray_view[0])
        return labelsArray

