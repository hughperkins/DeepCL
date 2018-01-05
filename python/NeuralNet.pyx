cdef class NeuralNet:
    cdef cDeepCL.NeuralNet *thisptr
    cdef object cl

    def __cinit__(self, DeepCL cl, planes = None, size = None):
#        print('__cinit__(planes,size)')
        self.cl = cl
        if planes == None and size == None:
            self.thisptr = cDeepCL.NeuralNet.instance(cl.thisptr)
        else:
            self.thisptr = cDeepCL.NeuralNet.instance3(cl.thisptr, planes, size)

    def __dealloc__(self):
        self.thisptr.deleteMe()

    def asString(self):
        cdef const char *result_charstar = self.thisptr.asNewCharStar()
        cdef str result = str(result_charstar.decode('UTF-8'))
        CppRuntimeBoundary.deepcl_deleteCharStar(result_charstar)
        return result

    def __str__(self):
#        print('__str__')
        return self.asString()

    def __unicode__(self):
#        print('__unicode__')
        return self.asString()

    def setBatchSize(self, int batchSize):
        self.thisptr.setBatchSize(batchSize) 
    def forward(self, images):
        cdef float[:] images_ = images.reshape(-1)
        self.thisptr.forward(&images_[0])
    #def forwardList(self, imagesList):
    #    cdef c_array.array imagesArray = array(floatArrayType, imagesList)
    #    cdef float[:] imagesArray_view = imagesArray
    #    self.thisptr.forward(&imagesArray_view[0])
    def backwardFromLabels(self, int[:] labels):
        return self.thisptr.backwardFromLabels(&labels[0]) 
    def backward(self, expectedOutput):
        cdef float[:] expectedOutput_ = expectedOutput.reshape(-1)
        return self.thisptr.backward(&expectedOutput_[0])
    def calcNumRight(self, int[:] labels):
        return self.thisptr.calcNumRight(&labels[0])
    def addLayer(self, LayerMaker2 layerMaker):
        self.thisptr.addLayer(layerMaker.baseptr)
    def getLayer(self, int index):
        cdef cDeepCL.Layer *cLayer = self.thisptr.getLayer(index)
        if cLayer == NULL:
            raise Exception('layer ' + str(index) + ' not found')
        layer = Layer()
        layer.set_thisptr(cLayer) # note: once neuralnet out of scope, these 
                                                    # are no longer valid
        # print('layer.getClassName()', layer.getClassName())
        # print('type(layer.getClassName()', type(layer.getClassName()))
        # print('type(layer.getClassName().decode("utf-8"))', type(layer.getClassName().decode('utf-8')))
        className = layer.getClassName()
        if className == 'SoftMaxLayer':
            layer.set_thisptr(<cDeepCL.Layer *>(0))
            layer = SoftMax()
            layer.set_thisptr(cLayer)
        elif className == 'RandomTranslations':
            layer.set_thisptr(<cDeepCL.Layer *>(0))
            layer = RandomTranslations()
            layer.set_thisptr(cLayer)
        elif className == 'ConvolutionalLayer':
            layer.set_thisptr(<cDeepCL.Layer *>(0))
            layer = ConvolutionalLayer()
            layer.set_thisptr(cLayer)
        elif className == 'PoolingLayer':
            layer.set_thisptr(<cDeepCL.Layer *>(0))
            layer = PoolingLayer()
            layer.set_thisptr(cLayer)
        elif className == 'ActivationLayer':
            layer.set_thisptr(<cDeepCL.Layer *>(0))
            layer = ActivationLayer()
            layer.set_thisptr(cLayer)
        return layer
    def getNetdef(self):
        netdefBits = []
        for i in range(self.getNumLayers()):
            layer = self.getLayer(i)
            layerNetdef = layer.getNetdefString()
            if layerNetdef != '':
                netdefBits.append(layerNetdef)
        return '-'.join(netdefBits)
    def getLastLayer(self):
        return self.getLayer(self.getNumLayers() - 1)
    def getNumLayers(self):
        return self.thisptr.getNumLayers()
    def getOutput(self):
        cdef const float *output = self.thisptr.getOutput()
        cdef int outputNumElements = self.thisptr.getOutputNumElements()
        lastLayer = self.getLastLayer()
        planes = lastLayer.getOutputPlanes()
        size = lastLayer.getOutputSize()
        batchSize = outputNumElements // planes // size // size
        outputArray = np.zeros((batchSize, planes, size, size), dtype=np.float32)
        # cdef c_array.array outputArray = array(floatArrayType, [0] * outputNumElements )
        outreshape = outputArray.reshape(-1)
        # outreshape = output
        for i in range(outputNumElements):
            outreshape[i] = output[i]
        # cdef c_array.array outputArray = array(floatArrayType, [0] * outputNumElements)
        #for i in range(outputNumElements):
        #    outputArray[i] = output[i]
        return outputArray
    def setTraining(self, training): # 1 is, we are training net, 0 is we are not
                            # used for example by randomtranslations layer (for now,
                            # used only by randomtranslations layer)
        self.thisptr.setTraining(training)
