cdef class NeuralNet:
    cdef cDeepCL.NeuralNet *thisptr

    def __cinit__(self, EasyCL cl, planes = None, size = None):
#        print( '__cinit__(planes,size)')
        if planes == None and size == None:
             self.thisptr = new cDeepCL.NeuralNet(cl.thisptr)
        else:
            self.thisptr = new cDeepCL.NeuralNet(cl.thisptr, planes, size)

    def __dealloc(self):
        del self.thisptr 

    def asString(self):
        return self.thisptr.asString()

#    def myprint(self):
#        self.thisptr.print()

    def setBatchSize( self, int batchSize ):
        self.thisptr.setBatchSize( batchSize ) 
    def forward( self, const float[:] images):
        self.thisptr.forward( &images[0] )
    def forwardList( self, imagesList):
        cdef c_array.array imagesArray = array('f', imagesList )
        cdef float[:] imagesArray_view = imagesArray
        self.thisptr.forward( &imagesArray_view[0] )
    def backwardFromLabels( self, int[:] labels):
        return self.thisptr.backwardFromLabels( &labels[0] ) 
    def backward( self, float[:] expectedOutput):
        return self.thisptr.backward( &expectedOutput[0] )
    def calcNumRight( self, int[:] labels ):
        return self.thisptr.calcNumRight( &labels[0] )
    def addLayer( self, LayerMaker2 layerMaker ):
        self.thisptr.addLayer( layerMaker.baseptr )
    def getLayer( self, int index ):
        cdef cDeepCL.Layer *cLayer = self.thisptr.getLayer( index )
        if cLayer == NULL:
            raise Exception('layer ' + str(index) + ' not found')
        layer = Layer()
        layer.set_thisptr( cLayer ) # note: once neuralnet out of scope, these 
                                                        # are no longer valid
        return layer
    def getNumLayers( self ):
        return self.thisptr.getNumLayers()
    def getOutput(self):
        cdef const float *output = self.thisptr.getOutput()
        cdef int outputSize = self.thisptr.getOutputSize()
        cdef c_array.array outputArray = array('f', [0] * outputSize )
        for i in range(outputSize):
            outputArray[i] = output[i]
        return outputArray
    def setTraining(self, training): # 1 is, we are training net, 0 is we are not
                            # used for example by randomtranslations layer (for now,
                            # used only by randomtranslations layer)
        self.thisptr.setTraining( training )

