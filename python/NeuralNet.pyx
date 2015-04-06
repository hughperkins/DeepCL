cdef class NeuralNet:
    cdef cDeepCL.NeuralNet *thisptr

    def __cinit__(self, planes = None, size = None):
#        print( '__cinit__(planes,size)')
        if planes == None and size == None:
             self.thisptr = new cDeepCL.NeuralNet()
        else:
            self.thisptr = new cDeepCL.NeuralNet(planes, size)

    def __dealloc(self):
        del self.thisptr 

    def asString(self):
        return self.thisptr.asString()

#    def myprint(self):
#        self.thisptr.print()

    def setBatchSize( self, int batchSize ):
        self.thisptr.setBatchSize( batchSize ) 
    #def propagate( self, const unsigned char[:] images):
    #    self.thisptr.propagate( &images[0] )
    def propagate( self, const float[:] images):
        self.thisptr.propagate( &images[0] )
    def propagateList( self, imagesList):
        cdef c_array.array imagesArray = array('f', imagesList )
        cdef float[:] imagesArray_view = imagesArray
        self.thisptr.propagate( &imagesArray_view[0] )
    def backPropFromLabels( self, float learningRate, int[:] labels):
        return self.thisptr.backPropFromLabels( learningRate, &labels[0] ) 
    def backProp( self, float learningRate, float[:] expectedResults):
        return self.thisptr.backProp( learningRate, &expectedResults[0] )
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
    def getResults(self):
        cdef const float *results = self.thisptr.getResults()
        cdef int resultsSize = self.thisptr.getResultsSize()
        cdef c_array.array resultsArray = array('f', [0] * resultsSize )
        for i in range(resultsSize):
            resultsArray[i] = results[i]
        return resultsArray
    def setTraining(self, training): # 1 is, we are training net, 0 is we are not
                            # used for example by randomtranslations layer (for now,
                            # used only by randomtranslations layer)
        self.thisptr.setTraining( training )

