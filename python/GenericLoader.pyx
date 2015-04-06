cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilePath ):
        cdef int N
        cdef int planes
        cdef int size
        cDeepCL.GenericLoader.getDimensions( toCppString( trainFilePath ), &N, &planes, &size )
        # print( N )
        return (N,planes,size)
    @staticmethod 
    def loaduc( trainFilepath, unsigned char[:] images, int[:] labels, startN, numExamples ):
        #(N, planes, size) = getDimensions(trainFilepath)
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        cDeepCL.GenericLoader.load( toCppString( trainFilepath ), &images[0], &labels[0], startN , numExamples )
        #return (images, labels)
    @staticmethod 
    def load( trainFilepath, float[:] images, int[:] labels, startN, numExamples ):
        (N, planes, size) = GenericLoader.getDimensions(toCppString(trainFilepath))
        #images = view.array(shape=(N,planes,size,size),itemsize=1,
        #cdef unsigned char *images
        #cdef int *labels
        #cdef unsigned char ucImages[numExamples * planes * size * size]
        print( (N, planes, size ) )
        cdef c_array.array ucImages = array('B', [0] * (numExamples * planes * size * size) )
        cdef unsigned char[:] ucImagesMv = ucImages
        cDeepCL.GenericLoader.load( toCppString(trainFilepath), &ucImagesMv[0], &labels[0], startN , numExamples )
        #return (images, labels)
        cdef int i
        cdef int total
        total = numExamples * planes * size * size
        print(total)
        for i in range(total):
            images[i] = ucImagesMv[i]


