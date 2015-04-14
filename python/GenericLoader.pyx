cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilePath ):
        cdef int N
        cdef int planes
        cdef int size
        cDeepCL.GenericLoader.getDimensions( toCppString( trainFilePath ), &N, &planes, &size )
        return (N,planes,size)
    @staticmethod 
    def load( trainFilepath, float[:] images, int[:] labels, startN, numExamples ):
        cDeepCL.GenericLoader.load( toCppString(trainFilepath), &images[0], &labels[0], startN , numExamples )


