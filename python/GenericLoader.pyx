cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilepath ):
        print 'GenericLoader.py getDimensions ', trainFilepath
        cdef int N
        cdef int planes
        cdef int size
        cdef const char *trainFilepath_charstar = trainFilepath
        cDeepCL.GenericLoader.getDimensions(trainFilepath_charstar, &N, &planes, &size)
        print 'finished calling'
        return (N,planes,size)
    @staticmethod 
    def load( trainFilepath, float[:] images, int[:] labels, startN, numExamples ):
        cdef const char *trainFilepath_charstar = trainFilepath
        cDeepCL.GenericLoader.load(trainFilepath_charstar, &images[0], &labels[0], startN , numExamples)
