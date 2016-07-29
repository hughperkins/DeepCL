cdef class GenericLoader:
    @staticmethod
    def getDimensions( trainFilepath ):
        print('GenericLoader.py getDimensions ', trainFilepath)
        cdef int N
        cdef int planes
        cdef int size
        cdef const char *trainFilepath_charstar
        trainFilepath_bytes = toCppString(trainFilepath)
        trainFilepath_charstar = trainFilepath_bytes
        cDeepCL.GenericLoader.getDimensions(trainFilepath_charstar, &N, &planes, &size)
        print('finished calling')
        return (N,planes,size)
    @staticmethod 
    def load( trainFilepath, images, int[:] labels, startN, numExamples ):
        cdef const char *trainFilepath_charstar
        cdef float[:] images_ = images.reshape(-1)
        trainFilepath_bytes = toCppString(trainFilepath)
        trainFilepath_charstar = trainFilepath_bytes
        cDeepCL.GenericLoader.load(trainFilepath_charstar, &images_[0], &labels[0], startN , numExamples)
