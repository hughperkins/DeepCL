cdef extern from "SGD.h":
    cdef cppclass SGD:
        SGD( EasyCL *cl ) except +
        void setLearningRate( float learningRate )
        void setMomentum( float momentum )

