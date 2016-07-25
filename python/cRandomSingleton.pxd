cdef extern from "util/RandomSingleton.h":
    cdef void randomsingleton_seed "RandomSingleton::seed"(unsigned int seed)
    cdef float randomsingleton_uniform "RandomSingleton::uniform"()

#    cdef cppclass RandomSingleton:
#        @staticmethod
#        void seed(unsigned int seed)
#        @staticmethod
#        float uniform()

