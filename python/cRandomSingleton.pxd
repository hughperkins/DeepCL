cdef extern from "util/RandomSingleton.h":
    cdef void randomsingleton_seed "RandomSingleton::seed"(unsigned long seed)
    cdef float randomsingleton_uniform "RandomSingleton::uniform"()

