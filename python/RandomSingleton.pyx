cimport cRandomSingleton

cdef class RandomSingleton:
    @staticmethod
    def seed(seed):
        cRandomSingleton.randomsingleton_seed(seed)

    @staticmethod
    def uniform():
        return cRandomSingleton.randomsingleton_uniform()

