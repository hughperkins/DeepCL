cdef extern from "EasyCL.h":
    cdef cppclass EasyCL:
        @staticmethod
        EasyCL *createForFirstGpuOtherwiseCpu()
        @staticmethod
        EasyCL *createForIndexedGpu( int gpu )

