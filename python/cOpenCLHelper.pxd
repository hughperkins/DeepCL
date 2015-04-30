cdef extern from "OpenCLHelper.h":
    cdef cppclass OpenCLHelper:
        @staticmethod
        OpenCLHelper *createForFirstGpuOtherwiseCpu()
        @staticmethod
        OpenCLHelper *createForIndexedGpu( int gpu )

