
cdef extern from "deepcl/CppRuntimeBoundary.h":
    cdef void deepcl_deleteCharStar(const char *charStar)
