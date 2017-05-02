#pragma once

#define STATIC static
#define VIRTUAL virtual

#include "clBLAS.h"
#include "DeepCLDllExport.h"

namespace easycl {
class EasyCL;
class CLWrapper;
}

class DeepCL_EXPORT ClBlasHelper {
    public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    ClBlasHelper();
    STATIC void Gemm(
        easycl::EasyCL *cl,
        clblasOrder order, clblasTranspose aTrans, clblasTranspose bTrans,
        int64 m, int64 k, int64 n,
        float alpha,
        easycl::CLWrapper *AWrapper, int64 aOffset,
        easycl::CLWrapper *BWrapper, int64 bOffset,
        float beta,
        easycl::CLWrapper *CWrapper, int64 cOffset
    );
    STATIC void Gemv(
        easycl::EasyCL *cl,
        clblasOrder order, clblasTranspose trans,
        int64 m, int64 n,
        float alpha,
        easycl::CLWrapper *AWrapper, int64 aOffset,
        easycl::CLWrapper *BWrapper, int64 bOffset,
        float beta,
        easycl::CLWrapper *CWrapper, int64 cOffset
    );

    // [[[end]]]
};

