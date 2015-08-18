#include "util/stringhelper.h"
#include "ClBlasHelper.h"

#include "EasyCL.h"

#include <iostream>
using namespace std;

#undef STATIC
#undef VIRTUAL
#define PUBLIC
#define STATIC
#define VIRTUAL

PUBLIC ClBlasHelper::ClBlasHelper() {
}

PUBLIC STATIC void ClBlasHelper::Gemm(
    EasyCL *cl,
    clblasOrder order, clblasTranspose aTrans, clblasTranspose bTrans,
    int64 m, int64 k, int64 n,
    float alpha,
    CLWrapper *AWrapper, int64 aOffset,
    CLWrapper *BWrapper, int64 bOffset,
    float beta,
    CLWrapper *CWrapper, int64 cOffset
        ) {
    if(!CWrapper->isOnDevice()) {
        if(beta == 0) {
            CWrapper->createOnDevice();
        } else {
            CWrapper->copyToDevice();
        }
    }
    size_t lda = ((order == clblasRowMajor) != (aTrans == clblasTrans)) ? k : m;
    size_t ldb = ((order == clblasRowMajor) != (bTrans == clblasTrans)) ? n : k;
    size_t ldc = order == clblasRowMajor ? n : m;
    cl_int err = clblasSgemm(
        order,
        aTrans, bTrans,
        m, n, k,
        alpha,
        AWrapper->getBuffer(), aOffset, lda,
        BWrapper->getBuffer(), bOffset, ldb,
        beta,
        CWrapper->getBuffer(), cOffset, ldc,
        1, cl->queue, 0, NULL, 0
   );
   if (err != CL_SUCCESS) {
       throw runtime_error("clblasSgemm() failed with " + toString(err));
   }    
}

PUBLIC STATIC void ClBlasHelper::Gemv(
    EasyCL *cl,
    clblasOrder order, clblasTranspose trans,
    int64 m, int64 n,
    float alpha,
    CLWrapper *AWrapper, int64 aOffset,
    CLWrapper *BWrapper, int64 bOffset,
    float beta,
    CLWrapper *CWrapper, int64 cOffset
        ) {
    if(!CWrapper->isOnDevice()) {
        if(beta == 0) {
            CWrapper->createOnDevice();
        } else {
            CWrapper->copyToDevice();
        }
    }
    int lda = order == clblasRowMajor ? n : m;
    cl_int err = clblasSgemv(
        order,
        trans,
        m, n,
        alpha,
        AWrapper->getBuffer(), aOffset, lda,
        BWrapper->getBuffer(), bOffset, 1,
        beta,
        CWrapper->getBuffer(), cOffset, 1,
        1, cl->queue, 0, NULL, 0
   );
   if (err != CL_SUCCESS) {
       throw runtime_error("clblasSgemv() failed with " + toString(err));
   }        
}

