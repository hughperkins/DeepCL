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

#ifndef _WIN32
    extern int clblasInitialized;
#endif

class ClblasNotInitializedException {
};

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
    #ifndef _WIN32  // not sure how to check this on Windows, but this is mostly to detect bugs during
                    // development/testing anyway.  we can fix any initialization-bugs on linux, and
                    // then it should work ok on Windows too
        if(!clblasInitialized) {
            cout << "Didnt initialize clBLAS" << endl;
            throw ClblasNotInitializedException();
        }
    #endif
    if(!CWrapper->isOnDevice()) {
        if(beta == 0) {
            CWrapper->createOnDevice();
        } else {
            CWrapper->copyToDevice();
        }
    }
    int64 lda = ((order == clblasRowMajor) != (aTrans == clblasTrans)) ? k : m;
    int64 ldb = ((order == clblasRowMajor) != (bTrans == clblasTrans)) ? n : k;
    int64 ldc = order == clblasRowMajor ? n : m;
    cl_int err = clblasSgemm(
        order,
        aTrans, bTrans,
        (size_t)m, (size_t)n, (size_t)k,
        alpha,
        AWrapper->getBuffer(), (size_t)aOffset, (size_t)lda,
        BWrapper->getBuffer(), (size_t)bOffset, (size_t)ldb,
        beta,
        CWrapper->getBuffer(), (size_t)cOffset, (size_t)ldc,
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
    #ifndef _WIN32
        if(!clblasInitialized) {
            cout << "Didnt initialize clBLAS" << endl;
            throw ClblasNotInitializedException();
        }
    #endif
    if(!CWrapper->isOnDevice()) {
        if(beta == 0) {
            CWrapper->createOnDevice();
        } else {
            CWrapper->copyToDevice();
        }
    }
    int64 lda = order == clblasRowMajor ? n : m;
    cl_int err = clblasSgemv(
        order,
        trans,
        (size_t)m, (size_t)n,
        alpha,
        AWrapper->getBuffer(), (size_t)aOffset, (size_t)lda,
        BWrapper->getBuffer(), (size_t)bOffset, 1,
        beta,
        CWrapper->getBuffer(), (size_t)cOffset, 1,
        1, cl->queue, 0, NULL, 0
   );
   if (err != CL_SUCCESS) {
       throw runtime_error("clblasSgemv() failed with " + toString(err));
   }        
}

