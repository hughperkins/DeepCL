#include "clblas/ClBlasHelper.h"
#include "clblas/ClBlasInstance.h"
#include "clBLAS.h"
#include "EasyCL.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

TEST(testClBlas, basic) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
    ClBlasInstance clblasInstance;
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    CWrap->createOnDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasRowMajor,
        clblasNoTrans, clblasNoTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
    cl->finish();
    CWrap->copyToHost();
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    cl->finish();

    delete CWrap;
    delete BWrap;
    delete AWrap;

    cl->finish();

    delete cl;
    clblasTeardown();
}

static void transpose(float *matrix, int rows, int cols) {
    float *tempMatrix = new float[rows * cols];
    for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            int pos1 = row * cols + col;
            int pos2 = col * rows + row;
//            float old = matrix[pos1];
            tempMatrix[pos2] = matrix[pos1];
        }
    }
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = tempMatrix[i];
    }
    delete[] tempMatrix;
}

TEST(testClBlas, transA) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
    transpose(A, 3, 2);
    for(int row=0; row < 2; row++) {
        for(int col=0; col < 3; col++) {
            cout << A[row*3 + col] << " ";
        }
        cout << endl;
    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasRowMajor,
        clblasTrans, clblasNoTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}

TEST(testClBlas, transB) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
    transpose(B, 2, 1);
    for(int row=0; row < 2; row++) {
        for(int col=0; col < 1; col++) {
            cout << B[row*1 + col] << " ";
        }
        cout << endl;
    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasRowMajor,
        clblasNoTrans, clblasTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}

TEST(testClBlas, colMajor) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
    transpose(A, 3, 2);
    transpose(B, 2, 1);
//    for(int row=0; row < 2; row++) {
//        for(int col=0; col < 1; col++) {
//            cout << B[row*1 + col] << " ";
//        }
//        cout << endl;
//    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasColumnMajor,
        clblasNoTrans, clblasNoTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    transpose(C, 1, 3);
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}
TEST(testClBlas, colMajor2) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5,
                 0, -2};
    float B[] = {3,2,8,
                 -1,0,4};

    float C[4*3];
    transpose(A, 4, 2);
    transpose(B, 2, 3);
//    for(int row=0; row < 2; row++) {
//        for(int col=0; col < 1; col++) {
//            cout << B[row*1 + col] << " ";
//        }
//        cout << endl;
//    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(4*2, A);
    CLWrapper *BWrap = cl->wrap(2*3, B);
    CLWrapper *CWrap = cl->wrap(4*3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasColumnMajor,
        clblasNoTrans, clblasNoTrans,
        4, 2, 3,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    transpose(C, 3, 4);
    EXPECT_EQ(1*3-1*3, C[0]);
    EXPECT_EQ(1*2+3*0, C[1]);
    EXPECT_EQ(1*8+4*3, C[2]);
    EXPECT_EQ(-8, C[11]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}
TEST(testClBlas, colMajorTransA) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
//    transpose(A, 3, 2);
    transpose(B, 2, 1);
//    for(int row=0; row < 2; row++) {
//        for(int col=0; col < 1; col++) {
//            cout << B[row*1 + col] << " ";
//        }
//        cout << endl;
//    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasColumnMajor,
        clblasTrans, clblasNoTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    transpose(C, 1, 3);
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}
TEST(testClBlas, colMajorTransB) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float A[] = {1, 3,
                 2, 7,
                 9, 5};
    float B[] = {3,
                 -1};

    float C[3];
    transpose(A, 3, 2);
//    transpose(B, 2, 1);
//    for(int row=0; row < 2; row++) {
//        for(int col=0; col < 1; col++) {
//            cout << B[row*1 + col] << " ";
//        }
//        cout << endl;
//    }
    ClBlasInstance clblasInstance;
//    ClBlasInstance::initializeIfNecessary();
    CLWrapper *AWrap = cl->wrap(6, A);
    CLWrapper *BWrap = cl->wrap(2, B);
    CLWrapper *CWrap = cl->wrap(3, C);
    AWrap->copyToDevice();
    BWrap->copyToDevice();
    ClBlasHelper::Gemm(
        cl,
        clblasColumnMajor,
        clblasNoTrans, clblasTrans,
        3, 2, 1,
        1,
        AWrap, 0,
        BWrap, 0,
        0,
        CWrap, 0
    );
//    cl->finish();
    CWrap->copyToHost();
    transpose(C, 1, 3);
    EXPECT_EQ(0, C[0]);
    EXPECT_EQ(-1, C[1]);
    EXPECT_EQ(22, C[2]);

    delete CWrap;
    delete BWrap;
    delete AWrap;

    delete cl;
}

