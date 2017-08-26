#include "clBLAS.h"

#include "ClBlasInstance.h"

#include <iostream>
using namespace std;

#define PUBLIC

PUBLIC ClBlasInstance::ClBlasInstance() {
    // cout << "initializing clblas" << endl;
    clblasSetup();
}

PUBLIC ClBlasInstance::~ClBlasInstance() {
    // cout << "clblas teardown" << endl;
    clblasTeardown();
}

//bool ClBlasInstance::initialized = false;

// assume single-threaded, at least for now
//void ClBlasInstance::initializeIfNecessary() {
//    if(!initialized) {
//        cout << "initializing clblas" << endl;
//        clblasSetup();
//        initialized = true;
//    }
//}

