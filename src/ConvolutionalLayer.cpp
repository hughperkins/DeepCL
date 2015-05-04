// this has moved to:
// conv/ConvolutionalLayer.cpp, ie
// https://github.com/hughperkins/DeepCL/blob/master/src/conv/ConvolutionalLayer.cpp
//
// but really if you're after the convolutional implementations themselves...
//
// opencl code is at:
// https://github.com/hughperkins/DeepCL/blob/master/cl/
// - in particular forward convolution is in the files called forward[something].cl
// - backward convolution is in the files called backward[something].cl
// - weights gradients calculations are in the files called backpropweights[something].cl
//
// c++ code that runs these files is at https://github.com/hughperkins/DeepCL/blob/master/src/conv
// - forward prop in the files called Forward[something].h (and .cpp)
// - backward prop in the files called Backward[something].h (and .cpp)
// - weights gradient calculations are in the files called BackpropWeights[something].h (and .cpp)

// quick summary of the different forward implementations:
//
// forward1: generic forward propagation, should probably work on most configurations, but
// doesnt use local memory, so slow
// forward2,3,4, etc ...: various configurations of using local memory to reduce transfers
//  from gpu global memory.  Each configuration will only work for certain image sizes etc
//  and so we need different implementations for different sizes
//  Up till now, most of the work has been going into go-board sized images, 19x19, and also
//  a bit of work into mnist-sized, around 28x28 or 24x24
//  Larger images probably need some kind of blocking structure most likely

