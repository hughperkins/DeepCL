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

