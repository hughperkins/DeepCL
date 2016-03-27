// Copyright Hugh Perkins 201, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU]

#ifdef TANH
    #define ACTIVATION_DERIV(output) (1 - output * output)
#elif defined SCALEDTANH
    #define ACTIVATION_DERIV(output) (0.66667f * (1.7159f - 1 / 1.7159f * output * output) )
#elif defined SIGMOID
    #define ACTIVATION_DERIV(output) (output * (1 - output) )
#elif defined RELU
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)
#elif defined ELU
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : output + 1)
#elif defined LINEAR
    #define ACTIVATION_DERIV(output) (1.0f)
#endif

//#ifdef ACTIVATION_DERIV
//void kernel applyActivationDeriv( 
//        const int N,
//        global float *inout) {
//    int globalId = get_global_id(0);
//    inout[globalId] = ACTIVATION_DERIV(inout[globalId]);
//}
//#endif

#ifdef ACTIVATION_DERIV
void kernel applyActivationDeriv( 
        const int N,
        global float *target, global const float *source) {
    int globalId = get_global_id(0);
    if (globalId < N) {
        target[globalId] *= ACTIVATION_DERIV(source[globalId]);
    }
  //  target[globalId] *= source[globalId];
}
#endif

#ifdef ACTIVATION_DERIV
void kernel backward( 
        const int N,
        global const float *inputs,
        global const float *gradOutput, 
        global float *gradInput) {
    int globalId = get_global_id(0);
    if (globalId < N) {
        gradInput[globalId] = ACTIVATION_DERIV(inputs[globalId]) * gradOutput[globalId];
            // probably not ideal to have the output and input separate?
    }
  //  target[globalId] *= source[globalId];
}
#endif

