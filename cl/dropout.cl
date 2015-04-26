// placeholder, for now
#ifdef gInverseDropRatio
kernel void propagateNaive(
        const int N, 
        global const unsigned char *mask,
        global const float *input,
        global float *output ) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    output[globalId] = mask[globalId] == 1 ? gInverseDropRatio * input[globalId] : 0.0f;
}
#endif

// placeholder, for now
#ifdef gDropRatio
kernel void backpropNaive(
        const int N,
        global const unsigned char *mask,
        global const float *errors,
        global float *output) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    output[globalId] = mask[globalId] == 1 ? errors[globalId] : 0.0f;
}
#endif

