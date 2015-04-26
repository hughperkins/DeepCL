// placeholder, for now
kernel void propagateNaive(
        const int N, 
        global const unsigned char *mask,
        global const float *input,
        global float *output ) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;
}

// placeholder, for now
kernel void backpropNaive() {
}

