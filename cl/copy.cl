// simply copies from one to the other...
// there might be something built-in to opencl for this
// anyway... :-)
kernel void copy(
        const int N,
        global const float *in,
        global float *out ) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    out[globalId] = in[globalId];
}

