// we expect that din1, din2 each have 4 values
// zeropad has same number of values as dimensions, ie 4
// 1 means zeropad that dimension (output will have same length as input)
// 0 means no zeropad (so output length will equal abs(d1-d2)+1 )
kernel void convolve( int batchSize,
         const int *din1, const int *din2, const int *dout, const int *zeropad,
        global const float *in1, const float *in2, global float *out ) {
    int globalId = get_global_id(0);
    
    const 

}

