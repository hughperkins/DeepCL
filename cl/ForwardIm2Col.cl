// from SpatialConvolutionMM.cu:

// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

//#define gPadding {{padding}}
//#define gStride {{stride}}
//#define gColSize {{colSize}}
//#define gFilterSize {{filterSize}}
//#define gSize {{size}}

// Kernel for fast unfold+copy
// (adapted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
kernel void im2col(
    const int n,
    global float const * im_data, int im_offset,
    global float* data_col) {
  global const float *data_im = im_data + im_offset;

  CL_KERNEL_LOOP(index, n) {
    int w_out = index % {{colSize}};
    index /= {{colSize}};
    int h_out = index % {{colSize}};
    int channel_in = index / {{colSize}};
    int channel_out = channel_in * {{filterSize}} * {{filterSize}};
    int h_in = h_out * {{stride}} - {{padding}};
    int w_in = w_out * {{stride}} - {{padding}};
    data_col += (channel_out * {{colSize}} + h_out) * {{colSize}} + w_out;
    data_im += (channel_in * {{size}} + h_in) * {{size}} + w_in;
    for (int i = 0; i < {{filterSize}}; ++i) {
      for (int j = 0; j < {{filterSize}}; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < {{size}} && w < {{size}}) ?
          data_im[i * {{size}} + j] : 0;
        data_col += {{colSize}} * {{colSize}};
      }
    }
  }
}

kernel void col2im(
    const int n,
    global float const *data_col,
    global float* im_data, int im_offset) {
  global float *data_im = im_data + im_offset;

  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0); index < (n); index += get_local_size(0) * get_num_groups(0)) {
    float val = 0;
    int w = index % {{size}} + {{padding}};
    int h = (index / {{size}}) % {{size}} + {{padding}};
    int c = index / ({{size}} * {{size}});
    // compute the start and end of the output
    int w_col_start = (w < {{filterSize}}) ? 0 : (w - {{filterSize}}) / {{stride}} + 1;
    int w_col_end = min(w / {{stride}} + 1, {{colSize}});
    int h_col_start = (h < {{filterSize}}) ? 0 : (h - {{filterSize}}) / {{stride}} + 1;
    int h_col_end = min(h / {{stride}} + 1, {{colSize}});

    int offset = (c * {{filterSize}} * {{filterSize}} + h * {{filterSize}} + w) * {{colSize}} * {{colSize}};
    int coeff_h_col = (1 - {{stride}} * {{filterSize}} * {{colSize}}) * {{colSize}};
    int coeff_w_col = (1 - {{stride}} * {{colSize}} * {{colSize}});
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

