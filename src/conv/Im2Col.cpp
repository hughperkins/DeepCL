#include "Im2Col.h"

#include <iostream>
#include <stdexcept>
using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIc
#define VIRTUAL
#define PUBLIC

PUBLIC Im2Col::Im2Col(EasyCL *cl, LayerDimensions dim) :
    cl(cl),
    dim(dim) {
}
PUBLIC void Im2Col::im2Col(CLWrapper *im, int64 im_offset, CLWrapper *columns) {
    throw runtime_error("not implemented yet");
}
PUBLIC void Im2Col::col2Im(CLWrapper *columns, CLWrapper *im, int64 im_offset) {
    throw runtime_error("not implemented yet");
}

