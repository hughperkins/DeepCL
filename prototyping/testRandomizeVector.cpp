#include <iostream>
using namespace std;

#include "ClConvolve_typedefs.h"
#include "test/WeightRandomizer.h"

using namespace ClConvolve;

int main( int argc, char *argv[] ) {
    vfloat a(10);
    WeightRandomizer::randomize( a.begin(), a.end(), -0.1f, 0.1f );
    for( vfloat::iterator it = a.begin(); it != a.end(); it++ ) {
        cout << it - a.begin() << ": " << *it << endl;
    }

    vint b(10);
    WeightRandomizer::randomizeInts( b.begin(), b.end(), 0, 10 );
    for( vint::iterator it = b.begin(); it != b.end(); it++ ) {
        cout << it - b.begin() << ": " << *it << endl;
    }

    return 0;
}

