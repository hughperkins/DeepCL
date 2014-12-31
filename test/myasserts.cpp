#include "myasserts.h"

using namespace std;

void assertEquals( float one, float two, float tolerance ) {
   float absdiff = one - two > 0 ? one - two : two - one;
   if( absdiff > tolerance ) {
      throw std::runtime_error( "assertEquals fail " + toString(one) + " != " + toString(two) );
   }
}


void assertEquals( float one, int two ) {
   assertEquals( one, (float)two );
}
void assertEquals( int one, float two ) {
   assertEquals( (float)one, two );
}

void assertLessThan( float expected, float actual ) {
    if( expected <= actual ) {
        throw std::runtime_error( "assertLessThan fail expected max " + toString(expected) + " but was " + toString(actual) );
    }
}

void assertEquals( int one, int two ) {
   assertEquals<int>( one, two );
}


