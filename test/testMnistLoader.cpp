#include <iostream>
using namespace std;

#include "MnistLoader.h"
#include "test/myasserts.h"

int main( int argc, char *argv[] ) {
    unsigned char fakedata[8];
    MnistLoader::writeUInt( fakedata, 0, 1234567890 );
    unsigned int returned = MnistLoader::readUInt( fakedata, 0 );
    assertEquals( (int)returned, 1234567890 );
    return 0;
}

