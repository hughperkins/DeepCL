#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

// - check it works on linux
// - check it works on windows
int main( int argc, char *argv[] ) {
    string mystring = R"DELIM(
    This is an example of a raw string.
    kernel void( global float *a ) {
        doSomething(a, b, c );
    };
    )DELIM";
    cout << mystring << endl;
    return 0;
}

