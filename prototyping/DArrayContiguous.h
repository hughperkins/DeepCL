// creates a D-dimensional array, allocated in contiguous space
// eg, can just pass it into GPU as a single array (or to mpi etc ... :-) )
#pragma once

#include <cassert>
#include <iostream>

template< typename T >
class DArrayContiguous {
public:
    const int D;
    int *size;
    int contiguousSize;
    T *array;
    DArrayContiguous( int d1 ) :
        D(1),
        size( new int[1] ) {
        size[0] = d1;
        init();
    }
    DArrayContiguous( int d1, int d2 ) :
        D(2),
        size( new int[2] ) {
        size[0] = d1;
        size[1] = d2;
        init();
    }
    DArrayContiguous( const int numDims, const int *dimSizes ) :
        D( numDims ),
        size( new int[numDims] ),
        array(0) {
        if( D == 0 ) {
            return;
        }
        for( int d = 0; d < D; d++ ) {
            size[d] = dimSizes[d];
        }
        init();
    }
    void init() {
        int contiguousSize = 1;
        for( int d = 0; d < D; d++ ) {
            contiguousSize *= size[d];
        }
        if( contiguousSize == 0 ) {
            return;
        }
        array = new T[contiguousSize];
        this->contiguousSize = contiguousSize;
        if( D == 1 ) {
            this->jagged = (void**)array;
        } else {
            // create jagged...
            this->jagged = new void *[size[0]];
            initJagged( (void **)this->jagged, 1, 0, contiguousSize / size[0] );
        }
    }
    void initJagged( void **thisjagged, int d, int arrayoffset, int hypercubeSize ) {
        std::cout << "initJagged d " << d << " arrayoffset " << arrayoffset << " hypercubesize " << hypercubeSize << std::endl;
        for( int i = 0; i < size[d]; i++ ) {
            if( d < D - 1 ) {
                thisjagged[i] = (void*)(new void *[size[d+1]]);
                initJagged( (void**)thisjagged[i], d+1, arrayoffset + hypercubeSize * i, hypercubeSize / size[d] );
            } else {
                ((T**)thisjagged[i] = &(array[arrayoffset + hypercubeSize * i ]);
            }
        }
    }
    T *getJagged1d() {
        assert(D==1);
        return reinterpret_cast<T*>(this->jagged);
    }
    T **getJagged2d() {
        assert(D==2);
        return reinterpret_cast<T**>(this->jagged);
    }
    T &operator()( int x ) {
        return array[x];
    }
    T &operator()( int x, int y ) {
        return array[0];
    }
    T &operator()( int x, int y, int z ) {
        return array[0];
    }
    T * getContiguous() {
        return array;
    }
    ~DArrayContiguous() {
        if( array != 0 ) {
            delete[] array;
        }
        if( size != 0 ) {
            delete[] size;
        }
    }
};

