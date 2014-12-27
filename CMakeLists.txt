cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)

include_directories( . )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )
link_libraries(dl)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

add_library( ClConvolve SHARED ClConvolve.h OpenCLHelper/OpenCLHelper.cpp LayerMaker.cpp NeuralNetMould.cpp
   EpochMaker.cpp
   OpenCLHelper/CLKernel.cpp OpenCLHelper/clew/src/clew.c )

file( COPY ClConvolve.cl DESTINATION ${CMAKE_BINARY_DIR} )

find_package(PNG)
link_libraries(${PNG_LIBRARY})

link_libraries(ClConvolve)

add_executable( test2layerfullyconnected test/test2layerfullyconnected.cpp )

#set( OPENCLHELPER $ENV{HOME}/git/OpenCLHelper)
add_executable( testarraysquare test/testarraysquare.cpp )
add_executable( testarraysquare_floats test/testarraysquare_floats.cpp )
add_executable( testClConvolve2 test/testClConvolve2.cpp )
#add_executable( testopencl test/testopencl.cpp )
add_executable( testboardshelper test/testboardshelper.cpp )


