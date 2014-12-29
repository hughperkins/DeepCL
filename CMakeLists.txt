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

add_custom_target( 
    clfiles
    cp ${CMAKE_CURRENT_SOURCE_DIR}/*.cl ${CMAKE_BINARY_DIR}
)

add_custom_target(
    git-submodule-update
    git submodule update --init --recursive
)

add_executable( testlogicaloperators test/testlogicaloperators.cpp )

#set( OPENCLHELPER $ENV{HOME}/git/OpenCLHelper)
add_executable( testarraysquare test/testarraysquare.cpp )
add_executable( testarraysquare_floats test/testarraysquare_floats.cpp )
#add_executable( testClConvolve2 test/testClConvolve2.cpp )
#add_executable( testopencl test/testopencl.cpp )
add_executable( testboardshelper test/testboardshelper.cpp )
add_executable( testsimpleconvolvenet test/testsimpleconvolvenet.cpp )
add_executable( testneuralnetmnist test/testneuralnetmnist.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve test/testneuralnetmnistconvolve.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve_simple test/testneuralnetmnistconvolve_simple.cpp stringhelper.cpp )
add_executable( test4dconvolve test/test4dconvolve.cpp )
add_executable( testsimpleconvolve test/testsimpleconvolve.cpp )
add_executable( test2layerconvolve test/test2layerconvolve.cpp stringhelper.cpp )
add_executable( printMnist test/printMnist.cpp stringhelper.cpp )

add_dependencies( testsimpleconvolve clfiles )
add_dependencies( testsimpleconvolvenet clfiles )
add_dependencies( testneuralnetmnistconvolve clfiles )
add_dependencies( test4dconvolve clfiles )

add_dependencies( ClConvolve git-submodule-update )

