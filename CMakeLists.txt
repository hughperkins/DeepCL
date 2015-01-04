cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)

include_directories( . )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )
link_libraries(dl)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

add_library( ClConvolve SHARED OpenCLHelper/OpenCLHelper.cpp LayerMaker.cpp NeuralNetMould.cpp
   EpochMaker.cpp ConvolutionalLayer.cpp
   OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c )

add_custom_command( 
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/thirdparty/clew/src/clew.c ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/OpenCLHelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/CLKernel.cpp
    COMMAND git submodule update --init --recursive
)
add_custom_target( git_submodules_update DEPENDS OpenCLHelper/thirdparty/clew/src/clew.c )
add_dependencies( ClConvolve git_submodules_update )

find_package(PNG)
link_libraries(${PNG_LIBRARY})

link_libraries(ClConvolve)

#add_custom_target(
#    git-submodule-update
#    git submodule update --init --recursive
#)

add_executable( testneuralnetmnist test/testneuralnetmnist.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve test/testneuralnetmnistconvolve.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve_simple test/testneuralnetmnistconvolve_simple.cpp stringhelper.cpp )
add_executable( test4dconvolve test/test4dconvolve.cpp )
add_executable( test2layerconvolve test/test2layerconvolve.cpp stringhelper.cpp )
add_executable( printMnist test/printMnist.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve-experimental test/testneuralnetmnistconvolve-experimental.cpp stringhelper.cpp )

#add_dependencies( ClConvolve git-submodule-update )

add_library( gtest SHARED thirdparty/gtest/gtest-all.cc )
target_link_libraries( gtest pthread)

add_executable( unittests test/testsimpleconvolve.cpp test/testfilehelper.cpp test/testMnistLoader.cpp test/testboardshelper.cpp test/testlogicaloperators.cpp test/testsimpleconvolvenet.cpp test/testmnistlearn_unit.cpp
test/testbackprop.cpp test/testcalcerrorsforupstream.cpp test/testbiasbackprop.cpp test/testsinglebatch.cpp
thirdparty/gtest/gtest_main.cc test/myasserts.cpp )
target_link_libraries( unittests gtest )
target_include_directories( unittests PRIVATE thirdparty/gtest )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

file( COPY ClConvolve.cl DESTINATION ${CMAKE_BINARY_DIR} )

add_custom_target( 
    clfiles
    cp ${CMAKE_CURRENT_SOURCE_DIR}/*.cl ${CMAKE_BINARY_DIR}
)

add_dependencies( ClConvolve clfiles )
#add_dependencies( testneuralnetmnistconvolve clfiles )
#add_dependencies( test4dconvolve clfiles )


