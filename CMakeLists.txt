cmake_minimum_required(VERSION 2.8)

if (NOT CMAKE_BUILD_TYPE)
    message("Setting build type to 'RelWithDebInfo'")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

include_directories( . )
include_directories( src )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )
link_libraries(dl)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set( CLConvolve_sources LayerMaker.cpp NeuralNetMould.cpp
    EpochMaker.cpp ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp LayerMaker.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
    BackpropWeights2Naive.cpp BackpropErrorsv2Naive.cpp BackpropWeights2Scratch.cpp
    CrossEntropyLoss.cpp SoftMaxLayer.cpp)
foreach(source ${CLConvolve_sources})
    set( CLConvolve_sources_prefixed ${CLConvolve_sources_prefixed} src/${source})
endforeach()

add_library( ClConvolve SHARED OpenCLHelper/OpenCLHelper.cpp ${CLConvolve_sources_prefixed}
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c )
#    BackpropErrors.cpp BackpropErrors1.cpp BackpropErrorsCpu.cpp BackpropErrors2.cpp
#    BackpropWeights.cpp BackpropWeightsScratchBias.cpp BackpropWeightsNaive.cpp BackpropWeightsCpu.cpp
add_custom_command( 
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/thirdparty/clew/src/clew.c ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/OpenCLHelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/CLKernel.cpp
    COMMAND git submodule update --init --force --recursive
)
add_custom_target( git_submodules_update DEPENDS OpenCLHelper/thirdparty/clew/src/clew.c )
add_dependencies( ClConvolve git_submodules_update )

find_package(PNG QUIET)
if( PNG_FOUND )
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." ON)
else()
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." OFF)
endif()

if( LIBPNGPP_AVAILABLE)
find_package(PNG)
link_libraries(${PNG_LIBRARY})
endif(LIBPNGPP_AVAILABLE)

link_libraries(ClConvolve)

execute_process( COMMAND python -V ERROR_VARIABLE python_test )
if( ${python_test} MATCHES "Python 2\\.7\\..*" )
    #message("Python 2.7 found")
    OPTION(PYTHON_AVAILABLE "Facilitates development, not needed for normal build." ON)
else()
    OPTION(PYTHON_AVAILABLE "Facilitates development, not needed for normal build." OFF)
endif()

if(PYTHON_AVAILABLE)
    add_custom_target( 
        cog
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp
    )
    add_dependencies( ClConvolve cog )
endif(PYTHON_AVAILABLE)

#add_custom_target(
#    git-submodule-update
#    git submodule update --init --recursive
#)

set( UNITTEST_SOURCES unittests test/testbackpropweights.cpp test/testpropagate.cpp test/testfilehelper.cpp
 test/testMnistLoader.cpp test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackproperrors.cpp test/testsinglebatch.cpp test/testmnistlearn_unit.cpp
 )
#
#
#test/testbackprop.cpp test/testbiasbackprop.cpp
#

#add_executable( testneuralnetmnist test/testneuralnetmnist.cpp stringhelper.cpp )
#add_executable( testneuralnetmnistconvolve test/testneuralnetmnistconvolve.cpp stringhelper.cpp )
#add_executable( testneuralnetmnistconvolve_simple test/testneuralnetmnistconvolve_simple.cpp stringhelper.cpp )
#add_executable( test2layerconvolve test/test2layerconvolve.cpp stringhelper.cpp )
add_executable( testneuralnetmnistconvolve-experimental test/testneuralnetmnistconvolve-experimental.cpp src/stringhelper.cpp )
if( PNG_AVAILABLE)
    #add_executable( test4dconvolve test/test4dconvolve.cpp )
    add_executable( printMnist test/printMnist.cpp stringhelper.cpp )
    set( UNITTEST_SOURCES ${UNITTEST_SOURCES} test/testboardshelper.cpp )
endif(PNG_AVAILABLE)

#add_dependencies( ClConvolve git-submodule-update )

add_library( gtest SHARED thirdparty/gtest/gtest-all.cc )
target_link_libraries( gtest pthread)

add_executable( ${UNITTEST_SOURCES}
thirdparty/gtest/gtest_main.cc test/myasserts.cpp )
target_link_libraries( unittests gtest )
target_include_directories( unittests PRIVATE thirdparty/gtest )

add_executable( loadkgs src/loadkgs.cpp src/stringhelper.cpp )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

#file( COPY ClConvolve.cl DESTINATION ${CMAKE_BINARY_DIR} )

add_custom_target( 
    clfiles
    cp ${CMAKE_CURRENT_SOURCE_DIR}/*.cl ${CMAKE_BINARY_DIR}
)

add_dependencies( ClConvolve clfiles )

#add_dependencies( testneuralnetmnistconvolve clfiles )
#add_dependencies( test4dconvolve clfiles )


