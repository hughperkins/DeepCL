cmake_minimum_required(VERSION 2.8)

#message( "[${CMAKE_SYSTEM_NAME}]" )
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" )
  set( ON_LINUX 1 )
endif()

if (NOT CMAKE_BUILD_TYPE)
    message("Setting build type to 'RelWithDebInfo'")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

OPTION(PROTOTYPING "Only for devs." OFF)

#find_package(MPI QUIET)
#if( MPI_FOUND )
#    OPTION(MPI_AVAILABLE "Parallelization over multiple hosts." ON)
#else()
#    OPTION(MPI_AVAILABLE "Parallelization over multiple hosts." OFF)
#endif()
#if( MPI_AVAILABLE )
#find_package(MPI REQUIRED)
#endif()

include_directories( . )
include_directories( src )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )

if( ON_LINUX )
link_libraries(dl)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set( CLConvolve_sources LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp LayerMaker.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
    BackpropWeights2Naive.cpp BackpropErrorsv2Naive.cpp BackpropWeights2Scratch.cpp
    CrossEntropyLoss.cpp SoftMaxLayer.cpp FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingPropagate.cpp PoolingPropagateCpu.cpp PoolingLayer.cpp PoolingBackprop.cpp
    PoolingBackpropCpu.cpp PoolingPropagateGpuNaive.cpp BackpropWeights2ScratchLarge.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp BackpropErrorsv2Cached.cpp PropagateByInputPlane.cpp
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp
 )
foreach(source ${CLConvolve_sources})
    set( CLConvolve_sources_prefixed ${CLConvolve_sources_prefixed} src/${source})
endforeach()

add_library( OpenCLHelper SHARED OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c )
link_libraries( OpenCLHelper )

add_executable( gpuinfo OpenCLHelper/gpuinfo.cpp OpenCLHelper/thirdparty/clew/src/clew.c 
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp )
if( ON_LINUX )
target_link_libraries( gpuinfo dl )
endif()

add_library( ClConvolve SHARED ${CLConvolve_sources_prefixed} )
#    BackpropErrors.cpp BackpropErrors1.cpp BackpropErrorsCpu.cpp BackpropErrors2.cpp
#    BackpropWeights.cpp BackpropWeightsScratchBias.cpp BackpropWeightsNaive.cpp BackpropWeightsCpu.cpp

execute_process( COMMAND gitx --version RESULT_VARIABLE git_test ERROR_QUIET )
if( ${git_test} EQUAL 0 )
    set( GIT_FOUND 1 )
endif()
if( GIT_FOUND )
message("git found")
endif()

if( GIT_FOUND )
add_custom_command( 
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/thirdparty/clew/src/clew.c ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/OpenCLHelper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHelper/CLKernel.cpp
    COMMAND git submodule update --init --force --recursive
)
add_custom_target( git_submodules_update DEPENDS OpenCLHelper/thirdparty/clew/src/clew.c )
add_dependencies( ClConvolve git_submodules_update )
endif()

find_package(PNG QUIET)
if( PNG_FOUND )
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." ON)
else()
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." OFF)
endif()

if( LIBPNGPP_AVAILABLE)
find_package(PNG REQUIRED)
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
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h
    )
    add_dependencies( ClConvolve cog )
endif(PYTHON_AVAILABLE)

#add_custom_target(
#    git-submodule-update
#    git submodule update --init --recursive
#)

set( UNITTEST_SOURCES test/testbackpropweights.cpp test/testpropagate.cpp test/testfilehelper.cpp
 test/testMnistLoader.cpp test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackproperrors.cpp test/testsinglebatch.cpp test/testmnistlearn_unit.cpp
 test/testpoolingpropagate.cpp test/testpoolingbackprop.cpp test/testNorbLoader.cpp
 test/teststringhelper.cpp test/testGtestGlobals.cpp test/testArgsParser.cpp
 src/stringhelper.cpp test/DimFromArgs.cpp
 )
#
#
#test/testbackprop.cpp test/testbiasbackprop.cpp
#

add_executable( clconvolve1 src/clconvolve1.cpp src/stringhelper.cpp )

add_executable( idx-to-mat test/idxToMat.cpp src/stringhelper.cpp )
add_executable( cifar-to-mat test/CifarToMat.cpp src/stringhelper.cpp test/CifarLoader.cpp )
add_executable( prepare-norb test/prepare-norb.cpp src/stringhelper.cpp )
#add_executable( testmnist-mpi test/testmnist-mpi.cpp src/stringhelper.cpp )
# target_link_libraries( testmnist-mpi ClConvolve )

#if( MPI_AVAILABLE )
    #add_definitions(-DMPI_AVAILABLE)

    #target_include_directories(testmnist-mpi PRIVATE ${MPI_C_INCLUDE_PATH})
    #target_link_libraries(testmnist-mpi ${MPI_CXX_LIBRARIES})
#endif()

if( LIBPNGPP_AVAILABLE)
    add_definitions(-DPNG_AVAILABLE)
#    add_executable( printMnist test/printMnist.cpp src/stringhelper.cpp )
    set( UNITTEST_SOURCES ${UNITTEST_SOURCES} test/testboardshelper.cpp )
#    add_executable( printmat test/printMat.cpp src/stringhelper.cpp )
    add_executable( testpatchextractor test/testPatchExtractor.cpp src/stringhelper.cpp src/PatchExtractor.cpp )
    add_executable( testtranslator test/testTranslator.cpp src/stringhelper.cpp src/Translator.cpp )
    add_executable( testgenericloader test/testGenericLoader.cpp src/stringhelper.cpp )
endif(LIBPNGPP_AVAILABLE)

if( ON_LINUX )
add_library( gtest SHARED thirdparty/gtest/gtest-all.cc )
target_link_libraries( gtest pthread)
else()
add_library( gtest thirdparty/gtest/gtest-all.cc )
endif()
target_include_directories( gtest PRIVATE thirdparty/gtest )

add_executable( unittests ${UNITTEST_SOURCES} test/gtest_main.cpp test/GtestGlobals.cpp test/myasserts.cpp
    test/TestArgsParser.cpp )
target_link_libraries( unittests gtest )
target_include_directories( unittests PRIVATE thirdparty/gtest )

if( PROTOTYPING )
    add_executable( prototyping prototyping/testreduce.cpp src/stringhelper.cpp thirdparty/gtest/gtest_main.cc )
    target_link_libraries( prototyping gtest )
    target_include_directories( prototyping PRIVATE thirdparty/gtest )
    add_executable( blocking prototyping/blocking.cpp src/stringhelper.cpp )
endif( PROTOTYPING )

target_link_libraries( unittests ClConvolve )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

#if( ON_LINUX )
#add_custom_target( 
#    clfiles
#    cp ${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cl ${CMAKE_BINARY_DIR}
#)
#add_dependencies( ClConvolve clfiles )
#else()
## no idea how to fix this, for Windows.  If you have any ideas, please send a pull request :-)
##add_custom_target( 
##    clfiles
##    copy ${CMAKE_CURRENT_SOURCE_DIR}\\cl\\*.cl ${CMAKE_BINARY_DIR}
##)
#endif()


