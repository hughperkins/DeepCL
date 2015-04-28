cmake_minimum_required(VERSION 2.8)

#message( "[${CMAKE_SYSTEM_NAME}]" )
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" )
  set( ON_LINUX 1 )
endif()
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" )
  set( ON_WINDOWS 1 )
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
include_directories( qlearning )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )

if( ON_LINUX )
link_libraries(dl)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

if( ON_WINDOWS)
link_libraries(winmm) # needed for timeGetTime
endif()

set( DeepCL_sources LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    ActivationFunction.cpp 
    SquareLossLayer.cpp LossLayer.cpp CrossEntropyLoss.cpp SoftMaxLayer.cpp 
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp 
    Propagate4.cpp 
    Backward.cpp BackwardCpu.cpp BackwardGpuNaive.cpp BackwardGpuCached.cpp 
    BackpropWeights2.cpp BackpropWeights2Cpu.cpp BackpropWeights2Naive.cpp 
    BackpropWeights2Scratch.cpp BackpropWeights2ScratchLarge.cpp
    FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingPropagate.cpp PoolingPropagateCpu.cpp PoolingLayer.cpp PoolingBackprop.cpp
    PoolingBackpropCpu.cpp PoolingPropagateGpuNaive.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp PropagateByInputPlane.cpp
    PoolingBackpropGpuNaive.cpp ../qlearning/QLearner.cpp ../qlearning/array_helper.cpp
    ForceBackpropLayerMaker.cpp ForceBackpropLayer.cpp MnistLoader.cpp
    OnDemandBatcher.cpp Batcher.cpp NetAction.cpp ActivationLayer.cpp ActivationMaker.cpp
    ActivationForward.cpp ActivationForwardCpu.cpp ActivationForwardGpuNaive.cpp
    ActivationBackward.cpp ActivationBackwardCpu.cpp ActivationBackwardGpuNaive.cpp
    DropoutMaker.cpp DropoutLayer.cpp DropoutPropagate.cpp DropoutBackprop.cpp
    DropoutPropagateCpu.cpp DropoutPropagateGpuNaive.cpp
    DropoutBackpropCpu.cpp DropoutBackpropGpuNaive.cpp
    CopyBuffer.cpp MultiplyBuffer.cpp
    Trainer.cpp SGD.cpp
 )
foreach(source ${DeepCL_sources})
    set( DeepCL_sources_prefixed ${DeepCL_sources_prefixed} src/${source})
endforeach()

set( OpenCLHelper_sources OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c )
add_library( OpenCLHelper SHARED ${OpenCLHelper_sources} )
link_libraries( OpenCLHelper )

add_executable( gpuinfo OpenCLHelper/gpuinfo.cpp OpenCLHelper/thirdparty/clew/src/clew.c 
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp )
if( ON_LINUX )
target_link_libraries( gpuinfo dl )
endif()

add_library( DeepCL SHARED ${DeepCL_sources_prefixed} )
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
add_dependencies( DeepCL git_submodules_update )
endif()

#find_package(Lua51 QUIET)
#if( LUA51_FOUND )
#    OPTION(LUA51_AVAILABLE "Builds lua wrappers." ON)
#else()
#    OPTION(LUA51_AVAILABLE "Builds lua wrappers." OFF)
#endif()

find_package(PNG QUIET)
if( PNG_FOUND )
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." ON)
else()
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." OFF)
endif()

#if( LUA51_AVAILABLE )
#    find_package(Lua51 REQUIRED)
#    add_library( luaDeepCL SHARED lua/DeepCL_wrap.cxx )
#    target_include_directories(luaDeepCL PRIVATE ${LUA_INCLUDE_DIR})
#    target_link_libraries(luaDeepCL ${LUA_LIBRARIES})
#endif()

if( LIBPNGPP_AVAILABLE)
find_package(PNG REQUIRED)
link_libraries(${PNG_LIBRARY})
endif(LIBPNGPP_AVAILABLE)

link_libraries(DeepCL)

execute_process( COMMAND python -V ERROR_VARIABLE python_test )
if( ${python_test} MATCHES "Python 2\\.7\\..*" )
    #message("Python 2.7 found")
    OPTION(DEV_RUN_COG "For developers: re-run cogapp." ON)
else()
    OPTION(DEV_RUN_COG "For developers: re-run cogapp." OFF)
endif()

if(DEV_RUN_COG)
    add_custom_target( 
        cog
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/*.cpp
${CMAKE_CURRENT_SOURCE_DIR}/qlearning/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/qlearning/*.h
    )
    add_dependencies( DeepCL cog )
endif(DEV_RUN_COG)

#add_custom_target(
#    git-submodule-update
#    git submodule update --init --recursive
#)

set( UNITTEST_SOURCES test/testbackpropweights.cpp test/testpropagate.cpp test/testfilehelper.cpp
 test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackward.cpp test/testsinglebatch.cpp 
 test/testpoolingpropagate.cpp test/testpoolingbackprop.cpp test/testNorbLoader.cpp
 test/teststringhelper.cpp test/testGtestGlobals.cpp 
 src/stringhelper.cpp test/DimFromArgs.cpp test/testMemset.cpp test/WeightRandomizer.cpp
 test/testCopyBuffer.cpp test/CopyBuffer.cpp test/PrintBuffer.cpp test/testCopyBlock.cpp
 test/SpeedTemplates.cpp test/testSpeedTemplates.cpp test/testCopyLocal.cpp
 test/testNetdefToNet.cpp test/testactivationpropagate.cpp test/testactivationbackprop.cpp
 test/testRandomSingleton.cpp test/testdropoutpropagate.cpp test/testdropoutbackprop.cpp
 )
#
#
#test/testbackprop.cpp test/testbiasbackprop.cpp
#

add_executable( deepclrun src/deepclrun.cpp src/stringhelper.cpp )
#add_executable( clconvolve1 src/deepclrun.cpp src/stringhelper.cpp ) # same as deepcl, just for 
              # backwards compatibility, so dont need to increment major version :-P

#add_executable( idx-to-mat test/idxToMat.cpp src/stringhelper.cpp )
add_executable( cifar-to-mat test/CifarToMat.cpp src/stringhelper.cpp test/CifarLoader.cpp )
add_executable( prepare-norb test/prepare-norb.cpp src/stringhelper.cpp )
#add_executable( testmnist-mpi test/testmnist-mpi.cpp src/stringhelper.cpp )
# target_link_libraries( testmnist-mpi DeepCL )

#if( MPI_AVAILABLE )
    #add_definitions(-DMPI_AVAILABLE)

    #target_include_directories(testmnist-mpi PRIVATE ${MPI_C_INCLUDE_PATH})
    #target_link_libraries(testmnist-mpi ${MPI_CXX_LIBRARIES})
#endif()

if( LIBPNGPP_AVAILABLE)
    add_definitions(-DPNG_AVAILABLE)
#    add_executable( printMnist test/printMnist.cpp src/stringhelper.cpp )
#    set( UNITTEST_SOURCES ${UNITTEST_SOURCES} test/testimageshelper.cpp )
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
#    add_executable( prototyping prototyping/testreduce.cpp src/stringhelper.cpp thirdparty/gtest/gtest_main.cc )
#    target_link_libraries( prototyping gtest )
#    target_include_directories( prototyping PRIVATE thirdparty/gtest )
#    add_executable( blocking prototyping/blocking.cpp src/stringhelper.cpp )
#    add_executable( testrawstring prototyping/testRawString.cpp )
#    add_executable( testpassdim prototyping/testPassDim.cpp )
#    add_executable( testperf prototyping/testperf.cpp test/TestArgsParser.cpp test/GtestGlobals.cpp
#       test/SpeedTemplates.cpp )
#    add_executable( testmemperf prototyping/testmemperf.cpp test/TestArgsParser.cpp test/GtestGlobals.cpp
#       test/SpeedTemplates.cpp )
#    add_executable( testfloat4 prototyping/testFloat4.cpp test/TestArgsParser.cpp test/GtestGlobals.cpp
#       test/SpeedTemplates.cpp )
#    add_executable( testrandomizevector prototyping/testRandomizeVector.cpp test/WeightRandomizer.cpp )
    add_executable( clconvolve-fixedweights prototyping/clconvolve-fixedweights.cpp src/stringhelper.cpp )
    add_executable( learn_scenario_image prototyping/qlearning/learnScenarioImage.cpp 
        prototyping/qlearning/ScenarioImage.cpp )
endif( PROTOTYPING )

target_link_libraries( unittests DeepCL )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

#if( ON_LINUX )
#add_custom_target( 
#    clfiles
#    cp ${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cl ${CMAKE_BINARY_DIR}
#)
#add_dependencies( DeepCL clfiles )
#else()
## no idea how to fix this, for Windows.  If you have any ideas, please send a pull request :-)
##add_custom_target( 
##    clfiles
##    copy ${CMAKE_CURRENT_SOURCE_DIR}\\cl\\*.cl ${CMAKE_BINARY_DIR}
##)
#endif()

find_package(Lua51 QUIET)
if(LUA51_FOUND)
    OPTION(BUILD_LUA_WRAPPERS "Builds Lua wrappers" ON)
else()
    OPTION(BUILD_LUA_WRAPPERS "Builds Lua wrappers" OFF)
endif()

if( BUILD_LUA_WRAPPERS )
add_subdirectory(lua)
endif()

find_package(PythonLibs QUIET)
if(PYTHONLIBS_FOUND)
    OPTION(BUILD_PYSWIG_WRAPPERS "Builds Python swig wrappers (not the Cython ones)." ON)
else()
    OPTION(BUILD_PYSWIG_WRAPPERS "Builds Python swig wrappers (not the Cython ones)." OFF)
endif()

if( BUILD_PYSWIG_WRAPPERS)
add_subdirectory(python_swig)
endif()


