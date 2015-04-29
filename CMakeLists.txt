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
mark_as_advanced( FORCE PROTOTYPING )
unset(BUILD_PYSWIG_WRAPPERS CACHE)
unset(BUILD_PYTHON_WRAPPERS CACHE)
unset(RUN_COG CACHE)
unset(RUN_SWIG CACHE)
unset(PYTHON_AVAILABLE CACHE)
unset(LUA51_AVAILABLE CACHE)

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
    Forward1.cpp Forward.cpp Forward2.cpp Forward3.cpp LayerDimensions.cpp
    ForwardExperimental.cpp ForwardAuto.cpp ForwardCpu.cpp 
    Forward4.cpp 
    Backward.cpp BackwardCpu.cpp BackwardGpuNaive.cpp BackwardGpuCached.cpp 
    BackpropWeights.cpp BackpropWeightsCpu.cpp BackpropWeightsNaive.cpp 
    BackpropWeightsScratch.cpp BackpropWeightsScratchLarge.cpp
    FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingForward.cpp PoolingForwardCpu.cpp PoolingLayer.cpp PoolingBackward.cpp
    PoolingBackwardCpu.cpp PoolingForwardGpuNaive.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    ForwardFc.cpp ForwardByInputPlane.cpp
    PoolingBackwardGpuNaive.cpp ../qlearning/QLearner.cpp ../qlearning/array_helper.cpp
    ForceBackpropLayerMaker.cpp ForceBackpropLayer.cpp MnistLoader.cpp
    OnDemandBatcher.cpp Batcher.cpp NetAction.cpp ActivationLayer.cpp ActivationMaker.cpp
    ActivationForward.cpp ActivationForwardCpu.cpp ActivationForwardGpuNaive.cpp
    ActivationBackward.cpp ActivationBackwardCpu.cpp ActivationBackwardGpuNaive.cpp
    DropoutMaker.cpp DropoutLayer.cpp DropoutForward.cpp DropoutBackward.cpp
    DropoutForwardCpu.cpp DropoutForwardGpuNaive.cpp
    DropoutBackwardCpu.cpp DropoutBackwardGpuNaive.cpp
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

OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." OFF)
mark_as_advanced(FORCE LIBPNGPP_AVAILABLE)

if( LIBPNGPP_AVAILABLE)
find_package(PNG REQUIRED)
link_libraries(${PNG_LIBRARY})
endif(LIBPNGPP_AVAILABLE)

link_libraries(DeepCL)

OPTION(DEV_RUN_COG "DeepCL developers should turn this on." OFF)

if(DEV_RUN_COG)
    add_custom_target( 
        cog
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/*.cpp
${CMAKE_CURRENT_SOURCE_DIR}/qlearning/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/qlearning/*.h
    )
    add_dependencies( DeepCL cog )
endif(DEV_RUN_COG)

set( UNITTEST_SOURCES test/testupdateweights.cpp test/testforward.cpp test/testfilehelper.cpp
 test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackward.cpp test/testsinglebatch.cpp 
 test/testpoolingforward.cpp test/testpoolingbackward.cpp test/testNorbLoader.cpp
 test/teststringhelper.cpp test/testGtestGlobals.cpp 
 src/stringhelper.cpp test/DimFromArgs.cpp test/testMemset.cpp test/WeightRandomizer.cpp
 test/testCopyBuffer.cpp test/CopyBuffer.cpp test/PrintBuffer.cpp test/testCopyBlock.cpp
 test/SpeedTemplates.cpp test/testSpeedTemplates.cpp test/testCopyLocal.cpp
 test/testNetdefToNet.cpp test/testactivationforward.cpp test/testactivationbackward.cpp
 test/testRandomSingleton.cpp test/testdropoutforward.cpp test/testdropoutbackward.cpp
 )

add_executable( deepclrun src/deepclrun.cpp src/stringhelper.cpp )

add_executable( cifar-to-mat test/CifarToMat.cpp src/stringhelper.cpp test/CifarLoader.cpp )
add_executable( prepare-norb test/prepare-norb.cpp src/stringhelper.cpp )

if( LIBPNGPP_AVAILABLE)
    add_definitions(-DPNG_AVAILABLE)
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
    add_executable( clconvolve-fixedweights prototyping/clconvolve-fixedweights.cpp src/stringhelper.cpp )
    add_executable( learn_scenario_image prototyping/qlearning/learnScenarioImage.cpp 
        prototyping/qlearning/ScenarioImage.cpp )
endif( PROTOTYPING )

target_link_libraries( unittests DeepCL )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

OPTION(BUILD_LUA_WRAPPERS "Builds Lua wrappers" OFF)

if( BUILD_LUA_WRAPPERS )
add_subdirectory(lua)
endif()


