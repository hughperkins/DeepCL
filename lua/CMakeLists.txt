cmake_minimum_required(VERSION 2.8)

find_package(Lua51 REQUIRED)

if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" )
  set( ON_LINUX 1 )
endif()
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" )
  set( ON_WINDOWS 1 )
endif()

if( ON_LINUX )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

# This is so we use the source tree at .., when we are building in the git clone etc
# but in a rock, where we've set inpkg.flag file, and copied the source into pkgsrc
# directory, we will use pkgsrc for our source code
if( EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inpkg.flag" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pkgsrc" )
set(SRCTREE ${CMAKE_CURRENT_SOURCE_DIR}/pkgsrc)
else()
set(SRCTREE ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
# message("Using source tree at: ${SRCTREE}")

find_package(SWIG QUIET)

OPTION(DEV_RUN_SWIG "DeepCL developers should turn this on." OFF)

set( lua_DeepCL_sources 
    LayerMaker.cpp NeuralNetMould.cpp
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
    PoolingForward.cpp PoolingForwardCpu.cpp PoolingLayer.cpp PoolingBackward.cpp
    PoolingBackwardCpu.cpp PoolingForwardGpuNaive.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp PropagateByInputPlane.cpp
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
     ../qlearning/QLearner.cpp ../qlearning/array_helper.cpp
 )

set(lua_OpenCLHelper_sources OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c)

foreach(source ${lua_DeepCL_sources})
    set( lua_DeepCL_sources_prefixed ${lua_DeepCL_sources_prefixed} ${SRCTREE}/src/${source})
endforeach()

foreach(source ${lua_OpenCLHelper_sources})
    set( lua_OpenCLHelper_sources_prefixed ${lua_OpenCLHelper_sources_prefixed} ${SRCTREE}/${source})
endforeach()

add_library( LuaDeepCL SHARED LuaDeepCL_wrap.cxx ${lua_DeepCL_sources_prefixed} ${lua_OpenCLHelper_sources_prefixed} )
target_include_directories( LuaDeepCL PRIVATE ${LUA_INCLUDE_DIR})
target_include_directories( LuaDeepCL PRIVATE ${SRCTREE}/src ${SRCTREE}/qlearning ${SRCTREE}/OpenCLHelper )
SET_TARGET_PROPERTIES( LuaDeepCL PROPERTIES PREFIX "") 

if(DEV_RUN_SWIG)
find_package(SWIG REQUIRED)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/LuaDeepCL_wrap.cxx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/LuaDeepCL.i
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND swig -c++ -lua LuaDeepCL.i
)
add_dependencies( LuaDeepCL run_swig )
endif()



#INCLUDE(${SWIG_USE_FILE})
#SET_SOURCE_FILES_PROPERTIES(LuaDeepCL.i PROPERTIES CPLUSPLUS ON)
##SET_SOURCE_FILES_PROPERTIES(LuaTest.i PROPERTIES SWIG_FLAGS "-lua")

#SWIG_ADD_MODULE(LuaDeepCL lua LuaDeepCL.i ${DeepCL_sources_prefixed} ${OpenCLHelper_sources_prefixed})
#target_include_directories(LuaDeepCL PRIVATE ../src ../qlearning ../OpenCLHelper )
#target_include_directories(LuaDeepCL PRIVATE ${LUA_INCLUDE_DIR})

#SET_TARGET_PROPERTIES(LuaDeepCL PROPERTIES PREFIX "") 

# prevents errors when building from luarocks (though we need to set the USING_LUAROCKS
# define)
if( USING_LUAROCKS )
ADD_CUSTOM_TARGET(install 
)
endif()



