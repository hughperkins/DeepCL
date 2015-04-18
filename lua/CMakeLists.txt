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

find_package(SWIG QUIET)
# INCLUDE(${SWIG_USE_FILE})

if(SWIG_EXECUTABLE AND ON_LINUX)
    OPTION(DEV_RUN_SWIG "Re-runs swig.  Only needed for development.  Linux only." ON)
else()
    OPTION(DEV_RUN_SWIG "Re-runs swig.  Only needed for development." OFF)
endif()

set( lua_DeepCL_sources LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
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
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp Propagate3_unfactorized.cpp
    PoolingBackpropGpuNaive.cpp ../qlearning/QLearner.cpp ../qlearning/array_helper.cpp
    ForceBackpropLayerMaker.cpp ForceBackpropLayer.cpp MnistLoader.cpp
    OnDemandBatcher.cpp Batcher.cpp NetAction.cpp
 )

set(lua_OpenCLHelper_sources OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c)

foreach(source ${lua_DeepCL_sources})
    set( lua_DeepCL_sources_prefixed ${lua_DeepCL_sources_prefixed} ${CMAKE_CURRENT_SOURCE_DIR}/../src/${source})
endforeach()

foreach(source ${lua_OpenCLHelper_sources})
    set( lua_OpenCLHelper_sources_prefixed ${lua_OpenCLHelper_sources_prefixed} ${CMAKE_CURRENT_SOURCE_DIR}/../${source})
endforeach()

add_library( LuaDeepCL SHARED LuaDeepCL_wrap.cxx ${lua_DeepCL_sources_prefixed} ${lua_OpenCLHelper_sources_prefixed} )
target_include_directories( LuaDeepCL PRIVATE ${LUA_INCLUDE_DIR})
target_include_directories( LuaDeepCL PRIVATE ../src ../qlearning ../OpenCLHelper )
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

if( FROM_LUAROCKS )
ADD_CUSTOM_TARGET(install 
)
endif()



