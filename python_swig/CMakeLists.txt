cmake_minimum_required(VERSION 2.8)

find_package(PythonLibs REQUIRED)

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

set( pyswig_DeepCL_sources 
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
 ../qlearning/QLearner.cpp ../qlearning/array_helper.cpp
 )

set(pyswig_OpenCLHelper_sources OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c)

foreach(source ${pyswig_DeepCL_sources})
    set( pyswig_DeepCL_sources_prefixed ${pyswig_DeepCL_sources_prefixed} ../src/${source})
endforeach()

foreach(source ${pyswig_OpenCLHelper_sources})
    set( pyswig_OpenCLHelper_sources_prefixed ${pyswig_OpenCLHelper_sources_prefixed} ../${source})
endforeach()

#SET_SOURCE_FILES_PROPERTIES(PyDeepCL.i PROPERTIES SWIG_FLAGS "-o;${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCLPYTHON_wrap.cxx")
#set(SWIGOPT_O ${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCLPYTHON_wrap.cxx)
#SET_SOURCE_FILES_PROPERTIES(PyDeepCL.i PROPERTIES CPLUSPLUS ON)
#message("swigopt_o ${SWIGOPT_O}")
# SET_PROPERTY(SOURCE PyDeepCL.i PROPERTY SWIG_FLAGS "-o" "${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCLPYTHON_wrap.cxx")
# SET(CMAKE_SWIG_FLAGS "-o ${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCLPYTHON_wrap.cxx")
# SET_SOURCE_FILES_PROPERTIES(example.i PROPERTIES SWIG_FLAGS "-includeall")
#set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR} )
#SWIG_ADD_MODULE(PyDeepCL python PyDeepCL.i ${DeepCL_sources_prefixed} ${OpenCLHelper_sources_prefixed})
#target_include_directories( _PyDeepCL PRIVATE ../src ../qlearning ../OpenCLHelper )
#target_include_directories( _PyDeepCL PRIVATE ${PYTHON_INCLUDE_DIR})

add_library( _PyDeepCL SHARED PyDeepCL_wrap.cxx ${pyswig_DeepCL_sources_prefixed} ${pyswig_OpenCLHelper_sources_prefixed} )
target_include_directories( _PyDeepCL PRIVATE ${PYTHON_INCLUDE_DIR})
target_include_directories( _PyDeepCL PRIVATE ../src ../qlearning ../OpenCLHelper )
SET_TARGET_PROPERTIES( _PyDeepCL PROPERTIES PREFIX "") 

if(DEV_RUN_SWIG)
find_package(SWIG REQUIRED)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCL_wrap.cxx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/PyDeepCL.i
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND swig -c++ -python PyDeepCL.i
)
add_dependencies( _PyDeepCL run_swig )
endif()


#ADD_CUSTOM_TARGET(install 
#)



