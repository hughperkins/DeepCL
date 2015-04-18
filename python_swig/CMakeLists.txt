cmake_minimum_required(VERSION 2.8)

find_package(PythonLibs REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -Wall")

find_package(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})
SET_SOURCE_FILES_PROPERTIES(PyDeepCL.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(LuaTest.i PROPERTIES SWIG_FLAGS "-lua")
SET_SOURCE_FILES_PROPERTIES(example.i PROPERTIES SWIG_FLAGS "-includeall")


set( DeepCL_sources LayerMaker.cpp NeuralNetMould.cpp
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

set(OpenCLHelper_sources OpenCLHelper/OpenCLHelper.cpp
    OpenCLHelper/deviceinfo_helper.cpp OpenCLHelper/platforminfo_helper.cpp
    OpenCLHelper/CLKernel.cpp OpenCLHelper/thirdparty/clew/src/clew.c)

foreach(source ${DeepCL_sources})
    set( DeepCL_sources_prefixed ${DeepCL_sources_prefixed} ../src/${source})
endforeach()

foreach(source ${OpenCLHelper_sources})
    set( OpenCLHelper_sources_prefixed ${OpenCLHelper_sources_prefixed} ../${source})
endforeach()

set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR} )
SWIG_ADD_MODULE(PyDeepCL python PyDeepCL.i ${DeepCL_sources_prefixed} ${OpenCLHelper_sources_prefixed})
target_include_directories( _PyDeepCL PRIVATE ../src ../qlearning ../OpenCLHelper )
target_include_directories( _PyDeepCL PRIVATE ${PYTHON_INCLUDE_DIR})

SET_TARGET_PROPERTIES( _PyDeepCL PROPERTIES PREFIX "") 

ADD_CUSTOM_TARGET(install 
)



