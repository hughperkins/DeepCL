cmake_minimum_required(VERSION 2.8.7)
project(DeepCL)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(ON_LINUX 1)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  set(ON_WINDOWS 1)
endif()

if (NOT CMAKE_BUILD_TYPE)
    message("Setting build type to 'RelWithDebInfo'")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

#OPTION(BUILD_PYTHON_WRAPPERS "Build Python wrappers.  Needs Python." ON)
OPTION(BUILD_JPEG_SUPPORT "Allows native loading of jpegs, via manifest file." ON)
OPTION(BUILD_INTERNAL_LUA "If using from Lua, set to 'OFF'" ON)
OPTION(MAINTAINER_OPTIONS "Show maintainer options" OFF)

if(MAINTAINER_OPTIONS)
    OPTION(BUILD_PYTHON_WRAPPERS "Build python wrappers.  Maintainers only." OFF.)
    OPTION(DEV_RUN_COG "DeepCL maintainers only, otherwise set to 'OFF'." OFF)
    OPTION(PROTOTYPING "Only for devs." OFF)
#    mark_as_advanced(FORCE PROTOTYPING)
    OPTION(LIBPNGPP_AVAILABLE "Some toys/samples only." OFF)
#    mark_as_advanced(FORCE LIBPNGPP_AVAILABLE)
else()
    unset(DEV_RUN_COG CACHE)
    unset(DEV_RUN_CYTHON CACHE)
    unset(BUILD_PYTHON_WRAPPERS CACHE)
    unset(LIBPNGPP_AVAILABLE CACHE)
    unset(PROTOTYPING CACHE)
endif()

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH "Installation prefix, default 'dist'" FORCE
   )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# remove old flags from the cache:
unset(LIBJPEG_AVAILABLE CACHE)
unset(PROVIDE_LUA_ENGINE CACHE)
unset(BUILD_PYSWIG_WRAPPERS CACHE)
unset(BUILD_LUA_WRAPPERS CACHE)
unset(RUN_COG CACHE)
unset(RUN_SWIG CACHE)
unset(DEV_RUN_SWIG CACHE)
unset(PYTHON_AVAILABLE CACHE)
unset(LUA51_AVAILABLE CACHE)
unset(SWIG_DIR CACHE)
unset(SWIG_EXECUTABLE CACHE)
unset(SWIG_VERSION CACHE)
unset(LUA_INCLUDE_DIR CACHE)

if(BUILD_JPEG_SUPPORT)
  set(LIBJPEG_AVAILABLE 1)
  add_definitions(-DLIBJPEG_FOUND)
endif(BUILD_JPEG_SUPPORT)

# SET(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE STRING "Installation directory." FORCE)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

INCLUDE("${CMAKE_MODULE_PATH}/build_EasyCL.cmake")
INCLUDE_DIRECTORIES(${EASYCL_INCLUDE_DIRS})
#LINK_DIRECTORIES(${EASYCL_LIBRARIES})

INCLUDE("${CMAKE_MODULE_PATH}/build_clBLAS.cmake")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
## TARGET_link_LIBRARIES(clBLAS ${EASYCL_LIBRARIES})

ADD_DEPENDENCIES(clBLAS-external EasyCL)

include_directories(.)
include_directories(src)
include_directories(qlearning)
# include_directories(test)

if(ON_LINUX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
endif()

if(ON_WINDOWS)
    link_libraries(winmm) # needed for timeGetTime
endif()

set(dirs clblas activate batch clmath conv dropout fc forcebackprop input layer loaders
   loss net netdef normalize patches pooling trainers util weights qlearning
   )
foreach(dir ${dirs})
    file(STRINGS src/${dir}/files.txt ${dir}_src)
#    message("${dir}_src ${${dir}_src}")
    foreach(source ${${dir}_src})
        set(deepcl_sources ${deepcl_sources} src/${dir}/${source})
    endforeach()
endforeach()
set(deepcl_sources ${deepcl_sources} src/DeepCL.cpp src/CppRuntimeBoundary.cpp)
#message("deepcl_sources ${deepcl_sources}")

#find_package(Lua51)
#include_directories(${LUA_INCLUDE_DIR})
#link_libraries(${LUA_LIBRARIES})

#include_directories(EasyCL/thirdparty/lua-5.1.5/src)

#file(STRINGS EasyCL/thirdparty/lua-5.1.5/files.txt lua_src1)
#foreach(source ${lua_src1})
#    set(lua_src ${lua_src} EasyCL/thirdparty/lua-5.1.5/src/${source})
#endforeach()

#set(EasyCL_sources EasyCL/EasyCL.cpp
#    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp EasyCL/CLWrapper.cpp EasyCL/util/easycl_stringhelper.cpp
#    EasyCL/templates/TemplatedKernel.cpp EasyCL/templates/LuaTemplater.cpp
#    EasyCL/CLKernel.cpp EasyCL/thirdparty/clew/src/clew.c
#    ${lua_src})
#add_library(EasyCL SHARED ${EasyCL_sources})
#link_libraries(EasyCL)
#target_link_libraries(EasyCL ${LUA_LIBRARIES})

#add_executable(gpuinfo EasyCL/gpuinfo.cpp EasyCL/thirdparty/clew/src/clew.c 
#    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp)
#if(ON_LINUX)
#    target_link_libraries(gpuinfo dl)
#endif()

if(LIBJPEG_AVAILABLE)
    find_package(JPEG REQUIRED)
    include_directories(${JPEG_INCLUDE_DIR})
    set(deepcl_sources ${deepcl_sources} src/util/JpegHelper.cpp src/loaders/ManifestLoaderv1.cpp)
endif(LIBJPEG_AVAILABLE)

add_library(DeepCL SHARED ${deepcl_sources})
if(ON_WINDOWS)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif(ON_WINDOWS)

target_link_libraries(DeepCL EasyCL)
target_link_libraries(DeepCL clBLAS)
if(LIBJPEG_AVAILABLE)
    target_link_libraries(DeepCL ${JPEG_LIBRARY})
endif(LIBJPEG_AVAILABLE)


#if(ON_LINUX)
#    execute_process(COMMAND git --version RESULT_VARIABLE git_test ERROR_QUIET)
#    if(${git_test} EQUAL 0)
#        set(GIT_FOUND 1)
#    endif()
#    if(GIT_FOUND)
#        add_custom_command(
#            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/thirdparty/clew/src/clew.c ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/EasyCL.cpp ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/CLKernel.cpp ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/CLWrapper.cpp
#            COMMAND git submodule update --init --force --recursive
#        )
#        add_custom_target(git_submodules_update DEPENDS EasyCL/thirdparty/clew/src/clew.c)
#        add_dependencies(DeepCL git_submodules_update)
#    endif()
#endif(ON_LINUX)

if(LIBPNGPP_AVAILABLE)
    find_package(PNG REQUIRED)
endif(LIBPNGPP_AVAILABLE)

#link_libraries(DeepCL)

if(DEV_RUN_COG)
    foreach(dir ${dirs} main)
        set(cog_dirs ${cog_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/src/${dir}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/${dir}/*.h)
    endforeach()
    add_custom_target(
        cog
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${cog_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(DeepCL cog)
endif(DEV_RUN_COG)

#ADD_DEPENDENCIES(DeepCL clBLAS)

set(UNITTEST_SOURCES test/testClBlas.cpp
 test/testDeepCL.cpp
 test/testupdateweights.cpp test/testforward.cpp test/testfilehelper.cpp
 test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackward.cpp test/testsinglebatch.cpp 
 test/testpoolingforward.cpp test/testpoolingbackward.cpp test/testNorbLoader.cpp
 test/teststringhelper.cpp test/testGtestGlobals.cpp 
 src/util/stringhelper.cpp test/DimFromArgs.cpp test/testMemset.cpp test/WeightRandomizer.cpp
 test/testCopyBuffer.cpp test/CopyBuffer.cpp test/PrintBuffer.cpp test/testCopyBlock.cpp
 test/testCopyLocal.cpp
 test/testNetdefToNet.cpp test/testactivationforward.cpp test/testactivationbackward.cpp
 test/testRandomSingleton.cpp test/testdropoutforward.cpp test/testdropoutbackward.cpp
 test/testsgd.cpp test/testCLMathWrapper.cpp test/testreducesegments.cpp
 test/NetTestHelper.cpp test/testGpuOp.cpp
 test/testdroplayer.cpp
)
if(LIBJPEG_AVAILABLE)
    set(UNITTEST_SOURCES ${UNITTEST_SOURCES} test/testjpeghelper.cpp)
    add_executable(mnist-to-jpegs test/mnist-to-jpegs.cpp src/util/stringhelper.cpp src/loaders/MnistLoader.cpp)
    target_link_libraries(mnist-to-jpegs DeepCL)
endif(LIBJPEG_AVAILABLE)

add_executable(deepcl_train src/main/train.cpp src/util/stringhelper.cpp)
add_executable(deepcl_predict src/main/predict.cpp src/util/stringhelper.cpp)

add_executable(cifar-to-mat test/CifarToMat.cpp src/util/stringhelper.cpp test/CifarLoader.cpp)
add_executable(prepare-norb test/prepare-norb.cpp src/util/stringhelper.cpp)
add_executable(mnist-to-floats test/mnist-to-floats.cpp src/util/stringhelper.cpp)
add_executable(mnist-to-pipe test/mnist-to-pipe.cpp src/util/stringhelper.cpp)

foreach(exe deepcl_train deepcl_predict cifar-to-mat prepare-norb mnist-to-floats mnist-to-pipe)
    target_link_libraries(${exe} DeepCL)
endforeach()

#target_link_libraries(cifar-to-mat ${LUA_LIBRARIES})

if(LIBPNGPP_AVAILABLE)
    add_definitions(-DPNG_AVAILABLE)
    add_executable(testpatchextractor test/testPatchExtractor.cpp src/util/stringhelper.cpp src/patches/PatchExtractor.cpp)
    add_executable(testtranslator test/testTranslator.cpp src/util/stringhelper.cpp src/patches/Translator.cpp)
    add_executable(testgenericloader test/testGenericLoader.cpp src/util/stringhelper.cpp)

    foreach(exe testpatchextractor testtranslator testgenericloader)
        target_link_libraries(${exe} DeepCL ${PNG_LIBRARY})
    endforeach()
endif(LIBPNGPP_AVAILABLE)

if(ON_LINUX)
    add_library(deepcl_gtest SHARED thirdparty/gtest/gtest-all.cc)
    target_link_libraries(deepcl_gtest pthread)
else()
    add_library(deepcl_gtest thirdparty/gtest/gtest-all.cc)
endif()
include_directories(thirdparty/gtest)

add_executable(deepcl_unittests ${UNITTEST_SOURCES} test/gtest_main.cpp test/DeepCLGtestGlobals.cpp
    test/TestArgsParser.cpp)
target_link_libraries(deepcl_unittests deepcl_gtest)

if(PROTOTYPING)
    add_executable(clconvolve-fixedweights prototyping/clconvolve-fixedweights.cpp src/util/stringhelper.cpp)
    add_executable(learn_scenario_image prototyping/qlearning/learnScenarioImage.cpp 
        prototyping/qlearning/ScenarioImage.cpp)
    target_link_libraries(clconvolve-fixedweights DeepCL)
    target_link_libraries(learn_scenario_image DeepCL)
endif(PROTOTYPING)

target_link_libraries(deepcl_unittests DeepCL)

if(LIBPNGPP_AVAILABLE)
    target_link_libraries(deepcl_unittests ${PNG_LIBRARY})
endif(LIBPNGPP_AVAILABLE)

add_executable(testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc)
target_link_libraries(testgtestsupp deepcl_gtest)

if(BUILD_PYTHON_WRAPPERS)
    add_subdirectory(python)
#    add_dependencies(PyDeepCL DeepCL)
#    target_link_libraries(PyDeepCL DeepCL)
endif()

INSTALL(DIRECTORY src/ DESTINATION include/deepcl FILES_MATCHING PATTERN *.h)
INSTALL(PROGRAMS src/activate.sh DESTINATION bin)
INSTALL(PROGRAMS src/activate.bat DESTINATION bin)
#INSTALL(DIRECTORY EasyCL/ DESTINATION include/easycl FILES_MATCHING PATTERN *.h)
INSTALL(TARGETS DeepCL deepcl_train deepcl_predict deepcl_unittests deepcl_gtest mnist-to-floats
        mnist-to-pipe cifar-to-mat
    EXPORT DeepCLTargets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
if(LIBJPEG_AVAILABLE)
    INSTALL(TARGETS mnist-to-jpegs
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
endif()

# Generate the project config file.
export(TARGETS DeepCL
  FILE "${PROJECT_BINARY_DIR}/DeepCLTargets.cmake")
export(PACKAGE DeepCL)

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/DeepCL)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR})
 
file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}"
   "${CMAKE_INSTALL_PREFIX}/include")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src/"
  "${PROJECT_SOURCE_DIR}/EasyCL/"
  "${PROJECT_SOURCE_DIR}/EasyCL/thirdparty/clew/include")
set(ABS_LIBRARY_FILE "${PROJECT_BINARY_DIR}/libDeepCL.so" 
  "${PROJECT_BINARY_DIR}/EasyCL/src/EasyCL-external-build/libEasyCL.so")
if(APPLE)
  set(ABS_LIBRARY_FILE "${PROJECT_BINARY_DIR}/libDeepCL.dylib" 
    "${PROJECT_BINARY_DIR}/EasyCL/src/EasyCL-external-build/libEasyCL.dylib")
endif(APPLE)
configure_file(DeepCLConfig.cmake.in
  "${PROJECT_BINARY_DIR}/DeepCLConfig.cmake" @ONLY)
set(CONF_INCLUDE_DIRS "\${DEEPCL_CMAKE_DIR}/${REL_INCLUDE_DIR}"
  "\${DEEPCL_CMAKE_DIR}/${REL_INCLUDE_DIR}/deepcl"
  "\${DEEPCL_CMAKE_DIR}/${REL_INCLUDE_DIR}/easycl")
set(ABS_LIBRARY_FILE "DeepCL" "EasyCL")
configure_file(DeepCLConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DeepCLConfig.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}/DeepCLTargets.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DeepCLConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
