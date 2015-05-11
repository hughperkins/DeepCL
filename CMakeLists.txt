cmake_minimum_required(VERSION 2.8.7)

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
include_directories( EasyCL )
include_directories( EasyCL/clew/include )

if( ON_LINUX )
link_libraries(dl)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

if( ON_WINDOWS)
link_libraries(winmm) # needed for timeGetTime
endif()

set(dirs activate batch clmath conv dropout fc forcebackprop input layer loaders
   loss net netdef normalize patches pooling trainers util weights qlearning)
foreach(dir ${dirs})
    file( STRINGS src/${dir}/files.txt ${dir}_src)
#    message("${dir}_src ${${dir}_src}")
    foreach(source ${${dir}_src})
        set( deepcl_sources ${deepcl_sources} src/${dir}/${source})
    endforeach()
endforeach()
#message("deepcl_sources ${deepcl_sources}")

set( EasyCL_sources EasyCL/EasyCL.cpp
    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp EasyCL/CLWrapper.cpp
    EasyCL/CLKernel.cpp EasyCL/thirdparty/clew/src/clew.c )
add_library( EasyCL SHARED ${EasyCL_sources} )
link_libraries( EasyCL )

add_executable( gpuinfo EasyCL/gpuinfo.cpp EasyCL/thirdparty/clew/src/clew.c 
    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp )
if( ON_LINUX )
target_link_libraries( gpuinfo dl )
endif()

add_library( DeepCL SHARED ${deepcl_sources} )

execute_process( COMMAND gitx --version RESULT_VARIABLE git_test ERROR_QUIET )
if( ${git_test} EQUAL 0 )
    set( GIT_FOUND 1 )
endif()
if( GIT_FOUND )
message("git found")
endif()

if( GIT_FOUND )
add_custom_command( 
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/thirdparty/clew/src/clew.c ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/EasyCL.cpp ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/CLKernel.cpp ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL/CLWrapper.cpp
    COMMAND git submodule update --init --force --recursive
)
add_custom_target( git_submodules_update DEPENDS EasyCL/thirdparty/clew/src/clew.c )
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
    foreach( dir ${dirs} main)
        set(cog_dirs ${cog_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/src/${dir}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/${dir}/*.h)
    endforeach()
    add_custom_target(
        cog
        python ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/cog-batteries -r ${cog_dirs} ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h ${CMAKE_CURRENT_SOURCE_DIR}/prototyping/*.cpp
    )
    add_dependencies( DeepCL cog )
endif(DEV_RUN_COG)

set( UNITTEST_SOURCES test/testupdateweights.cpp test/testforward.cpp test/testfilehelper.cpp
 test/testsimpleconvolvenet.cpp test/testlogicaloperators.cpp 
 test/testbackward.cpp test/testsinglebatch.cpp 
 test/testpoolingforward.cpp test/testpoolingbackward.cpp test/testNorbLoader.cpp
 test/teststringhelper.cpp test/testGtestGlobals.cpp 
 src/util/stringhelper.cpp test/DimFromArgs.cpp test/testMemset.cpp test/WeightRandomizer.cpp
 test/testCopyBuffer.cpp test/CopyBuffer.cpp test/PrintBuffer.cpp test/testCopyBlock.cpp
 test/SpeedTemplates.cpp test/testSpeedTemplates.cpp test/testCopyLocal.cpp
 test/testNetdefToNet.cpp test/testactivationforward.cpp test/testactivationbackward.cpp
 test/testRandomSingleton.cpp test/testdropoutforward.cpp test/testdropoutbackward.cpp
 test/testsgd.cpp test/testCLMathWrapper.cpp test/testreducesegments.cpp
 )

add_executable( deepclrun src/main/deepclrun.cpp src/util/stringhelper.cpp )

add_executable( cifar-to-mat test/CifarToMat.cpp src/util/stringhelper.cpp test/CifarLoader.cpp )
add_executable( prepare-norb test/prepare-norb.cpp src/util/stringhelper.cpp )

if( LIBPNGPP_AVAILABLE)
    add_definitions(-DPNG_AVAILABLE)
    add_executable( testpatchextractor test/testPatchExtractor.cpp src/util/stringhelper.cpp src/PatchExtractor.cpp )
    add_executable( testtranslator test/testTranslator.cpp src/util/stringhelper.cpp src/Translator.cpp )
    add_executable( testgenericloader test/testGenericLoader.cpp src/util/stringhelper.cpp )
endif(LIBPNGPP_AVAILABLE)

if( ON_LINUX )
add_library( gtest SHARED thirdparty/gtest/gtest-all.cc )
target_link_libraries( gtest pthread)
else()
add_library( gtest thirdparty/gtest/gtest-all.cc )
endif()
include_directories( thirdparty/gtest )

add_executable( unittests ${UNITTEST_SOURCES} test/gtest_main.cpp test/GtestGlobals.cpp
    test/TestArgsParser.cpp )
target_link_libraries( unittests gtest )
# target_include_directories( unittests PRIVATE thirdparty/gtest )

if( PROTOTYPING )
    add_executable( clconvolve-fixedweights prototyping/clconvolve-fixedweights.cpp src/util/stringhelper.cpp )
    add_executable( learn_scenario_image prototyping/qlearning/learnScenarioImage.cpp 
        prototyping/qlearning/ScenarioImage.cpp )
endif( PROTOTYPING )

target_link_libraries( unittests DeepCL )

add_executable( testgtestsupp test/testgtestsupp.cpp thirdparty/gtest/gtest_main.cc )
target_link_libraries( testgtestsupp gtest )
# target_include_directories( testgtestsupp PRIVATE thirdparty/gtest )

OPTION(BUILD_LUA_WRAPPERS "Builds Lua wrappers" OFF)

if( BUILD_LUA_WRAPPERS )
add_subdirectory(lua)
endif()


