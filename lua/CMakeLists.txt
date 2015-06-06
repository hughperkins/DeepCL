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

set(dirs activate batch clmath conv dropout fc forcebackprop input layer loaders
   loss net netdef normalize patches pooling trainers util weights qlearning
   speedtemplates templated)
foreach(dir ${dirs})
    file( STRINGS ${SRCTREE}/src/${dir}/files.txt ${dir}_src)
#    message("${dir}_src ${${dir}_src}")
    foreach(source ${${dir}_src})
        set( lua_deepcl_sources ${lua_deepcl_sources} ${SRCTREE}/src/${dir}/${source})
    endforeach()
endforeach()
#message("deepcl_sources ${lua_deepcl_sources}")

set(lua_EasyCL_sources EasyCL/EasyCL.cpp
    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp EasyCL/CLWrapper.cpp
    EasyCL/CLKernel.cpp EasyCL/thirdparty/clew/src/clew.c)

foreach(source ${lua_EasyCL_sources})
    set( lua_EasyCL_sources_prefixed ${lua_EasyCL_sources_prefixed} ${SRCTREE}/${source})
endforeach()

add_library( LuaDeepCL SHARED LuaDeepCL_wrap.cxx ${lua_deepcl_sources} ${lua_EasyCL_sources_prefixed} )
target_include_directories( LuaDeepCL PRIVATE ${LUA_INCLUDE_DIR})
target_include_directories( LuaDeepCL PRIVATE ${SRCTREE}/src ${SRCTREE}/qlearning ${SRCTREE}/EasyCL )
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

#SWIG_ADD_MODULE(LuaDeepCL lua LuaDeepCL.i ${DeepCL_sources_prefixed} ${EasyCL_sources_prefixed})
#target_include_directories(LuaDeepCL PRIVATE ../src ../qlearning ../EasyCL )
#target_include_directories(LuaDeepCL PRIVATE ${LUA_INCLUDE_DIR})

#SET_TARGET_PROPERTIES(LuaDeepCL PROPERTIES PREFIX "") 

# prevents errors when building from luarocks (though we need to set the USING_LUAROCKS
# define)
if( USING_LUAROCKS )
ADD_CUSTOM_TARGET(install 
)
endif()



