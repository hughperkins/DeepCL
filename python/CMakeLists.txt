CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/../dist" CACHE PATH "Installation prefix, default 'dist'" FORCE
    )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

include(UseCython)

include_directories(.)
include_directories(../EasyCL)
include_directories(../EasyCL/thirdparty/clew/include)
include_directories(../src)

if(UNIX)
#    target_link_libraries(EasyCL dl)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

if(WIN32)
    link_libraries(winmm) # needed for timeGetTime
endif()

set_source_files_properties(PyDeepCL.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(PyDeepCL PyDeepCL.pyx CyWrappers.cpp)

if(WIN32)
	SET(DEEPCL_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/DeepCL.lib)
else()
	SET(DEEPCL_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}DeepCL${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

target_link_libraries(PyDeepCL ${DEEPCL_LIBRARIES})

