CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)

OPTION(DEV_RUN_CYTHON "DeepCL maintainers only, otherwise set to 'OFF'." OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/../dist" CACHE PATH "Installation prefix, default 'dist'" FORCE
    )
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

mark_as_advanced(CLEAR PYTHON_LIBRARY)
mark_as_advanced(CLEAR PYTHON_INCLUDE_DIR)

include_directories(.)
include_directories(../EasyCL)
include_directories(../EasyCL/thirdparty/clew/include)
include_directories(../src)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

if(WIN32)
    link_libraries(winmm) # needed for timeGetTime
endif()

if(DEV_RUN_CYTHON)
    include(UseCython)
    set_source_files_properties(PyDeepCL.pyx PROPERTIES CYTHON_IS_CXX TRUE)
    cython_add_module(PyDeepCL PyDeepCL.pyx CyWrappers.cpp)
else()
    find_package( PythonLibs REQUIRED )
    include_directories(${PYTHON_INCLUDE_DIRS})
	python_add_module(PyDeepCL PyDeepCL.cxx CyWrappers.cpp)
	target_link_libraries(PyDeepCL ${PYTHON_LIBRARY})
endif()

if(WIN32)
#	SET(DEEPCL_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/DeepCL.lib ${CMAKE_INSTALL_PREFIX}/lib/EasyCL.lib)
else()
#	SET(DEEPCL_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}DeepCL${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

#target_link_libraries(PyDeepCL ${DEEPCL_LIBRARIES})
target_link_libraries(PyDeepCL DeepCL)

#install(PROGRAMS test_deepcl.py test_lowlevel.py test_qlearning.py
#    DESTINATION bin)
if(WIN32)
	install(TARGETS PyDeepCL
		RUNTIME DESTINATION bin
		ARCHIVE DESTINATION lib
		LIBRARY DESTINATION bin )
else()
	install(TARGETS PyDeepCL
		RUNTIME DESTINATION bin
		ARCHIVE DESTINATION lib
		LIBRARY DESTINATION lib )
endif()

