cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)

include_directories( . )
include_directories( OpenCLHelper )
include_directories( OpenCLHelper/clew/include )
link_libraries(dl)

add_library( ClConvolve SHARED ClConvolve.h OpenCLHelper/OpenCLHelper.cpp
   OpenCLHelper/CLKernel.cpp OpenCLHelper/clew/src/clew.c )

file( COPY ClConvolve.cl DESTINATION ${CMAKE_BINARY_DIR} )

find_package(PNG)
link_libraries(${PNG_LIBRARY})

#set( OPENCLHELPER $ENV{HOME}/git/OpenCLHelper)
add_executable( testarraysquare test/testarraysquare.cpp OpenCLHelper/OpenCLHelper.cpp
   OpenCLHelper/CLKernel.cpp OpenCLHelper/clew/src/clew.c )
#add_executable( testopencl test/testopencl.cpp )
add_executable( testboardshelper test/testboardshelper.cpp )


